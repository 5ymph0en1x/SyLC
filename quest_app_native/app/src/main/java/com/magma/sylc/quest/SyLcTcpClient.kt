package com.magma.sylc.quest

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.ByteBuffer

/**
 * Reconnect delay policy, kept separate from the socket so it can be reasoned
 * about (and tested) on its own: back off while failures are immediate, and
 * reset once a connection has proved it can live, so a link that drops after an
 * hour reconnects promptly instead of inheriting an old penalty.
 */
class ReconnectBackoff(
    private val initialDelayMs: Long = 250L,
    private val maxDelayMs: Long = 3_000L,
    private val stableConnectionNs: Long = 5_000_000_000L,
) {
    var delayMs: Long = initialDelayMs
        private set

    /** Records how long the attempt lasted; returns how long to wait now. */
    fun onSessionEnded(livedNs: Long): Long {
        val stable = livedNs >= stableConnectionNs
        val waitMs = if (stable) initialDelayMs else delayMs
        delayMs = if (stable) {
            initialDelayMs
        } else {
            (delayMs * 2).coerceAtMost(maxDelayMs)
        }
        return waitMs
    }

    fun reset() {
        delayMs = initialDelayMs
    }
}

class SyLcTcpClient(
    private val host: String,
    private val port: Int,
    private val onVideoFrame: (
        ptsMs: Long,
        isKeyframe: Boolean,
        payload: ByteArray,
        offset: Int,
        length: Int
    ) -> Unit,
    private val onAudioFrame: ((
        ptsMs: Long,
        payload: ByteArray,
        offset: Int,
        length: Int
    ) -> Unit)? = null,
    private val feedbackProvider: (() -> SyLcProtocol.BandwidthFeedback)? = null,
    private val onStreamFormat: ((SyLcProtocol.StreamFormat) -> Unit)? = null,
    private val onConnected: () -> Unit = {},
    private val onConnectionError: (Throwable) -> Unit = {}
) {
    private var socket: Socket? = null
    private var job: Job? = null
    private var seqCounter = 0L
    @Volatile private var disconnectRequested = false
    @Volatile var lastError: Throwable? = null
        private set
    private val TAG = "SyLcTcpClient"
    
    // Zero-allocation buffers
    private val receiveBuffer = ByteArray(MAX_PACKET_BYTES)
    private val reassemblyBuffer = ByteArray(MAX_ACCESS_UNIT_BYTES)
    private var reassemblySize = 0
    
    suspend fun connect() {
        disconnectRequested = false
        lastError = null
        val ownerJob = currentCoroutineContext()[Job]
        job = ownerJob
        val backoff = ReconnectBackoff(
            RECONNECT_INITIAL_DELAY_MS, RECONNECT_MAX_DELAY_MS, STABLE_CONNECTION_NS
        )

        try {
            withContext(Dispatchers.IO) {
                while (isActive && !disconnectRequested) {
                    val connectedAtNs = System.nanoTime()
                    try {
                        runConnectedSession()
                    } catch (error: CancellationException) {
                        throw error
                    } catch (error: Exception) {
                        if (!disconnectRequested) {
                            lastError = error
                            Log.w(
                                TAG,
                                "Connection interrupted; retrying in ${backoff.delayMs}ms",
                                error
                            )
                            onConnectionError(error)
                        }
                    } finally {
                        closeSocketQuietly()
                    }

                    if (!isActive || disconnectRequested) break
                    delay(backoff.onSessionEnded(System.nanoTime() - connectedAtNs))
                }
            }
        } finally {
            closeSocketQuietly()
            if (job === ownerJob) job = null
            Log.i(TAG, "Disconnected.")
        }
    }

    private suspend fun runConnectedSession() = coroutineScope {
        Log.i(TAG, "Connecting to $host:$port")
        val activeSocket = Socket().apply {
            tcpNoDelay = true
            receiveBufferSize = SOCKET_RECEIVE_BUFFER_BYTES
            keepAlive = true
            connect(
                InetSocketAddress(host, this@SyLcTcpClient.port),
                CONNECT_TIMEOUT_MS
            )
        }
        socket = activeSocket
        val inputStream = DataInputStream(activeSocket.getInputStream())
        val outputStream = DataOutputStream(activeSocket.getOutputStream())

        Log.i(TAG, "TCP connected; sending HELLO.")
        outputStream.write(SyLcProtocol.packHello(++seqCounter))
        outputStream.flush()
        var handshakeComplete = false

        fun completeHandshake() {
            if (handshakeComplete) return
            handshakeComplete = true
            lastError = null
            Log.i(TAG, "SyLC handshake complete")
            onConnected()
        }

        // One outbound coroutine owns DataOutputStream after HELLO. It carries
        // decoder-pressure feedback and doubles as the connection heartbeat.
        launch {
            while (isActive && !disconnectRequested && !activeSocket.isClosed) {
                delay(FEEDBACK_INTERVAL_MS)
                val feedback = feedbackProvider?.invoke()
                val packet = if (feedback != null) {
                    SyLcProtocol.packBandwidthFeedback(++seqCounter, feedback)
                } else {
                    SyLcProtocol.packControl("keepalive", ++seqCounter)
                }
                outputStream.write(packet)
                outputStream.flush()
            }
        }

        // Read loop. EOF/reset propagates out of this session, cancels the
        // feedback child, closes this exact socket, then enters bounded retry.
        while (isActive && !disconnectRequested) {
            val frameLength = inputStream.readInt()
            if (frameLength > receiveBuffer.size || frameLength <= 0) {
                throw IllegalStateException("Invalid protocol frame length: $frameLength")
            }

            inputStream.readFully(receiveBuffer, 0, frameLength)
            val buffer = ByteBuffer.wrap(receiveBuffer, 0, frameLength)
            val header = SyLcProtocol.parseHeader(buffer)
            val payloadLength = header.length.toInt()
            val payloadOffset = buffer.position()

            if (!header.magic.contentEquals(SyLcProtocol.MAGIC) ||
                header.ver != SyLcProtocol.VER ||
                payloadLength < 0 ||
                payloadOffset + payloadLength != frameLength
            ) {
                throw IllegalStateException("Malformed protocol frame")
            }

            when (header.type) {
                SyLcProtocol.PT_VIDEO -> {
                    // Receiving media is sufficient proof when interoperating
                    // with a sender predating explicit HELLO_ACK support.
                    completeHandshake()
                    if (header.fragCnt == 1 && header.fragIdx == 0) {
                        onVideoFrame(
                            header.ptsMs,
                            (header.flags and 1) != 0,
                            receiveBuffer,
                            payloadOffset,
                            payloadLength
                        )
                        continue
                    }

                    if (header.fragIdx == 0) reassemblySize = 0
                    if (reassemblySize + payloadLength > reassemblyBuffer.size) {
                        reassemblySize = 0
                        throw IllegalStateException("Video reassembly buffer overflow")
                    }
                    System.arraycopy(
                        receiveBuffer,
                        payloadOffset,
                        reassemblyBuffer,
                        reassemblySize,
                        payloadLength
                    )
                    reassemblySize += payloadLength

                    if (header.fragIdx == header.fragCnt - 1 && reassemblySize > 0) {
                        onVideoFrame(
                            header.ptsMs,
                            (header.flags and 1) != 0,
                            reassemblyBuffer,
                            0,
                            reassemblySize
                        )
                        reassemblySize = 0
                    }
                }
                SyLcProtocol.PT_AUDIO -> {
                    completeHandshake()
                    onAudioFrame?.invoke(
                        header.ptsMs,
                        receiveBuffer,
                        payloadOffset,
                        payloadLength
                    )
                }
                SyLcProtocol.PT_CONTROL -> {
                    val controlPayload = String(
                        receiveBuffer,
                        payloadOffset,
                        payloadLength
                    )
                    Log.d(TAG, "Received PT_CONTROL: $controlPayload")
                }
                SyLcProtocol.PT_HELLO_ACK -> {
                    // The sender describes what it is really streaming here.
                    // An empty payload means an older sender: the parser
                    // then yields the historical 3840x1080 / 24 fps values.
                    onStreamFormat?.invoke(
                        SyLcProtocol.parseStreamFormat(
                            receiveBuffer, payloadOffset, payloadLength
                        )
                    )
                    completeHandshake()
                }
                SyLcProtocol.PT_BYE -> throw java.io.EOFException("Server ended the stream")
            }
        }
    }

    private fun closeSocketQuietly() {
        val activeSocket = socket
        socket = null
        try {
            activeSocket?.close()
        } catch (_: Exception) {
            // Closing is best-effort; the retry loop owns the next socket.
        }
    }

    fun disconnect() {
        disconnectRequested = true
        closeSocketQuietly()
        job?.cancel()
        job = null
    }

    companion object {
        private const val MAX_PACKET_BYTES = 12 * 1024 * 1024
        private const val MAX_ACCESS_UNIT_BYTES = 12 * 1024 * 1024
        // Enough kernel receive window for several 500-Mbps/24-fps access
        // units without expanding the Java AU pool.
        private const val SOCKET_RECEIVE_BUFFER_BYTES = 8 * 1024 * 1024
        private const val FEEDBACK_INTERVAL_MS = 500L
        private const val CONNECT_TIMEOUT_MS = 2_000
        private const val RECONNECT_INITIAL_DELAY_MS = 250L
        private const val RECONNECT_MAX_DELAY_MS = 3_000L
        private const val STABLE_CONNECTION_NS = 5_000_000_000L
    }
}
