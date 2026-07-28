package com.magma.sylc.quest

import android.util.Log
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.Inet4Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.nio.ByteBuffer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Finds the PC on the local network so the address never has to be typed.
 *
 * No sender change was needed for this: WifiTransport already answers any
 * well-formed PT_HELLO with a unicast PT_HELLO_ACK sent back to the datagram's
 * source. Broadcasting a HELLO therefore makes the sender reveal its address,
 * and the reply's source is the answer.
 *
 * The sender binds 0.0.0.0, so the discovered host is equally valid for the TCP
 * media connection. When the session runs over USB-C there is no UDP listener
 * and discovery simply finds nothing -- which costs nothing, because that path
 * uses 127.0.0.1 through `adb reverse` anyway.
 */
object CastDiscovery {

    private const val TAG = "CastDiscovery"
    private const val REPLY_BUFFER_BYTES = 2048

    data class Result(val host: String, val port: Int)

    /**
     * Broadcasts a HELLO and returns the first sender that answers.
     * Returns null on timeout. Safe to call repeatedly.
     */
    suspend fun discover(
        port: Int = CastSettings.DEFAULT_PORT,
        timeoutMs: Int = DEFAULT_TIMEOUT_MS,
    ): Result? = withContext(Dispatchers.IO) {
        val targets = broadcastAddresses()
        if (targets.isEmpty()) {
            Log.i(TAG, "No broadcast-capable interface; skipping discovery")
            return@withContext null
        }

        var socket: DatagramSocket? = null
        try {
            socket = DatagramSocket().apply {
                broadcast = true
                soTimeout = timeoutMs
            }
            val hello = SyLcProtocol.packHelloDatagram(seq = 1)
            for (target in targets) {
                try {
                    socket.send(DatagramPacket(hello, hello.size, InetSocketAddress(target, port)))
                    Log.i(TAG, "Discovery HELLO -> ${target.hostAddress}:$port")
                } catch (error: Exception) {
                    Log.d(TAG, "Broadcast to ${target.hostAddress} failed", error)
                }
            }

            val deadline = System.nanoTime() + timeoutMs * 1_000_000L
            val buffer = ByteArray(REPLY_BUFFER_BYTES)
            while (System.nanoTime() < deadline) {
                val reply = DatagramPacket(buffer, buffer.size)
                socket.receive(reply)   // throws SocketTimeoutException at soTimeout
                if (!isHelloAck(buffer, reply.length)) continue
                val host = reply.address?.hostAddress ?: continue
                Log.i(TAG, "Sender discovered at $host:$port")
                return@withContext Result(host, port)
            }
            null
        } catch (error: Exception) {
            Log.i(TAG, "Discovery found no sender (${error.javaClass.simpleName})")
            null
        } finally {
            try {
                socket?.close()
            } catch (_: Exception) {
                // Best effort; the socket is ours alone and short-lived.
            }
        }
    }

    private fun isHelloAck(buffer: ByteArray, length: Int): Boolean {
        if (length < SyLcProtocol.HEADER_SIZE) return false
        return try {
            val header = SyLcProtocol.parseHeader(ByteBuffer.wrap(buffer, 0, length))
            header.magic.contentEquals(SyLcProtocol.MAGIC) &&
                header.ver == SyLcProtocol.VER &&
                header.type == SyLcProtocol.PT_HELLO_ACK
        } catch (_: Exception) {
            false
        }
    }

    /** Every IPv4 broadcast address this headset can currently reach. */
    private fun broadcastAddresses(): List<InetAddress> {
        val addresses = mutableListOf<InetAddress>()
        try {
            for (nic in NetworkInterface.getNetworkInterfaces()) {
                if (!nic.isUp || nic.isLoopback) continue
                for (binding in nic.interfaceAddresses) {
                    val broadcast = binding.broadcast ?: continue
                    if (binding.address is Inet4Address) addresses.add(broadcast)
                }
            }
        } catch (error: Exception) {
            Log.w(TAG, "Could not enumerate interfaces", error)
        }
        // Fall back to the limited broadcast address when an interface reports
        // no broadcast of its own (some Quest Wi-Fi configurations).
        if (addresses.isEmpty()) {
            try {
                addresses.add(InetAddress.getByName("255.255.255.255"))
            } catch (_: Exception) {
                // Nothing more to try.
            }
        }
        return addresses
    }

    const val DEFAULT_TIMEOUT_MS = 1_200
}
