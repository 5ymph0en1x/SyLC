package com.magma.sylc.quest

import android.util.Log
import android.view.Surface
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch

/**
 * One receiving session: transport, decoder and audio, wired together once.
 *
 * The flat and immersive screens used to each own a copy of this wiring --
 * acquire an access unit from the pool, bounds-check it, copy, queue, feed PCM,
 * answer bandwidth feedback, apply the announced stream format, start audio at
 * the right moment. Two copies of a real-time pipeline drift: a fix or a tuning
 * applied to one silently misses the other. The mechanism lives here now, and
 * the screens only say what to display through [Listener].
 *
 * Threading matches the pieces it owns: [Listener] callbacks arrive on the
 * transport reader or a MediaCodec callback thread, never on the main thread.
 * Callers that touch views must post them themselves, exactly as before.
 */
class CastSession(
    private val settings: CastSettings,
    private val listener: Listener,
) {

    interface Listener {
        /** The very first access unit of a session, before it is decoded. */
        fun onFirstVideoUnit(lengthBytes: Int) {}

        /** Every access unit accepted into the decoder; for throughput stats. */
        fun onVideoUnit(lengthBytes: Int) {}

        /** A decoded picture reached the surface. */
        fun onFirstFrameRendered() {}

        /** Handshake completed (or media started arriving from an older sender). */
        fun onConnected() {}

        /** The link dropped; the transport retries on its own. */
        fun onConnectionError(error: Throwable) {}

        fun onDecoderError(message: String) {}

        /** The sender announced what it is streaming. Already applied. */
        fun onStreamFormat(format: SyLcProtocol.StreamFormat) {}

        /** The transport loop ended; [error] is the last failure, if any. */
        fun onStopped(error: Throwable?) {}
    }

    private var decoder: HevcDecoder? = null
    private var client: SyLcTcpClient? = null
    private var audio: AudioPlayer? = null

    val isActive: Boolean get() = client != null

    /** What the decoder is currently configured for. */
    val streamFormat: SyLcProtocol.StreamFormat
        get() = decoder?.streamFormat() ?: settings.lastStreamFormat

    /**
     * Builds the pipeline against [surface] and connects, retrying in the
     * background until [stop]. Calling it twice is a no-op on the second call.
     */
    fun start(surface: Surface, scope: CoroutineScope) {
        if (isActive) {
            Log.w(TAG, "Session already running; ignoring duplicate start")
            return
        }

        val audioPlayer = AudioPlayer()
        audio = audioPlayer

        val hevc = HevcDecoder(
            surface = surface,
            streamFormat = settings.lastStreamFormat,
            audioClockUs = { audioPlayer.currentMediaTimeUs() },
            onFirstFrameRendered = { listener.onFirstFrameRendered() },
            onDecoderError = { message -> listener.onDecoderError(message) },
        )
        decoder = hevc
        hevc.start()

        var sawFirstUnit = false
        val transport = SyLcTcpClient(
            host = settings.host,
            port = settings.port,
            onVideoFrame = { ptsMs, isKeyframe, payload, offset, length ->
                if (!sawFirstUnit) {
                    sawFirstUnit = true
                    Log.i(TAG, "First HEVC access unit received ($length bytes)")
                    listener.onFirstVideoUnit(length)
                }
                if (submitVideoUnit(hevc, ptsMs, isKeyframe, payload, offset, length)) {
                    listener.onVideoUnit(length)
                }
            },
            onAudioFrame = { ptsMs, payload, offset, length ->
                // Pooled copy: no per-packet allocation on the receive path.
                audioPlayer.play(ptsMs, payload, offset, length)
            },
            onStreamFormat = { format ->
                settings.lastStreamFormat = format
                hevc.applyStreamFormat(format)
                listener.onStreamFormat(format)
            },
            feedbackProvider = {
                decoder?.takeBandwidthFeedback()
                    ?: SyLcProtocol.BandwidthFeedback(queueDepth = 0, underrun = false)
            },
            onConnected = {
                // AudioTrack initialization is independent of any View. Start it
                // before the transport reader can dispatch the first PCM packet.
                audioPlayer.start()
                listener.onConnected()
            },
            onConnectionError = { error -> listener.onConnectionError(error) },
        )
        client = transport

        scope.launch {
            try {
                transport.connect()
            } finally {
                listener.onStopped(transport.lastError)
            }
        }
    }

    /**
     * Copies one access unit into a pooled buffer and hands it to the decoder.
     * Returns false when the unit was dropped, so callers do not count it.
     */
    private fun submitVideoUnit(
        hevc: HevcDecoder,
        ptsMs: Long,
        isKeyframe: Boolean,
        payload: ByteArray,
        offset: Int,
        length: Int,
    ): Boolean {
        // A null frame means the decoder is behind: dropping here keeps the
        // transport reader from blocking, which would stall the sender.
        val frame = hevc.acquireFrame() ?: return false
        return try {
            if (length > frame.data.size) {
                Log.e(TAG, "Access unit too large: $length > ${frame.data.size}")
                hevc.releaseFrame(frame)
                false
            } else {
                System.arraycopy(payload, offset, frame.data, 0, length)
                frame.length = length
                frame.ptsMs = ptsMs
                frame.isKeyframe = isKeyframe
                hevc.queueFrame(frame)   // takes ownership of the pooled frame
                true
            }
        } catch (error: RuntimeException) {
            Log.e(TAG, "Unable to queue the access unit", error)
            hevc.releaseFrame(frame)
            false
        }
    }

    /** Tears the pipeline down. Safe to call more than once, and from any state. */
    fun stop() {
        client?.disconnect()
        client = null
        decoder?.stop()
        decoder = null
        audio?.stop()
        audio = null
    }

    private companion object {
        private const val TAG = "CastSession"
    }
}
