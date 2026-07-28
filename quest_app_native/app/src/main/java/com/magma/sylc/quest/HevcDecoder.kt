package com.magma.sylc.quest

import android.media.MediaCodec
import android.media.MediaFormat
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class HevcDecoder(
    private val surface: Surface,
    streamFormat: SyLcProtocol.StreamFormat = SyLcProtocol.StreamFormat.DEFAULT,
    private val audioClockUs: () -> Long? = { null },
    private val onFirstFrameRendered: () -> Unit = {},
    private val onDecoderError: (String) -> Unit = {}
) {
    private val TAG = "HevcDecoder"

    /** What the sender announced; drives codec geometry and pacing. */
    @Volatile private var format: SyLcProtocol.StreamFormat = streamFormat
    private var codec: MediaCodec? = null
    @Volatile private var isConfigured = false
    private val firstFrameRendered = AtomicBoolean(false)
    private val queuedFrameCount = AtomicLong(0)
    private val renderedFrameCount = AtomicLong(0)
    private val droppedFrameCount = AtomicLong(0)
    private val pressureSinceFeedback = AtomicBoolean(false)
    private val recoveryIdrSinceFeedback = AtomicBoolean(false)
    private var statsWindowStartedNs = 0L

    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null

    // Frame object for zero-allocation pooling
    class Frame(val data: ByteArray) {
        var length: Int = 0
        var ptsMs: Long = 0
        var isKeyframe: Boolean = false
    }

    // Four bounded reusable AUs are sufficient for an asynchronous decoder and
    // avoid the previous 120 MiB Java-heap reservation (6 x 20 MiB).
    private val framePool = ArrayBlockingQueue<Frame>(FRAME_POOL_SIZE)
    
    // Frames waiting to be decoded
    private val pendingFrames = ConcurrentLinkedQueue<Frame>()
    
    // Available input buffer indices from MediaCodec
    private val availableInputBuffers = ConcurrentLinkedQueue<Int>()

    fun start() {
        if (codec != null || handlerThread != null) {
            stop()
        }
        try {
            // Initialize frame pool
            framePool.clear()
            firstFrameRendered.set(false)
            for (i in 0 until FRAME_POOL_SIZE) {
                framePool.offer(Frame(ByteArray(MAX_ACCESS_UNIT_BYTES)))
            }
            pendingFrames.clear()
            availableInputBuffers.clear()
            queuedFrameCount.set(0)
            renderedFrameCount.set(0)
            droppedFrameCount.set(0)
            statsWindowStartedNs = System.nanoTime()

            handlerThread = HandlerThread("CodecThread")
            handlerThread?.start()
            handler = Handler(handlerThread!!.looper)

            pressureSinceFeedback.set(false)
            recoveryIdrSinceFeedback.set(false)
            startFirstWorkingDecoder()
        } catch (fatal: Exception) {
            Log.e(TAG, "Error starting MediaCodec", fatal)
            stop()
            onDecoderError("Unable to start the HEVC decoder")
        }
    }

    /**
     * Walk the capability-ordered decoder candidates (hardware first, software
     * last) and start the first one that configures. A PQ (Main10) stream on a
     * device whose hardware decoder lacks Main10 — or lies about it, like the
     * emulator's goldfish — thus lands on the software c2.android decoder
     * instead of dying with an opaque error_14.
     */
    private fun startFirstWorkingDecoder() {
        val names = selectDecoderNames()
        if (names.isEmpty()) {
            stop()
            onDecoderError(
                if (format.isPq) "HDR (HEVC Main10) is not supported by this device's video decoder"
                else "No HEVC decoder available on this device"
            )
            return
        }
        var lastError: Exception? = null
        for (name in names) {
            for (hints in booleanArrayOf(true, false)) {
                try {
                    codec = configureCodec(preferPerformanceHints = hints, codecName = name)
                    isConfigured = true
                    Log.i(
                        TAG,
                        "MediaCodec started (Async, hints=$hints): ${codec?.name}, " +
                            "${MAX_ACCESS_UNIT_BYTES / (1024 * 1024)} MiB AU pool"
                    )
                    return
                } catch (error: Exception) {
                    Log.w(TAG, "Decoder $name (hints=$hints) failed to start", error)
                    lastError = error
                    releaseCodecQuietly()
                }
            }
        }
        Log.e(TAG, "Every HEVC decoder candidate failed", lastError)
        stop()
        onDecoderError(
            if (format.isPq) "HDR (HEVC Main10) failed on every decoder of this device"
            else "Unable to start the HEVC decoder"
        )
    }

    /**
     * HEVC decoder names able to take this stream, hardware-accelerated first.
     * A PQ stream requires the Main10 profile in the decoder's declared
     * capabilities; SDR accepts any HEVC decoder (historical behavior).
     */
    private fun selectDecoderNames(): List<String> {
        val announced = format
        val needMain10 = announced.isPq
        val infos = android.media.MediaCodecList(android.media.MediaCodecList.REGULAR_CODECS)
            .codecInfos
            .filter { !it.isEncoder && it.supportedTypes.any {
                t -> t.equals(MediaFormat.MIMETYPE_VIDEO_HEVC, ignoreCase = true) } }
        fun isHw(info: android.media.MediaCodecInfo): Boolean =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) info.isHardwareAccelerated
            else !(info.name.startsWith("OMX.google.") || info.name.startsWith("c2.android."))
        fun capable(info: android.media.MediaCodecInfo): Boolean = try {
            val caps = info.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_HEVC)
            val profileOk = !needMain10 || caps.profileLevels.any {
                it.profile == android.media.MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10
            }
            val sizeOk = caps.videoCapabilities
                ?.isSizeSupported(announced.width, announced.height) != false
            profileOk && sizeOk
        } catch (_: Exception) {
            false
        }
        val capable = infos.filter(::capable)
        var ordered = capable.filter(::isHw) + capable.filterNot(::isHw)
        if (ordered.isEmpty() && needMain10 && infos.isNotEmpty()) {
            // No decoder CLAIMS Main10 (the emulator's images declare none, yet
            // AOSP's software libhevc usually decodes 10-bit anyway). Best
            // effort: try them all, SOFTWARE FIRST — a hardware decoder that
            // does not claim the profile tends to accept the config and then
            // silently decode nothing, whereas software either works or fails
            // loudly at configure time.
            Log.w(TAG, "No decoder claims HEVC Main10 — best-effort attempts, software first")
            ordered = infos.filterNot(::isHw) + infos.filter(::isHw)
        }
        Log.i(TAG, "HEVC decoder candidates for ${announced.width}x${announced.height}" +
            (if (needMain10) " Main10" else "") + ": " + ordered.joinToString { it.name })
        return ordered.map { it.name }
    }

    private fun configureCodec(preferPerformanceHints: Boolean,
                               codecName: String? = null): MediaCodec {
        val candidate = if (codecName != null) MediaCodec.createByCodecName(codecName)
                        else MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_HEVC)
        try {
            candidate.setCallback(object : MediaCodec.Callback() {
                override fun onInputBufferAvailable(mc: MediaCodec, index: Int) {
                    availableInputBuffers.offer(index)
                    processPendingFrames()
                }

                override fun onOutputBufferAvailable(mc: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
                    try {
                        if (index >= 0) {
                            val render = info.size != 0
                            val presented = if (render) {
                                releaseVideoAtAudioTime(mc, index, info.presentationTimeUs)
                            } else {
                                mc.releaseOutputBuffer(index, false)
                                false
                            }
                            if (presented) {
                                renderedFrameCount.incrementAndGet()
                                maybeLogPerformance()
                            }
                            if (presented && firstFrameRendered.compareAndSet(false, true)) {
                                Log.i(TAG, "First decoded frame rendered")
                                onFirstFrameRendered()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error releasing output buffer", e)
                        if (isConfigured) {
                            onDecoderError("Unable to render the decoded frame")
                        }
                    }
                }

                override fun onError(mc: MediaCodec, e: MediaCodec.CodecException) {
                    Log.e(TAG, "MediaCodec Error", e)
                    if (isConfigured) onDecoderError(e.diagnosticInfo)
                }

                override fun onOutputFormatChanged(mc: MediaCodec, format: MediaFormat) {
                    Log.i(TAG, "Output format changed: $format")
                }
            }, handler)

            val announced = format
            val format = MediaFormat.createVideoFormat(
                MediaFormat.MIMETYPE_VIDEO_HEVC,
                announced.width,
                announced.height
            )
            format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, MAX_ACCESS_UNIT_BYTES)
            format.setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED)
            if (announced.isPq) {
                // HDR cast: the sender announced (and the bitstream VUI carries)
                // an HEVC Main10 BT.2020/ST2084 stream. Declaring it lets the
                // decoder/compositor apply its PQ handling (the Quest panels are
                // not true HDR, but the compositor's tone mapping + contrast
                // enhancement read this intent).
                format.setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT2020)
                format.setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_ST2084)
            } else {
                // SDR cast (8-bit Blu-ray/MVC/SBS planes). Labelling these
                // BT.2020 changed their chromaticities on Meta's compositor;
                // BT.709 limited-range SDR is the faithful declaration.
                format.setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT709)
                format.setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
            }
            format.setInteger(MediaFormat.KEY_FRAME_RATE, announced.fps)

            if (preferPerformanceHints) {
                // KEY_LOW_LATENCY only exists from API 30. The constant is a
                // plain string inlined at compile time, so setting it lower down
                // is harmless rather than fatal — but it is also meaningless, and
                // saying so beats leaving a reader to wonder.
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    format.setInteger(MediaFormat.KEY_LOW_LATENCY, 1)
                }
                format.setInteger(MediaFormat.KEY_PRIORITY, 0)
                format.setFloat(MediaFormat.KEY_OPERATING_RATE, selectOperatingRate(candidate))
            }

            candidate.configure(format, surface, null, 0)
            candidate.start()
            return candidate
        } catch (error: Exception) {
            try {
                candidate.release()
            } catch (_: Exception) {
                // Candidate never became externally visible; preserve the root error.
            }
            throw error
        }
    }

    private fun selectOperatingRate(candidate: MediaCodec): Float {
        val announced = format
        val videoCapabilities = candidate.codecInfo
            .getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_HEVC)
            .videoCapabilities ?: return announced.fps.toFloat()
        // Ask for headroom above the stream's own cadence so a burst after a
        // seek is absorbed, but never below it.
        val candidates = intArrayOf(
            announced.fps * 2, (announced.fps * 3) / 2, announced.fps
        )
        val selected = candidates.firstOrNull { rate ->
            videoCapabilities.areSizeAndRateSupported(
                announced.width,
                announced.height,
                rate.toDouble()
            )
        } ?: announced.fps
        Log.i(TAG, "Selected HEVC operating rate: $selected fps")
        return selected.toFloat()
    }

    /**
     * Adopt a stream description announced after the decoder was started.
     * Restarts MediaCodec only when the geometry or cadence actually moved:
     * a redundant announcement must not interrupt playback.
     */
    fun applyStreamFormat(announced: SyLcProtocol.StreamFormat) {
        val current = format
        if (current == announced) return
        Log.i(
            TAG,
            "Stream format changed: ${current.width}x${current.height}@${current.fps} " +
                "-> ${announced.width}x${announced.height}@${announced.fps} " +
                "(${announced.stereo})"
        )
        format = announced
        if (isConfigured) start()
    }

    fun streamFormat(): SyLcProtocol.StreamFormat = format

    /**
     * AudioTrack is the receiver's master clock. Future video is scheduled on
     * the Surface against it; stale video is dropped instead of letting lip-sync
     * drift grow. Without an audio anchor, MediaCodec remains a normal immediate
     * renderer, so video-only sources retain their existing behavior.
     */
    private fun releaseVideoAtAudioTime(
        mc: MediaCodec,
        index: Int,
        videoPtsUs: Long
    ): Boolean {
        // Never turn decoder/audio warm-up latency into a black screen. The
        // first decoded picture is our visual acquisition point; only
        // subsequent pictures may be dropped to converge on the audio clock.
        if (!firstFrameRendered.get()) {
            mc.releaseOutputBuffer(index, true)
            return true
        }

        val audioUs = try {
            audioClockUs()
        } catch (error: Exception) {
            Log.w(TAG, "Audio clock provider failed", error)
            null
        }
        if (audioUs == null) {
            mc.releaseOutputBuffer(index, true)
            return true
        }

        val deltaUs = videoPtsUs - audioUs
        if (deltaUs < -VIDEO_LATE_DROP_US) {
            mc.releaseOutputBuffer(index, false)
            droppedFrameCount.incrementAndGet()
            // This is presentation-clock convergence, not decoder congestion.
            // Reporting it as transport pressure previously walked 500 Mbps down
            // to 100 Mbps after seeks even with an empty MediaCodec queue.
            maybeLogPerformance()
            return false
        }
        if (deltaUs in VIDEO_SCHEDULE_THRESHOLD_US..VIDEO_MAX_SCHEDULE_US) {
            val renderTimeNs = System.nanoTime() + deltaUs * 1000L
            mc.releaseOutputBuffer(index, renderTimeNs)
        } else {
            // A discontinuity larger than the scheduling window is handled by
            // AudioPlayer's PTS re-anchor. Do not hold scarce codec buffers for it.
            mc.releaseOutputBuffer(index, true)
        }
        return true
    }

    /**
     * Obtains a free frame from the pool.
     * Returns null immediately if the decoder is lagging; the TCP reader never
     * blocks and memory use remains bounded.
     */
    fun acquireFrame(): Frame? {
        if (!isConfigured) return null
        // Use poll() instead of take() to drop frames immediately if the decoder 
        // is lagging behind. This prevents blocking the TCP read loop, which would 
        // otherwise cause a TCP zero-window and blow up the PC sender with WSAENOBUFS.
        return framePool.poll().also {
            if (it == null) {
                droppedFrameCount.incrementAndGet()
                pressureSinceFeedback.set(true)
                recoveryIdrSinceFeedback.set(true)
                maybeLogPerformance()
            }
        }
    }

    fun releaseFrame(frame: Frame) {
        frame.length = 0
        frame.ptsMs = 0
        frame.isKeyframe = false
        framePool.offer(frame)
    }

    /**
     * Enqueues the frame for decoding.
     */
    fun queueFrame(frame: Frame) {
        if (!isConfigured) {
            releaseFrame(frame)
            return
        }
        val posted = handler?.post {
            pendingFrames.offer(frame)
            processPendingFrames()
        } == true
        if (!posted) releaseFrame(frame)
    }

    private fun processPendingFrames() {
        if (!isConfigured) return
        val c = codec ?: return

        while (availableInputBuffers.isNotEmpty() && pendingFrames.isNotEmpty()) {
            val index = availableInputBuffers.poll() ?: break
            val frame = pendingFrames.poll()
            
            if (frame == null) {
                availableInputBuffers.offer(index)
                break
            }

            try {
                val inputBuffer = c.getInputBuffer(index)
                if (inputBuffer != null) {
                    inputBuffer.clear()
                    
                    if (frame.length > inputBuffer.capacity()) {
                        droppedFrameCount.incrementAndGet()
                        pressureSinceFeedback.set(true)
                        recoveryIdrSinceFeedback.set(true)
                        Log.e(
                            TAG,
                            "Dropping oversized HEVC AU: ${frame.length} > " +
                                "${inputBuffer.capacity()}"
                        )
                        c.queueInputBuffer(index, 0, 0, frame.ptsMs * 1000L, 0)
                        continue
                    }
                    inputBuffer.put(frame.data, 0, frame.length)
                    
                    val flags = if (frame.isKeyframe) {
                        MediaCodec.BUFFER_FLAG_KEY_FRAME
                    } else {
                        0
                    }

                    c.queueInputBuffer(index, 0, frame.length, frame.ptsMs * 1000L, flags)
                    queuedFrameCount.incrementAndGet()
                }
            } catch (e: Exception) {
                pressureSinceFeedback.set(true)
                recoveryIdrSinceFeedback.set(true)
                Log.e(TAG, "Error queuing input buffer", e)
            } finally {
                // Always return frame to pool
                releaseFrame(frame)
            }
        }
    }

    /**
     * Atomically sample receiver pressure for the sender's adaptive ladder.
     * `underrun` is the wire-compatible v1 name; here it means that the receiver
     * dropped or could not queue at least one AU since the previous report.
     */
    fun takeBandwidthFeedback(): SyLcProtocol.BandwidthFeedback {
        // Frames checked out by the TCP callback or currently being copied into
        // MediaCodec are normal pipeline occupancy. Only AUs genuinely waiting
        // for an input buffer represent queue pressure.
        val depth = pendingFrames.size.coerceAtMost(FRAME_POOL_SIZE)
        return SyLcProtocol.BandwidthFeedback(
            queueDepth = depth,
            underrun = pressureSinceFeedback.getAndSet(false),
            needsIdr = recoveryIdrSinceFeedback.getAndSet(false)
        )
    }

    @Synchronized
    private fun maybeLogPerformance() {
        val now = System.nanoTime()
        val started = statsWindowStartedNs
        if (started == 0L || now - started < STATS_INTERVAL_NS) return
        val elapsedSeconds = (now - started) / 1_000_000_000.0
        val queued = queuedFrameCount.getAndSet(0)
        val rendered = renderedFrameCount.getAndSet(0)
        val dropped = droppedFrameCount.getAndSet(0)
        statsWindowStartedNs = now
        Log.i(
            TAG,
            "pipeline queued=${"%.1f".format(queued / elapsedSeconds)} fps, " +
                "rendered=${"%.1f".format(rendered / elapsedSeconds)} fps, " +
                "dropped=$dropped"
        )
    }

    fun stop() {
        isConfigured = false
        releaseCodecQuietly()

        var pending = pendingFrames.poll()
        while (pending != null) {
            releaseFrame(pending)
            pending = pendingFrames.poll()
        }
        availableInputBuffers.clear()
        handler?.removeCallbacksAndMessages(null)

        handlerThread?.quitSafely()
        handlerThread?.join(500)
        handlerThread = null
        handler = null
    }

    private fun releaseCodecQuietly() {
        val activeCodec = codec
        codec = null
        try {
            activeCodec?.stop()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping MediaCodec", e)
        }
        try {
            activeCodec?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing MediaCodec", e)
        }
    }

    companion object {
        private const val FRAME_POOL_SIZE = 4
        private const val MAX_ACCESS_UNIT_BYTES = 12 * 1024 * 1024
        private const val VIDEO_SCHEDULE_THRESHOLD_US = 2_000L
        private const val VIDEO_MAX_SCHEDULE_US = 250_000L
        private const val VIDEO_LATE_DROP_US = 120_000L
        private const val STATS_INTERVAL_NS = 1_000_000_000L
    }
}
