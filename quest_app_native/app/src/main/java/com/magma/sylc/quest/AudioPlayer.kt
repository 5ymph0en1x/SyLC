package com.magma.sylc.quest

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.max

/**
 * Timestamp-aware PCM renderer and A/V master clock.
 *
 * AudioTap deliberately sends a small amount of audio ahead of the PC playhead.
 * Packets therefore stay in this bounded queue until AudioTrack consumes them;
 * their PTS is not discarded. Video asks [currentMediaTimeUs] for the position
 * that has actually left AudioTrack and schedules its Surface presentation
 * against that clock.
 */
class AudioPlayer {
    private data class TimedPcm(val ptsUs: Long, val bytes: ByteArray, val length: Int)

    private val lifecycleLock = Any()
    private val frameQueue = ArrayBlockingQueue<TimedPcm>(MAX_QUEUED_FRAMES)

    // Video pools its access units; audio used to allocate a fresh ByteArray for
    // every packet, i.e. tens of short-lived arrays per second feeding the GC in
    // a real-time path. Same discipline on both media now.
    private val pcmPool = ArrayBlockingQueue<ByteArray>(MAX_QUEUED_FRAMES + 2)

    @Volatile private var audioTrack: AudioTrack? = null
    @Volatile private var isRunning = false
    @Volatile private var playbackThread: Thread? = null
    @Volatile private var generation = 0L
    @Volatile private var activeSampleRate = DEFAULT_SAMPLE_RATE
    @Volatile private var anchorPtsUs = CLOCK_UNSET
    @Volatile private var anchorHeadFrames = 0L
    @Volatile private var lastObservedHeadFrames = 0L
    @Volatile private var lastHeadAdvanceNs = 0L

    fun start(sampleRate: Int = DEFAULT_SAMPLE_RATE) {
        synchronized(lifecycleLock) {
            if (isRunning) return
            if (playbackThread?.isAlive == true) {
                Log.w(TAG, "Refusing to start while the previous audio worker is still exiting")
                return
            }

            try {
                val minBuffer = AudioTrack.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_OUT_STEREO,
                    AudioFormat.ENCODING_PCM_16BIT
                )
                if (minBuffer <= 0) {
                    throw IllegalStateException("AudioTrack minimum buffer query failed: $minBuffer")
                }
                val targetBuffer = sampleRate * BYTES_PER_FRAME * TARGET_BUFFER_MS / 1000
                val bufferSize = max(minBuffer * 2, targetBuffer)

                val track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_MOVIE)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                            .build()
                    )
                    .setBufferSizeInBytes(bufferSize)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
                    .build()

                if (track.state != AudioTrack.STATE_INITIALIZED) {
                    track.release()
                    throw IllegalStateException("AudioTrack was not initialized")
                }

                drainQueueToPool()
                activeSampleRate = sampleRate
                anchorPtsUs = CLOCK_UNSET
                anchorHeadFrames = 0L
                lastObservedHeadFrames = 0L
                lastHeadAdvanceNs = System.nanoTime()
                generation += 1
                val workerGeneration = generation
                audioTrack = track
                isRunning = true
                track.play()

                playbackThread = thread(
                    start = true,
                    name = "SyLC-Audio-$workerGeneration",
                    priority = Thread.MAX_PRIORITY
                ) {
                    playbackLoop(track, sampleRate, workerGeneration)
                }
                Log.i(TAG, "Audio player started at $sampleRate Hz, buffer=$bufferSize bytes")
            } catch (error: Exception) {
                isRunning = false
                audioTrack = null
                Log.e(TAG, "Failed to start audio player", error)
            }
        }
    }

    /**
     * Queue one immutable PCM packet. The TCP receive buffer is reused, so callers
     * must pass an owned byte array. Overflow discards the oldest audio rather than
     * growing latency without bound; the worker detects the resulting PTS jump and
     * atomically re-anchors AudioTrack.
     */
    fun play(ptsMs: Long, pcm: ByteArray) = play(ptsMs, pcm, 0, pcm.size)

    /**
     * Queues PCM copied out of a transport buffer, borrowing the destination
     * from an internal pool so the receive path allocates nothing per packet.
     */
    fun play(ptsMs: Long, source: ByteArray, offset: Int, length: Int) {
        if (!isRunning || length <= 0) return
        if (offset < 0 || offset + length > source.size) {
            Log.w(TAG, "Ignoring PCM packet with out-of-range bounds")
            return
        }
        val buffer = borrow(length)
        System.arraycopy(source, offset, buffer, 0, length)
        val packet = TimedPcm(ptsMs * 1000L, buffer, length)
        if (!frameQueue.offer(packet)) {
            recycle(frameQueue.poll())
            if (!frameQueue.offer(packet)) {
                Log.w(TAG, "Audio queue remained full after dropping its oldest packet")
                recycle(packet)
            }
        }
    }

    /** A pooled buffer of at least [length] bytes, or a fresh one if none fits. */
    private fun borrow(length: Int): ByteArray {
        while (true) {
            val candidate = pcmPool.poll() ?: return ByteArray(maxOf(length, PCM_BUFFER_BYTES))
            if (candidate.size >= length) return candidate
            // Undersized (an unusually large packet grew the working set): drop
            // it and keep looking rather than copying past its end.
        }
    }

    private fun drainQueueToPool() {
        while (true) {
            val queued = frameQueue.poll() ?: return
            recycle(queued)
        }
    }

    private fun recycle(packet: TimedPcm?) {
        val bytes = packet?.bytes ?: return
        if (bytes.size >= PCM_BUFFER_BYTES) pcmPool.offer(bytes)
    }

    /**
     * Media PTS currently consumed by AudioTrack, or null until the first packet
     * establishes an anchor. playbackHeadPosition is the number of frames actually
     * presented by the Android audio pipeline, making this more useful than packet
     * arrival time as the master clock.
     */
    fun currentMediaTimeUs(): Long? {
        val track = audioTrack ?: return null
        val pts = anchorPtsUs
        if (!isRunning || pts == CLOCK_UNSET) return null
        val head = unsignedPlaybackHead(track)
        val now = System.nanoTime()
        if (head != lastObservedHeadFrames) {
            lastObservedHeadFrames = head
            lastHeadAdvanceNs = now
        } else if (now - lastHeadAdvanceNs > AUDIO_CLOCK_STALE_NS) {
            return null
        }
        val elapsedFrames = unsignedFrameDelta(head, anchorHeadFrames)
        return pts + elapsedFrames * 1_000_000L / activeSampleRate
    }

    private fun playbackLoop(track: AudioTrack, sampleRate: Int, workerGeneration: Long) {
        try {
            while (isRunning && generation == workerGeneration) {
                val packet = frameQueue.poll(100, TimeUnit.MILLISECONDS) ?: continue
                if (!isRunning || generation != workerGeneration) break

                val hadAnchor = anchorPtsUs != CLOCK_UNSET
                val currentUs = currentMediaTimeUs()
                if (!hadAnchor) {
                    reanchor(track, packet.ptsUs)
                } else if (currentUs != null) {
                    if (abs(packet.ptsUs - currentUs) > DISCONTINUITY_US) {
                        reanchor(track, packet.ptsUs)
                    } else if (packet.ptsUs < currentUs - LATE_AUDIO_DROP_US) {
                        Log.w(
                            TAG,
                            "Dropping stale audio packet: pts=${packet.ptsUs / 1000}ms, " +
                                "clock=${currentUs / 1000}ms"
                        )
                        recycle(packet)
                        continue
                    }
                }

                var offset = 0
                try {
                    while (
                        offset < packet.length &&
                        isRunning &&
                        generation == workerGeneration
                    ) {
                        val written = track.write(
                            packet.bytes,
                            offset,
                            packet.length - offset,
                            AudioTrack.WRITE_BLOCKING
                        )
                        if (written < 0) {
                            throw IllegalStateException("AudioTrack.write failed: $written")
                        }
                        if (written == 0) break
                        offset += written
                    }
                } finally {
                    recycle(packet)
                }
            }
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (error: Exception) {
            if (isRunning && generation == workerGeneration) {
                Log.e(TAG, "Error in audio playback loop", error)
            }
        } finally {
            Log.i(TAG, "Audio player thread $workerGeneration exiting")
        }
    }

    private fun reanchor(track: AudioTrack, ptsUs: Long) {
        try {
            track.pause()
            track.flush()
            anchorHeadFrames = unsignedPlaybackHead(track)
            lastObservedHeadFrames = anchorHeadFrames
            lastHeadAdvanceNs = System.nanoTime()
            anchorPtsUs = ptsUs
            track.play()
            Log.i(TAG, "Audio clock anchored at ${ptsUs / 1000}ms")
        } catch (error: IllegalStateException) {
            Log.w(TAG, "Unable to reset AudioTrack clock cleanly", error)
            anchorHeadFrames = unsignedPlaybackHead(track)
            lastObservedHeadFrames = anchorHeadFrames
            lastHeadAdvanceNs = System.nanoTime()
            anchorPtsUs = ptsUs
        }
    }

    fun stop() {
        synchronized(lifecycleLock) {
            val track = audioTrack
            val worker = playbackThread

            generation += 1
            isRunning = false
            playbackThread = null
            audioTrack = null
            drainQueueToPool()
            anchorPtsUs = CLOCK_UNSET

            // pause+flush unblocks a WRITE_BLOCKING call before we join. The worker
            // captures this exact AudioTrack instance, so it can never write into a
            // later session's track even if a vendor implementation exits slowly.
            try {
                track?.pause()
                track?.flush()
            } catch (error: IllegalStateException) {
                Log.d(TAG, "AudioTrack already stopped while flushing", error)
            }
            worker?.interrupt()
            if (worker != null && worker !== Thread.currentThread()) {
                try {
                    worker.join(WORKER_JOIN_TIMEOUT_MS)
                } catch (_: InterruptedException) {
                    Thread.currentThread().interrupt()
                }
                if (worker.isAlive) {
                    Log.e(TAG, "Audio worker did not stop within $WORKER_JOIN_TIMEOUT_MS ms")
                }
            }

            try {
                track?.stop()
            } catch (error: IllegalStateException) {
                Log.d(TAG, "AudioTrack already stopped", error)
            } finally {
                track?.release()
            }
            Log.i(TAG, "Audio player stopped")
        }
    }

    private fun unsignedPlaybackHead(track: AudioTrack): Long {
        return track.playbackHeadPosition.toLong() and UINT32_MASK
    }

    private fun unsignedFrameDelta(current: Long, anchor: Long): Long {
        return (current - anchor) and UINT32_MASK
    }

    companion object {
        private const val TAG = "AudioPlayer"
        private const val DEFAULT_SAMPLE_RATE = 48_000
        private const val BYTES_PER_FRAME = 4 // stereo, signed 16-bit
        private const val TARGET_BUFFER_MS = 60
        private const val MAX_QUEUED_FRAMES = 12
        // AudioTap sends ~20 ms chunks; 16 KiB covers 85 ms of 48 kHz stereo
        // 16-bit, so pooled buffers are reused rather than reallocated.
        private const val PCM_BUFFER_BYTES = 16 * 1024
        private const val DISCONTINUITY_US = 500_000L
        private const val LATE_AUDIO_DROP_US = 120_000L
        private const val AUDIO_CLOCK_STALE_NS = 750_000_000L
        private const val WORKER_JOIN_TIMEOUT_MS = 1_500L
        private const val CLOCK_UNSET = Long.MIN_VALUE
        private const val UINT32_MASK = 0xFFFF_FFFFL
    }
}
