package com.magma.sylc.quest

import android.util.Log
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

object SyLcProtocol {
    val MAGIC = byteArrayOf('S'.code.toByte(), 'Y'.code.toByte(), 'L'.code.toByte(), 'C'.code.toByte())
    const val VER = 1

    const val PT_HELLO = 1
    const val PT_HELLO_ACK = 2
    const val PT_VIDEO = 3
    const val PT_AUDIO = 4
    const val PT_CONTROL = 5
    const val PT_BYE = 6

    const val HEADER_SIZE = 27

    data class Header(
        val magic: ByteArray,
        val ver: Int,
        val type: Int,
        val flags: Int,
        val seq: Long,
        val ptsMs: Long,
        val fragIdx: Int,
        val fragCnt: Int,
        val length: Long
    )

    data class BandwidthFeedback(
        val queueDepth: Int,
        val underrun: Boolean,
        val needsIdr: Boolean = false
    )

    /**
     * What the sender says it is actually streaming, announced in the
     * PT_HELLO_ACK payload. The receiver used to hardcode 3840x1080 at 24 fps in
     * both the codec format and the panel size, which silently mis-sizes and
     * mis-paces anything else. Senders that predate this simply send an empty
     * payload, so [parseStreamFormat] falls back to those same values.
     */
    data class StreamFormat(
        val width: Int,
        val height: Int,
        val fps: Int,
        val stereo: String,
        /** Transfer hint: "pq" = HEVC Main10 BT.2020/ST2084 (HDR cast);
         *  empty = SDR BT.709 (every sender before the field existed). */
        val hdr: String = ""
    ) {
        /** True when the frame carries both eyes side by side. */
        val isSideBySide: Boolean get() = stereo.equals(STEREO_LEFT_RIGHT, ignoreCase = true)

        /** True when the sender announced a PQ (ST 2084) Main10 stream. */
        val isPq: Boolean get() = hdr.equals(HDR_PQ, ignoreCase = true)

        companion object {
            val DEFAULT = StreamFormat(3840, 1080, 24, STEREO_LEFT_RIGHT)
        }
    }

    const val STEREO_LEFT_RIGHT = "lr"
    const val STEREO_MONO = "mono"
    const val HDR_PQ = "pq"

    /**
     * Reads the stream description from a PT_HELLO_ACK payload.
     * Any missing, malformed or out-of-range field keeps its default, so a
     * partial or absent announcement can never leave the decoder unconfigured.
     */
    fun parseStreamFormat(
        payload: ByteArray,
        offset: Int,
        length: Int
    ): StreamFormat {
        val fallback = StreamFormat.DEFAULT
        if (length <= 0) return fallback
        return try {
            val json = JSONObject(String(payload, offset, length, Charsets.UTF_8))
            val width = json.optInt("width", fallback.width)
            val height = json.optInt("height", fallback.height)
            val fps = json.optInt("fps", fallback.fps)
            val stereo = json.optString("stereo", fallback.stereo)
            val hdr = json.optString("hdr", fallback.hdr)
            StreamFormat(
                width = if (width in 16..16384) width else fallback.width,
                height = if (height in 16..16384) height else fallback.height,
                fps = if (fps in 1..480) fps else fallback.fps,
                stereo = if (stereo.isNullOrBlank()) fallback.stereo else stereo,
                hdr = hdr ?: ""
            )
        } catch (error: Exception) {
            Log.w("SyLcProtocol", "Unreadable HELLO_ACK payload; keeping defaults", error)
            fallback
        }
    }

    fun parseHeader(buffer: ByteBuffer): Header {
        buffer.order(ByteOrder.BIG_ENDIAN)
        val magic = ByteArray(4)
        buffer.get(magic)
        val ver = buffer.get().toInt() and 0xFF
        val type = buffer.get().toInt() and 0xFF
        val flags = buffer.get().toInt() and 0xFF
        
        // struct ">I" is unsigned 32-bit int, Kotlin doesn't have UInt before 1.5 without experimental, 
        // using Long to avoid negative values
        val seq = buffer.int.toLong() and 0xFFFFFFFFL
        val ptsMs = buffer.long
        val fragIdx = buffer.short.toInt() and 0xFFFF
        val fragCnt = buffer.short.toInt() and 0xFFFF
        val length = buffer.int.toLong() and 0xFFFFFFFFL

        return Header(magic, ver, type, flags, seq, ptsMs, fragIdx, fragCnt, length)
    }

    fun packHello(seq: Long): ByteArray {
        return pack(PT_HELLO, 0, seq, 0, 0, 1, ByteArray(0))
    }

    fun packControl(kind: String, seq: Long): ByteArray {
        return packControlJson(seq, JSONObject().put("kind", kind))
    }

    fun packBandwidthFeedback(seq: Long, feedback: BandwidthFeedback): ByteArray {
        val json = JSONObject()
            .put("kind", "bwfeedback")
            .put("queue_depth", feedback.queueDepth.coerceAtLeast(0))
            .put("underrun", feedback.underrun)
            .put("needs_idr", feedback.needsIdr)
        return packControlJson(seq, json)
    }

    private fun packControlJson(seq: Long, json: JSONObject): ByteArray {
        val payload = json.toString().toByteArray(Charsets.UTF_8)
        return pack(PT_CONTROL, 0, seq, 0, 0, 1, payload)
    }

    /** A HELLO as a bare datagram: UDP carries message boundaries itself, so
     *  the 4-byte stream framing TCP needs would corrupt the header there. */
    fun packHelloDatagram(seq: Long): ByteArray =
        packDatagram(PT_HELLO, 0, seq, 0, 0, 1, ByteArray(0))

    private fun pack(
        type: Int, flags: Int, seq: Long, ptsMs: Long,
        fragIdx: Int, fragCnt: Int, payload: ByteArray
    ): ByteArray {
        val pkt = packDatagram(type, flags, seq, ptsMs, fragIdx, fragCnt, payload)

        // Stream framing for TCP: [4 bytes length] + pkt
        val framedBuffer = ByteBuffer.allocate(4 + pkt.size)
        framedBuffer.order(ByteOrder.BIG_ENDIAN)
        framedBuffer.putInt(pkt.size)
        framedBuffer.put(pkt)

        return framedBuffer.array()
    }

    private fun packDatagram(
        type: Int, flags: Int, seq: Long, ptsMs: Long,
        fragIdx: Int, fragCnt: Int, payload: ByteArray
    ): ByteArray {
        val buffer = ByteBuffer.allocate(HEADER_SIZE + payload.size)
        buffer.order(ByteOrder.BIG_ENDIAN)

        buffer.put(MAGIC)
        buffer.put(VER.toByte())
        buffer.put(type.toByte())
        buffer.put(flags.toByte())
        buffer.putInt(seq.toInt())
        buffer.putLong(ptsMs)
        buffer.putShort(fragIdx.toShort())
        buffer.putShort(fragCnt.toShort())
        buffer.putInt(payload.size)
        buffer.put(payload)

        return buffer.array()
    }
}
