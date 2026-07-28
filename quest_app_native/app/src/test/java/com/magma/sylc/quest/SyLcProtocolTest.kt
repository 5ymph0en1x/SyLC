package com.magma.sylc.quest

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Wire-format tests. This is pure logic and needs no headset, yet it guards the
 * seam where a receiver and a sender written in different languages have to
 * agree byte for byte.
 */
class SyLcProtocolTest {

    /** Builds a header exactly as cast_sender/protocol.py `>4sBBBIqHHI` does. */
    private fun senderHeader(
        type: Int,
        flags: Int = 0,
        seq: Long = 1,
        ptsMs: Long = 0,
        fragIdx: Int = 0,
        fragCnt: Int = 1,
        payload: ByteArray = ByteArray(0),
    ): ByteArray {
        val buffer = ByteBuffer.allocate(SyLcProtocol.HEADER_SIZE + payload.size)
        buffer.order(ByteOrder.BIG_ENDIAN)
        buffer.put(SyLcProtocol.MAGIC)
        buffer.put(SyLcProtocol.VER.toByte())
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

    @Test
    fun `header size matches the sender struct layout`() {
        // ">4sBBBIqHHI" = 4 + 1 + 1 + 1 + 4 + 8 + 2 + 2 + 4
        assertEquals(27, SyLcProtocol.HEADER_SIZE)
    }

    @Test
    fun `parses every field of a sender header`() {
        val raw = senderHeader(
            type = SyLcProtocol.PT_VIDEO,
            flags = 1,
            seq = 4_294_967_295L,   // u32 max: must not surface as -1
            ptsMs = 1_234_567L,
            fragIdx = 3,
            fragCnt = 7,
            payload = byteArrayOf(9, 8, 7),
        )

        val header = SyLcProtocol.parseHeader(ByteBuffer.wrap(raw))

        assertArrayEquals(SyLcProtocol.MAGIC, header.magic)
        assertEquals(SyLcProtocol.VER, header.ver)
        assertEquals(SyLcProtocol.PT_VIDEO, header.type)
        assertEquals(1, header.flags)
        assertEquals(4_294_967_295L, header.seq)
        assertEquals(1_234_567L, header.ptsMs)
        assertEquals(3, header.fragIdx)
        assertEquals(7, header.fragCnt)
        assertEquals(3L, header.length)
    }

    @Test
    fun `negative presentation timestamps survive the round trip`() {
        val raw = senderHeader(type = SyLcProtocol.PT_AUDIO, ptsMs = -4_000L)
        assertEquals(-4_000L, SyLcProtocol.parseHeader(ByteBuffer.wrap(raw)).ptsMs)
    }

    @Test
    fun `framed packets carry a length prefix, datagrams do not`() {
        val framed = SyLcProtocol.packHello(seq = 5)
        val datagram = SyLcProtocol.packHelloDatagram(seq = 5)

        // TCP is a byte stream and needs explicit message boundaries.
        assertEquals(4 + SyLcProtocol.HEADER_SIZE, framed.size)
        assertEquals(
            SyLcProtocol.HEADER_SIZE,
            ByteBuffer.wrap(framed, 0, 4).order(ByteOrder.BIG_ENDIAN).int
        )
        // UDP already delimits messages; a prefix would corrupt the header.
        assertEquals(SyLcProtocol.HEADER_SIZE, datagram.size)
        assertArrayEquals(
            SyLcProtocol.MAGIC,
            datagram.copyOfRange(0, 4)
        )
        assertEquals(
            SyLcProtocol.PT_HELLO,
            SyLcProtocol.parseHeader(ByteBuffer.wrap(datagram)).type
        )
    }

    @Test
    fun `bandwidth feedback is a control packet carrying its fields`() {
        val framed = SyLcProtocol.packBandwidthFeedback(
            seq = 2,
            feedback = SyLcProtocol.BandwidthFeedback(
                queueDepth = 3, underrun = true, needsIdr = true
            )
        )
        val body = framed.copyOfRange(4, framed.size)
        val header = SyLcProtocol.parseHeader(ByteBuffer.wrap(body))
        assertEquals(SyLcProtocol.PT_CONTROL, header.type)

        val json = String(
            body, SyLcProtocol.HEADER_SIZE, header.length.toInt(), Charsets.UTF_8
        )
        assertTrue(json.contains("\"kind\":\"bwfeedback\""))
        assertTrue(json.contains("\"queue_depth\":3"))
        assertTrue(json.contains("\"underrun\":true"))
        assertTrue(json.contains("\"needs_idr\":true"))
    }

    @Test
    fun `negative queue depth is clamped rather than sent`() {
        val framed = SyLcProtocol.packBandwidthFeedback(
            seq = 1,
            feedback = SyLcProtocol.BandwidthFeedback(queueDepth = -5, underrun = false)
        )
        val body = framed.copyOfRange(4, framed.size)
        val header = SyLcProtocol.parseHeader(ByteBuffer.wrap(body))
        val json = String(
            body, SyLcProtocol.HEADER_SIZE, header.length.toInt(), Charsets.UTF_8
        )
        assertTrue(json.contains("\"queue_depth\":0"))
    }

    @Test
    fun `stream format is read from a hello ack payload`() {
        val payload = """{"width":1920,"height":1080,"fps":60,"stereo":"mono"}"""
            .toByteArray(Charsets.UTF_8)

        val format = SyLcProtocol.parseStreamFormat(payload, 0, payload.size)

        assertEquals(1920, format.width)
        assertEquals(1080, format.height)
        assertEquals(60, format.fps)
        assertFalse(format.isSideBySide)
    }

    @Test
    fun `stream format honours the payload offset`() {
        val json = """{"width":7680,"height":2160,"fps":30,"stereo":"lr"}"""
        val framed = ByteArray(8) { 0x7F } + json.toByteArray(Charsets.UTF_8)

        val format = SyLcProtocol.parseStreamFormat(framed, 8, framed.size - 8)

        assertEquals(7680, format.width)
        assertEquals(2160, format.height)
        assertTrue(format.isSideBySide)
    }

    @Test
    fun `an absent or unusable announcement keeps the historical defaults`() {
        val default = SyLcProtocol.StreamFormat.DEFAULT

        // A sender predating the announcement sends an empty payload.
        assertEquals(default, SyLcProtocol.parseStreamFormat(ByteArray(0), 0, 0))

        val garbage = "not json at all".toByteArray(Charsets.UTF_8)
        assertEquals(default, SyLcProtocol.parseStreamFormat(garbage, 0, garbage.size))
    }

    @Test
    fun `implausible announced values fall back field by field`() {
        val payload = """{"width":0,"height":1080,"fps":100000}"""
            .toByteArray(Charsets.UTF_8)

        val format = SyLcProtocol.parseStreamFormat(payload, 0, payload.size)

        // Width and fps are out of range and revert; height was fine and stands.
        assertEquals(SyLcProtocol.StreamFormat.DEFAULT.width, format.width)
        assertEquals(1080, format.height)
        assertEquals(SyLcProtocol.StreamFormat.DEFAULT.fps, format.fps)
    }

    @Test
    fun `hdr announcement parses to isPq and is absent by default`() {
        val pq = """{"width":3840,"height":1080,"fps":24,"stereo":"lr","hdr":"pq"}"""
            .toByteArray(Charsets.UTF_8)
        val fmt = SyLcProtocol.parseStreamFormat(pq, 0, pq.size)
        assertEquals("pq", fmt.hdr)
        assertEquals(true, fmt.isPq)

        // Senders that never send the field: SDR, exactly as before.
        val sdr = """{"width":3840,"height":1080,"fps":24,"stereo":"lr"}"""
            .toByteArray(Charsets.UTF_8)
        val fmt2 = SyLcProtocol.parseStreamFormat(sdr, 0, sdr.size)
        assertEquals(false, fmt2.isPq)
        assertEquals(false, SyLcProtocol.StreamFormat.DEFAULT.isPq)
    }
}
