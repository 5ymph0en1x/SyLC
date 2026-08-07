"""Wire protocol for SyLC Cast (PC <-> Quest 3).

Pure Python, no I/O: this module only turns structured fields into bytes
(`pack_*`) and bytes back into structured dicts (`parse`), plus fragmentation
helpers for video units that exceed a transport's MTU. It is the single
source of truth for the wire format; transports (Wi-Fi UDP, USB-C) build on
top of it without knowing the byte layout themselves.

Header layout (big-endian, struct format ">4sBBBIqHHI"), followed by payload:

    MAGIC     4s   b"SYLC"
    VER       B    protocol version (currently 1)
    TYPE      B    packet type (PT_* constants below)
    FLAGS     B    packet flags; for PT_VIDEO bit0 = keyframe
    SEQ       I    sequence number (u32)
    PTS_MS    q    presentation timestamp in milliseconds (i64)
    FRAG_IDX  H    fragment index, 0-based (u16)
    FRAG_CNT  H    total fragment count sharing this SEQ; 1 if not fragmented
    LEN       I    payload length in bytes (u32)
    payload   LEN bytes

Non-fragmented packets (audio, control, bye, and any video unit that fits in
one datagram) use FRAG_IDX=0, FRAG_CNT=1. `fragment_video` splits a video
payload larger than the MTU into multiple datagrams that share the same
SEQ/PTS_MS/FLAGS, with FRAG_CNT set to the total piece count; `Reassembler`
buffers those by SEQ and delivers the concatenated payload once every
fragment for that SEQ has arrived.

`pack_control` encodes its kwargs (kind + params, e.g. pos_ms for seek,
paused for play/pause, queue_depth/underrun for bwfeedback) as a small JSON
payload; `parse` decodes it back under the "control" key.
"""

import json
import struct

MAGIC = b"SYLC"
VER = 1

# packet types
PT_HELLO = 1
PT_HELLO_ACK = 2
PT_VIDEO = 3
PT_AUDIO = 4
PT_CONTROL = 5
PT_BYE = 6

_HEADER_FMT = ">4sBBBIqHHI"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _pack(pkt_type, flags, seq, pts_ms, frag_idx, frag_cnt, payload):
    header = pack_header(
        pkt_type, flags, seq, pts_ms, frag_idx, frag_cnt, len(payload)
    )
    return header + payload


def pack_header(pkt_type: int, flags: int, seq: int, pts_ms: int,
                frag_idx: int, frag_cnt: int, payload_length: int) -> bytes:
    """Pack only the fixed wire header.

    Stream transports use this to write ``length + header + payload`` as
    separate buffers.  That avoids copying multi-megabyte HEVC access units
    twice merely to prepend 31 bytes of TCP framing.
    """
    return struct.pack(
        _HEADER_FMT, MAGIC, VER, pkt_type, flags, seq, pts_ms,
        frag_idx, frag_cnt, payload_length,
    )


def pack_video(seq: int, pts_ms: int, flags: int, payload: bytes) -> bytes:
    """Pack a single (non-fragmented) video datagram. flags bit0 = keyframe.

    For payloads that may exceed the transport MTU, use fragment_video
    instead.
    """
    return _pack(PT_VIDEO, flags, seq, pts_ms, 0, 1, payload)


def pack_audio(seq: int, pts_ms: int, payload: bytes) -> bytes:
    return _pack(PT_AUDIO, 0, seq, pts_ms, 0, 1, payload)


def pack_control(kind: str, **kw) -> bytes:
    body = {"kind": kind, **kw}
    payload = json.dumps(body).encode("utf-8")
    return _pack(PT_CONTROL, 0, 0, 0, 0, 1, payload)


def pack_bye(seq: int) -> bytes:
    """Pack a PT_BYE datagram: announces a transport's clean shutdown to its peer."""
    return _pack(PT_BYE, 0, seq, 0, 0, 1, b"")


def pack_hello(seq: int) -> bytes:
    """Pack a PT_HELLO datagram: announces a receiver's presence to its peer (handshake
    greeting) -- the counterpart to pack_bye's clean-shutdown announcement. No payload:
    any handshake metadata is a higher-level concern than the wire format."""
    return _pack(PT_HELLO, 0, seq, 0, 0, 1, b"")


def pack_hello_ack(seq: int, stream: dict | None = None) -> bytes:
    """Confirm that the peer is a live SyLC server, not merely an accepted TCP socket.

    The optional `stream` mapping announces what this session actually sends:
    width, height, fps and the stereo layout. The receiver used to hardcode
    3840x1080 at 24 fps in both its codec format and its screen size, so any
    other geometry or cadence would have been mis-sized or mis-paced. Senders
    may still answer with no payload -- receivers keep their defaults then.
    """
    payload = b""
    if stream:
        payload = json.dumps(stream, separators=(",", ":")).encode("utf-8")
    return _pack(PT_HELLO_ACK, 0, seq, 0, 0, 1, payload)


def stream_announcement(width: int, height: int, fps: int,
                        stereo: str = "lr", hdr: str | None = None) -> dict:
    """Build the `stream` mapping for :func:`pack_hello_ack`.

    `hdr` announces the stream's transfer when it is NOT plain SDR — "pq" for
    an HEVC Main10 BT.2020/ST2084 session. Omitted (None) = SDR BT.709, which
    is also what receivers that predate the field assume.
    """
    out = {
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "stereo": stereo,
    }
    if hdr:
        out["hdr"] = str(hdr)
    return out


def parse(datagram: bytes) -> dict:
    """Parse one datagram's header (+ payload) into a dict.

    Raises ValueError on bad MAGIC, unsupported VER, a truncated header, or
    a payload whose length doesn't match the header's LEN field.
    """
    if len(datagram) < HEADER_SIZE:
        raise ValueError(
            f"truncated header: {len(datagram)} bytes < {HEADER_SIZE}"
        )
    magic, ver, pkt_type, flags, seq, pts_ms, frag_idx, frag_cnt, length = \
        struct.unpack(_HEADER_FMT, datagram[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if ver != VER:
        raise ValueError(f"unsupported version: {ver}")

    payload = datagram[HEADER_SIZE:]
    if len(payload) != length:
        raise ValueError(
            f"truncated payload: header LEN={length}, got {len(payload)}"
        )

    out = {
        "type": pkt_type,
        "seq": seq,
        "pts_ms": pts_ms,
        "flags": flags,
        "frag_idx": frag_idx,
        "frag_cnt": frag_cnt,
        "payload": payload,
    }
    if pkt_type == PT_CONTROL:
        out["control"] = json.loads(payload.decode("utf-8"))
    return out


def fragment_video(seq: int, pts_ms: int, flags: int, payload: bytes, mtu: int) -> list:
    """Split a video payload into <= mtu-sized datagrams sharing seq/pts_ms/flags."""
    chunks = [payload[i:i + mtu] for i in range(0, len(payload), mtu)] or [b""]
    frag_cnt = len(chunks)
    return [
        _pack(PT_VIDEO, flags, seq, pts_ms, idx, frag_cnt, chunk)
        for idx, chunk in enumerate(chunks)
    ]


class Reassembler:
    """Buffers fragmented video/audio datagrams (as produced by
    fragment_video) by seq, delivering the reassembled unit once every
    fragment for that seq has arrived.

    Bounded by construction: at most one seq is ever buffered at a time (the
    most recently started one). A datagram whose seq differs from the one
    currently being assembled means the previous, incomplete one is stale
    and unrecoverable, so it is dropped immediately in favor of the new
    seq -- partial fragments never accumulate across more than one seq.

    Use one Reassembler per logical stream (e.g. one for video, one for
    audio); it tracks state by seq only, not by packet type.
    """

    def __init__(self):
        self._seq = None
        self._type = None
        self._pts_ms = None
        self._flags = None
        self._frag_cnt = 1
        self._parts = {}

    # More fragments than any real access unit needs (a 4 MiB unit at the
    # smallest sane MTU). Beyond it the header is corrupt or adversarial.
    MAX_FRAG_CNT = 4096

    def push(self, dg: dict) -> list:
        # Validate the fragment geometry BEFORE touching any state: a corrupt
        # datagram must neither crash the delivery join (a frag_idx >= frag_cnt
        # could otherwise satisfy the length check with a hole in the parts)
        # nor destroy an in-progress assembly by resetting onto garbage.
        cnt = dg["frag_cnt"]
        if not 1 <= cnt <= self.MAX_FRAG_CNT or not 0 <= dg["frag_idx"] < cnt:
            return []
        if dg["seq"] != self._seq:
            # New (or restarted) seq: whatever was pending for the old one
            # is incomplete and unrecoverable now. Start fresh.
            self._seq = dg["seq"]
            self._type = dg["type"]
            self._pts_ms = dg["pts_ms"]
            self._flags = dg["flags"]
            self._frag_cnt = dg["frag_cnt"]
            self._parts = {}

        self._parts[dg["frag_idx"]] = dg["payload"]
        if len(self._parts) < self._frag_cnt:
            return []

        payload = b"".join(self._parts[i] for i in range(self._frag_cnt))
        done = {
            "type": self._type,
            "seq": self._seq,
            "pts_ms": self._pts_ms,
            "flags": self._flags,
            "frag_idx": 0,
            "frag_cnt": 1,
            "payload": payload,
        }
        self._seq = None
        self._parts = {}
        return [done]
