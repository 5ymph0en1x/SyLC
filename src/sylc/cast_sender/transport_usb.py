"""USB-C TCP transport for SyLC Cast (PC -> Quest 3).

Wraps a single asyncio TCP server (asyncio.start_server) bound to the
PC-side end of an `adb reverse` port forward. On the real link, with the
Quest connected over USB-C and the PC listening on some port P:

    adb reverse tcp:P tcp:P

forwards connections the Quest app makes to its own localhost:P out over
USB-C to the PC's localhost:P -- i.e. the SERVER (this class) binds and
listens on the PC side; the Quest app is the one that connects, exactly
as it would to reach a real network peer. No adb is involved in the tests
in this package: they connect over plain 127.0.0.1 TCP directly.

All wire-format knowledge stays in cast_sender.protocol, same as
transport_wifi.WifiTransport: this module only decides *when* to call its
pack_* helpers and *where* the resulting bytes go, plus how to dispatch
whatever parse() hands back for inbound data. The difference from Wi-Fi is
the transport underneath: TCP is a single stable, ordered byte stream, not
discrete datagrams, so:

  - No fragmentation. fragment_video is a UDP-MTU concern; over TCP every
    video/audio unit is sent as exactly one pack_video/pack_audio packet.
  - No loss/reorder/dedup handling. TCP already guarantees in-order,
    exactly-once delivery for as long as the connection is alive.
  - Packets need their own delimiting, since TCP has no datagram
    boundaries: each is framed with a 4-byte big-endian length prefix
    (struct.pack(">I", len(pkt)) + pkt) so the reader can split the
    stream back into whole packets.
  - No PT_BYE announcement on stop(): closing the writer already delivers
    a TCP FIN, which the peer's next read observes as a clean EOF -- a
    stream close is itself the "goodbye", unlike connectionless UDP where
    the peer has no other way to learn the sender is gone.
  - No separate liveness TIMEOUT either (Task 14): TCP is connection-
    oriented, so its own EOF/reset IS the client-lost signal -- when the
    read loop ends because the peer closed or reset the connection,
    on_client_lost() fires immediately, once, right there (see _read_loop).
    A reconnect is simply the next accepted connection (_on_connected runs
    again), which fires on_client the same way the first one did.

One client connects at a time (the real deployment is a single Quest over
one adb-forwarded port). A new connection simply replaces the previous
peer for send_*/dispatch purposes -- there is no fight over who "the"
client is because there is only ever supposed to be one.
"""

import asyncio
import json
import logging
import socket
import struct

from . import protocol as P
from .transport_base import TransportServer

logger = logging.getLogger(__name__)

_LEN_FMT = ">I"
_LEN_SIZE = struct.calcsize(_LEN_FMT)


def _frame(pkt: bytes) -> bytes:
    """Length-prefix one already-packed protocol.py packet for the wire."""
    return struct.pack(_LEN_FMT, len(pkt)) + pkt


def _hello_payload(msg: dict) -> dict:
    """Best-effort JSON decode of a PT_HELLO payload; {} if empty or
    unparseable (mirrors transport_wifi._hello_payload -- the handshake
    payload format is the client's to define, this transport only needs
    to hand something back to on_client without crashing on it)."""
    if not msg["payload"]:
        return {}
    try:
        return json.loads(msg["payload"].decode("utf-8"))
    except ValueError:
        return {}


class UsbTransport(TransportServer):
    """TCP implementation of TransportServer: one listening server, one peer.

    send_video/send_audio each pack a single protocol.py packet (no
    fragmentation -- TCP is a stream, not MTU-bounded) and write it
    length-prefixed to the current peer's StreamWriter.
    """

    def __init__(self):
        super().__init__()
        self._server = None
        self._writer = None        # current peer's StreamWriter
        self._reader_task = None   # Task running _read_loop for that peer
        self._seq = 0
        self._active = False
        # Bound the amount asyncio may retain after the kernel send buffer fills.
        # HEVC AUs are large, so byte-based pressure is much more meaningful than
        # counting Python callbacks/NALs.
        self._max_write_buffer = 16 * 1024 * 1024

    async def start(self, host: str, port: int) -> None:
        self._server = await asyncio.start_server(self._on_connected, host, port)
        self._active = True

    def _sock_addr(self):
        """The (host, port) this server is actually bound to."""
        return self._server.sockets[0].getsockname()

    # Client -> server frames are HELLO / control JSON, a few hundred bytes at
    # most. A length prefix beyond this is a desynced or garbage stream, and
    # honoring it would make readexactly() buffer up to 4 GiB.
    _MAX_INBOUND_BYTES = 1 * 1024 * 1024

    def _next_seq(self) -> int:
        # Masked to the header's u32: an unmasked Python int would eventually
        # overflow struct.pack(">I") and kill every subsequent send.
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return self._seq

    async def _on_connected(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
        """Called by asyncio.start_server once per accepted connection.
        The connection itself is what makes this the peer -- send_video/
        send_audio become live as soon as a client is connected, with no
        need to wait for an in-band PT_HELLO first."""
        previous = self._writer
        self._writer = writer
        self._reader_task = asyncio.current_task()
        transport = writer.transport
        if transport is not None:
            transport.set_write_buffer_limits(
                high=self._max_write_buffer,
                low=self._max_write_buffer // 4,
            )
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                # The platform default is commonly only 64-256 KiB. At
                # 500 Mbps that is a few milliseconds and needlessly turns a
                # brief scheduler stall into Python-side backpressure.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                logger.debug("[cast-usb] unable to enlarge TCP send buffer", exc_info=True)
        if previous is not None and previous is not writer and not previous.is_closing():
            previous.close()
        try:
            await self._read_loop(reader, writer)
        finally:
            # Never leave a dead StreamWriter published: doing so caused every
            # subsequent frame to hit asyncio's noisy socket.send() error path.
            if self._writer is writer:
                self._writer = None
            if not writer.is_closing():
                writer.close()

    async def _read_loop(self, reader: asyncio.StreamReader,
                          writer: asyncio.StreamWriter) -> None:
        while True:
            try:
                hdr = await reader.readexactly(_LEN_SIZE)
                (n,) = struct.unpack(_LEN_FMT, hdr)
                if not 0 < n <= self._MAX_INBOUND_BYTES:
                    # Desynced/garbage stream: treat exactly like a reset --
                    # the except below fires on_client_lost and this
                    # connection ends; a real client reconnects cleanly.
                    raise ConnectionResetError(
                        f"inbound frame length {n} outside 1..{self._MAX_INBOUND_BYTES}")
                pkt = await reader.readexactly(n)
            except (asyncio.IncompleteReadError, ConnectionError):
                # Peer closed / reset mid-frame: TCP's own EOF/RST IS the
                # client-lost signal (Task 14) -- no separate heartbeat needed
                # over a connection-oriented transport. NOT reached by a
                # stop()-driven cancellation: that raises CancelledError at
                # this same await, which this except clause does not catch,
                # so a deliberate teardown propagates straight out without
                # firing on_client_lost.
                if self._writer is writer and self.on_client_lost is not None:
                    try:
                        self.on_client_lost()
                    except Exception:
                        # A raising callback must not leave this connection's task
                        # exiting with an unretrieved exception -- swallow + log,
                        # same defensive posture as WifiTransport._check_liveness.
                        logger.exception("[cast-usb] on_client_lost callback raised")
                return  # stop serving this connection
            self._dispatch(pkt, writer)

    def _dispatch(self, pkt: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            msg = P.parse(pkt)
        except ValueError:
            return  # malformed frame -- drop, keep serving

        if msg["type"] == P.PT_HELLO:
            # Confirm the application-level peer before the Quest starts
            # AudioTrack or declares a transient ADB socket fully connected.
            writer.write(_frame(P.pack_hello_ack(self._next_seq(), self.stream_format)))
            if self.on_client is not None:
                self.on_client(writer.get_extra_info("peername"), _hello_payload(msg))
        elif msg["type"] == P.PT_CONTROL:
            if self.on_control is not None:
                self.on_control(msg)

    def send_video(self, pts_ms: int, nal: bytes, keyframe: bool) -> bool:
        writer = self._writer
        if writer is None or writer.is_closing():
            return False
        transport = writer.transport
        if (transport is not None and
                transport.get_write_buffer_size() >= self._max_write_buffer):
            return False
        seq = self._next_seq()
        flags = 1 if keyframe else 0
        header = P.pack_header(P.PT_VIDEO, flags, seq, pts_ms, 0, 1, len(nal))
        # Three writes preserve the exact TCP byte stream while retaining the
        # original Python bytes object for the large payload (zero extra
        # multi-megabyte concatenation/copy on the GIL).
        writer.write(struct.pack(_LEN_FMT, len(header) + len(nal)))
        writer.write(header)
        writer.write(nal)
        return True

    def send_audio(self, pts_ms: int, pcm: bytes) -> bool:
        writer = self._writer
        if writer is None or writer.is_closing():
            return False
        transport = writer.transport
        if (transport is not None and
                transport.get_write_buffer_size() >= self._max_write_buffer):
            return False
        seq = self._next_seq()
        header = P.pack_header(P.PT_AUDIO, 0, seq, pts_ms, 0, 1, len(pcm))
        writer.write(struct.pack(_LEN_FMT, len(header) + len(pcm)))
        writer.write(header)
        writer.write(pcm)
        return True

    async def stop(self) -> None:
        if not self._active:
            return
        self._active = False

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass  # torn down peer may already be gone/erroring -- fine
            self._reader_task = None

        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
