"""Wi-Fi transports for SyLC Cast (PC -> Quest 3).

Two classes live here:

  * WifiTransport -- the original pure-UDP datagram transport. Kept for its
    loopback tests and as a possible future low-latency path, but NOT what a
    Wi-Fi session speaks: the Quest receiver has no UDP media path.
  * WifiMediaTransport -- what kind="wifi" actually uses: the USB-C TCP media
    server plus a UDP endpoint on the same port that answers the Quest's
    broadcast discovery PT_HELLO (see the class docstring for the history).

All wire-format knowledge stays in cast_sender.protocol: this module only
decides *when* to call its pack_*/fragment_video helpers and *where* the
resulting bytes go, plus how to dispatch whatever parse() hands back.

Real UDP is adversarial -- reordering, loss, duplication, and outright noise
from anything else that happens to hit this port -- so every inbound
datagram is parsed defensively: a malformed one is dropped silently and
never crashes the endpoint or bubbles out of datagram_received.
"""

import asyncio
import json
import logging
import socket
import time

from . import protocol as P
from .transport_base import TransportServer
from .transport_usb import UsbTransport

logger = logging.getLogger(__name__)


class _Endpoint(asyncio.DatagramProtocol):
    """Forwards inbound datagrams to the owning WifiTransport, which holds
    all the actual parsing/dispatch logic."""

    def __init__(self, owner: "WifiTransport"):
        self._owner = owner

    def datagram_received(self, data, addr):
        self._owner._on_datagram(data, addr)


def _hello_payload(msg: dict) -> dict:
    """Best-effort JSON decode of a PT_HELLO payload; {} if empty or
    unparseable. The handshake payload format is the client's to define --
    this transport only needs to hand something back to on_client without
    crashing on it."""
    if not msg["payload"]:
        return {}
    try:
        return json.loads(msg["payload"].decode("utf-8"))
    except ValueError:
        return {}


class WifiTransport(TransportServer):
    """UDP implementation of TransportServer: one bound endpoint, one peer.

    Video is sent via protocol.fragment_video, so a single encoded NAL
    larger than `mtu` still arrives as several MTU-sized datagrams sharing
    one sequence number; audio is assumed to always fit in one datagram.

    Liveness (Task 14): UDP is connectionless, so the only way to notice a
    client is gone is inbound silence. `_last_rx` is the monotonic time of
    the last inbound datagram (or of set_peer()/a PT_HELLO establishing the
    peer in the first place -- see set_peer()/_on_datagram); a self-
    rescheduling IO-loop check (`_schedule_liveness_check`) fires
    `on_client_lost()` exactly ONCE if the known peer has gone silent longer
    than `client_timeout_s`. A `_client_lost_fired` latch prevents repeat
    fires for the same silence episode; it clears on the next inbound
    datagram (a genuine PT_HELLO reconnect, or anything else -- e.g. a
    keepalive), so a later silence is detected fresh again.
    """

    CLIENT_TIMEOUT_S = 2.0   # a known peer silent longer than this is presumed gone

    def __init__(self, mtu: int = 1200, client_timeout_s: float = None):
        super().__init__()
        self._mtu = mtu
        self._transport = None
        self._peer = None
        self._seq = 0
        self._active = False
        self._client_timeout_s = self.CLIENT_TIMEOUT_S if client_timeout_s is None else client_timeout_s
        self._check_interval_s = max(0.02, self._client_timeout_s / 4.0)
        self._last_rx = None                # monotonic time of the last proof-of-life; None = unknown
        self._client_lost_fired = False     # latch: fire on_client_lost at most once per silence episode
        self._liveness_handle = None        # the pending call_later for _schedule_liveness_check

    async def start(self, host: str, port: int) -> None:
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _Endpoint(self), local_addr=(host, port),
        )
        # A fragmented 4K keyframe is a burst of hundreds of MTU-sized
        # datagrams; the platform's default UDP send buffer (commonly 64 KiB)
        # silently drops the tail of such a burst before it ever reaches the
        # wire. Give the kernel room to absorb one whole access unit.
        sock = transport.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            except OSError:
                logger.debug("[cast-wifi] unable to enlarge UDP send buffer", exc_info=True)
        self._transport = transport
        self._active = True
        self._schedule_liveness_check()

    def set_peer(self, host: str, port: int) -> None:
        """Test/handshake hook: where send_video/send_audio/PT_BYE go. Also
        counts as proof the peer is known/alive right now -- same as any
        inbound datagram -- so the liveness timeout has a reference point even
        when a caller wires the peer directly instead of via a real PT_HELLO."""
        self._peer = (host, port)
        self._last_rx = time.monotonic()
        self._client_lost_fired = False

    def _sock_addr(self):
        """The (host, port) this endpoint is actually bound to."""
        return self._transport.get_extra_info("sockname")

    def _next_seq(self) -> int:
        # Masked to the header's u32: an unmasked Python int would eventually
        # overflow struct.pack(">I") and kill every subsequent send.
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return self._seq

    def send_video(self, pts_ms: int, nal: bytes, keyframe: bool) -> None:
        if self._transport is None or self._peer is None:
            return
        seq = self._next_seq()
        flags = 1 if keyframe else 0
        for dg in P.fragment_video(seq, pts_ms, flags, nal, self._mtu):
            self._transport.sendto(dg, self._peer)

    def send_audio(self, pts_ms: int, pcm: bytes) -> None:
        if self._transport is None or self._peer is None:
            return
        seq = self._next_seq()
        self._transport.sendto(P.pack_audio(seq, pts_ms, pcm), self._peer)

    def _on_datagram(self, data: bytes, addr) -> None:
        try:
            msg = P.parse(data)
        except ValueError:
            return  # malformed datagram -- drop silently, keep serving

        # Proof of life for the current peer (Task 14): refresh the liveness
        # timestamp and clear a previously-fired client-lost latch, so a
        # subsequent silence is detected fresh -- covers both a genuine
        # PT_HELLO reconnect and a plain keepalive. Scoped to the CURRENT peer
        # (or a HELLO, which re-establishes one): a well-formed datagram from
        # some OTHER host on the LAN must not keep a dead peer looking alive.
        if msg["type"] == P.PT_HELLO or addr == self._peer:
            self._last_rx = time.monotonic()
            self._client_lost_fired = False

        if msg["type"] == P.PT_HELLO:
            if self._peer is not None and addr != self._peer:
                logger.info("[cast-wifi] peer switched by HELLO: %s -> %s",
                            self._peer, addr)
            self._peer = addr
            if self._transport is not None:
                self._transport.sendto(
                    P.pack_hello_ack(self._next_seq(), self.stream_format), addr
                )
            if self.on_client is not None:
                self.on_client(addr, _hello_payload(msg))
        elif msg["type"] == P.PT_CONTROL:
            if self.on_control is not None:
                self.on_control(msg)

    def _schedule_liveness_check(self) -> None:
        """Self-rescheduling IO-loop check (same idiom as
        loopback_receiver._schedule_drain): polls _check_liveness every
        _check_interval_s. No-ops once the transport is no longer active;
        the pending call is cancelled explicitly in stop()."""
        if not self._active:
            return
        self._check_liveness()
        loop = asyncio.get_running_loop()
        self._liveness_handle = loop.call_later(self._check_interval_s, self._schedule_liveness_check)

    def _check_liveness(self) -> None:
        """Fire on_client_lost() ONCE if a known peer has gone silent past
        client_timeout_s. No-op while no peer is known yet (nothing to lose),
        before any inbound datagram/set_peer has established a reference
        time, or once already fired for the current silence episode."""
        if self._peer is None or self._last_rx is None or self._client_lost_fired:
            return
        if time.monotonic() - self._last_rx > self._client_timeout_s:
            self._client_lost_fired = True
            if self.on_client_lost is not None:
                try:
                    self.on_client_lost()
                except Exception:
                    # A raising callback must NEVER kill _schedule_liveness_check's
                    # own reschedule (the call_later line right after this method
                    # returns) -- that would silently and permanently stop ALL
                    # future liveness checks for this transport, not just this one.
                    logger.exception("[cast-wifi] on_client_lost callback raised")

    async def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._liveness_handle is not None:
            self._liveness_handle.cancel()
            self._liveness_handle = None
        if self._transport is not None and self._peer is not None:
            self._transport.sendto(P.pack_bye(self._next_seq()), self._peer)
            # Give the proactor a real (if tiny) slice of wall-clock time to
            # observe the write's completion before we close -- on Windows,
            # ProactorEventLoop's datagram transport only nulls out its
            # socket once a pending write's completion callback has run, and
            # closing while one is still in flight otherwise leaves that to
            # __del__ (an "unclosed transport" ResourceWarning at GC time,
            # not a clean explicit close).
            await asyncio.sleep(0.01)
        if self._transport is not None:
            self._transport.close()
            await asyncio.sleep(0.01)  # let the close's own teardown callback run
            self._transport = None
        self._peer = None


class _DiscoveryEndpoint(asyncio.DatagramProtocol):
    """Forwards inbound discovery datagrams to the owning WifiMediaTransport."""

    def __init__(self, owner: "WifiMediaTransport"):
        self._owner = owner

    def datagram_received(self, data, addr):
        self._owner._on_discovery_datagram(data, addr)


class WifiMediaTransport(UsbTransport):
    """What a Wi-Fi cast session ACTUALLY is on the wire, matched to the Quest
    receiver app:

      - media/control run over the SAME TCP server as USB-C. The Quest's only
        media path is SyLcTcpClient (TCP; there is no UDP media receiver in the
        app at all), and TCP is what its reassembly/feedback/reconnect logic is
        built on;
      - plus one UDP endpoint on the SAME port whose only job is answering the
        Quest's broadcast discovery PT_HELLO with a unicast PT_HELLO_ACK --
        CastDiscovery.kt takes the reply's SOURCE ADDRESS as "the PC" and then
        TCP-connects to it on this same port.

    History: the session transport for kind="wifi" used to be the pure-UDP
    WifiTransport above. That made Wi-Fi casting impossible end to end --
    discovery succeeded (the UDP endpoint ACKed the broadcast), then the
    Quest's TCP connection to the same port found no listener, while the
    sender aimed its UDP media stream at the Quest's already-closed discovery
    socket. The UDP WifiTransport class remains for its loopback tests and as
    a possible future low-latency path; it is simply no longer what a Wi-Fi
    session speaks.

    A failed UDP bind is deliberately non-fatal: the TCP server is up, so a
    manually-entered host still casts -- only broadcast discovery is lost.
    """

    def __init__(self):
        super().__init__()
        self._discovery = None

    async def start(self, host: str, port: int) -> None:
        await super().start(host, port)
        # Bind the discovery endpoint to the port the TCP server ACTUALLY got
        # (they must match: the Quest broadcasts to, then TCP-connects to, ONE
        # configured port; port=0 in tests would otherwise split them).
        tcp_port = self._sock_addr()[1]
        loop = asyncio.get_running_loop()
        try:
            udp, _ = await loop.create_datagram_endpoint(
                lambda: _DiscoveryEndpoint(self), local_addr=(host, tcp_port),
            )
            self._discovery = udp
        except OSError:
            logger.exception(
                "[cast-wifi] discovery endpoint failed to bind udp:%s:%d -- "
                "casting still works with a manually-entered host", host, tcp_port)

    def _on_discovery_datagram(self, data: bytes, addr) -> None:
        try:
            msg = P.parse(data)
        except ValueError:
            return  # not ours -- drop silently, keep answering real HELLOs
        if msg["type"] == P.PT_HELLO and self._discovery is not None:
            logger.info("[cast-wifi] discovery HELLO from %s -- ACKing", addr)
            self._discovery.sendto(
                P.pack_hello_ack(self._next_seq(), self.stream_format), addr)

    async def stop(self) -> None:
        if self._discovery is not None:
            self._discovery.close()
            await asyncio.sleep(0)      # let the close's teardown callback run
            self._discovery = None
        await super().stop()
