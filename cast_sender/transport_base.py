"""Transport interface for SyLC Cast (PC -> Quest 3).

A TransportServer moves the packets produced/consumed by cast_sender.protocol
between the PC sender and a single connected client. It knows nothing about
the wire byte layout -- that stays entirely in protocol.py -- only about
delivery: when to fragment/send outbound video and audio, and how to route
whatever comes back in (a client handshake, or in-band control messages).

This is the seam between the controller (Task 12, which just calls
send_video/send_audio and wires on_control/on_client) and each concrete link:
Wi-Fi UDP (transport_wifi.py) today, USB-C (transport_usb.py) later. Both
implement this same ABC so the controller never has to know which one it's
talking to.
"""

import abc


class TransportServer(abc.ABC):
    def __init__(self):
        self.on_control = None      # callable(dict) set by controller -- receives parsed PT_CONTROL dicts
        self.on_client = None       # callable(addr, hello:dict) -- fired on PT_HELLO handshake
        self.on_client_lost = None  # callable() set by controller (Task 14) -- fired ONCE when the
                                     # connected client is presumed gone: Wi-Fi = an inbound-liveness
                                     # timeout, USB-C = the TCP reader hitting EOF/reset (the connection
                                     # close IS the signal there, no timeout needed).
        self.stream_format = None   # dict from protocol.stream_announcement(), or None.
                                     # Sent in PT_HELLO_ACK so the receiver sizes its decoder and
                                     # screen from what we actually send instead of assuming.

    @abc.abstractmethod
    async def start(self, host: str, port: int) -> None:
        """Open the transport and begin serving. Idempotent-start is not
        required; callers are expected to start each instance once."""

    @abc.abstractmethod
    def send_video(self, pts_ms: int, nal: bytes, keyframe: bool) -> bool | None:
        """Send one encoded video access unit, fragmenting if the transport
        needs to. Not a coroutine: implementations must not block the
        caller (e.g. the render thread) waiting on the network. Implementations
        may return False when local byte backpressure rejects the unit."""

    @abc.abstractmethod
    def send_audio(self, pts_ms: int, pcm: bytes) -> bool | None:
        """Send one chunk of encoded/PCM audio."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Tear down the transport (notify the peer if possible, release
        sockets/tasks). Must be safe to call more than once."""
