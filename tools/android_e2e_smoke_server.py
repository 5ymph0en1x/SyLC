"""Feed the Android receiver a short deterministic TCP smoke-test stream.

This is intentionally tiny and dependency-free beyond the project itself. It
is useful after rebuilding the APK: establish ``adb reverse tcp:47420
tcp:47420``, start this script, then press Connect in the Android receiver.
The server exercises the HELLO handshake, receiver feedback, PCM playback,
HEVC MediaCodec configuration, and a clean TCP teardown.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sylc.cast_sender.transport_usb import UsbTransport  # noqa: E402


async def run(host: str, port: int, duration: float) -> None:
    transport = UsbTransport()
    connected = asyncio.Event()
    feedback_count = 0

    def on_client(peer, _metadata) -> None:
        print(f"client={peer}", flush=True)
        connected.set()

    def on_control(message) -> None:
        nonlocal feedback_count
        control = message.get("control", {})
        if control.get("kind") == "bwfeedback":
            feedback_count += 1
        print(f"control={control}", flush=True)

    transport.on_client = on_client
    transport.on_control = on_control
    await transport.start(host, port)
    print(f"ready={host}:{port}", flush=True)

    try:
        await asyncio.wait_for(connected.wait(), timeout=30.0)
        hevc = (
            PROJECT_ROOT / "tests" / "cast" / "fixtures" / "sbs_cbr.h265"
        ).read_bytes()
        transport.send_video(0, hevc, keyframe=True)

        packet_duration = 0.020
        packets = max(1, round(duration / packet_duration))
        for index in range(packets):
            # 20 ms, signed PCM16 LE, stereo, 48 kHz.
            transport.send_audio(index * 20, bytes(3_840))
            await asyncio.sleep(packet_duration)

        # Leave time for the receiver's final 500-ms feedback interval.
        await asyncio.sleep(0.75)
        print(f"feedback_count={feedback_count}", flush=True)
    finally:
        await transport.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47_420)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    asyncio.run(run(args.host, args.port, args.duration))


if __name__ == "__main__":
    main()
