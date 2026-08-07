"""Exercise the complete native MVC demux + edge264 path on a real media file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path, help="MKV file containing an MVC track")
    parser.add_argument(
        "--dll",
        type=Path,
        default=ROOT / "runtime" / "edge264.dll",
        help="edge264 DLL to load (default: runtime directory)",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--seek-ms", type=int, default=30_000)
    parser.add_argument("--seek-frames", type=int, default=180)
    parser.add_argument("--restart-cycles", type=int, default=5)
    return parser.parse_args()


def drain(decoder, retained_y=None):
    decoded = 0
    mvc_frames = 0
    while True:
        got, frame = decoder.get_frame()
        if not got:
            break
        decoded += 1
        mvc_frames += int(frame.has_mvc)
        assert frame.base_view.y_plane.shape == (
            frame.base_view.height,
            frame.base_view.width,
        )
        if frame.has_mvc:
            assert frame.dependent_view.y_plane.shape == (
                frame.dependent_view.height,
                frame.dependent_view.width,
            )
            assert frame.picture_order_cnt_mvc == frame.picture_order_cnt, (
                frame.picture_order_cnt,
                frame.picture_order_cnt_mvc,
            )
        if retained_y is None:
            retained_y = frame.base_view.y_plane
    return decoded, mvc_frames, retained_y


def decode_units(decoder, demuxer, ring, count, require_keyframe=False):
    decoded = 0
    mvc_frames = 0
    retained_y = None
    errors = []
    started = not require_keyframe
    units_read = 0

    for _ in range(count):
        if not demuxer.read_next_into_ring(ring):
            break
        ok, base, dependent, _timestamp, keyframe, _sequence = ring.pop()
        assert ok
        units_read += 1
        if not started and not keyframe:
            continue
        started = True
        result = decoder.decode_access_unit_pair(base, dependent)
        if result:
            errors.append((result, decoder.get_last_error()))
        got, got_mvc, retained_y = drain(decoder, retained_y)
        decoded += got
        mvc_frames += got_mvc

    decoder.bump_frames()
    got, got_mvc, retained_y = drain(decoder, retained_y)
    decoded += got
    mvc_frames += got_mvc
    return {
        "decoded": decoded,
        "mvc": mvc_frames,
        "retained_y": retained_y,
        "errors": errors,
        "started": started,
        "units_read": units_read,
    }


def main() -> int:
    args = parse_args()
    media = args.media.resolve()
    dll = args.dll.resolve()
    if not media.is_file():
        raise FileNotFoundError(media)
    if not dll.is_file():
        raise FileNotFoundError(dll)

    os.environ["SYLC_EDGE264_DLL"] = str(dll)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "runtime"))

    import mvc_demuxer_cpp as native
    from sylc.mvc_decoder import convert_avcc_to_annexb

    assert hasattr(native, "MVCDecoder"), "MVCDecoder is not exposed by the .pyd"
    print("runtime:", native.edge264_runtime_status())

    decoder = native.MVCDecoder()
    assert decoder.init(args.workers), decoder.get_last_error()

    demuxer = native.MVCMatroskaDemuxer()
    assert demuxer.open(str(media)), f"Cannot open {media}"
    info = demuxer.get_video_info()
    assert info.hasMVC, "Input is not reported as MVC"
    print(f"video: {info.width}x{info.height} {info.fps:.3f} fps MVC")

    headers = convert_avcc_to_annexb(demuxer.get_codec_private())
    assert headers
    assert decoder.decode_annexb_stream(headers) == 0, decoder.get_last_error()

    ring = native.FrameRingBuffer(capacity=32)
    first = decode_units(decoder, demuxer, ring, args.frames)
    minimum = max(1, min(first["units_read"], args.frames) - 8)
    assert first["decoded"] >= minimum, first
    assert first["mvc"] == first["decoded"], first
    assert not first["errors"], first["errors"][:5]
    retained = first["retained_y"]
    checksum = int(retained[::64, ::64].sum()) if retained is not None else -1
    print(
        f"initial: {first['decoded']} MVC frames, "
        f"retained checksum={checksum}, errors=0"
    )

    decoder.close()
    assert not decoder.is_initialized()

    assert demuxer.seek(args.seek_ms)
    assert decoder.init(args.workers), decoder.get_last_error()
    assert decoder.decode_annexb_stream(headers) == 0, decoder.get_last_error()
    ring.clear()
    seek = decode_units(
        decoder,
        demuxer,
        ring,
        args.seek_frames,
        require_keyframe=True,
    )
    assert seek["started"], "No keyframe found after seek"
    assert seek["decoded"] > 0, seek
    assert seek["mvc"] == seek["decoded"], seek
    assert not seek["errors"], seek["errors"][:5]
    print(f"seek @{args.seek_ms} ms: {seek['decoded']} MVC frames, errors=0")
    decoder.close()

    for _ in range(args.restart_cycles):
        assert decoder.init(args.workers), decoder.get_last_error()
        assert decoder.decode_annexb_stream(headers) == 0, decoder.get_last_error()
        decoder.close()
    print(f"restart cycles: {args.restart_cycles}")

    demuxer.close()
    print("native edge264 probe: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
