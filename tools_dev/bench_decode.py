"""Time the native MVC decode path and prove it stays bit-exact.

Access units are demuxed once and held in memory, so the timed region contains
decode work only -- no container I/O, no GUI, no presentation. That matches the
methodology behind the worker measurements recorded in
``mvc_decoder._native_edge264_thread_count``.

Every run also reports a CRC over the decoded planes. Two builds that decode the
same file must produce the same CRC; a build that is faster and *different* is a
regression, not an optimization.

    python tools_dev/bench_decode.py G:/APE/Video/Ruin.mkv
    python tools_dev/bench_decode.py G:/APE/Video/Ruin.mkv --sweep 0,2,3,4,6,8
    python tools_dev/bench_decode.py G:/APE/Video/Ruin.mkv --dll .perf_backup/edge264.dll.bak
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path, help="MKV/SSIF file carrying an MVC track")
    parser.add_argument(
        "--dll",
        type=Path,
        default=ROOT / "runtime" / "edge264.dll",
        help="edge264 DLL under test (default: runtime directory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="edge264 worker count for a single run (default: 3)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="comma-separated worker counts to compare in one session",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=600,
        help="access-unit pairs to preload and decode (default: 600)",
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=60_000,
        help="seek here before capturing, to skip titles/black (default: 60000)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="timed passes per configuration; the best is reported (default: 3)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the CRC pass (slightly faster, loses the correctness signal)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the results to this JSON file",
    )
    parser.add_argument(
        "--crc-out",
        type=Path,
        default=None,
        help=(
            "write the per-frame CRC list (first configuration) to this file. "
            "Comparing two builds needs the list, not a digest of it: a run "
            "that flushes one fewer tail frame has a different digest but "
            "identical pixels."
        ),
    )
    return parser.parse_args()


def preload(native, media: Path, start_ms: int, count: int) -> Tuple[bytes, List[Tuple[bytes, bytes]], object]:
    """Demux ``count`` access-unit pairs into memory, starting at a keyframe."""
    demuxer = native.MVCMatroskaDemuxer()
    if not demuxer.open(str(media)):
        raise RuntimeError(f"cannot open {media}")
    info = demuxer.get_video_info()
    if not info.hasMVC:
        raise RuntimeError("input is not reported as MVC")

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "runtime"))
    from sylc.mvc_decoder import convert_avcc_to_annexb

    headers = convert_avcc_to_annexb(demuxer.get_codec_private())
    if not headers:
        raise RuntimeError("empty codec private data")

    if start_ms > 0 and not demuxer.seek(start_ms):
        raise RuntimeError(f"seek to {start_ms} ms failed")

    ring = native.FrameRingBuffer(capacity=32)
    units: List[Tuple[bytes, bytes]] = []
    started = start_ms <= 0
    while len(units) < count:
        if not demuxer.read_next_into_ring(ring):
            break
        ok, base, dependent, _ts, keyframe, _seq = ring.pop()
        if not ok:
            break
        if not started:
            if not keyframe:
                continue
            started = True
        units.append((bytes(base), bytes(dependent)))

    demuxer.close()
    if not units:
        raise RuntimeError("no access units captured")
    return headers, units, info


def _crc_frame(frame) -> int:
    """CRC of one decoded frame, both views, all planes.

    Deliberately per-frame rather than a running total: a run that flushes one
    fewer tail frame must read as "599 frames, same pixels", not as a pixel
    mismatch.
    """
    crc = 0
    for view in (frame.base_view, getattr(frame, "dependent_view", None)):
        if view is None:
            continue
        for name in ("y_plane", "cb_plane", "cr_plane"):
            plane = getattr(view, name, None)
            if plane is None:
                continue
            try:
                crc = zlib.crc32(memoryview(plane).cast("B"), crc)
            except (TypeError, ValueError):
                # Non-contiguous view: fall back to a copy.
                import numpy as np

                crc = zlib.crc32(np.ascontiguousarray(plane).tobytes(), crc)
    return crc


def run_once(
    native,
    headers: bytes,
    units: Sequence[Tuple[bytes, bytes]],
    workers: int,
    verify: bool,
) -> dict:
    """One timed decode pass. CRC work happens outside the timed segments."""
    decoder = native.MVCDecoder()
    if not decoder.init(workers):
        raise RuntimeError(decoder.get_last_error())
    if decoder.decode_annexb_stream(headers) != 0:
        raise RuntimeError(decoder.get_last_error())

    per_unit: List[float] = []
    frame_crcs: List[int] = []
    decoded = 0
    mvc = 0
    errors = 0
    elapsed = 0.0

    for base, dependent in units:
        t0 = time.perf_counter()
        rc = decoder.decode_access_unit_pair(base, dependent)
        frames = []
        while True:
            got, frame = decoder.get_frame()
            if not got:
                break
            frames.append(frame)
        t1 = time.perf_counter()

        elapsed += t1 - t0
        per_unit.append((t1 - t0) * 1000.0)
        if rc:
            errors += 1
        for frame in frames:
            decoded += 1
            mvc += int(frame.has_mvc)
            if verify:
                frame_crcs.append(_crc_frame(frame))

    t0 = time.perf_counter()
    decoder.bump_frames()
    tail = []
    while True:
        got, frame = decoder.get_frame()
        if not got:
            break
        tail.append(frame)
    elapsed += time.perf_counter() - t0
    for frame in tail:
        decoded += 1
        mvc += int(frame.has_mvc)
        if verify:
            frame_crcs.append(_crc_frame(frame))

    decoder.close()

    per_unit.sort()

    def pct(p: float) -> float:
        if not per_unit:
            return 0.0
        idx = min(len(per_unit) - 1, int(round(p / 100.0 * (len(per_unit) - 1))))
        return per_unit[idx]

    return {
        "workers": workers,
        "decoded": decoded,
        "mvc": mvc,
        "errors": errors,
        "seconds": elapsed,
        "fps": decoded / elapsed if elapsed > 0 else 0.0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mean_ms": statistics.fmean(per_unit) if per_unit else 0.0,
        "frame_crcs": frame_crcs if verify else None,
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
    # The bench drives the worker count explicitly; the app-level override must
    # not leak in from the shell and silently flatten a sweep.
    os.environ.pop("SYLC_EDGE264_THREADS", None)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "runtime"))

    import mvc_demuxer_cpp as native

    print(f"dll     : {dll}")
    print(f"runtime : {native.edge264_runtime_status()}")

    headers, units, info = preload(native, media, args.start_ms, args.frames)
    total_bytes = sum(len(b) + len(d) for b, d in units)
    print(
        f"media   : {info.width}x{info.height} {info.fps:.3f} fps MVC | "
        f"{len(units)} AU pairs preloaded ({total_bytes / 1e6:.1f} MB)"
    )

    if args.sweep:
        configs = [int(x) for x in args.sweep.split(",") if x.strip()]
    else:
        configs = [3 if args.workers is None else args.workers]

    verify = not args.no_verify
    results = []
    reference: Optional[List[int]] = None

    print()
    header = (
        f"{'workers':>7} {'fps':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} "
        f"{'frames':>7}  {'pixels':<10}"
    )
    print(header)
    print("-" * len(header))

    for workers in configs:
        best = None
        for _ in range(max(1, args.repeat)):
            run = run_once(native, headers, units, workers, verify)
            if best is None or run["fps"] > best["fps"]:
                best = run
        assert best is not None

        verdict = "-"
        if verify:
            crcs = best["frame_crcs"] or []
            if reference is None:
                reference = crcs
                verdict = f"ref {len(crcs)}f"
                if args.crc_out:
                    args.crc_out.write_text(
                        "\n".join(f"{c:08x}" for c in crcs), encoding="utf-8")
            else:
                # Compare the common prefix: a shorter tail is a flush
                # difference, not a decode difference.
                n = min(len(reference), len(crcs))
                diff = [i for i in range(n) if reference[i] != crcs[i]]
                if diff:
                    verdict = f"DIFF @{diff[0]}"
                    best["pixel_mismatch"] = True
                elif len(crcs) != len(reference):
                    verdict = f"ok ({len(crcs)}f)"
                else:
                    verdict = "bit-exact"

        results.append(best)
        print(
            f"{best['workers']:>7} {best['fps']:>8.1f} {best['p50_ms']:>8.2f} "
            f"{best['p95_ms']:>8.2f} {best['p99_ms']:>8.2f} {best['decoded']:>7}  {verdict:<10}"
        )
        if best["errors"]:
            print(f"        !! {best['errors']} decode errors")

    mismatches = [r for r in results if r.get("pixel_mismatch")]
    if mismatches:
        print("\nFAIL: configurations disagree on decoded pixels.")
    elif verify and len(results) > 1:
        print("\nAll configurations agree on decoded pixels.")

    if args.json:
        # The per-frame CRC list is large and only meaningful within a session.
        slim = [{k: v for k, v in r.items() if k != "frame_crcs"} for r in results]
        for row, run in zip(slim, results):
            crcs = run.get("frame_crcs") or []
            row["crc_frames"] = len(crcs)
            row["crc_digest"] = f"{zlib.crc32(repr(crcs).encode()):08x}" if crcs else None
        args.json.write_text(
            json.dumps({"dll": str(dll), "media": str(media), "results": slim}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
