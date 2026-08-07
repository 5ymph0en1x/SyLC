r"""Reproducible micro-profile of SyLC's production 2D->3D stages.

The native extension must be built with ``_depth_infer_benchmark``.  The tool
prefers ``build_py314/python/Release`` over ``runtime`` so a development build
can be measured without replacing the packaged module.

Examples::

    .venv\Scripts\python tools_dev\profile_synth3d.py
    .venv\Scripts\python tools_dev\profile_synth3d.py --mode inference -n 60
    .venv\Scripts\python tools_dev\profile_synth3d.py --mode cpu -n 12
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD_MODULE = ROOT / "build_py314" / "python" / "Release"
RUNTIME = ROOT / "runtime"


def _load_native(ort_dir: Path):
    sys.path.insert(0, str(BUILD_MODULE if BUILD_MODULE.exists() else RUNTIME))
    if os.name == "nt":
        # Keep the handles alive for the duration of the process.
        handles = []
        for directory in (RUNTIME, ort_dir):
            if directory.exists():
                handles.append(os.add_dll_directory(str(directory)))
        globals()["_DLL_HANDLES"] = handles
    import mvc_demuxer_cpp as native
    return native


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min_ms": ordered[0],
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
    }


def _timed(fn, iterations: int, warmup: int = 2) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return _stats(samples)


def _scene(width: int, height: int):
    y, x = np.mgrid[0:height, 0:width]
    base = (
        0.45
        + 0.22 * np.sin(x * (8.0 * np.pi / max(1, width)))
        + 0.18 * np.cos(y * (6.0 * np.pi / max(1, height)))
        + 0.12 * (((x - 0.38 * width) ** 2 + (y - 0.55 * height) ** 2)
                  < (0.13 * min(width, height)) ** 2)
    ).astype(np.float32)
    base = np.clip(base, 0.0, 1.0)
    return (
        np.ascontiguousarray(np.roll(base, (0, -3), axis=(0, 1)).ravel()),
        np.ascontiguousarray(base.ravel()),
        np.ascontiguousarray(np.roll(base, (0, 3), axis=(0, 1)).ravel()),
    )


def profile_inference(native, ort_dir: Path, iterations: int, warmup: int):
    rgb = np.random.default_rng(42).integers(
        0, 256, (1080, 1920, 3), dtype=np.uint8)
    cases = (
        ("Quality Scope", "da3_base_756x322.onnx", 756, 322),
        ("Quality square", "da3_base_756.onnx", 756, 756),
        ("Balanced Scope", "da3_base_518x210.onnx", 518, 210),
        ("Performance Scope", "da3_small_518x210.onnx", 518, 210),
    )
    results = {}
    for label, filename, width, height in cases:
        model = ROOT / "models" / filename
        if not model.exists():
            continue
        raw = dict(native._depth_infer_benchmark(
            str(model), str(ort_dir), rgb, width, width, height,
            warmup, iterations))
        raw["warmup_first_ms"] = raw["warmup_ms"][0] if warmup else 0.0
        raw.pop("samples_ms", None)
        raw.pop("warmup_ms", None)
        results[label] = raw
        print(
            f"{label:<20} {raw['provider']:<9} "
            f"p50={raw['p50_ms']:7.2f}  p95={raw['p95_ms']:7.2f}  "
            f"mean={raw['mean_ms']:7.2f} ms  init={raw['init_ms']:8.1f} ms")
    return results


def profile_cpu(native, iterations: int, threads: int):
    cases = (
        ("Scope 756x322", 756, 322),
        ("Square 756x756", 756, 756),
        ("Scope 518x210", 518, 210),
        ("Square 518x518", 518, 518),
    )
    results = {}
    for label, width, height in cases:
        prev, cur, nxt = _scene(width, height)
        depth = np.ascontiguousarray(cur.reshape(height, width))
        confidence = np.full((height, width), 0.85, dtype=np.float32)
        rgb = np.repeat(
            np.clip(depth[..., None] * 255.0, 0, 255).astype(np.uint8),
            3, axis=2)
        depth_q16 = np.clip(depth * 65535.0, 0, 65535).astype(np.uint16)
        flow_x = np.full(width * height, 0.35, dtype=np.float32)
        flow_y = np.full(width * height, -0.20, dtype=np.float32)
        reliability = np.full(width * height, 0.85, dtype=np.float32)
        motion = np.abs(cur - prev).astype(np.float32)
        boundary = np.zeros(width * height, dtype=np.float32)

        stabilizer = native.DepthStabilizer(width * height)
        stabilizer.worker_threads = threads
        stabilizer.set_source_dt_ms(41.7)
        stabilizer.set_update_dt_ms(41.7)
        stabilizer.step(cur, motion, 0.0, confidence.ravel(), boundary)

        stages = {
            "flow_one_direction": _timed(
                lambda: native._synth3d_estimate_flow_test(
                    prev, cur, width, height, threads), iterations),
            "flow_two_directions_plus_fusion": _timed(
                lambda: native._synth3d_fuse_flow_test(
                    prev, cur, nxt, width, height, 1.0, threads), iterations),
            "boundary_plus_refine": _timed(
                lambda: native._synth3d_refine_depth_test(
                    depth, depth, confidence, threads), iterations),
            "geometry_map": _timed(
                lambda: native._synth3d_build_geometry_test(
                    depth_q16, rgb, confidence, threads), iterations),
            "stabilizer_step": _timed(
                lambda: stabilizer.step(
                    cur, motion, 0.02, confidence.ravel(), boundary), iterations),
            "stabilizer_reproject": _timed(
                lambda: stabilizer.reproject(
                    flow_x, flow_y, reliability, width, height), iterations),
        }
        results[label] = stages
        print(f"\n{label} ({threads} threads)")
        for name, timing in stages.items():
            print(
                f"  {name:<36} p50={timing['p50_ms']:7.2f}  "
                f"p95={timing['p95_ms']:7.2f} ms")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "inference", "cpu"),
                        default="all")
    parser.add_argument("-n", "--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--ort-dir", type=Path, default=ROOT / "ort_tensorrt")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.threads <= 0:
        parser.error("iterations/threads must be positive and warmup non-negative")

    native = _load_native(args.ort_dir.resolve())
    print(f"native: {native.__file__}")
    report = {"native": str(native.__file__)}
    if args.mode in ("all", "inference"):
        print("\nPersistent DepthEngine inference")
        report["inference"] = profile_inference(
            native, args.ort_dir.resolve(), args.iterations, args.warmup)
    if args.mode in ("all", "cpu"):
        print("\nProduction CPU stages")
        report["cpu"] = profile_cpu(native, args.iterations, args.threads)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
