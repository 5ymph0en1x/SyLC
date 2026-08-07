r"""Profile temporal look-ahead stereo reconstruction on a real video pair.

The tool is intentionally an offline visual/metric gate before the D3D11
implementation.  It uses the shipped Depth Anything 3 TensorRT path and the
same native CPU flow as SharedDepthService, then compares the established
nearest-background fill against pixels recovered from a future frame.

Example:
    .venv\Scripts\python.exe tools_dev\profile_lookahead_stereo.py ^
      "H:\movie.mkv" --seek 600 --future-frames 1 ^
      --output tools_dev\lookahead_oblivion
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build_py314" / "python" / "Release"))
sys.path.insert(1, str(ROOT / "src"))
_DLL_HANDLES = [
    os.add_dll_directory(str(path))
    for path in (ROOT / "runtime", ROOT / "ort_tensorrt")
    if path.exists()
]

import mvc_demuxer_cpp as native  # noqa: E402
from sylc.lookahead_stereo_reconstruction import (  # noqa: E402
    bgr_to_yuv,
    estimate_future_to_current_flow,
    reconstruction_metrics,
    reconstruct_stereo,
    yuv_to_bgr,
)


PRESETS = {
    "quality": ("da3_base_756x322.onnx", 756, 322),
    "balanced": ("da3_base_518x210.onnx", 518, 210),
    "performance": ("da3_small_518x210.onnx", 518, 210),
}


class Timer:
    def __init__(self, samples: dict[str, list[float]], name: str):
        self.samples = samples
        self.name = name
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.samples.setdefault(self.name, []).append(
            (time.perf_counter() - self.started) * 1000.0)


def _distribution(values):
    if not values:
        return {}
    ordered = sorted(float(v) for v in values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1,
                           max(0, int(np.ceil(0.95 * len(ordered))) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _decode_pair(video: Path, seek: float, future_frames: int):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not fps > 0.0:
            raise RuntimeError("video reports no usable frame rate")
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seek) * 1000.0)
        frames = []
        for _ in range(future_frames + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("decode ended before the look-ahead pair")
            frames.append(frame)
        return frames[0], frames[-1], fps
    finally:
        capture.release()


def _normalize_depth_pair(current: np.ndarray, future: np.ndarray):
    current = np.asarray(current, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    lo, hi = np.percentile(current, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1.0e-6:
        return np.full_like(current, 0.5), np.full_like(future, 0.5)
    scale = 1.0 / float(hi - lo)
    return (np.clip((current - lo) * scale, 0.0, 1.0),
            np.clip((future - lo) * scale, 0.0, 1.0))


def _depth(model: Path, ort_dir: Path, frame: np.ndarray,
           width: int, height: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.asarray(native.depth_infer_test(
        str(model), str(ort_dir), rgb, width, width, height),
        dtype=np.float32)


def _panel(image: np.ndarray, label: str, width: int = 756):
    frame = np.asarray(image)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[1] != width:
        target_h = max(1, round(frame.shape[0] * width / frame.shape[1]))
        frame = cv2.resize(frame, (width, target_h), interpolation=cv2.INTER_AREA)
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (8, 8, 8), -1)
    cv2.putText(frame, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return frame


def _confidence_panel(left, right):
    confidence = np.maximum(left.confidence, right.confidence)
    heat = cv2.applyColorMap(
        np.clip(confidence * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO)
    mask = left.holes | right.holes
    heat[~mask] = (20, 20, 20)
    return heat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--seek", type=float, default=600.0)
    parser.add_argument("--future-frames", type=int, default=1)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="quality")
    parser.add_argument("--strength-pct", type=float, default=1.5)
    parser.add_argument("--convergence", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--flow-backend", choices=("auto", "nvof", "cpu"),
                        default="auto")
    parser.add_argument("--nvof-perf", choices=("fast", "medium", "slow"),
                        default="medium")
    parser.add_argument("--nvof-grid", choices=(1, 2, 4), type=int, default=4)
    parser.add_argument("--flow-iterations", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tools_dev" / "lookahead_profile")
    args = parser.parse_args()
    if not args.video.exists():
        parser.error(f"video is missing: {args.video}")
    if args.future_frames < 1 or args.future_frames > 8:
        parser.error("future-frames must be in 1..8")
    if args.flow_iterations < 1 or args.flow_iterations > 500:
        parser.error("flow-iterations must be in 1..500")

    model_name, width, height = PRESETS[args.preset]
    model = ROOT / "models" / model_name
    ort_dir = ROOT / "ort_tensorrt"
    if not model.exists():
        parser.error(f"model is missing: {model}")
    samples: dict[str, list[float]] = {}

    with Timer(samples, "decode_ms"):
        current_bgr, future_bgr, fps = _decode_pair(
            args.video, args.seek, args.future_frames)
    current_yuv = bgr_to_yuv(current_bgr, width, height)
    future_yuv = bgr_to_yuv(future_bgr, width, height)

    # depth_infer_test creates an isolated engine for each call.  This makes
    # depth_ms a cold-path number; the live hot TensorRT time remains the
    # synth3d status' infer_ms.  Flow/reconstruction below are hot-path costs.
    with Timer(samples, "depth_current_cold_ms"):
        current_depth_raw = _depth(model, ort_dir, current_bgr, width, height)
    with Timer(samples, "depth_future_cold_ms"):
        future_depth_raw = _depth(model, ort_dir, future_bgr, width, height)
    current_depth, future_depth = _normalize_depth_pair(
        current_depth_raw, future_depth_raw)

    nvof_session = None
    use_nvof = (hasattr(native, "NvofFlow") and
                args.flow_backend in ("auto", "nvof"))
    if use_nvof:
        with Timer(samples, "flow_init_ms"):
            nvof_session = native.NvofFlow(
                width, height, args.nvof_perf, args.nvof_grid)
    flow_kwargs = dict(
        native_module=native, threads=args.threads,
        backend=args.flow_backend, nvof_perf=args.nvof_perf,
        nvof_grid=args.nvof_grid, nvof_session=nvof_session)
    # The first execute initializes NVOFA's internal temporal resources.  The
    # live service pays it once at session start, never on a displayed frame.
    if use_nvof:
        estimate_future_to_current_flow(
            current_yuv[..., 0], future_yuv[..., 0], width, height,
            **flow_kwargs)
    flow = None
    for _ in range(args.flow_iterations):
        with Timer(samples, "flow_ms"):
            flow = estimate_future_to_current_flow(
                current_yuv[..., 0], future_yuv[..., 0], width, height,
                **flow_kwargs)
    assert flow is not None
    with Timer(samples, "reconstruct_ms"):
        result = reconstruct_stereo(
            current_yuv, future_yuv, current_depth, flow,
            future_depth=future_depth,
            strength_pct=args.strength_pct,
            convergence=args.convergence)

    metrics = reconstruction_metrics(result)
    flow_valid = flow.reliability > 0.08
    metrics.update({
        "flow_reliability_mean": float(flow.reliability.mean()),
        "flow_valid_pct": 100.0 * float(flow_valid.mean()),
        "flow_magnitude_mean_px": (
            float(np.hypot(flow.x, flow.y)[flow_valid].mean())
            if flow_valid.any() else 0.0),
        "lookahead_delay_ms": 1000.0 * args.future_frames / fps,
    })

    args.output.mkdir(parents=True, exist_ok=True)
    current_panel = cv2.resize(current_bgr, (width, height),
                               interpolation=cv2.INTER_AREA)
    future_panel = cv2.resize(future_bgr, (width, height),
                              interpolation=cv2.INTER_AREA)
    panels = [
        _panel(current_panel, f"Current t={args.seek:.3f}s"),
        _panel(future_panel,
               f"Future +{args.future_frames} frame(s) / "
               f"{metrics['lookahead_delay_ms']:.1f} ms"),
        _panel(yuv_to_bgr(result.left.baseline), "Left - spatial baseline"),
        _panel(yuv_to_bgr(result.left.lookahead), "Left - lookahead"),
        _panel(yuv_to_bgr(result.right.baseline), "Right - spatial baseline"),
        _panel(yuv_to_bgr(result.right.lookahead), "Right - lookahead"),
        _panel(_confidence_panel(result.left, result.right),
               "Accepted future evidence (both eyes)"),
    ]
    contact = np.vstack(panels)
    image_path = args.output / "lookahead_ab.png"
    if not cv2.imwrite(str(image_path), contact):
        raise RuntimeError(f"could not write {image_path}")

    report = {
        "video": str(args.video.resolve()),
        "seek_s": args.seek,
        "source_fps": fps,
        "future_frames": args.future_frames,
        "preset": args.preset,
        "grid": [width, height],
        "strength_pct": args.strength_pct,
        "convergence": args.convergence,
        "provider": (("NVOFA/CUDA" if use_nvof
                       else "native synth3d_flow CPU") + " + DepthEngine"),
        "timings": {name: _distribution(values)
                    for name, values in samples.items()},
        "metrics": metrics,
        "artifacts": {"contact_sheet": str(image_path.resolve())},
    }
    report_path = args.output / "lookahead_profile.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
