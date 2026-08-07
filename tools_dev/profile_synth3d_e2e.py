r"""Exercise the complete native 2D->3D path with a paced synthetic movie.

Unlike the micro-profiler, this includes the D3D11 prep/readback ring, mailbox,
overlapped inference/flow/stabilization, geometry upload and final warp.  It
creates a small real Win32 window because DXGI presentation is part of the path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build_py314" / "python" / "Release"))
sys.path.insert(1, str(ROOT / "src"))
_DLL_HANDLES = [
    os.add_dll_directory(str(path))
    for path in (ROOT / "runtime", ROOT / "ort_tensorrt")
    if path.exists()
]

import mvc_demuxer_cpp as native
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget


CASES = {
    "quality_scope": ("da3_base_756x322.onnx", 756, 322),
    "quality_square": ("da3_base_756.onnx", 756, 756),
    "balanced_scope": ("da3_base_518x210.onnx", 518, 210),
    "performance_scope": ("da3_small_518x210.onnx", 518, 210),
}


def parse_status(line: str) -> dict[str, str]:
    fields = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def synthetic_frames(width: int, height: int, count: int = 48):
    yy, xx = np.mgrid[0:height, 0:width]
    base = 48.0 + 44.0 * np.sin(xx * 2.0 * np.pi / width)
    base += 28.0 * np.cos(yy * 4.0 * np.pi / height)
    frames = []
    for index in range(count):
        frame = np.roll(base, 5 * index, axis=1).copy()
        left = (73 + 9 * index) % (width - 420)
        top = (height // 3) + int(35 * np.sin(index * 0.24))
        frame[top:top + 300, left:left + 380] += 105.0
        frames.append(np.clip(frame, 16, 235).astype(np.uint8))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(CASES))
    parser.add_argument("--seconds", type=float, default=14.0)
    parser.add_argument("--fps", type=float, default=23.976)
    parser.add_argument("--sync-interval", type=int, choices=(0, 1), default=1)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--static-frame", action="store_true",
                        help="repeat one frame so CPU/GPU output can be compared")
    parser.add_argument("--capture-npz", type=Path,
                        help="save final left/right luma planes for quality comparison")
    args = parser.parse_args()
    if args.seconds <= 2.0 or args.fps <= 0.0:
        parser.error("seconds must be > 2 and fps must be positive")

    filename, grid_width, grid_height = CASES[args.case]
    model = ROOT / "models" / filename
    ort_dir = ROOT / "ort_tensorrt"
    if not model.exists():
        parser.error(f"model is missing: {model}")

    app = QApplication.instance() or QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle(f"SyLC 2D→3D profile — {args.case}")
    window.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
    window.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
    window.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
    window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    window.resize(480, 270)
    window.move(20, 20)
    window.show()
    app.processEvents()

    renderer = native.NativeRenderer()
    if not renderer.initialize(int(window.winId()), 480, 270):
        raise RuntimeError(renderer.last_error())
    renderer.set_source_aspect(16.0 / 9.0)
    if not renderer.set_synth3d(
            True, 1.5, 0.5, False, str(model), str(ort_dir), False,
            grid_width, grid_width, grid_height,
            0.0, 0.0, False, False, True):
        raise RuntimeError(renderer.last_error())

    width, height = 1920, 1080
    frames = synthetic_frames(width, height)
    if args.static_frame:
        frames = frames[:1]
    u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    v = np.full_like(u, 128)
    frame_period = 1.0 / args.fps
    total_frames = int(args.seconds * args.fps)
    started = time.perf_counter()
    next_tick = started
    samples = []
    captured = None
    try:
        for index in range(total_frames):
            y = frames[index % len(frames)]
            renderer.set_video_time_ms(index * frame_period * 1000.0)
            if not renderer.set_yuv_frame(y, u, v):
                raise RuntimeError(renderer.last_error())
            if not renderer.present(args.sync_interval):
                raise RuntimeError(renderer.last_error())
            app.processEvents()
            if index % max(1, int(args.fps)) == 0:
                status = renderer.synth3d_status()
                fields = parse_status(status)
                if fields.get("state") == "running" and fields.get("fps") != "0.0":
                    samples.append(fields)
                print(f"{index / args.fps:5.1f}s  {status}")
            next_tick += frame_period
            delay = next_tick - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
        if args.capture_npz:
            left = np.asarray(renderer.synth3d_read_plane(0)).copy()
            right = np.asarray(renderer.synth3d_read_plane(3)).copy()
            np.savez_compressed(args.capture_npz, left=left, right=right)
            captured = {
                "path": str(args.capture_npz),
                "left_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
                "right_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
            }
    finally:
        final_status = renderer.synth3d_status()
        renderer.set_synth3d(False)
        renderer.shutdown()
        window.close()
        app.processEvents()

    elapsed = time.perf_counter() - started
    metric_names = (
        "fps", "flow_ms", "infer_ms", "stab_ms", "obs_ms", "guard_ms",
        "reproj_ms", "step_ms", "realign_ms", "owner_ms", "owner_local_ms",
        "owner_prop_ms", "pack_ms", "inwait_ms",
        "reswait_ms", "joinwait_ms", "cycle_ms", "pass_ms", "taplat_ms",
        "source_ms", "update_ms", "age_ms")
    summary = {}
    # Drop the first running status window: it includes lazy TensorRT warm-up.
    stable_samples = samples[1:] if len(samples) > 1 else samples
    for name in metric_names:
        values = []
        for sample in stable_samples:
            try:
                values.append(float(sample[name]))
            except (KeyError, ValueError):
                pass
        if values:
            summary[name] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    result = {
        "case": args.case,
        "grid": [grid_width, grid_height],
        "requested_fps": args.fps,
        "presented_fps": total_frames / elapsed,
        "summary": summary,
        "final_status": final_status,
    }
    if captured:
        result["capture"] = captured
    print("\nSUMMARY")
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
