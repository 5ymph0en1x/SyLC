r"""D3D11/NVOFA-era live lookahead contract probe.

Runs the exact production Synth3D pixel shader at 1920x1080 with a deterministic
depth step and a one-frame future reveal.  It reports the CPU->GPU future upload
cost, resolve/present cost, changed output pixels, and verifies that evidence is
single-use by rendering the same current frame once more without re-arming it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build_py314" / "python" / "Release"))
_DLL_HANDLES = [
    os.add_dll_directory(str(path))
    for path in (ROOT / "runtime", ROOT / "ort_tensorrt")
    if path.exists()
]

import mvc_demuxer_cpp as native
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget


def _pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = QWidget()
    window.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
    window.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
    window.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
    window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    window.resize(480, 270)
    window.show()
    app.processEvents()

    renderer = native.NativeRenderer()
    if not renderer.initialize(int(window.winId()), 480, 270):
        raise RuntimeError(renderer.last_error())
    grid_w, grid_h = 756, 322
    if not renderer.set_synth3d(
            True, 1.5, 0.5, False, "", "", False,
            grid_w, grid_w, grid_h, 0.0, 0.0,
            False, False, False):
        raise RuntimeError(renderer.last_error())

    width, height = 1920, 1080
    yy, xx = np.mgrid[0:height, 0:width]
    background = 72.0 + 0.055 * xx + 13.0 * np.sin(yy / 43.0)
    current = np.clip(background, 16, 220).astype(np.uint8)
    future = current.copy()
    # Near object shifts right in t+1, exposing real textured background on its
    # left edge. The current frame has no access to those values after overwrite.
    current[300:820, 690:1120] = 198
    future[300:820, 742:1172] = 198
    u = np.full((height // 2, width // 2), 128, np.uint8)
    v = np.full_like(u, 128)
    future_u = u.copy()
    future_v = v.copy()

    depth = np.full((grid_h, grid_w), int(0.18 * 65535), np.uint16)
    depth[90:245, 272:441] = int(0.90 * 65535)
    renderer.synth3d_set_test_depth(depth)
    flow_x = np.zeros((81, 189), np.float32)
    flow_y = np.zeros_like(flow_x)
    flow_q = np.full_like(flow_x, 0.98)

    def render(pts, arm):
        upload_ms = None
        if arm:
            t0 = time.perf_counter()
            ok = renderer.synth3d_set_lookahead_frame(
                future, future_u, future_v, flow_x, flow_y, flow_q,
                pts, pts + 1000.0 / 23.976, 1.0)
            upload_ms = (time.perf_counter() - t0) * 1000.0
            if not ok:
                raise RuntimeError(renderer.last_error())
        renderer.set_video_time_ms(pts)
        if not renderer.set_yuv_frame(current, u, v):
            raise RuntimeError(renderer.last_error())
        t0 = time.perf_counter()
        if not renderer.present(0):
            raise RuntimeError(renderer.last_error())
        present_ms = (time.perf_counter() - t0) * 1000.0
        app.processEvents()
        return upload_ms, present_ms

    try:
        render(0.0, False)
        baseline_l = np.asarray(renderer.synth3d_read_plane(0)).copy()
        baseline_r = np.asarray(renderer.synth3d_read_plane(3)).copy()

        render(100.0, True)
        reveal_l = np.asarray(renderer.synth3d_read_plane(0)).copy()
        reveal_r = np.asarray(renderer.synth3d_read_plane(3)).copy()

        # Evidence must have been consumed by the preceding present.
        render(200.0, False)
        rollback_l = np.asarray(renderer.synth3d_read_plane(0)).copy()
        rollback_r = np.asarray(renderer.synth3d_read_plane(3)).copy()

        upload_samples = []
        present_samples = []
        baseline_present_samples = []
        for index in range(36):
            pair_pts = 300.0 + index * 100.0
            # Alternate order so immediate-context queueing/resource hazards do
            # not systematically charge the second draw to one variant.
            if index % 2 == 0:
                upload_ms, present_ms = render(pair_pts, True)
                _, baseline_present_ms = render(pair_pts + 50.0, False)
            else:
                _, baseline_present_ms = render(pair_pts, False)
                upload_ms, present_ms = render(pair_pts + 50.0, True)
            if index >= 6:  # discard D3D cache warm-up
                upload_samples.append(upload_ms)
                present_samples.append(present_ms)
                baseline_present_samples.append(baseline_present_ms)

        delta_l = np.abs(reveal_l.astype(np.int16) - baseline_l.astype(np.int16))
        delta_r = np.abs(reveal_r.astype(np.int16) - baseline_r.astype(np.int16))
        result = {
            "resolution": [width, height],
            "grid": [grid_w, grid_h],
            "flow_texture": [flow_x.shape[1], flow_x.shape[0]],
            "changed_pixels": {
                "left": int(np.count_nonzero(delta_l)),
                "right": int(np.count_nonzero(delta_r)),
                "both_pct": 100.0 * float(
                    np.count_nonzero(delta_l) + np.count_nonzero(delta_r)) /
                    float(2 * width * height),
            },
            "single_use_rollback_exact": bool(
                np.array_equal(rollback_l, baseline_l) and
                np.array_equal(rollback_r, baseline_r)),
            "future_upload_ms": {
                "median": statistics.median(upload_samples),
                "p95": _pct(upload_samples, 0.95),
            },
            # Present(0) measures CPU submission/queue interaction, not GPU
            # execution. Keep both distributions for regression diagnostics,
            # but do not subtract them as a claimed shader cost.
            "cpu_present_submit_ms_lookahead": {
                "median": statistics.median(present_samples),
                "p95": _pct(present_samples, 0.95),
            },
            "cpu_present_submit_ms_baseline": {
                "median": statistics.median(baseline_present_samples),
                "p95": _pct(baseline_present_samples, 0.95),
            },
            "status": renderer.synth3d_status(),
        }
        print(json.dumps(result, indent=2))
        if not result["single_use_rollback_exact"]:
            raise RuntimeError("lookahead evidence leaked into a later frame")
        if not (result["changed_pixels"]["left"] or
                result["changed_pixels"]["right"]):
            raise RuntimeError("the live lookahead shader changed no hole pixel")
    finally:
        renderer.set_synth3d(False)
        renderer.shutdown()
        window.close()
        app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
