"""Measure what the per-frame Python layer actually costs.

The native pipeline moved decode, demux and rendering into C++, so the Python
that still runs once per presented frame is the widget's uniform/state
forwarding plus the plane upload. This times that exact sequence against the
41.7 ms budget of a 24 fps frame, to answer whether any Python-side
optimization (vectorization, numba, micro-tuning) can still move the needle.

    python tools_dev/bench_frame_python.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runtime"))

import numpy as np

import mvc_demuxer_cpp as m

if not getattr(m, "NATIVE_RENDERER_AVAILABLE", False):
    print("FAIL: module built without the native renderer")
    sys.exit(3)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

WIDTH, HEIGHT = 1920, 1080
ITERATIONS = 600
FRAME_BUDGET_MS = 1000.0 / 23.976

app = QApplication(sys.argv)
win = QWidget()
win.resize(960, 540)
win.setAttribute(Qt.WA_NativeWindow, True)
win.setAttribute(Qt.WA_PaintOnScreen, True)
win.setAttribute(Qt.WA_NoSystemBackground, True)
win.show()

r = m.NativeRenderer()
if not r.initialize(int(win.winId()), 960, 540):
    print(f"FAIL: initialize -> {r.last_error()}")
    sys.exit(4)
print(f"backend: {r.backend_info()}")

cw, ch = WIDTH // 2, HEIGHT // 2
y = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
u = np.full((ch, cw), 128, dtype=np.uint8)
v = np.full((ch, cw), 128, dtype=np.uint8)


def timed(label, fn, n=ITERATIONS):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
    print(f"{label:<38} p50={p50:7.4f} ms  p99={p99:7.4f} ms  "
          f"mean={statistics.fmean(samples):7.4f} ms")
    return statistics.fmean(samples)


print(f"\nPer-frame Python cost, {WIDTH}x{HEIGHT}, {ITERATIONS} iterations")
print(f"(one 23.976 fps frame = {FRAME_BUDGET_MS:.1f} ms)\n")

# The per-frame state forwarding the widget performs before every upload.
def state_calls():
    r.set_uniforms(1, 0, 0.0, 0.0, 1.0, 1.0, 200.0, 2.2, 0.0)
    r.set_source_aspect(16.0 / 9.0)
    r.set_color_params(0, 0)


state_ms = timed("widget state forwarding (3 calls)", state_calls)
upload_ms = timed("set_yuv_frame (6 planes, 1080p)",
                  lambda: r.set_yuv_frame(y, u, v, y, u, v))

print()
total = state_ms + upload_ms
print(f"{'TOTAL per-frame Python (excl. present)':<38} {total:7.4f} ms")
print(f"{'share of a 23.976 fps frame':<38} {100.0 * total / FRAME_BUDGET_MS:7.2f} %")
print()
print("set_yuv_frame is a C++ memcpy of 6 planes behind one pybind11 call;")
print("only the call overhead itself is Python. Anything Python-side can win")
print("is bounded by the state-forwarding line above.")
