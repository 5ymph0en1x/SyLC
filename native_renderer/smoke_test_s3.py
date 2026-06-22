"""S3 smoke test — concurrency primitive (the seek/pause GPU-race mitigation).

Run AFTER rebuilding mvc_demuxer_cpp.pyd.

Proves the renderer is safe when the presenter thread and the GUI thread touch it
concurrently — the exact situation the live pipeline creates:
  - a WORKER thread uploads an animated synthetic frame and calls present() in a
    tight loop (present-on-arrival, like the decode/presenter thread will),
  - the GUI thread concurrently calls resize() (every ~1s) and pause()/resume()
    (every ~2s), like a window resize and a seek.

The internal render mutex must serialize all of this with ZERO crashes/errors.
Visually: the color bands scroll; during PAUSE the image FREEZES (present holds the
last frame though the worker keeps uploading); on RESUME it jumps to the latest.

PASS = many frames presented, 0 errors, clean shutdown, no access violation.

Usage:
    .venv\\Scripts\\python.exe native_renderer\\smoke_test_s3.py
"""
import sys
import time
import threading
import numpy as np

try:
    import mvc_demuxer_cpp as m
except Exception as e:
    print(f"[S3] FAIL: cannot import mvc_demuxer_cpp: {e}")
    sys.exit(2)
if not getattr(m, "NATIVE_RENDERER_AVAILABLE", False) or not hasattr(m, "NativeRenderer"):
    print("[S3] FAIL: module built WITHOUT the native renderer.")
    sys.exit(3)

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt

COLORS = {"white": (235, 128, 128), "red": (81, 90, 240),
          "green": (145, 54, 34), "blue": (41, 240, 110)}
ORDER_L = ["white", "red", "green", "blue"]
ORDER_R = ["blue", "green", "red", "white"]
W, H = 1920, 1080
CW, CH = W // 2, H // 2

def bands(order, w, h, comp):
    out = np.empty((h, w), dtype=np.uint8)
    for i, name in enumerate(order):
        out[i * h // 4:(i + 1) * h // 4, :] = COLORS[name][comp]
    return np.ascontiguousarray(out)

BASE = {
    "yl": bands(ORDER_L, W, H, 0), "ul": bands(ORDER_L, CW, CH, 1), "vl": bands(ORDER_L, CW, CH, 2),
    "yr": bands(ORDER_R, W, H, 0), "ur": bands(ORDER_R, CW, CH, 1), "vr": bands(ORDER_R, CW, CH, 2),
}

app = QApplication(sys.argv)
win = QWidget()
win.setWindowTitle("SyLC native renderer — S3 (concurrent present + resize + pause)")
win.resize(1280, 720)
win.setAttribute(Qt.WA_NativeWindow, True)
win.setAttribute(Qt.WA_PaintOnScreen, True)
win.setAttribute(Qt.WA_NoSystemBackground, True)
win.show()

r = m.NativeRenderer()
sz = win.size()
if not r.initialize(int(win.winId()), sz.width(), sz.height()):
    print(f"[S3] FAIL: initialize() -> {r.last_error()}")
    sys.exit(4)
print(f"[S3] backend_info: {r.backend_info()}")
r.set_uniforms(2, 0, 0.0, 0.0, 1.0, 1.0, 2.5)  # SBS so L|R both visible while scrolling

stop = threading.Event()
errors = [0]
frames = [0]

def render_loop():
    off = 0
    while not stop.is_set():
        c = off // 2
        yl = np.ascontiguousarray(np.roll(BASE["yl"], off, axis=0))
        h, w = yl.shape[:2]
        ok = r.set_yuv_frame(
            yl,
            np.ascontiguousarray(np.roll(BASE["ul"], c, axis=0)),
            np.ascontiguousarray(np.roll(BASE["vl"], c, axis=0)),
            np.ascontiguousarray(np.roll(BASE["yr"], off, axis=0)),
            np.ascontiguousarray(np.roll(BASE["ur"], c, axis=0)),
            np.ascontiguousarray(np.roll(BASE["vr"], c, axis=0)),
            int(w), int(h),
        )
        if not ok:
            errors[0] += 1
        if not r.present():
            errors[0] += 1
        frames[0] += 1
        off = (off + 8) % H
        # present(1) blocks on vsync when playing; while paused it returns
        # instantly, so cap the spin to keep CPU sane during the pause windows.
        time.sleep(0.003)

worker = threading.Thread(target=render_loop, daemon=True)
worker.start()

state = {"t": 0}

def gui_tick():
    state["t"] += 1
    t = state["t"]
    if t % 10 == 0:  # ~every 1s: resize swapchain + window together
        w, h = (1024, 576) if (t // 10) % 2 else (1280, 720)
        if not r.resize(w, h):
            errors[0] += 1
            print(f"[S3] resize error: {r.last_error()}")
        else:
            win.resize(w, h)
            print(f"[S3] resize {w}x{h}")
    if t % 20 == 5:  # ~every 2s: simulate a seek (pause/resume)
        if r.is_paused():
            r.resume(); print("[S3] RESUME")
        else:
            r.pause(); print("[S3] PAUSE (image should freeze)")
    if t >= 80:      # ~8s
        gui_timer.stop()
        stop.set()
        worker.join(timeout=3.0)
        alive = worker.is_alive()
        r.shutdown()
        ok = (errors[0] == 0) and not alive
        print(f"[S3] frames={frames[0]} errors={errors[0]} worker_stuck={alive}")
        print("[S3] PASS" if ok else "[S3] FAIL")
        app.exit(0 if ok else 7)

gui_timer = QTimer()
gui_timer.timeout.connect(gui_tick)
gui_timer.start(100)  # 10 Hz
sys.exit(app.exec())
