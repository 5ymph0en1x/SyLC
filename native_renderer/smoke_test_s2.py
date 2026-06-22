"""S2 smoke test for the native D3D11 renderer — shaded YUV draw + parity.

Run AFTER rebuilding mvc_demuxer_cpp.pyd (BUILD_NATIVE_RENDERER ON).

Uploads a SYNTHETIC stereo frame and draws it through the exact extracted shader,
cycling all four stereo modes. This is the pixel-parity gate vs the current Qt path.

Synthetic frame (BT.601 limited-range YUV, so it round-trips through the shader):
  LEFT eye  = 4 horizontal bands, top->bottom: WHITE, RED, GREEN, BLUE
  RIGHT eye = reversed:            top->bottom: BLUE,  GREEN, RED, WHITE

What you should SEE (window is letterboxed/pillarboxed to the mode's aspect):
  - 2D        : white(top) / red / green / blue(bottom). Wrong colors (e.g. green
                tint everywhere) => YUV matrix bug; white at BOTTOM => vertical flip.
  - Framepack : LEFT bands on top, a thin BLACK GAP (~45px @1920x2205), RIGHT bands
                below — so the seam is blue-meets-blue (L bottom = blue, R top = blue).
  - SBS       : LEFT bands on the left half, RIGHT bands on the right half.
  - TAB       : LEFT bands top half, RIGHT bands bottom half (spline-resampled).

The mode name is printed as it switches (every ~3s). Compare against the same
frame shown by the Qt renderer for exact parity.

Usage:
    .venv\\Scripts\\python.exe native_renderer\\smoke_test_s2.py
Env:
    SYLC_SDR_WHITE  scRGB white multiplier (default 2.5 ~= 200 nits on HDR). Raise
                    if the image looks dim on your HDR display; 1.0 == 80 nits.
"""
import os
import sys
import numpy as np

try:
    import mvc_demuxer_cpp as m
except Exception as e:
    print(f"[S2] FAIL: cannot import mvc_demuxer_cpp: {e}")
    sys.exit(2)

if not getattr(m, "NATIVE_RENDERER_AVAILABLE", False) or not hasattr(m, "NativeRenderer"):
    print("[S2] FAIL: module built WITHOUT the native renderer. Rebuild with -DBUILD_NATIVE_RENDERER=ON.")
    sys.exit(3)

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt

# BT.601 limited-range YUV for pure colors (matches the shader's inverse matrix).
#                       Y    U(Cb) V(Cr)
COLORS = {
    "white": (235, 128, 128),
    "red":   (81,  90,  240),
    "green": (145, 54,  34),
    "blue":  (41,  240, 110),
}
ORDER_L = ["white", "red", "green", "blue"]
ORDER_R = ["blue", "green", "red", "white"]

W, H = 1920, 1080
CW, CH = W // 2, H // 2

def bands(order, w, h, comp):
    out = np.empty((h, w), dtype=np.uint8)
    for i, name in enumerate(order):
        y0 = i * h // 4
        y1 = (i + 1) * h // 4
        out[y0:y1, :] = COLORS[name][comp]
    return np.ascontiguousarray(out)

Y_L, U_L, V_L = bands(ORDER_L, W, H, 0), bands(ORDER_L, CW, CH, 1), bands(ORDER_L, CW, CH, 2)
Y_R, U_R, V_R = bands(ORDER_R, W, H, 0), bands(ORDER_R, CW, CH, 1), bands(ORDER_R, CW, CH, 2)

SDR_WHITE = float(os.environ.get("SYLC_SDR_WHITE", "2.5"))

app = QApplication(sys.argv)
win = QWidget()
win.setWindowTitle("SyLC native renderer — S2 (synthetic stereo frame)")
win.resize(1280, 720)
win.setAttribute(Qt.WA_NativeWindow, True)
win.setAttribute(Qt.WA_PaintOnScreen, True)
win.setAttribute(Qt.WA_NoSystemBackground, True)
win.show()

r = m.NativeRenderer()
sz = win.size()
if not r.initialize(int(win.winId()), sz.width(), sz.height()):
    print(f"[S2] FAIL: initialize() -> {r.last_error()}")
    sys.exit(4)
print(f"[S2] backend_info: {r.backend_info()}")
print(f"[S2] is_hdr={r.is_hdr()}  sdr_white={SDR_WHITE}")

h, w = Y_L.shape[:2]
if not r.set_yuv_frame(Y_L, U_L, V_L, Y_R, U_R, V_R, int(w), int(h)):
    print(f"[S2] FAIL: set_yuv_frame -> {r.last_error()}")
    sys.exit(5)

MODES = [(0, "2D (L only)"),
         (1, "FRAMEPACK (L top / 45px gap / R bottom)"),
         (2, "SBS (L left | R right)"),
         (3, "TAB (L top / R bottom)")]

state = {"frame": 0, "mode_idx": -1, "errors": 0}

def set_mode(i):
    mode, label = MODES[i]
    r.set_uniforms(mode, 0, 0.0, 0.0, 1.0, 1.0, SDR_WHITE)
    print(f"[S2] stereo_mode = {mode}  -> {label}")

def tick():
    # Switch mode every ~180 frames (~3s @ 60Hz).
    new_idx = (state["frame"] // 180) % len(MODES)
    if new_idx != state["mode_idx"]:
        state["mode_idx"] = new_idx
        set_mode(new_idx)
    if not r.present():
        state["errors"] += 1
        print(f"[S2] present error: {r.last_error()}")
    state["frame"] += 1
    if state["frame"] >= 180 * len(MODES):  # one full cycle
        timer.stop()
        r.shutdown()
        ok = state["errors"] == 0
        print(f"[S2] presented {state['frame']} frames, {state['errors']} errors")
        print("[S2] PASS (now confirm the visuals above match the Qt renderer)" if ok else "[S2] FAIL")
        app.exit(0 if ok else 6)

timer = QTimer()
timer.timeout.connect(tick)
timer.start(16)
sys.exit(app.exec())
