"""S1 smoke test for the native D3D11 renderer.

Run AFTER rebuilding mvc_demuxer_cpp.pyd with BUILD_NATIVE_RENDERER=ON.

What it proves (Stage S1):
  - the module exposes NativeRenderer (NATIVE_RENDERER_AVAILABLE == True),
  - a flip-model FP16 scRGB swapchain is created on a real HWND,
  - is_hdr() reports whether scRGB was accepted (True on an HDR-enabled display),
  - present() runs for a few seconds (window shows solid black),
  - resize() (ResizeBuffers) works without a crash.

It does NOT validate decoded video — that's S2+. This is the "does the HDR
swapchain come up and present" gate.

Usage:
    .venv\\Scripts\\python.exe native_renderer\\smoke_test_s1.py
Optional:  set SYLC_S1_HDR_CHECK=1 to FAIL if is_hdr() is False (use only when
you know the target display is in HDR mode).
"""
import os
import sys

try:
    import mvc_demuxer_cpp as m
except Exception as e:
    print(f"[S1] FAIL: cannot import mvc_demuxer_cpp: {e}")
    sys.exit(2)

if not getattr(m, "NATIVE_RENDERER_AVAILABLE", False) or not hasattr(m, "NativeRenderer"):
    print("[S1] FAIL: module built WITHOUT the native renderer "
          "(NATIVE_RENDERER_AVAILABLE is False). Rebuild with -DBUILD_NATIVE_RENDERER=ON.")
    sys.exit(3)

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt

app = QApplication(sys.argv)

win = QWidget()
win.setWindowTitle("SyLC native renderer — S1 smoke test (solid black = OK)")
win.resize(960, 540)
# Force a native HWND we can hand to D3D11, and keep Qt from painting over it.
win.setAttribute(Qt.WA_NativeWindow, True)
win.setAttribute(Qt.WA_PaintOnScreen, True)
win.setAttribute(Qt.WA_NoSystemBackground, True)
win.show()

hwnd = int(win.winId())
print(f"[S1] HWND = {hwnd:#x}")

r = m.NativeRenderer()
size = win.size()
if not r.initialize(hwnd, size.width(), size.height()):
    print(f"[S1] FAIL: initialize() -> {r.last_error()}")
    sys.exit(4)

print(f"[S1] backend_info: {r.backend_info()}")
print(f"[S1] is_hdr      : {r.is_hdr()}")

state = {"frames": 0, "resized": False, "errors": 0}

def tick():
    if not r.present():
        state["errors"] += 1
        print(f"[S1] present() error: {r.last_error()}")
    state["frames"] += 1

    # Exercise ResizeBuffers once, midway.
    if state["frames"] == 120 and not state["resized"]:
        if r.resize(1280, 720):
            print("[S1] resize(1280,720): OK")
        else:
            state["errors"] += 1
            print(f"[S1] resize FAILED: {r.last_error()}")
        state["resized"] = True

    if state["frames"] >= 240:  # ~4s @ 60Hz
        timer.stop()
        ok = (state["errors"] == 0)
        if os.environ.get("SYLC_S1_HDR_CHECK") == "1" and not r.is_hdr():
            ok = False
            print("[S1] FAIL: SYLC_S1_HDR_CHECK=1 but is_hdr() is False")
        r.shutdown()
        print(f"[S1] presented {state['frames']} frames, {state['errors']} errors")
        print("[S1] PASS" if ok else "[S1] FAIL")
        app.exit(0 if ok else 5)

timer = QTimer()
timer.timeout.connect(tick)
timer.start(16)  # ~60 Hz; present(1) blocks on vsync anyway

sys.exit(app.exec())
