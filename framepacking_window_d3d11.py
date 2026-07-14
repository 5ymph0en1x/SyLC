# -*- coding: utf-8 -*-
"""
Framepacking 3D Window - D3D11 Native Version for HDR Preservation

This replaces the OpenGL-based framepacking_window.py with a D3D11 native
implementation that preserves HDR in fullscreen mode.

Key differences from OpenGL version:
- Uses QRhiWidget with D3D11 backend instead of QOpenGLWidget
- HDR is preserved in fullscreen (no DXGI copy overhead)
- Same API as the OpenGL version for drop-in replacement
"""

import ctypes
from ctypes import wintypes, byref
import logging

from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent

logger = logging.getLogger(__name__)


def apply_borderless_dwm(hwnd, enable):
    """Kill (or restore) the DWM window border + rounded corners on a top-level
    window. On Windows 11, DWM draws a thin border and rounds the corners of
    EVERY top-level window — even one stripped of WS_CAPTION/WS_THICKFRAME and
    sized to the whole monitor — which is the light 'liseret' seen all around a
    borderless fake-fullscreen window. `enable=True` squares the corners and
    sets the border colour to NONE; `enable=False` restores the defaults.

    Returns the border-colour call's HRESULT (0 = S_OK). Harmless on Windows 10
    / pre-22000 (unsupported attribute → non-zero HRESULT, ignored)."""
    try:
        dwmapi = ctypes.windll.dwmapi
        dwmapi.DwmSetWindowAttribute.argtypes = [
            wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long  # HRESULT

        DWMWA_WINDOW_CORNER_PREFERENCE = 33
        DWMWA_BORDER_COLOR = 34
        DWMWCP_DEFAULT = 0
        DWMWCP_DONOTROUND = 1
        DWMWA_COLOR_DEFAULT = 0xFFFFFFFF
        DWMWA_COLOR_NONE = 0xFFFFFFFE

        corner = ctypes.c_uint(DWMWCP_DONOTROUND if enable else DWMWCP_DEFAULT)
        color = ctypes.c_uint(DWMWA_COLOR_NONE if enable else DWMWA_COLOR_DEFAULT)
        h = wintypes.HWND(int(hwnd))
        dwmapi.DwmSetWindowAttribute(h, DWMWA_WINDOW_CORNER_PREFERENCE,
                                     byref(corner), ctypes.sizeof(corner))
        hr = dwmapi.DwmSetWindowAttribute(h, DWMWA_BORDER_COLOR,
                                          byref(color), ctypes.sizeof(color))
        return hr
    except Exception as e:
        logger.debug(f"[DWM] borderless attribute skipped: {e}")
        return -1


FRAMEPACK_WIDTH = 1920
FRAMEPACK_HEIGHT = 2205
TARGET_ASPECT = FRAMEPACK_WIDTH / FRAMEPACK_HEIGHT


class Framepacking3DWindow(QMainWindow):
    """
    D3D11-native 3D framepacking window with HDR support.

    Drop-in replacement for the OpenGL version with same API.
    """
    visibilityChanged = Signal(bool)

    def __init__(self, parent=None, use_yuv_shader=True, display_widget=None):
        super().__init__(parent)
        self.setWindowTitle("3D Frame-Packed Output (D3D11 HDR)")
        self.resize(960, 1102)
        self.setStyleSheet("background-color: black;")

        # Use the provided widget, or create the native C++ D3D11 renderer widget.
        if display_widget:
            self.display_widget = display_widget
        else:
            from native_renderer.native_framepack_widget import NativeFramepackWidget
            self.display_widget = NativeFramepackWidget(self)
            self.display_widget.set_stereo_mode('framepack')

        self.setCentralWidget(self.display_widget)

        self.is_fullscreen = False
        self.use_yuv_shader = use_yuv_shader

        # Fullscreen state (Win32 API for true fullscreen)
        self._is_fake_fullscreen = False
        self._saved_style = None
        self._saved_rect = None

        logger.info("[D3D11-WINDOW] Framepacking3DWindow created (D3D11 HDR mode)")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_F:
            self.toggle_fullscreen()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape and self._is_fake_fullscreen:
            self.toggle_fullscreen()
            event.accept()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_fullscreen()
            event.accept()

    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self._is_fake_fullscreen:
            self.exit_fake_fullscreen()
        else:
            self.enter_fake_fullscreen()

    def enter_fake_fullscreen(self):
        """Enter fullscreen using Win32 API (preserves HDR with D3D11)."""
        if self._is_fake_fullscreen:
            return

        try:
            user32 = ctypes.windll.user32
            dwmapi = ctypes.windll.dwmapi

            GWL_STYLE = -16
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_MINIMIZEBOX = 0x00020000
            WS_MAXIMIZEBOX = 0x00010000
            WS_SYSMENU = 0x00080000
            SWP_FRAMECHANGED = 0x0020
            SWP_SHOWWINDOW = 0x0040
            SWP_NOZORDER = 0x0004  # CRITICAL: Don't change Z-order (avoids exclusive fullscreen detection)
            MONITOR_DEFAULTTONEAREST = 2

            hwnd = int(self.winId())

            # Save current state
            self._saved_style = user32.GetWindowLongW(hwnd, GWL_STYLE)

            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, byref(rect))
            self._saved_rect = (rect.left, rect.top,
                               rect.right - rect.left,
                               rect.bottom - rect.top)

            # Get monitor info
            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', wintypes.DWORD),
                    ('rcMonitor', wintypes.RECT),
                    ('rcWork', wintypes.RECT),
                    ('dwFlags', wintypes.DWORD),
                ]

            hMonitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            user32.GetMonitorInfoW(hMonitor, byref(mi))

            mon_x = mi.rcMonitor.left
            mon_y = mi.rcMonitor.top
            mon_w = mi.rcMonitor.right - mi.rcMonitor.left
            mon_h = mi.rcMonitor.bottom - mi.rcMonitor.top

            # Remove window decorations
            new_style = self._saved_style & ~(WS_CAPTION | WS_THICKFRAME |
                                               WS_MINIMIZEBOX | WS_MAXIMIZEBOX |
                                               WS_SYSMENU)
            user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)

            # Resize to fullscreen with SWP_NOZORDER to preserve HDR
            user32.SetWindowPos(hwnd, 0, mon_x, mon_y, mon_w, mon_h,
                               SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER)

            # Win11: suppress the DWM border + rounded corners (the light
            # 'liseret' around a borderless window) so 3D reaches the edge.
            apply_borderless_dwm(hwnd, True)

            # Force DWM composition refresh to ensure HDR state is preserved
            try:
                dwmapi.DwmFlush()
            except Exception:
                pass

            # Refresh SDR white level in case display settings changed
            if hasattr(self.display_widget, 'refresh_sdr_white_level'):
                self.display_widget.refresh_sdr_white_level()

            self._is_fake_fullscreen = True
            self.is_fullscreen = True
            self.visibilityChanged.emit(True)

            logger.info(f"[D3D11-WINDOW] Entered fullscreen: {mon_w}x{mon_h} (HDR preserved with SWP_NOZORDER)")

        except Exception as e:
            logger.error(f"[D3D11-WINDOW] Failed to enter fullscreen: {e}")

    def exit_fake_fullscreen(self):
        """Exit fullscreen mode."""
        if not self._is_fake_fullscreen:
            return

        try:
            user32 = ctypes.windll.user32
            dwmapi = ctypes.windll.dwmapi
            GWL_STYLE = -16
            SWP_FRAMECHANGED = 0x0020
            SWP_SHOWWINDOW = 0x0040
            SWP_NOZORDER = 0x0004  # Preserve Z-order

            hwnd = int(self.winId())

            # Restore the DWM border + rounded corners we suppressed on enter
            apply_borderless_dwm(hwnd, False)

            # Restore style
            if self._saved_style is not None:
                user32.SetWindowLongW(hwnd, GWL_STYLE, self._saved_style)

            # Restore geometry with SWP_NOZORDER
            if self._saved_rect is not None:
                x, y, w, h = self._saved_rect
                user32.SetWindowPos(hwnd, 0, x, y, w, h,
                                   SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER)

            # Force DWM composition refresh
            try:
                dwmapi.DwmFlush()
            except Exception:
                pass

            self._is_fake_fullscreen = False
            self.is_fullscreen = False
            self.visibilityChanged.emit(False)

            logger.info("[D3D11-WINDOW] Exited fullscreen (HDR preserved)")

        except Exception as e:
            logger.error(f"[D3D11-WINDOW] Failed to exit fullscreen: {e}")

    def display_frame(self, qimage):
        """Display a QImage frame (compatibility method)."""
        # This method exists for compatibility but D3D11 version uses YUV directly
        pass

    def showEvent(self, event):
        super().showEvent(event)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibilityChanged.emit(False)

    def closeEvent(self, event):
        # Exit fullscreen before closing
        if self._is_fake_fullscreen:
            self.exit_fake_fullscreen()
        # Release the native renderer's D3D11 resources (no-op if the widget lacks it).
        try:
            if hasattr(self.display_widget, 'shutdown'):
                self.display_widget.shutdown()
        except Exception:
            pass
        super().closeEvent(event)


# Check availability
def is_d3d11_available():
    """Check if the native C++ D3D11 renderer is available."""
    try:
        import mvc_demuxer_cpp as _m
        return bool(getattr(_m, "NATIVE_RENDERER_AVAILABLE", False))
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    print(f"Native D3D11 renderer available: {is_d3d11_available()}")
    window = Framepacking3DWindow()
    window.show()
    sys.exit(app.exec())
