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

# Import our D3D11 widget
from framepacking_widget_d3d11 import (
    FramepackingDisplayWidgetD3D11,
    HAS_RHI_WIDGET,
    check_hdr_support
)

logger = logging.getLogger(__name__)

# Re-export the widget class with the expected name for compatibility
FramepackingDisplayWidget = FramepackingDisplayWidgetD3D11

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

        # Use provided widget or create D3D11 widget
        if display_widget:
            self.display_widget = display_widget
        else:
            self.display_widget = FramepackingDisplayWidgetD3D11(self)
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
        super().closeEvent(event)


# Check availability
def is_d3d11_available():
    """Check if D3D11 HDR rendering is available."""
    return HAS_RHI_WIDGET and check_hdr_support()[0]


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    available, msg = check_hdr_support()
    print(f"D3D11 HDR Available: {available}")
    print(f"Message: {msg}")

    if available:
        window = Framepacking3DWindow()
        window.show()
        sys.exit(app.exec())
