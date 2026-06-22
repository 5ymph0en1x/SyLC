"""Centralized fullscreen logic for SyLC main and framepack windows.

Qt's showFullScreen() is avoided because it can trigger SDR fallback on HDR
monitors and recreate the window, breaking MPV. We keep the existing Win32
borderless approach but centralize it so it is not duplicated.
"""
import ctypes
import logging
from ctypes import wintypes, byref, c_void_p, c_int, c_uint

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger("SyLC.Fullscreen")

# Win32 constants
GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
SWP_NOZORDER = 0x0004
MONITOR_DEFAULTTONEAREST = 2

HWND_NOTOPMOST = c_void_p(-2)
HWND_TOP = c_void_p(0)


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.DWORD),
        ('rcMonitor', wintypes.RECT),
        ('rcWork', wintypes.RECT),
        ('dwFlags', wintypes.DWORD),
    ]


class FullscreenManager(QObject):
    """Manages fake-fullscreen state for a main window and optional framepack window."""

    toggled = Signal(bool)

    def __init__(self, main_window: QMainWindow, framepack_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.framepack_window = framepack_window
        self._state = {
            'main': False,
            'framepack': False,
        }
        self._saved = {
            'main': {},
            'framepack': {},
        }
        self._user32 = ctypes.WinDLL('user32', use_last_error=True)

        self._user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.GetWindowLongW.restype = wintypes.LONG
        self._user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        self._user32.SetWindowLongW.restype = wintypes.LONG
        self._user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self._user32.GetWindowRect.restype = wintypes.BOOL
        self._user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        self._user32.MonitorFromWindow.restype = wintypes.HMONITOR
        self._user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
        self._user32.GetMonitorInfoW.restype = wintypes.BOOL
        self._user32.SetWindowPos.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_uint]
        self._user32.SetWindowPos.restype = ctypes.c_bool

    def _window_name(self, window):
        """Return the internal key for a window or raise ValueError."""
        if window is self.main_window:
            return 'main'
        if window is self.framepack_window:
            return 'framepack'
        raise ValueError("Window is not managed by this FullscreenManager")

    def _hwnd(self, window):
        wid = int(window.winId())
        return c_void_p(wid) if wid else None

    def _log_fail(self, operation):
        """Log a warning with the last Win32 error and return False."""
        err = ctypes.get_last_error()
        logger.warning(f"[FULLSCREEN] Win32 call failed during {operation}: error {err}")
        return False

    def _check_bool(self, result, operation):
        """Check a Win32 BOOL-style result (non-zero = success)."""
        if result:
            return True
        return self._log_fail(operation)

    def _check_long(self, result, operation):
        """Check a Win32 LONG-style result where 0 can be valid.

        GetWindowLongW/SetWindowLongW return 0 on failure and set last error.
        We clear the error before the call so a non-zero last error after a
        zero result reliably indicates failure.
        """
        if result != 0:
            return True
        err = ctypes.get_last_error()
        if err == 0:
            return True
        logger.warning(f"[FULLSCREEN] Win32 call failed during {operation}: error {err}")
        return False

    def _get_window_rect(self, hwnd):
        rect = wintypes.RECT()
        if not self._check_bool(self._user32.GetWindowRect(hwnd, byref(rect)), "GetWindowRect"):
            return None
        return rect

    def _monitor_rect(self, window):
        hwnd = int(window.winId())
        h_monitor = self._user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not h_monitor:
            logger.warning("[FULLSCREEN] MonitorFromWindow returned no monitor")
            return None
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if not self._check_bool(self._user32.GetMonitorInfoW(h_monitor, byref(mi)), "GetMonitorInfoW"):
            return None
        return (
            mi.rcMonitor.left,
            mi.rcMonitor.top,
            mi.rcMonitor.right - mi.rcMonitor.left,
            mi.rcMonitor.bottom - mi.rcMonitor.top,
        )

    def enter(self, window):
        name = self._window_name(window)
        if self._state[name]:
            return

        hwnd_val = int(window.winId())
        if hwnd_val == 0:
            logger.warning(f"[FULLSCREEN] Cannot enter {name} fullscreen: winId() is 0")
            return
        hwnd = c_void_p(hwnd_val)

        ctypes.set_last_error(0)
        style = self._user32.GetWindowLongW(hwnd, GWL_STYLE)
        if not self._check_long(style, "GetWindowLongW(style)"):
            return

        ctypes.set_last_error(0)
        exstyle = self._user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if not self._check_long(exstyle, "GetWindowLongW(exstyle)"):
            return

        rect = self._get_window_rect(hwnd)
        if rect is None:
            return

        self._saved[name] = {
            'style': style,
            'exstyle': exstyle,
            'rect': (
                rect.left, rect.top,
                rect.right - rect.left, rect.bottom - rect.top,
            ),
        }

        new_style = style & ~(
            WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU
        )
        ctypes.set_last_error(0)
        if not self._check_long(self._user32.SetWindowLongW(hwnd, GWL_STYLE, new_style), "SetWindowLongW(style)"):
            self._saved[name].clear()
            return

        mon = self._monitor_rect(window)
        if mon is None:
            # Try to restore style so the window is not left in a bad state.
            ctypes.set_last_error(0)
            self._user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            self._saved[name].clear()
            return
        mon_x, mon_y, mon_w, mon_h = mon

        if not self._check_bool(
            self._user32.SetWindowPos(
                hwnd, HWND_TOP,
                mon_x, mon_y, mon_w, mon_h,
                SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER
            ),
            "SetWindowPos(enter)",
        ):
            ctypes.set_last_error(0)
            self._user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            self._saved[name].clear()
            return

        self._state[name] = True
        logger.info(f"[FULLSCREEN] Entered {name} fullscreen {mon_w}x{mon_h}")

    def exit(self, window):
        name = self._window_name(window)
        if not self._state[name]:
            return

        hwnd_val = int(window.winId())
        if hwnd_val == 0:
            logger.warning(f"[FULLSCREEN] Cannot exit {name} fullscreen: winId() is 0")
            self._state[name] = False
            self._saved[name].clear()
            return
        hwnd = c_void_p(hwnd_val)

        saved = self._saved[name]
        restore_style = saved.get('style')
        restore_exstyle = saved.get('exstyle')
        restore_rect = saved.get('rect')

        if restore_style is not None:
            ctypes.set_last_error(0)
            if not self._check_long(self._user32.SetWindowLongW(hwnd, GWL_STYLE, restore_style), "SetWindowLongW(style)"):
                return
        if restore_exstyle is not None:
            ctypes.set_last_error(0)
            if not self._check_long(self._user32.SetWindowLongW(hwnd, GWL_EXSTYLE, restore_exstyle), "SetWindowLongW(exstyle)"):
                pass  # Non-critical; continue to restore position.
        if restore_rect is not None:
            x, y, w, h = restore_rect
            # Use HWND_NOTOPMOST to ensure the window leaves any topmost state,
            # but pass SWP_NOZORDER so the relative Z order is otherwise unchanged.
            if not self._check_bool(
                self._user32.SetWindowPos(
                    hwnd, HWND_NOTOPMOST,
                    int(x), int(y), int(w), int(h),
                    SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER
                ),
                "SetWindowPos(exit)",
            ):
                return

        self._state[name] = False
        self._saved[name].clear()
        logger.info(f"[FULLSCREEN] Exited {name} fullscreen")

    def toggle(self, window):
        name = self._window_name(window)
        if self._state[name]:
            self.exit(window)
        else:
            self.enter(window)
        self.toggled.emit(self._state[name])

    def is_fullscreen(self, window):
        name = self._window_name(window)
        return self._state[name]

    def sync_framepack(self, enter):
        if not self.framepack_window or not self.framepack_window.isVisible():
            return
        if enter:
            self.enter(self.framepack_window)
        else:
            self.exit(self.framepack_window)
