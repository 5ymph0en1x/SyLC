# -*- coding: utf-8 -*-
"""Window presentation, navigation visibility and Qt input coordination."""

import logging
import time

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget


logger = logging.getLogger(__name__)


class WindowPresentationMixin:
    def _controls_shown(self):
        """True if the nav bar is actually visible (accounts for the fullscreen opacity trick)."""
        hud = getattr(self, 'stereo_hud', None)
        if hud is not None and hud.active:
            return bool(hud.desired_visible)
        if not self.controls_overlay.isVisible():
            return False
        eff = self.controls_overlay.graphicsEffect()
        if eff and eff.opacity() < 0.1:
            return False
        if self.controls_overlay.windowOpacity() < 0.1:
            return False
        return True

    def _controls_busy(self):
        """True if the user is interacting with the bar (hovering it or an open dropdown) — never auto-hide then."""
        if getattr(self, '_is_scrubbing', False):
            return True

        try:
            slider = self.controls_overlay.time_slider
            if slider.isSliderDown():
                return True
            volume = getattr(self.controls_overlay, 'volume_slider', None)
            if volume is not None and getattr(volume, '_dragging', False):
                return True
        except Exception:
            pass

        hud = getattr(self, 'stereo_hud', None)
        hud_active = bool(hud is not None and hud.active)
        
        # --- DÉTECTION DES FENÊTRES EXTÉRIEURES ---
        has_external = False
        fp = getattr(self, 'framepacking_window', None)
        if fp and getattr(fp, 'isVisible', lambda: False)():
            has_external = True
        eyes = getattr(self, 'eye_windows', None)
        if eyes and any(e and getattr(e, 'isVisible', lambda: False)() for e in eyes):
            has_external = True
        
        hide_native = hud_active and not has_external

        if hud_active:
            if (getattr(hud, '_popup', None) is not None
                    or getattr(hud, '_pressed_widget', None) is not None):
                return True
            # La soupape anti-drapeau-collé doit lire l'horloge d'interaction
            # DU HUD : _nav_poll_tick rafraîchit _nav_last_activity à chaque
            # tick occupé (120 ms), donc une fenêtre de 10 s mesurée sur
            # _nav_last_activity ne peut jamais expirer — un `interacting`
            # resté vrai épinglait les barres pour toujours.
            if (getattr(hud, 'interacting', False)
                    and time.monotonic() - getattr(hud, 'last_interaction', 0.0) < 10.0):
                return True
            # En mode HUD pur (vidéo 3D directement dans la fenêtre principale), on s'arrête ici
            if hide_native:
                return False

        try:
            # Le hover Qt n'est fiable que SANS HUD stéréo : en mode HUD (mono
            # comme pilote/FramePack), la barre capturée/remappée garde un
            # WA_UnderMouse périmé à True, et chaque tick occupé rafraîchit
            # l'échéance — les deux barres par œil ne disparaissaient JAMAIS.
            # Un survol réel continue d'épingler la barre : ses événements
            # Enter/MouseMove passent par l'eventFilter → _mark_activity().
            if not hud_active and self.controls_overlay.underMouse():
                return True
                
            for combo in (self.controls_overlay.audio_track_combo,
                          self.controls_overlay.subtitle_track_combo,
                          self.controls_overlay.stereo_mode_combo):
                if combo.view().isVisible():
                    return True
                    
            for name in ('export_menu', 'export_mvhevc_menu',
                         'export_eye_order_menu', 'cast_menu',
                         'synth3d_menu', 'synth3d_preset_menu',
                         'synth3d_depth_preset_menu'):
                menu = getattr(self.controls_overlay, name, None)
                if menu is not None and menu.isVisible():
                    return True
        except Exception:
            pass
        return False

    def _mark_activity(self):
        """Register mouse activity inside the window: show the bar and reset the idle clock."""
        self._nav_last_activity = time.monotonic()
        if not self._controls_shown():
            self.show_controls()

    def _nav_is_fullscreen(self):
        """Return the effective fullscreen state used by nav rendering.

        The player deliberately implements HDR-safe fullscreen with Win32
        styles, so Qt's isFullScreen() remains False for the normal fullscreen
        path. Navigation opacity/heartbeat decisions must include that state.
        """
        return bool(self.isFullScreen() or getattr(self, '_is_fake_fullscreen', False))

    def _cursor_on_our_window(self):
        """True only if the top-level window actually UNDER the cursor belongs
        to this app. frameGeometry().contains() alone is blind to z-order: with
        the player lying BEHIND the user's work window, every mouse move over
        that shared screen area popped the (Tool-window) nav bar above the
        other app — 'the player keeps putting itself on top every few seconds'.
        WindowFromPoint sees the real stacking; GetCursorPos gives physical
        pixels so no DPI conversion of QCursor.pos() is needed."""
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            pt = wintypes.POINT()
            if not user32.GetCursorPos(ctypes.byref(pt)):
                return True
            user32.WindowFromPoint.argtypes = [wintypes.POINT]
            user32.WindowFromPoint.restype = wintypes.HWND
            hwnd = user32.WindowFromPoint(pt)
            if not hwnd:
                return False
            GA_ROOT = 2
            user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
            user32.GetAncestor.restype = wintypes.HWND
            root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
            # Covers the main window AND every other top-level of ours (nav bar,
            # framepack output, eye windows...) — mpv's embedded child resolves
            # to the main window via GA_ROOT.
            return QWidget.find(int(root)) is not None
        except Exception:
            return True     # non-Windows / API failure: keep legacy behavior

    def _is_cursor_over_framepack(self, pos):
        """True if the cursor is over the framepacking 3D window (or any eye window)."""
        fp = getattr(self, 'framepacking_window', None)
        if fp is not None and fp.isVisible():
            try:
                if fp.frameGeometry().contains(pos):
                    return True
            except Exception:
                pass
        for ew in getattr(self, 'eye_windows', None) or ():
            try:
                if ew is not None and ew.isVisible() and ew.frameGeometry().contains(pos):
                    return True
            except Exception:
                pass
        return False

    def _nav_poll_tick(self):
        """SINGLE source of truth for the nav bar: show on movement inside the window,
        hide after 3 s (playing) / 5 s (paused) of no movement inside the window."""
        try:
            if not self.has_media:
                self._nav_had_media = False
                return  # before/after playback: leave the bar as-is
            if not self._nav_had_media:
                # Playback just started: show the bar, then let it auto-hide normally.
                self._nav_had_media = True
                self._nav_last_cursor = QCursor.pos()
                self._mark_activity()
                return
            pos = QCursor.pos()
            moved = (pos - self._nav_last_cursor).manhattanLength() > 1
            self._nav_last_cursor = pos
            inside = (self.isVisible() and not self.isMinimized()
                      and (self.frameGeometry().contains(pos)
                           or self._is_cursor_over_framepack(pos))
                      and self._cursor_on_our_window())
            if moved and inside:
                self._mark_activity()
                return
            if self._controls_shown():
                if self._controls_busy():
                    self._nav_last_activity = time.monotonic()  # defer while interacting
                    return
                hud = getattr(self, 'stereo_hud', None)
                # The stereoscopic HUD uses the requested five-second dwell:
                # long enough to aim at duplicated controls, deterministic in
                # both play/pause states. The 2D policy stays 3 s playing / 5 s
                # paused.
                timeout = (5.0 if hud is not None and hud.active
                           else (3.0 if self.is_playing else 5.0))
                if (time.monotonic() - self._nav_last_activity) >= timeout:
                    self.hide_controls()
        except Exception:
            pass

    def show_controls(self):
        # Showing the bar is an activity edge even when it comes from pause,
        # fullscreen, media load, or a delayed 3D transition rather than a
        # mouse move. Without this refresh the poller can hide it again on its
        # very next 120 ms tick using a stale deadline.
        self._nav_last_activity = time.monotonic()

        # Stop both retired timer paths. _nav_poll_tick is the single hide
        # authority and re-checks active drags/popups before every transition.
        self._mouse_inactivity_timer.stop()
        self._ensure_controls_timer_initialized()
        self.controls_hide_timer.stop()

        # V14b: Restore opacity and ensure visibility
        opacity_effect = self.controls_overlay.graphicsEffect()
        if opacity_effect:
            opacity_effect.setOpacity(1.0)
            
        hud = getattr(self, 'stereo_hud', None)
        if hud is not None:
            hud.set_desired_visible(True)
            
        if hud is None or not hud.active:
            if not self.controls_overlay.isVisible():
                self.controls_overlay.show()
            self.controls_overlay.setWindowOpacity(1.0)
            self.controls_overlay.raise_()
        else:
            # HUD active: trigger a fresh capture so the bar appears on
            # all 3D display surfaces immediately (bypasses stale caches).
            if hud is not None:
                hud.sync(force=True)
            self.controls_overlay.setWindowOpacity(1.0)
            
        self.setCursor(Qt.CursorShape.ArrowCursor)

        # V14b RENDER HEARTBEAT: Stop heartbeat when controls are visible (UI activity is sufficient)
        if self._render_heartbeat_timer.isActive():
            self._render_heartbeat_timer.stop()

    def hide_controls(self):
        self._mouse_inactivity_timer.stop()
        self._ensure_controls_timer_initialized()
        self.controls_hide_timer.stop()

        # --- DÉTECTION DES FENÊTRES EXTÉRIEURES (Mode Pilote) ---
        has_external = False
        fp = getattr(self, 'framepacking_window', None)
        if fp and getattr(fp, 'isVisible', lambda: False)():
            has_external = True
        eyes = getattr(self, 'eye_windows', None)
        if eyes and any(e and getattr(e, 'isVisible', lambda: False)() for e in eyes):
            has_external = True

        hud = getattr(self, 'stereo_hud', None)
        if hud is not None:
            # L'ordre de masquage est envoyé au HUD (effacera la barre sur le projecteur)
            hud.set_desired_visible(False)
            
        if hud is None or not hud.active:
            if self._nav_is_fullscreen():
                from PySide6.QtWidgets import QGraphicsOpacityEffect
                opacity_effect = self.controls_overlay.graphicsEffect()
                if not opacity_effect:
                    opacity_effect = QGraphicsOpacityEffect(self.controls_overlay)
                    self.controls_overlay.setGraphicsEffect(opacity_effect)
                opacity_effect.setOpacity(0.0)
            else:
                self.controls_overlay.setWindowOpacity(0.0)
        else:
            # HUD actif : On efface la texture 3D du projecteur (et des autres écrans)
            for w in self._display_widgets():
                try:
                    if hasattr(w, 'clear_hud'):
                        w.clear_hud()
                except Exception:
                    pass
            QApplication.processEvents()
            
            # FIX : On applique la disparition à la barre native APRÈS 5 SECONDES, même en mode pilote !
            self.controls_overlay.setWindowOpacity(0.0)

        # La souris disparaît de l'écran principal
        self.setCursor(Qt.CursorShape.BlankCursor)

        if (self._nav_is_fullscreen() and not self._render_heartbeat_timer.isActive()
                and not self.mvc_mode_active and not getattr(self, '_hevc_mode_active', False)):
            self._render_heartbeat_timer.start()

    def _on_mouse_inactivity(self):
        """Deprecated: the nav bar's auto-hide is now driven solely by _nav_poll_tick
        (single source of truth). Kept as a no-op so the legacy 3 s timer can't double-hide."""
        return

    def _render_heartbeat(self):
        """V14b: Maintain rendering smoothness when controls are hidden in fullscreen.

        When UI elements are hidden, Windows DWM may reduce compositor activity.
        Force window-level operations to keep the compositor active.
        """
        # V7b++ STUTTER FIX: Skip in MVC mode - D3D11 widget handles its own rendering
        # The heartbeat was designed for MPV rendering, not for MVC/D3D11 mode.
        # GUI-HOG FIX (2026-07): the HEVC path also drives the native D3D11 renderer and
        # (unlike MVC) does NOT set mvc_mode_active, so this 120 Hz repaint()+processEvents()
        # would run in HEVC fullscreen — a re-entrant GUI-thread hog. Treat HEVC like MVC.
        if self.mvc_mode_active or getattr(self, '_hevc_mode_active', False):
            return

        if self.is_playing:
            # Force a window operation to keep DWM compositor engaged
            # This triggers the same code path as having visible UI elements
            self.video_widget.repaint()  # Immediate repaint, not deferred

            # Also process any pending events to maintain event loop cadence
            from PySide6.QtCore import QCoreApplication
            QCoreApplication.processEvents()

    def toggle_fullscreen(self):
        """Toggle fullscreen using Win32 API to preserve HDR and MPV connection.

        CRITICAL: Qt's showFullScreen() triggers SDR mode on HDR displays.
        CRITICAL: Qt's setWindowFlags() recreates window and breaks MPV.
        Solution: Use Win32 API to modify window style without recreating it.
        """
        import ctypes
        from ctypes import wintypes, byref, c_void_p, c_int, c_uint

        user32 = ctypes.windll.user32
        
        # Define SetWindowPos argument types for proper casting
        user32.SetWindowPos.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_uint]
        user32.SetWindowPos.restype = ctypes.c_bool

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
        HWND_TOPMOST = c_void_p(-1)
        HWND_NOTOPMOST = c_void_p(-2)

        hwnd = c_void_p(int(self.winId()))

        if self._is_fake_fullscreen:
            # === EXIT FULLSCREEN ===
            if self._render_heartbeat_timer.isActive():
                self._render_heartbeat_timer.stop()

            # Restore the DWM window border + rounded corners we suppressed
            try:
                from sylc.framepacking_window_d3d11 import apply_borderless_dwm
                apply_borderless_dwm(int(self.winId()), False)
            except Exception:
                pass

            # Restore original window style
            if hasattr(self, '_saved_style'):
                user32.SetWindowLongW(int(self.winId()), GWL_STYLE, self._saved_style)
            if hasattr(self, '_saved_exstyle'):
                user32.SetWindowLongW(int(self.winId()), GWL_EXSTYLE, self._saved_exstyle)

            # Restore position and size
            if hasattr(self, '_saved_rect'):
                x, y, w, h = self._saved_rect
                user32.SetWindowPos(hwnd, HWND_NOTOPMOST, int(x), int(y), int(w), int(h), SWP_FRAMECHANGED | SWP_SHOWWINDOW)

            self._is_fake_fullscreen = False
            self.controls_overlay.set_fullscreen_icon(False)
            
            # Optimize for windowed: disable flip model to reduce compositor stuttering
            if self.player:
                try:
                    self.player['d3d11-flip'] = 'no'
                    logger.info("[HDR] Windowed: d3d11-flip=no for smooth playback")
                except Exception as e:
                    logger.warning(f"[HDR] Could not set d3d11-flip: {e}")

            # Sync framepacking window
            if self.framepacking_window and self.mvc_mode_active and self.framepacking_window.isVisible():
                self.framepacking_window.exit_fake_fullscreen()
                self.framepacking_window.raise_()

            QTimer.singleShot(100, self._apply_windowed_video_settings)
            logger.info("[FULLSCREEN-WIN32] Exited fake fullscreen")
        else:
            # === ENTER FULLSCREEN ===
            # Save current window state via Win32
            self._saved_style = user32.GetWindowLongW(int(self.winId()), GWL_STYLE)
            self._saved_exstyle = user32.GetWindowLongW(int(self.winId()), GWL_EXSTYLE)

            rect = wintypes.RECT()
            user32.GetWindowRect(int(self.winId()), byref(rect))
            self._saved_rect = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)

            # Get EXACT monitor dimensions via Win32 (avoids DPI scaling issues)
            # MonitorFromWindow + GetMonitorInfo gives us the true pixel dimensions
            MONITOR_DEFAULTTONEAREST = 2
            
            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', wintypes.DWORD),
                    ('rcMonitor', wintypes.RECT),
                    ('rcWork', wintypes.RECT),
                    ('dwFlags', wintypes.DWORD),
                ]
            
            hMonitor = user32.MonitorFromWindow(int(self.winId()), MONITOR_DEFAULTTONEAREST)
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
            
            # Use rcMonitor (full monitor area) not rcWork (excludes taskbar)
            mon_x = mi.rcMonitor.left
            mon_y = mi.rcMonitor.top
            mon_w = mi.rcMonitor.right - mi.rcMonitor.left
            mon_h = mi.rcMonitor.bottom - mi.rcMonitor.top
            
            logger.info(f"[FULLSCREEN-WIN32] Monitor geometry: {mon_x},{mon_y} {mon_w}x{mon_h}")

            # Remove window decorations (borderless)
            new_style = self._saved_style & ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
            user32.SetWindowLongW(int(self.winId()), GWL_STYLE, new_style)

            # Resize to cover full monitor (HDR preserved via __COMPAT_LAYER)
            HWND_TOP = c_void_p(0)
            SWP_NOZORDER = 0x0004
            user32.SetWindowPos(
                hwnd, HWND_TOP,
                mon_x, mon_y, mon_w, mon_h,
                SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER
            )

            # Windows 11 draws a thin border + rounded corners around any
            # top-level window (the white 'liseret' all around fake-fullscreen).
            # Suppress both so the video reaches the true screen edge.
            try:
                from sylc.framepacking_window_d3d11 import apply_borderless_dwm
                apply_borderless_dwm(int(self.winId()), True)
            except Exception:
                pass

            self._is_fake_fullscreen = True
            self.controls_overlay.set_fullscreen_icon(True)
            
            # Optimize for fullscreen: flip model for best performance
            if self.player:
                try:
                    self.player['d3d11-flip'] = 'yes'
                    logger.info("[HDR] Fullscreen: d3d11-flip=yes for optimal performance")
                except Exception as e:
                    logger.warning(f"[HDR] Could not set d3d11-flip: {e}")

            # Sync framepacking window -- but leave a Glasses layout alone (fix
            # round 1, Critical 2b): same reasoning as the visibility-changed
            # handler this mirrors, F-fullscreen on the main window must not
            # force the detached window back to 'framepack' mid-Glasses-session.
            if self.framepacking_window and self.mvc_mode_active and self.framepacking_window.isVisible():
                if getattr(self, 'current_stereo_mode', None) != 'glasses':
                    self.framepacking_window.display_widget.set_stereo_mode('framepack')
                self.framepacking_window.enter_fake_fullscreen()
                self.framepacking_window.raise_()

            logger.info(f"[FULLSCREEN-WIN32] Entered fake fullscreen {mon_w}x{mon_h} (HDR preserved)")

        # Win32 style fullscreen does not guarantee a Qt WindowStateChange;
        # explicitly reposition/show the Tool-window navigation bar after the
        # native resize has reached Qt's layout system.
        QTimer.singleShot(50, self._refresh_nav_after_window_transition)

    def _refresh_nav_after_window_transition(self):
        """Re-anchor and reveal navigation after an HDR-safe Win32 resize."""
        if self.isHidden() or self.isMinimized():
            return
        self._update_overlays_geometry()
        if self.has_media:
            self._mark_activity()

    def _apply_fullscreen_video_settings(self):
        """Apply optimal MPV settings for fullscreen HDR playback."""
        if not self.player:
            return
        try:
            self.player['video-sync'] = 'display-resample'
            
            # Force MPV to reset HDR/brightness settings after fullscreen transition
            # Method 1: Toggle gamma briefly
            self.player['gamma'] = 1
            self.player['gamma'] = 0
            
            # Method 2: Force video output reconfiguration
            try:
                self.player.command('vo-cmdline', 'd3d11-exclusive-fs=no')
            except:
                pass
            
            # Method 3: Re-apply HDR settings
            self.player['target-colorspace-hint'] = 'yes'
            self.player['target-trc'] = 'auto'
            self.player['target-prim'] = 'auto'
            
            # Method 4: Force DWM composition refresh
            try:
                import ctypes
                dwmapi = ctypes.windll.dwmapi
                dwmapi.DwmFlush()
                
                # Also try toggling a DWM window attribute to force HDR refresh
                hwnd = int(self.winId())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = ctypes.c_int(1)
                dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
                value = ctypes.c_int(0)
                dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))
                
                logger.info("[FULLSCREEN] Forced DWM refresh")
            except Exception as e:
                logger.warning(f"[FULLSCREEN] DWM refresh failed: {e}")
            
            logger.info("[FULLSCREEN] Applied fullscreen video settings")
        except Exception as e:
            logger.warning(f"[FULLSCREEN] Could not apply settings: {e}")

    def _refresh_windows_hdr_brightness(self):
        """Force Windows to re-apply HDR SDR brightness setting via DisplayConfig API."""
        try:
            import ctypes
            from ctypes import wintypes, Structure, byref, sizeof

            class LUID(Structure):
                _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

            class DISPLAYCONFIG_DEVICE_INFO_HEADER(Structure):
                _fields_ = [
                    ("type", wintypes.UINT),
                    ("size", wintypes.UINT),
                    ("adapterId", LUID),
                    ("id", wintypes.UINT),
                ]

            class DISPLAYCONFIG_SDR_WHITE_LEVEL(Structure):
                _fields_ = [
                    ("header", DISPLAYCONFIG_DEVICE_INFO_HEADER),
                    ("SDRWhiteLevel", wintypes.DWORD),
                ]

            class DISPLAYCONFIG_PATH_INFO(Structure):
                _fields_ = [
                    ("sourceInfo", ctypes.c_ubyte * 20),
                    ("targetInfo", ctypes.c_ubyte * 48),
                    ("flags", wintypes.UINT),
                ]

            class DISPLAYCONFIG_MODE_INFO(Structure):
                _fields_ = [
                    ("infoType", wintypes.UINT),
                    ("id", wintypes.UINT),
                    ("adapterId", LUID),
                    ("info", ctypes.c_ubyte * 64),
                ]

            QDC_ONLY_ACTIVE_PATHS = 0x00000002
            GET_SDR_WHITE_LEVEL = 0x0000000B
            SET_SDR_WHITE_LEVEL = 0x0000000C

            numPath = wintypes.UINT(0)
            numMode = wintypes.UINT(0)

            result = ctypes.windll.user32.GetDisplayConfigBufferSizes(
                QDC_ONLY_ACTIVE_PATHS, byref(numPath), byref(numMode))

            if result == 0 and numPath.value > 0:
                pathArray = (DISPLAYCONFIG_PATH_INFO * numPath.value)()
                modeArray = (DISPLAYCONFIG_MODE_INFO * numMode.value)()

                result = ctypes.windll.user32.QueryDisplayConfig(
                    QDC_ONLY_ACTIVE_PATHS, byref(numPath), pathArray,
                    byref(numMode), modeArray, None)

                if result == 0:
                    for i in range(numPath.value):
                        path_bytes = bytes(pathArray[i].targetInfo)
                        adapterId = LUID()
                        adapterId.LowPart = int.from_bytes(path_bytes[0:4], 'little')
                        adapterId.HighPart = int.from_bytes(path_bytes[4:8], 'little', signed=True)
                        targetId = int.from_bytes(path_bytes[8:12], 'little')

                        sdrLevel = DISPLAYCONFIG_SDR_WHITE_LEVEL()
                        sdrLevel.header.type = GET_SDR_WHITE_LEVEL
                        sdrLevel.header.size = sizeof(DISPLAYCONFIG_SDR_WHITE_LEVEL)
                        sdrLevel.header.adapterId = adapterId
                        sdrLevel.header.id = targetId

                        if ctypes.windll.user32.DisplayConfigGetDeviceInfo(byref(sdrLevel)) == 0:
                            currentLevel = sdrLevel.SDRWhiteLevel
                            logger.info(f"[HDR-FIX] SDR white level: {currentLevel}")
                            sdrLevel.header.type = SET_SDR_WHITE_LEVEL
                            ctypes.windll.user32.DisplayConfigSetDeviceInfo(byref(sdrLevel))
                            logger.info("[HDR-FIX] Re-applied SDR white level")
                            break

        except Exception as e:
            logger.warning(f"[HDR-FIX] Could not refresh HDR: {e}")

    def _apply_windowed_video_settings(self):
        """Apply optimal MPV settings for windowed playback."""
        if not self.player:
            return
        try:
            self.player['video-sync'] = 'display-resample'
            logger.info("[WINDOWED] Applied windowed video settings")
        except Exception as e:
            logger.warning(f"[WINDOWED] Could not apply settings: {e}")
        
        # Refresh Windows HDR brightness after exiting fullscreen
        QTimer.singleShot(200, self._refresh_windows_hdr_brightness)

    def _enter_borderless_fullscreen_win32(self):
        """Enter borderless fullscreen using Win32 API directly.
        
        This avoids Qt's setWindowFlags() which destroys and recreates the window,
        breaking the MPV player connection. Instead, we modify the window style
        directly via Windows API.
        """
        import ctypes
        from ctypes import wintypes
        
        # Win32 constants
        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        WS_SYSMENU = 0x00080000
        WS_EX_DLGMODALFRAME = 0x00000001
        WS_EX_CLIENTEDGE = 0x00000200
        WS_EX_STATICEDGE = 0x00020000
        SWP_FRAMECHANGED = 0x0020
        SWP_NOZORDER = 0x0004
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2
        SWP_NOACTIVATE = 0x0010
        
        user32 = ctypes.windll.user32
        
        # Get window handle
        hwnd = int(self.winId())
        
        # Save current window state
        self._saved_style = user32.GetWindowLongW(int(self.winId()), GWL_STYLE)
        self._saved_exstyle = user32.GetWindowLongW(int(self.winId()), GWL_EXSTYLE)
        
        # Get window rect before changing
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        self._saved_rect = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        
        # Remove window decorations
        new_style = self._saved_style & ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
        new_exstyle = self._saved_exstyle & ~(WS_EX_DLGMODALFRAME | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE)
        
        user32.SetWindowLongW(int(self.winId()), GWL_STYLE, new_style)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_exstyle)
        
        # Get screen geometry
        from PySide6.QtGui import QGuiApplication
        screen = self.screen() or QGuiApplication.primaryScreen()
        screen_geo = screen.geometry()
        
        # Set topmost and resize to fullscreen
        # SWP_SHOWWINDOW is critical to ensure window is visible after style change
        SWP_SHOWWINDOW = 0x0040

        x = int(screen_geo.x())
        y = int(screen_geo.y())
        w = int(screen_geo.width())
        h = int(screen_geo.height())

        logger.info(f"[FULLSCREEN-WIN32] Setting window to {x},{y} {w}x{h}")

        result = user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            x, y, w, h,
            SWP_FRAMECHANGED | SWP_SHOWWINDOW
        )

        logger.info(f"[FULLSCREEN-WIN32] Entered borderless fullscreen {w}x{h} (SetWindowPos result={result})")

    def _exit_borderless_fullscreen_win32(self):
        """Exit borderless fullscreen using Win32 API directly."""
        import ctypes
        
        SWP_FRAMECHANGED = 0x0020
        HWND_NOTOPMOST = -2
        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        
        user32 = ctypes.windll.user32
        hwnd = int(self.winId())
        
        # Restore window styles
        if hasattr(self, '_saved_style'):
            user32.SetWindowLongW(int(self.winId()), GWL_STYLE, self._saved_style)
        if hasattr(self, '_saved_exstyle'):
            user32.SetWindowLongW(int(self.winId()), GWL_EXSTYLE, self._saved_exstyle)
        
        # Restore position and size, remove topmost
        if hasattr(self, '_saved_rect'):
            x, y, w, h = self._saved_rect
            user32.SetWindowPos(
                hwnd, HWND_NOTOPMOST,
                x, y, w, h,
                SWP_FRAMECHANGED
            )
        
        logger.info("[FULLSCREEN-WIN32] Exited borderless fullscreen")

    def dragEnterEvent(self, event):
        """Accept drag of files/folders (including a Blu-ray drive or BDMV folder)."""
        try:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
        except Exception:
            pass

    def dropEvent(self, event):
        """Play a dropped file, or auto-detect the 3D feature from a dropped folder/disc."""
        try:
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path:
                    event.acceptProposedAction()
                    self.play_file(path)  # smart: file, folder, drive root, or BDMV
        except Exception as e:
            logger.warning(f"[DROP] {e}")

    def resizeEvent(self, event):
        """Repositions overlays on window resize."""
        super().resizeEvent(event)
        self._update_overlays_geometry()

        # Fix: Safely handle MPV resize commands
        if self.player:
            try:
                self.player.command_async('auto', ['set', 'video-zoom', 0])
                self.player.command_async('auto', ['set', 'video-pan-x', 0])
                self.player.command_async('auto', ['set', 'video-pan-y', 0])
            except Exception:
                pass

    def moveEvent(self, event):
        """Repositions overlays on window move."""
        super().moveEvent(event)
        self._update_overlays_geometry()

    def _update_overlays_geometry(self):
        """Updates the geometry of all floating overlays."""
        if not self.isVisible(): return

        # Calculate global geometry for the video area
        # We want overlays to cover the video_container
        global_pos = self.video_container.mapToGlobal(QPoint(0, 0))
        w = self.video_container.width()
        h = self.video_container.height()

        # Info Overlay (Full Screen)
        self.info_overlay.move(global_pos)
        self.info_overlay.resize(w, h)

        # Loading Overlay (Full Screen)
        self.loading_overlay.move(global_pos)
        self.loading_overlay.resize(w, h)

        # Controls Overlay (Bottom Floating)
        ctrl_h = self.controls_overlay.sizeHint().height()
        margin_bottom = 20
        margin_side = 10

        # Compute bar width: at least 600px or window width minus margins,
        # but NEVER wider than the window itself (prevents overflow off-screen
        # when the window is narrower than 600px).
        ctrl_w = max(600, w - (margin_side * 2))
        ctrl_w = min(ctrl_w, w)
        ctrl_x = global_pos.x() + (w - ctrl_w) // 2
        ctrl_y = global_pos.y() + h - ctrl_h - margin_bottom

        # HARD INVARIANT — the bar must never be wider than the client area.
        # resize() alone gets clamped UP to the overlay's layout-minimum width,
        # which (with the reserved 166px badge slot + widened stereo combo) can
        # exceed a ~1332px window and push the fullscreen button past the right
        # edge. Pinning maximumWidth = ctrl_w and clearing the layout-driven
        # window minimum forces the bar to exactly ctrl_w; its flexible middle
        # (the VU meter) absorbs the difference, shrinking toward 0 before any
        # overflow can occur. See _checkpoints/fix_bar_overflow.
        self.controls_overlay.setMinimumWidth(0)
        self.controls_overlay.setMaximumWidth(ctrl_w)
        self.controls_overlay.move(ctrl_x, ctrl_y)
        self.controls_overlay.resize(ctrl_w, ctrl_h)

        hud = getattr(self, 'stereo_hud', None)
        if hud is not None:
            # A resize changes both the capture resolution and the texture's
            # aspect-correct eye-space rectangle.
            hud.sync(force=True)

        # Monitoring Overlay (Top Right)
        self._update_monitoring_overlay_geometry()
        self._update_metrics_overlay_geometry()

        # Ensure visibility/z-order
        if self.controls_overlay.isVisible():
            self.controls_overlay.raise_()
        if self.info_overlay.isVisible():
            self.info_overlay.raise_()

    def mouseMoveEvent(self, event):
        """Nav bar: any movement over the window counts as activity (moves over child
        widgets like the D3D11 video are caught globally by _nav_poll_tick)."""
        self._mark_activity()
        super().mouseMoveEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMinimized():
                self.info_overlay.hide()
                self.monitoring_overlay.hide()
            elif not self.isHidden():
                # Reposition overlays after unminimize / fullscreen toggle:
                # the video_container may have shifted relative to the frame
                # without a dedicated resize/move event on Windows.
                self._update_overlays_geometry()
                if self.has_media:
                    self.show_controls()
                else:
                    self.info_overlay.show()
                self._refresh_monitoring_overlay()
        super().changeEvent(event)

    def enterEvent(self, event):
        """Mouse entering the player is navigation activity."""
        self._mouse_outside_window = False
        self._mouse_inactivity_timer.stop()
        if self.has_media:
            self._mark_activity()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Start the poller's idle interval when the pointer leaves the player."""
        self._mouse_outside_window = True
        if self.is_playing and self.controls_overlay.isVisible():
            # Check if mouse is over a popup (ComboBox dropdown)
            audio_combo = self.controls_overlay.audio_track_combo
            subtitle_combo = self.controls_overlay.subtitle_track_combo
            stereo_combo = self.controls_overlay.stereo_mode_combo

            for combo in [audio_combo, subtitle_combo, stereo_combo]:
                if combo.view().isVisible():
                    # Mouse is in a popup - don't start hide timer
                    return

            self._nav_last_activity = time.monotonic()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()

        # Space -> Play/Pause
        if key == Qt.Key.Key_Space:
            self.toggle_play()
            event.accept()
            return

        # Escape -> Exit Fullscreen
        if key == Qt.Key.Key_Escape:
            if self._is_fake_fullscreen:
                self.toggle_fullscreen()
            event.accept()
            return

        # [ / ] -> adjust A/V sync offset live (delay video to match the heard audio)
        if key in (Qt.Key.Key_BracketRight, Qt.Key.Key_BracketLeft):
            delta = 0.05 if key == Qt.Key.Key_BracketRight else -0.05
            sync_decoder = None
            if (getattr(self, '_hevc_mode_active', False)
                    and getattr(self, 'hevc_thread', None) is not None
                    and hasattr(self.hevc_thread, 'adjust_av_offset')):
                sync_decoder = self.hevc_thread
            elif (self.mvc_decoder_thread and self.mvc_mode_active
                    and hasattr(self.mvc_decoder_thread, 'adjust_av_offset')):
                sync_decoder = self.mvc_decoder_thread
            if sync_decoder is not None:
                off = sync_decoder.adjust_av_offset(delta)
                # V60: persist the trim so every future decoder thread starts with it
                self._app_settings['av_sync_offset_s'] = off
                self._save_app_settings()
                if off >= 0:
                    self.show_3d_notification(f"A/V sync — video delayed by {off*1000:.0f} ms", success=True)
                else:
                    self.show_3d_notification(f"A/V sync — video advanced by {-off*1000:.0f} ms", success=True)
            event.accept()
            return

        super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        # V15: Handle combo popup visibility changes
        if watched.property("is_combo_popup") and event.type() == QEvent.Type.Hide:
            self._on_combo_popup_closed()
            return super().eventFilter(watched, event)

        if watched is self.controls_overlay:
            if event.type() in (QEvent.Type.Enter, QEvent.Type.MouseMove):
                # Direct event handling gives immediate feedback; the global
                # cursor poll remains the sole authority for hiding.
                self._mouse_inactivity_timer.stop()
                self._ensure_controls_timer_initialized()
                self.controls_hide_timer.stop()
                if self.has_media:
                    self._mark_activity()
            elif event.type() == QEvent.Type.Leave and self.is_playing:
                # V15: Mouse left controls overlay
                # Check if it went to a popup (ComboBox dropdown)
                audio_combo = self.controls_overlay.audio_track_combo
                subtitle_combo = self.controls_overlay.subtitle_track_combo
                stereo_combo = self.controls_overlay.stereo_mode_combo

                for combo in [audio_combo, subtitle_combo, stereo_combo]:
                    if combo.view().isVisible():
                        # Mouse is in a popup - don't start hide timer
                        return super().eventFilter(watched, event)

                # Mouse left to main window area: begin a fresh poller-owned
                # idle interval, without arming a competing QTimer.
                if not self._mouse_outside_window:
                    self._nav_last_activity = time.monotonic()
        return super().eventFilter(watched, event)

    def _update_metrics_overlay_geometry(self):
        if not hasattr(self, 'metrics_overlay') or not self.metrics_overlay.parent():
            return
        self.metrics_overlay.adjustSize()
        margin = 20
        self.metrics_overlay.move(margin, margin)

    def _update_monitoring_overlay_geometry(self):
        if not hasattr(self, 'monitoring_overlay'):
            return
        margin = 20
        width = self.monitoring_overlay.width()
        height = self.monitoring_overlay.height()

        # Convert local position to global screen coordinates for the Tool window
        local_pos = QPoint(self.width() - width - margin, margin)
        global_pos = self.mapToGlobal(local_pos)

        self.monitoring_overlay.move(global_pos)


__all__ = ['WindowPresentationMixin']
