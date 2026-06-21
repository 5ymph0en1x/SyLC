"""S5a: NativeFramepackWidget — a drop-in replacement for FramepackingDisplayWidgetD3D11
backed by the native C++ D3D11 renderer.

When SYLC_NATIVE_RENDER=1, the player injects this as the detached framepack window's
display_widget, so the real 3D output is produced by the native renderer instead of Qt
RHI (removing the tobytes/QByteArray upload copy and the Qt RHI overhead). The Qt path
remains the default; this is opt-in and revertible.

It implements the subset of the widget contract the player actually calls on the
display widget (verified by grep): set_frame_yuv_views, set_stereo_mode,
pause_rendering, resume_rendering, clear_textures, set_subtitle, clear_subtitle,
plus the deprecated set_frame_fast (no-op) and refresh_sdr_white_level.

Frame delivery runs on the GUI thread (the frameYUVReady QueuedConnection slot),
reusing the player's existing pacing + serialization. The decode-thread raw-pointer
push (Copy #1 elimination) is the subsequent step (S5b).
"""
import logging
import os
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt

logger = logging.getLogger("SyLC.NativeWidget")

_MODE = {'2d': 0, 'framepack': 1, 'sbs': 2, 'tab': 3}


class NativeFramepackWidget(QWidget):
    def __init__(self, parent=None, sdr_white=1.0):
        super().__init__(parent)
        # Own native HWND for the D3D11 swapchain; don't let Qt paint over it.
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setAttribute(Qt.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._r = None                 # NativeRenderer, or False if unavailable
        self._stereo_mode = 1          # framepack default
        self.current_stereo_mode = 1   # public attr the player syncs/reads
        self._sdr_white = float(sdr_white) if sdr_white else 1.0
        self._hdr = False
        self._gamma = 0.0
        self._rendering_paused = False
        self._sub = None               # (rgba_ndarray, (x,y,w,h) normalized) or None
        self._fail_logged = False

        # Public attrs some call sites read on the Qt widget.
        self.has_video = False

    # --- Qt overrides ---------------------------------------------------------
    def paintEngine(self):
        return None  # rendering goes through D3D11, not Qt's paint system

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._r and self._r is not False:
            s = event.size()
            try:
                self._r.resize(max(1, s.width()), max(1, s.height()))
            except Exception:
                pass

    # --- native renderer lifecycle -------------------------------------------
    def _ensure(self):
        if self._r is not None:
            return self._r is not False
        try:
            import mvc_demuxer_cpp as m
            if not getattr(m, "NATIVE_RENDERER_AVAILABLE", False) or not hasattr(m, "NativeRenderer"):
                logger.warning("[NATIVE-WIDGET] module built without NativeRenderer")
                self._r = False
                return False

            # SDR vs HDR from the display's SDR white level (>1.01 => HDR).
            self._hdr = self._sdr_white > 1.01
            env_hdr = os.environ.get("SYLC_NATIVE_HDR")
            if env_hdr is not None:
                self._hdr = env_hdr == "1"
            self._gamma = 2.4 if self._hdr else 0.0
            try:
                self._gamma = float(os.environ.get("SYLC_NATIVE_GAMMA", str(self._gamma)))
            except ValueError:
                pass

            r = m.NativeRenderer()
            sz = self.size()
            if not r.initialize(int(self.winId()), max(1, sz.width()), max(1, sz.height()), self._hdr):
                logger.warning(f"[NATIVE-WIDGET] initialize failed: {r.last_error()}")
                self._r = False
                return False
            logger.info(f"[NATIVE-WIDGET] {r.backend_info()} | hdr={self._hdr} gamma={self._gamma} sdr_white={self._sdr_white}")
            self._r = r
            return True
        except Exception as e:
            logger.warning(f"[NATIVE-WIDGET] disabled: {e}")
            self._r = False
            return False

    # --- contract: frame delivery --------------------------------------------
    def set_frame_yuv_views(self, y_l_or_tuple, u_l_or_right=None, v_l=None,
                            y_r=None, u_r=None, v_r=None):
        if self._rendering_paused:
            return
        if isinstance(y_l_or_tuple, tuple):
            yl, ul, vl = y_l_or_tuple
            if isinstance(u_l_or_right, tuple):
                yr, ur, vr = u_l_or_right
            else:
                yr = ur = vr = None
        else:
            yl, ul, vl = y_l_or_tuple, u_l_or_right, v_l
            yr, ur, vr = y_r, u_r, v_r
        if not self._ensure():
            return
        try:
            rect = self._sub[1] if self._sub else (0.0, 0.0, 1.0, 1.0)
            self._r.set_uniforms(self._stereo_mode, 1 if self._sub else 0,
                                 rect[0], rect[1], rect[2], rect[3],
                                 self._sdr_white, self._gamma)
            if self._sub is not None:
                self._r.set_subtitle_rgba(self._sub[0])
            self._r.set_yuv_frame(yl, ul, vl, yr, ur, vr)
            self._r.present()
            self.has_video = True
        except Exception as e:
            if not self._fail_logged:
                logger.warning(f"[NATIVE-WIDGET] frame delivery failed: {e}")
                self._fail_logged = True

    # --- contract: control ----------------------------------------------------
    def set_stereo_mode(self, mode_str):
        self._stereo_mode = _MODE.get(str(mode_str).lower(), 1)
        self.current_stereo_mode = self._stereo_mode

    def pause_rendering(self):
        self._rendering_paused = True
        if self._r and self._r is not False:
            try:
                self._r.pause()
            except Exception:
                pass

    def resume_rendering(self):
        self._rendering_paused = False
        if self._r and self._r is not False:
            try:
                self._r.resume()
            except Exception:
                pass

    def clear_textures(self):
        self.has_video = False
        if self._r and self._r is not False:
            try:
                self._r.clear_frame()
                self._r.present()
            except Exception:
                pass

    def set_subtitle(self, rgba_array, x, y, w, h, video_width=1920, video_height=1080):
        try:
            vw = float(video_width) or 1920.0
            vh = float(video_height) or 1080.0
            nx, ny = x / vw, y / vh
            nw, nh = w / vw, h / vh
            self._sub = (np.ascontiguousarray(rgba_array, dtype=np.uint8), (nx, ny, nw, nh))
        except Exception as e:
            logger.warning(f"[NATIVE-WIDGET] set_subtitle failed: {e}")
            self._sub = None

    def clear_subtitle(self):
        self._sub = None

    # --- deprecated / no-ops the player may still call ------------------------
    def set_frame_fast(self, *args, **kwargs):
        pass  # legacy packed-array path; unused in the YUV pipeline

    def refresh_sdr_white_level(self):
        pass  # native picks SDR/HDR at init from the white level

    def shutdown(self):
        if self._r and self._r is not False:
            try:
                self._r.shutdown()
            except Exception:
                pass
        self._r = False
