"""NativeFramepackWidget — the player's sole video display widget, backed by the
native C++ D3D11 renderer.

The player injects this for both the embedded 2D view and the detached framepack
window, so all video output is produced by the native renderer (no tobytes/QByteArray
upload copy, no Qt RHI overhead). It replaced the former Qt RHI widget
(FramepackingDisplayWidgetD3D11), now removed.

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
import time
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt

logger = logging.getLogger("SyLC.NativeWidget")

_MODE = {'2d': 0, 'framepack': 1, 'sbs': 2, 'tab': 3}


def _pct(sorted_vals, q):
    """p-quantile of an already-sorted list (nearest-rank). 0.0 on empty."""
    if not sorted_vals:
        return 0.0
    idx = int(q * (len(sorted_vals) - 1) + 0.5)
    if idx >= len(sorted_vals):
        idx = len(sorted_vals) - 1
    return sorted_vals[idx]


def query_sdr_white_level():
    """Windows SDR white level as an scRGB multiplier (1.0 = SDR display, ~2.0-3.5
    for HDR). Extracted from the Qt widget so the native renderer path is
    self-sufficient for HDR brightness (no dependency on the Qt widget)."""
    import ctypes
    from ctypes import Structure, c_uint32, c_int32, byref, sizeof
    try:
        class DISPLAYCONFIG_DEVICE_INFO_HEADER(Structure):
            _fields_ = [("type", c_uint32), ("size", c_uint32),
                        ("adapterId_LowPart", c_uint32), ("adapterId_HighPart", c_int32),
                        ("id", c_uint32)]

        class DISPLAYCONFIG_SDR_WHITE_LEVEL(Structure):
            _fields_ = [("header", DISPLAYCONFIG_DEVICE_INFO_HEADER), ("SDRWhiteLevel", c_uint32)]

        QDC_ONLY_ACTIVE_PATHS = 0x00000002
        DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL = 0x0B
        user32 = ctypes.windll.user32

        num_paths = c_uint32(0)
        num_modes = c_uint32(0)
        if (user32.GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, byref(num_paths),
                                               byref(num_modes)) != 0 or num_paths.value == 0):
            return 1.0

        class DISPLAYCONFIG_PATH_INFO(Structure):
            _fields_ = [("data", c_uint32 * 18)]

        class DISPLAYCONFIG_MODE_INFO(Structure):
            _fields_ = [("data", c_uint32 * 16)]

        paths = (DISPLAYCONFIG_PATH_INFO * num_paths.value)()
        modes = (DISPLAYCONFIG_MODE_INFO * num_modes.value)()
        if user32.QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, byref(num_paths), paths,
                                     byref(num_modes), modes, None) != 0:
            return 1.0
        if num_paths.value > 0:
            pd = paths[0].data
            sdr = DISPLAYCONFIG_SDR_WHITE_LEVEL()
            sdr.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL
            sdr.header.size = sizeof(DISPLAYCONFIG_SDR_WHITE_LEVEL)
            sdr.header.adapterId_LowPart = pd[8]
            sdr.header.adapterId_HighPart = pd[9]
            sdr.header.id = pd[10]
            if user32.DisplayConfigGetDeviceInfo(byref(sdr)) == 0:
                mult = (sdr.SDRWhiteLevel / 1000.0) / 80.0   # scRGB 1.0 = 80 nits
                logger.info(f"[NATIVE-WIDGET] SDR white level multiplier: {mult:.2f}")
                return mult
        return 1.0
    except Exception as e:
        logger.debug(f"[NATIVE-WIDGET] SDR white level query failed: {e}")
        return 1.0


class NativeFramepackWidget(QWidget):
    def __init__(self, parent=None, sdr_white=None):
        super().__init__(parent)
        # Own native HWND for the D3D11 swapchain; don't let Qt paint over it.
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setAttribute(Qt.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._r = None                 # NativeRenderer, or False if unavailable
        self._stereo_mode = 1          # framepack default
        self.current_stereo_mode = 1   # public attr the player syncs/reads
        # Self-sufficient HDR: query the display's SDR white level when not given,
        # so we no longer depend on the Qt widget having done it.
        self._sdr_white = float(sdr_white) if sdr_white is not None else query_sdr_white_level()
        self._sdr_white_level = self._sdr_white   # alias: some call sites read _sdr_white_level
        # Decide SDR vs HDR from the display's SDR white level (>1.01 => HDR), with the
        # SYLC_NATIVE_HDR override, AT CONSTRUCTION so the player can read _hdr BEFORE the
        # first frame (HEVC transfer_sel selection). _ensure() recomputes it identically.
        self._hdr = self._sdr_white > 1.01
        _env_hdr = os.environ.get("SYLC_NATIVE_HDR")
        if _env_hdr is not None:
            self._hdr = _env_hdr == "1"
        self._gamma = 0.0
        self._rendering_paused = False
        self._sub = None               # (rgba_ndarray, (x,y,w,h) normalized, disparity) or None
        self._sub_dirty = False        # upload the RGBA to the GPU only when it changed
        self._sub_depth_override = None  # BD3D dynamic depth (OFMD); None = per-cue value
        self._uniforms_take_disparity = True   # probed once; False on an older renderer build
        self._fail_logged = False
        # 10-bit HEVC (uint16 planes) routing. plane_scale rescales a 10-bit value
        # stored low in an R16 texel back to [0,1]: 65535/1023 ~= 64.06 (yuv420p10le).
        # The player overwrites plane_scale per-source (Task 8).
        self.plane_scale = 65535.0 / 1023.0
        self._have_yuv16 = True                # False after an old .pyd rejects set_yuv_frame16
        self._yuv16_unsupported_logged = False
        # C2: display-aspect override forwarded to the renderer each frame. > 0 forces the
        # display aspect (half-SBS/half-TAB: the packed frame carries the ORIGINAL 2D dims,
        # so each squeezed eye must still display at that aspect); 0.0 = derive from the
        # uploaded eye dimensions. The player sets it per-source (Task C2).
        self.source_aspect = 0.0
        self._have_source_aspect = True        # False after an old .pyd rejects set_source_aspect
        self._source_aspect_unsupported_logged = False
        # HDR10/PQ selectors forwarded to the renderer each frame (next to plane_scale/
        # source_aspect). The player sets them per-source in _try_start_hevc; 0/0 = legacy
        # (byte-identical for MVC/H.264/8-bit). Reset to 0/0 in _stop_hevc_decoder.
        self.yuv_matrix_sel = 0
        self.transfer_sel = 0
        self._have_color_params = True         # False after an old .pyd rejects set_color_params
        self._color_params_unsupported_logged = False

        # Public attrs some call sites read on the Qt widget.
        self.has_video = False

        # --- [HEVC-METER] instrumentation (SYLC_HEVC_DIAG=1, silent otherwise) ---
        # Measures the GUI-thread cost of one frame: slot-to-slot cadence, the native
        # YUV upload call, and present() (vsync) — reported every ~5 s as p50/p99/max.
        self._diag = os.environ.get("SYLC_HEVC_DIAG") == "1"
        self._diag_slot = []       # slot-to-slot intervals (ms)
        self._diag_upload = []     # set_yuv_frame[16] duration (ms)
        self._diag_present = []    # present() duration (ms)
        self._diag_last_slot = None
        self._diag_win = None

    # --- Qt overrides ---------------------------------------------------------
    def paintEngine(self):
        return None  # rendering goes through D3D11, not Qt's paint system

    def _phys(self, w, h):
        """Logical (Qt) -> PHYSICAL pixels for the D3D11 backbuffer. The swapchain and
        the HWND client area live in physical pixels; on a HiDPI display Qt's sizes are
        logical, so passing them raw would make the backbuffer smaller than the client
        area (DXGI STRETCH upscales it -> blur). The C++ present() also self-heals to the
        true GetClientRect, so this keeps the two in agreement instead of fighting."""
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        return max(1, int(round(w * dpr))), max(1, int(round(h * dpr)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._r and self._r is not False:
            s = event.size()
            pw, ph = self._phys(s.width(), s.height())
            try:
                self._r.resize(pw, ph)
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
            pw, ph = self._phys(sz.width(), sz.height())
            if not r.initialize(int(self.winId()), pw, ph, self._hdr):
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
        if self._diag:
            _t_slot = time.perf_counter()
            if self._diag_last_slot is not None:
                self._diag_slot.append((_t_slot - self._diag_last_slot) * 1000.0)
            self._diag_last_slot = _t_slot
            if self._diag_win is None:
                self._diag_win = _t_slot
        try:
            rect = self._sub[1] if self._sub else (0.0, 0.0, 1.0, 1.0)
            disp = (self._sub_depth_override if self._sub_depth_override is not None
                    else (self._sub[2] if self._sub else 0.0))
            if self._uniforms_take_disparity:
                try:
                    self._r.set_uniforms(self._stereo_mode, 1 if self._sub else 0,
                                         rect[0], rect[1], rect[2], rect[3],
                                         self._sdr_white, self._gamma, disp)
                except TypeError:
                    # renderer built before the subtitle_disparity uniform
                    self._uniforms_take_disparity = False
            if not self._uniforms_take_disparity:
                self._r.set_uniforms(self._stereo_mode, 1 if self._sub else 0,
                                     rect[0], rect[1], rect[2], rect[3],
                                     self._sdr_white, self._gamma)
            # C2: forward the display-aspect override each frame (next to the uniforms). An
            # old .pyd without set_source_aspect raises AttributeError/TypeError -> disable
            # it (logged once); geometry then derives the aspect from planes as before.
            if self._have_source_aspect:
                try:
                    self._r.set_source_aspect(float(self.source_aspect))
                except (AttributeError, TypeError):
                    self._have_source_aspect = False
                    if not self._source_aspect_unsupported_logged:
                        logger.warning("[NATIVE-WIDGET] set_source_aspect unavailable "
                                       "(old .pyd); deriving aspect from planes")
                        self._source_aspect_unsupported_logged = True
            # HDR10/PQ: forward the two color selectors each frame (same old-.pyd probe
            # idiom). An old .pyd without set_color_params raises AttributeError/TypeError
            # -> disable it (logged once); rendering then stays on the legacy 0/0 path.
            if self._have_color_params:
                try:
                    self._r.set_color_params(int(self.yuv_matrix_sel), int(self.transfer_sel))
                except (AttributeError, TypeError):
                    self._have_color_params = False
                    if not self._color_params_unsupported_logged:
                        logger.warning("[NATIVE-WIDGET] set_color_params unavailable "
                                       "(old .pyd); HDR/PQ color disabled (legacy render)")
                        self._color_params_unsupported_logged = True
            # The subtitle texture persists on the GPU (slot t0) — upload only
            # when the image actually changed, not on every frame.
            if self._sub is not None and self._sub_dirty:
                self._r.set_subtitle_rgba(self._sub[0])
                self._sub_dirty = False
            # Route by plane dtype: uint16 (10-bit HEVC) -> R16 path with plane_scale;
            # uint8 -> the existing R8 path. Same TypeError/AttributeError-probe idiom
            # as _uniforms_take_disparity: an old .pyd without set_yuv_frame16 drops
            # 10-bit frames (logged once) instead of crashing; 8-bit is unaffected.
            is16 = (yl is not None and getattr(yl, 'dtype', None) == np.uint16)
            _t_up0 = time.perf_counter() if self._diag else 0.0
            if is16:
                if not self._have_yuv16:
                    return
                try:
                    self._r.set_yuv_frame16(yl, ul, vl, yr, ur, vr, float(self.plane_scale))
                except (AttributeError, TypeError):
                    self._have_yuv16 = False
                    if not self._yuv16_unsupported_logged:
                        logger.warning("[NATIVE-WIDGET] set_yuv_frame16 unavailable "
                                       "(old .pyd); dropping 10-bit frames")
                        self._yuv16_unsupported_logged = True
                    return
            else:
                self._r.set_yuv_frame(yl, ul, vl, yr, ur, vr)
            if self._diag:
                _t_up1 = time.perf_counter()
                self._diag_upload.append((_t_up1 - _t_up0) * 1000.0)
            self._r.present()
            self.has_video = True
            if self._diag:
                self._diag_present.append((time.perf_counter() - _t_up1) * 1000.0)
                if self._diag_win is not None and (time.perf_counter() - self._diag_win) >= 5.0:
                    _ss, _su, _sp = (sorted(self._diag_slot), sorted(self._diag_upload),
                                     sorted(self._diag_present))
                    logger.info(
                        f"[HEVC-METER] widget slot ms p50={_pct(_ss, 0.5):.1f} "
                        f"p99={_pct(_ss, 0.99):.1f} max={(_ss[-1] if _ss else 0.0):.1f} | "
                        f"upload ms p50={_pct(_su, 0.5):.2f} p99={_pct(_su, 0.99):.2f} "
                        f"max={(_su[-1] if _su else 0.0):.2f} | present ms "
                        f"p50={_pct(_sp, 0.5):.2f} p99={_pct(_sp, 0.99):.2f} "
                        f"max={(_sp[-1] if _sp else 0.0):.2f} | n={len(_ss)}")
                    self._diag_slot, self._diag_upload, self._diag_present = [], [], []
                    self._diag_win = time.perf_counter()
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
        # C2: reset the display-aspect override so the next source derives aspect from
        # planes again until the player re-sets it.
        self.source_aspect = 0.0
        if self._r and self._r is not False:
            try:
                self._r.clear_frame()
                self._r.present()
            except Exception:
                pass

    def set_subtitle(self, rgba_array, x, y, w, h, video_width=1920, video_height=1080,
                     disparity=0.0):
        """disparity: stereoscopic overlay depth — horizontal disparity normalized
        to eye width; > 0 floats the subtitle in FRONT of the screen (each eye view
        is shifted by half, in opposite directions). 0.0 = screen depth."""
        try:
            vw = float(video_width) or 1920.0
            vh = float(video_height) or 1080.0
            nx, ny = x / vw, y / vh
            nw, nh = w / vw, h / vh
            self._sub = (np.ascontiguousarray(rgba_array, dtype=np.uint8),
                         (nx, ny, nw, nh), float(disparity))
            self._sub_dirty = True
        except Exception as e:
            logger.warning(f"[NATIVE-WIDGET] set_subtitle failed: {e}")
            self._sub = None

    def clear_subtitle(self):
        self._sub = None

    def set_subtitle_depth(self, disparity):
        """Dynamic depth override for the overlay (BD3D per-GOP offset metadata).

        Applies on top of whatever subtitle is displayed, without re-uploading
        the bitmap. Pass None to clear (per-cue authored disparity applies)."""
        self._sub_depth_override = None if disparity is None else float(disparity)

    # --- deprecated / no-ops the player may still call ------------------------
    def set_frame_fast(self, *args, **kwargs):
        pass  # legacy packed-array path; unused in the YUV pipeline

    def refresh_sdr_white_level(self):
        pass  # native picks SDR/HDR at init from the white level

    def shutdown(self):
        # Release the D3D11 renderer but stay RE-INITIALIZABLE. The framepack window is a
        # session singleton the player creates once and reuses (never recreated / never
        # nulled). Closing the detached window (X / Alt-F4 / app-exit closeEvent) fires
        # Framepacking3DWindow.closeEvent -> shutdown(); a later MultiView relaunch reuses
        # THIS same widget. Reset to None (the "not yet initialized" state) instead of the
        # sticky False "permanently unavailable" sentinel, so _ensure() lazily rebuilds the
        # renderer on the next delivered frame — and rebinds the swapchain to the CURRENT
        # winId, which Qt recreates when the window is closed and reshown. Leaving it False
        # made _ensure() early-return False forever -> every frame dropped -> a completely
        # WHITE native surface on replay. Idempotent: a 2nd shutdown finds _r None (or False)
        # and no-ops; on app exit the decoder is stopped before this runs, so no stray frame
        # re-triggers _ensure().
        r = self._r
        self._r = None
        if r and r is not False:
            try:
                r.shutdown()
            except Exception:
                pass
