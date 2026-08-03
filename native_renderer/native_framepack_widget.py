"""NativeFramepackWidget — the player's sole video display widget, backed by the
native C++ D3D11 renderer.

The player injects this for both the embedded 2D view and the detached framepack
window, so all video output is produced by the native renderer (no tobytes/QByteArray
upload copy, no Qt RHI overhead). It replaced the former Qt RHI widget
(FramepackingDisplayWidgetD3D11), now removed.

It implements the subset of the widget contract the player actually calls on the
display widget (verified by grep): set_frame_yuv_views, set_stereo_mode,
pause_rendering, resume_rendering, clear_textures, set_subtitle, clear_subtitle,
set_hud, clear_hud, plus the deprecated set_frame_fast (no-op) and
refresh_sdr_white_level.

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

# 'glasses' deliberately shares SBS's value: it IS the side-by-side arrangement.
# The distinct key exists so the window can choose a 3840x1080 geometry for it
# (see Framepacking3DWindow.apply_output_geometry), not so the shader differs.
_MODE = {'2d': 0, 'framepack': 1, 'sbs': 2, 'tab': 3, 'glasses': 2}


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
        # Dual-output policy: the detached framepack renderer owns vsync; the
        # simultaneous main-window preview uses interval 0 so its second
        # swapchain never stalls the GUI frame-delivery slot.
        self.present_vsync = True
        self._present_interval_supported = True
        self._present_interval_warned = False
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
        # Playback HUD is a second, independent RGBA layer. It is authored once
        # in one-eye coordinates and duplicated by the native shader for every
        # packed stereo layout, while subtitles retain their own depth/texture.
        self._hud = None               # (rgba, rect, disparity, opacity) or None
        self._hud_dirty = False
        self._hud_state_dirty = False
        self._have_hud_api = True
        self._hud_api_warned = False
        self._last_eye_size = None      # decoded eye plane dimensions for hit-testing/aspect
        self.eye_view = None      # 'left'/'right' in Dual Projector windows
        self._uniforms_take_disparity = True   # probed once; False on an older renderer build
        self._fail_logged = False
        self._renderer_failures = 0
        self._next_renderer_retry = 0.0
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
        # Exact media PTS of the frame currently being delivered. The player
        # writes it beside present_vsync immediately before this slot. It is
        # forwarded independently from GUI arrival/compute time.
        self.video_time_ms = -1.0
        self._have_video_time = True

        # 2D->3D AI synthesis (spec 2026-07-28): pushed to the renderer on change only.
        self.synth3d_enabled = False
        self.synth3d_strength = 1.5      # % of image width
        self.synth3d_convergence = 0.5
        self.synth3d_depth_view = False
        self.synth3d_diagnostics = False
        self.synth3d_model_path = ""
        self.synth3d_ort_dir = ""
        # Square inference grid the model was exported for, set by the host next
        # to synth3d_model_path (round 4 depth presets: 756 or 518). 0 = the host
        # named none, so the renderer's own default applies -- the binding
        # rejects a non-positive side.
        self.synth3d_side = 0
        # Optional fixed rectangular export selected after the shared worker has
        # observed stable encoded mattes. Zero keeps the square preset above.
        self.synth3d_grid_width = 0
        self.synth3d_grid_height = 0
        self.synth3d_crop_top = 0.0
        self.synth3d_crop_bottom = 0.0
        self.synth3d_auto_convergence = False
        self.synth3d_temporal_fill = False
        self._have_synth3d = True        # old-.pyd probe, same idiom as _have_yuv16
        self._synth3d_takes_side = True  # False after an old .pyd rejects side=
        self._synth3d_takes_rect = True  # False after a pre-aspect .pyd rejects ROI=
        self._synth3d_takes_auto_conv = True  # False after a pre-round-5 .pyd rejects it
        self._synth3d_takes_temporal_fill = True  # idem, round 5a
        self._synth3d_pushed = None
        # Optional MatAnyone 2 alpha. The player shares one immutable result
        # across all presentation surfaces; each D3D11 renderer uploads it only
        # when the worker's sequence changes.
        self._synth3d_human_matte = None
        self._synth3d_human_matte_key = None
        self._synth3d_human_matte_uploaded = object()
        self._have_synth3d_human_matte = True

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
                if not self._r.resize(pw, ph):
                    self._invalidate_renderer("resize")
            except Exception as exc:
                self._invalidate_renderer("resize exception", exc)

    # --- native renderer lifecycle -------------------------------------------
    def _invalidate_renderer(self, operation, exception=None, renderer=None):
        """Release a failed D3D11 instance and retry later with backoff.

        DXGI device removal/reset, display hot-plug and suspend/resume all surface
        as false/exception from upload, resize or Present. Retaining that renderer
        leaves a permanently black/stale surface, so rebuild the whole device and
        swapchain on a subsequent frame.
        """
        r = renderer if renderer is not None else self._r
        self._r = None
        self.has_video = False
        if self._sub is not None:
            self._sub_dirty = True
        if self._hud is not None:
            self._hud_dirty = True
        self._hud_state_dirty = True
        # The rebuilt instance has never received the synth3d state -- force a
        # re-push on the next delivered frame instead of trusting the stale cache
        # from the old (now-discarded) renderer object. _have_synth3d (the old-.pyd
        # verdict) stays as-is: it describes this .pyd build, not this instance.
        self._synth3d_pushed = None
        self._synth3d_human_matte_uploaded = object()
        error = str(exception) if exception is not None else ""
        if not error and r and r is not False:
            try:
                error = str(r.last_error())
            except Exception:
                error = ""
        if r and r is not False:
            try:
                r.shutdown()
            except Exception:
                pass
        self._renderer_failures += 1
        delay = min(5.0, 0.25 * (2 ** min(self._renderer_failures - 1, 5)))
        self._next_renderer_retry = time.monotonic() + delay
        logger.warning(
            f"[NATIVE-WIDGET] {operation} failed"
            f"{': ' + error if error else ''}; rebuilding D3D11 in {delay:.2f}s"
        )
        return False

    def _ensure(self):
        if self._r is not None:
            return self._r is not False
        if time.monotonic() < self._next_renderer_retry:
            return False
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
                return self._invalidate_renderer("initialize", renderer=r)
            logger.info(f"[NATIVE-WIDGET] {r.backend_info()} | hdr={self._hdr} gamma={self._gamma} sdr_white={self._sdr_white}")
            self._r = r
            # A rebuilt D3D instance starts with transparent/default HUD state.
            # Re-push both the texture and its rect on the next frame.
            if self._hud is not None:
                self._hud_dirty = True
            self._hud_state_dirty = True
            self._renderer_failures = 0
            self._next_renderer_retry = 0.0
            self._fail_logged = False
            return True
        except Exception as e:
            return self._invalidate_renderer("initialize exception", e)

    # --- contract: frame delivery --------------------------------------------
    def set_frame_yuv_views(self, y_l_or_tuple, u_l_or_right=None, v_l=None,
                            y_r=None, u_r=None, v_r=None):
        if self._rendering_paused:
            return False
        if isinstance(y_l_or_tuple, tuple):
            yl, ul, vl = y_l_or_tuple
            if isinstance(u_l_or_right, tuple):
                yr, ur, vr = u_l_or_right
            else:
                yr = ur = vr = None
        else:
            yl, ul, vl = y_l_or_tuple, u_l_or_right, v_l
            yr, ur, vr = y_r, u_r, v_r

        try:
            if yl is not None and getattr(yl, 'ndim', 0) >= 2:
                self._last_eye_size = (int(yl.shape[1]), int(yl.shape[0]))
        except Exception:
            pass

        # A 2D presentation samples only t1..t3. Do not upload the unused right
        # eye to an embedded preview: this halves its CPU->GPU traffic when the
        # framepack window is being presented simultaneously. Same reasoning when
        # synth3d is active: the renderer synthesizes the right eye itself, so a
        # duplicated right-eye upload from the source would be wasted bandwidth.
        if self._stereo_mode == 0 or (self.synth3d_enabled and self._have_synth3d):
            yr = ur = vr = None

        if not self._ensure():
            return False
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
            # A Dual Projector window renders ONE eye through the '2d' path,
            # where the shader forces eyeSign = 0 and therefore drops the
            # subtitle disparity entirely. Shifting the rect here restores the
            # authored depth without a shader change: each eye moves by half
            # the disparity, in opposite directions -- the same split the
            # framepack shader performs internally.
            if self.eye_view in ('left', 'right') and disp:
                half = 0.5 * float(disp)
                rect = (rect[0] + (half if self.eye_view == 'left' else -half),
                        rect[1], rect[2], rect[3])
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
            # 2D->3D AI synthesis: push the state to the renderer only when it changed
            # (the call locks the renderer mutex and (re)configures the warp pipeline;
            # a per-frame call would churn it needlessly). Same old-.pyd probe idiom as
            # set_source_aspect/set_color_params above.
            if self._have_synth3d:
                state = (bool(self.synth3d_enabled), float(self.synth3d_strength),
                         float(self.synth3d_convergence), bool(self.synth3d_depth_view),
                         bool(self.synth3d_diagnostics),
                         str(self.synth3d_model_path), str(self.synth3d_ort_dir),
                         int(self.synth3d_side),
                         int(self.synth3d_grid_width),
                         int(self.synth3d_grid_height),
                         float(self.synth3d_crop_top),
                         float(self.synth3d_crop_bottom),
                         bool(self.synth3d_auto_convergence),
                         bool(self.synth3d_temporal_fill))
                if state != self._synth3d_pushed:
                    kwargs = dict(model_path=state[5], ort_dir=state[6],
                                  diagnostics=state[4])
                    # Per-shot auto-convergence (round 5). A pre-round-5 .pyd
                    # has no such parameter: retry without it (manual
                    # convergence continues) -- same probe idiom as side=.
                    if self._synth3d_takes_auto_conv and state[12]:
                        kwargs['auto_convergence'] = True
                    # Temporal disocclusion plate (round 5a), same idiom.
                    if self._synth3d_takes_temporal_fill and state[13]:
                        kwargs['temporal_fill'] = True
                    # The grid is part of the shared depth service's key, so a
                    # preset switch must reach the renderer with it. A .pyd from
                    # before round 4 has no side= parameter: retry without it
                    # (the renderer then uses its own 756 default) instead of
                    # losing synthesis entirely -- same probe idiom as
                    # _uniforms_take_disparity.
                    if self._synth3d_takes_side and state[7] > 0:
                        kwargs['side'] = state[7]
                    if (self._synth3d_takes_rect
                            and state[8] > 0 and state[9] > 0):
                        kwargs.update(grid_width=state[8],
                                      grid_height=state[9],
                                      crop_top=state[10],
                                      crop_bottom=state[11])
                    try:
                        ok = self._r.set_synth3d(*state[:4], **kwargs)
                    except (AttributeError, TypeError) as exc:
                        if (isinstance(exc, TypeError)
                                and 'temporal_fill' in kwargs):
                            # Only temporal_fill is unsupported: retry without
                            # it on the next frame (stretch fill continues).
                            self._synth3d_takes_temporal_fill = False
                            logger.warning(
                                "[NATIVE-WIDGET] set_synth3d has no "
                                "temporal_fill= (old .pyd); stretch "
                                "disocclusion fill only")
                        elif (isinstance(exc, TypeError)
                                and 'auto_convergence' in kwargs):
                            # Only auto_convergence is unsupported: leave
                            # _synth3d_pushed alone so the next frame retries
                            # without it (manual convergence still applies).
                            self._synth3d_takes_auto_conv = False
                            logger.warning(
                                "[NATIVE-WIDGET] set_synth3d has no "
                                "auto_convergence= (old .pyd); manual "
                                "convergence only")
                        elif isinstance(exc, TypeError) and 'grid_width' in kwargs:
                            # A renderer old enough not to understand rectangular
                            # grids cannot safely open the rectangular ONNX path.
                            # Do not cache the attempted state; the host will keep
                            # square operation on builds whose status has no crop.
                            self._synth3d_takes_rect = False
                            logger.warning("[NATIVE-WIDGET] set_synth3d has no "
                                           "rectangular-grid/ROI arguments "
                                           "(old .pyd); adaptive aspect disabled")
                        elif isinstance(exc, TypeError) and 'side' in kwargs:
                            # Only side= is unsupported: leave _synth3d_pushed
                            # alone so the next frame retries without it.
                            self._synth3d_takes_side = False
                            logger.warning("[NATIVE-WIDGET] set_synth3d has no "
                                           "side= (old .pyd); depth presets fall "
                                           "back to the renderer's default grid")
                        else:
                            self._have_synth3d = False
                            logger.warning("[NATIVE-WIDGET] set_synth3d unavailable "
                                           "(old .pyd); 2D->3D synthesis disabled")
                    else:
                        # A False return means the renderer refused the push (e.g. before
                        # initialize) -- do not cache it as pushed, so it retries next frame.
                        if ok:
                            self._synth3d_pushed = state
            # MatAnyone 2 refines ownership only; depth remains the source of
            # geometry. A stale/absent result explicitly disarms t4 so the
            # shader falls back to its normal robust warp without retaining a
            # contour from an earlier scene or seek.
            if self._have_synth3d_human_matte:
                matte_key = self._synth3d_human_matte_key
                if matte_key != self._synth3d_human_matte_uploaded:
                    frame = self._synth3d_human_matte
                    try:
                        setter = getattr(self._r, 'synth3d_set_human_matte', None)
                        if setter is None:
                            setter = self._r.synth3d_set_test_matte
                        if frame is None:
                            ok = setter(None)
                        else:
                            ok = setter(frame.alpha, 'contour')
                    except (AttributeError, TypeError):
                        self._have_synth3d_human_matte = False
                        logger.warning("[NATIVE-WIDGET] human-matte API unavailable "
                                       "(old .pyd); MatAnyone 2 guidance disabled")
                    else:
                        # The production binding returns bool. Pre-production
                        # builds exposed the same operation as a void method;
                        # ``None`` therefore also means the upload completed.
                        if ok is not False:
                            self._synth3d_human_matte_uploaded = matte_key
            # The subtitle texture persists on the GPU (slot t0) — upload only
            # when the image actually changed, not on every frame.
            if self._sub is not None and self._sub_dirty:
                if not self._r.set_subtitle_rgba(self._sub[0]):
                    return self._invalidate_renderer("subtitle upload")
                self._sub_dirty = False
            if not self._push_hud_to_renderer():
                return self._invalidate_renderer("HUD upload/state")
            if self._have_video_time:
                try:
                    self._r.set_video_time_ms(float(self.video_time_ms))
                except (AttributeError, TypeError):
                    # Compatibility with a pyd predating the timed frame path.
                    self._have_video_time = False
            # Route by plane dtype: uint16 (10-bit HEVC) -> R16 path with plane_scale;
            # uint8 -> the existing R8 path. Same TypeError/AttributeError-probe idiom
            # as _uniforms_take_disparity: an old .pyd without set_yuv_frame16 drops
            # 10-bit frames (logged once) instead of crashing; 8-bit is unaffected.
            is16 = (yl is not None and getattr(yl, 'dtype', None) == np.uint16)
            _t_up0 = time.perf_counter() if self._diag else 0.0
            if is16:
                if not self._have_yuv16:
                    return False
                try:
                    uploaded = self._r.set_yuv_frame16(
                        yl, ul, vl, yr, ur, vr, float(self.plane_scale))
                except (AttributeError, TypeError):
                    self._have_yuv16 = False
                    if not self._yuv16_unsupported_logged:
                        logger.warning("[NATIVE-WIDGET] set_yuv_frame16 unavailable "
                                       "(old .pyd); dropping 10-bit frames")
                        self._yuv16_unsupported_logged = True
                    return False
            else:
                uploaded = self._r.set_yuv_frame(yl, ul, vl, yr, ur, vr)
            if not uploaded:
                return self._invalidate_renderer(
                    "16-bit YUV upload" if is16 else "YUV upload")
            if self._diag:
                _t_up1 = time.perf_counter()
                self._diag_upload.append((_t_up1 - _t_up0) * 1000.0)
            interval = 1 if self.present_vsync else 0
            if self._present_interval_supported:
                try:
                    presented = self._r.present(interval)
                except TypeError:
                    # Compatibility with a pyd built before the optional
                    # sync_interval argument. Correctness is preserved, though
                    # the secondary preview remains blocking until rebuilt.
                    self._present_interval_supported = False
                    presented = self._r.present()
                    if not self._present_interval_warned:
                        logger.warning("[NATIVE-WIDGET] renderer lacks present(sync_interval); "
                                       "dual-output preview uses compatibility vsync")
                        self._present_interval_warned = True
            else:
                presented = self._r.present()
            if not presented:
                return self._invalidate_renderer("Present")
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
            return True
        except Exception as e:
            return self._invalidate_renderer("frame delivery exception", e)

    # --- contract: control ----------------------------------------------------
    def set_lookahead_advisory(self, cut_in_ms, storm_in_ms):
        """Two-filter look-ahead advisory → renderer → depth service.
        None means 'no upcoming event'. False when the renderer or its
        synth3d session is not up (harmless: the pump retries next tick)."""
        r = self._r
        if r is None or not hasattr(r, 'synth3d_set_lookahead'):
            return False
        # None -> -1e9 (the C++ NONE sentinel). NEVER -1.0: small negative
        # delays are real hold-window data ("the cut landed a frame ago").
        _NONE = -1.0e9
        try:
            return bool(r.synth3d_set_lookahead(
                _NONE if cut_in_ms is None else float(cut_in_ms),
                _NONE if storm_in_ms is None else float(storm_in_ms)))
        except Exception:
            return False

    def set_stereo_mode(self, mode_str):
        self._stereo_mode = _MODE.get(str(mode_str).lower(), 1)
        self.current_stereo_mode = self._stereo_mode

    def _push_hud_to_renderer(self):
        """Push changed HUD state/texture to the live renderer.

        The compatibility probe keeps a pre-HUD .pyd playable: video and
        subtitles continue normally, while the host can fall back to the Qt bar.
        """
        if not self._have_hud_api or not self._r or self._r is False:
            return True
        try:
            if self._hud_state_dirty:
                if self._hud is None:
                    self._r.set_hud_state(False)
                else:
                    _rgba, rect, disparity, opacity = self._hud
                    self._r.set_hud_state(True, rect[0], rect[1], rect[2], rect[3],
                                          disparity, opacity)
                self._hud_state_dirty = False
            if self._hud is not None and self._hud_dirty:
                if not self._r.set_hud_rgba(self._hud[0]):
                    return False
                self._hud_dirty = False
            return True
        except (AttributeError, TypeError):
            self._have_hud_api = False
            self._hud_state_dirty = False
            self._hud_dirty = False
            if not self._hud_api_warned:
                logger.warning("[NATIVE-WIDGET] stereo HUD API unavailable (old .pyd)")
                self._hud_api_warned = True
            return True
        except Exception as exc:
            logger.warning(f"[NATIVE-WIDGET] stereo HUD push failed: {exc}")
            return False

    def set_hud(self, rgba_array, x, y, w, h, disparity=0.003, opacity=1.0):
        """Stage an HxWx4 HUD in normalized one-eye video coordinates."""
        try:
            rect = (float(x), float(y), float(w), float(h))
            # This texture is normally uploaded on the next video-frame slot,
            # after the caller's capture buffer may have been destroyed.  A
            # forced copy is therefore part of this API's ownership contract;
            # np.ascontiguousarray() is insufficient when the input is already
            # contiguous because it aliases the caller's memory.
            owned_rgba = np.array(rgba_array, dtype=np.uint8, order="C", copy=True)
            if owned_rgba.ndim != 3 or owned_rgba.shape[2] != 4:
                raise ValueError("HUD texture must have shape HxWx4")
            self._hud = (owned_rgba, rect,
                         float(disparity), max(0.0, min(1.0, float(opacity))))
            self._hud_dirty = True
            self._hud_state_dirty = True
        except Exception as exc:
            logger.warning(f"[NATIVE-WIDGET] set_hud failed: {exc}")
            self._hud = None
            self._hud_state_dirty = True

    def clear_hud(self):
        if self._hud is None and not self._hud_state_dirty:
            return
        self._hud = None
        self._hud_dirty = False
        self._hud_state_dirty = True

    def refresh_hud(self):
        """Upload/present a changed HUD while media is paused.

        Video playback already pushes it from set_frame_yuv_views; this path is
        used only when no new frame is expected, and presents with interval 0.
        """
        if not self._r or self._r is False or self._rendering_paused:
            return False
        if not self._push_hud_to_renderer():
            return self._invalidate_renderer("HUD refresh")
        if not self.has_video:
            return True
        try:
            if not self._r.present(0):
                return self._invalidate_renderer("HUD refresh Present")
            return True
        except Exception as exc:
            return self._invalidate_renderer("HUD refresh exception", exc)

    def stereo_hud_video_uv(self, x, y):
        """Map one Qt-local output point to normalized one-eye video UV.

        This mirrors native_renderer.cpp's aspect viewport and the HLSL's
        SBS/TAB/FramePack split, including the 45-pixel HDMI gap and per-eye
        letterbox. None means the point is in a black bar/gap.
        """
        ow, oh = float(max(1, self.width())), float(max(1, self.height()))
        px, py = float(x), float(y)
        if px < 0.0 or py < 0.0 or px >= ow or py >= oh:
            return None

        eye_aspect = 0.0
        try:
            eye_aspect = float(self.source_aspect)
        except Exception:
            pass
        if eye_aspect <= 0.0 and self._last_eye_size and self._last_eye_size[1]:
            eye_aspect = self._last_eye_size[0] / self._last_eye_size[1]
        if eye_aspect <= 0.0:
            eye_aspect = 16.0 / 9.0

        target_aspect = 1920.0 / 2205.0 if self._stereo_mode == 1 else eye_aspect
        out_aspect = ow / oh
        vx = vy = 0.0
        vw, vh = ow, oh
        if out_aspect > target_aspect:
            vw = oh * target_aspect
            vx = (ow - vw) * 0.5
        else:
            vh = ow / target_aspect
            vy = (oh - vh) * 0.5
        if px < vx or px >= vx + vw or py < vy or py >= vy + vh:
            return None
        sx, sy = (px - vx) / vw, (py - vy) / vh

        if self._stereo_mode == 2:       # SBS
            return ((sx * 2.0) if sx < 0.5 else ((sx - 0.5) * 2.0), sy)
        if self._stereo_mode == 3:       # TAB
            return (sx, (sy * 2.0) if sy < 0.5 else ((sy - 0.5) * 2.0))
        if self._stereo_mode != 1:       # 2D / one-eye projector
            return (sx, sy)

        top = 1080.0 / 2205.0
        gap_end = 1125.0 / 2205.0
        if sy < top:
            slot_y = sy / top
        elif sy > gap_end:
            slot_y = (sy - gap_end) / top
        else:
            return None
        slot_aspect = 1920.0 / 1080.0
        if eye_aspect >= slot_aspect:
            hfill, vfill = 1.0, slot_aspect / eye_aspect
        else:
            hfill, vfill = eye_aspect / slot_aspect, 1.0
        hbar, vbar = (1.0 - hfill) * 0.5, (1.0 - vfill) * 0.5
        if sx < hbar or sx > 1.0 - hbar or slot_y < vbar or slot_y > 1.0 - vbar:
            return None
        return ((sx - hbar) / hfill, (slot_y - vbar) / vfill)

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

    def set_synth3d_human_matte(self, frame):
        """Stage one shared MatAnyone result for upload on the next frame.

        ``frame`` is intentionally duck-typed (sequence, generation, alpha) so
        this display module does not import or own the worker service.
        """
        if frame is None:
            key = None
        else:
            key = (int(frame.generation), int(frame.sequence))
        if key == self._synth3d_human_matte_key:
            return
        self._synth3d_human_matte = frame
        self._synth3d_human_matte_key = key

    def clear_textures(self):
        self.has_video = False
        self.set_synth3d_human_matte(None)
        # C2: reset the display-aspect override so the next source derives aspect from
        # planes again until the player re-sets it.
        self.source_aspect = 0.0
        if self._r and self._r is not False:
            try:
                self._r.clear_frame()
                if not self._r.present():
                    self._invalidate_renderer("clear Present")
            except Exception as exc:
                self._invalidate_renderer("clear exception", exc)

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
        self._renderer_failures = 0
        self._next_renderer_retry = 0.0
        if r and r is not False:
            try:
                r.shutdown()
            except Exception:
                pass
