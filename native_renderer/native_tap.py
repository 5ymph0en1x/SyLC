"""Native-renderer A/B tap (S4) — diagnostic, env-gated, zero impact when off.

When SYLC_NATIVE_TAP=1, the player mirrors every decoded stereo frame it sends to
the Qt widgets into a SEPARATE plain-QWidget window driven by the native C++
NativeRenderer. This lets you watch real 3D content rendered by BOTH pipelines side
by side, for live parity validation on actual MVC discs — without modifying the
shipping Qt render path.

Runs entirely on the GUI thread (the frame slot's thread), reusing the player's
existing decode-thread pacing + QueuedConnection serialization — so it inherits the
same safety the current path has. This is the safe partial-win integration
(eliminates the tobytes/QByteArray upload copy); the full decode-thread push that
also removes the edge264->numpy copy is the later cut-over step.

Env:
    SYLC_NATIVE_TAP=1     enable the tap window
    SYLC_SDR_WHITE=<f>    scRGB white multiplier (default 2.5 ~= 200 nits)
"""
import logging

logger = logging.getLogger("SyLC.NativeTap")


class NativeRendererTap:
    def __init__(self, sdr_white: float = 2.5):
        self._r = None          # NativeRenderer, or False if init failed
        self._win = None
        self._sdr = sdr_white   # fallback only
        self._sdr_override = None  # set from SYLC_SDR_WHITE if present (manual A/B)
        self._gamma = 2.2       # EOTF exponent (SYLC_NATIVE_GAMMA); <=0 disables
        self._fail_logged = False

    def _ensure(self, sdr_white=None) -> bool:
        if self._r is not None:
            return self._r is not False
        try:
            import os
            import mvc_demuxer_cpp as m
            if not getattr(m, "NATIVE_RENDERER_AVAILABLE", False) or not hasattr(m, "NativeRenderer"):
                logger.warning("[NATIVE-TAP] module built without NativeRenderer")
                self._r = False
                return False

            from PySide6.QtWidgets import QWidget
            from PySide6.QtCore import Qt

            if "SYLC_SDR_WHITE" in os.environ:
                try:
                    self._sdr_override = float(os.environ["SYLC_SDR_WHITE"])
                    logger.info(f"[NATIVE-TAP] SDR white override = {self._sdr_override}")
                except ValueError:
                    self._sdr_override = None
            win = QWidget()
            win.setWindowTitle("SyLC — NATIVE RENDERER (A/B tap)")
            win.resize(960, 1102)  # ~1920x2205 framepack aspect, half scale
            win.setAttribute(Qt.WA_NativeWindow, True)
            win.setAttribute(Qt.WA_PaintOnScreen, True)
            win.setAttribute(Qt.WA_NoSystemBackground, True)
            win.show()

            # Decide SDR vs HDR from the display's SDR white level (the same value
            # the Qt widget queries): > ~1.01 means Windows HDR is ON. On an SDR
            # display, use an 8-bit gamma swapchain with NO EOTF — the shader's
            # gamma-domain output displays correctly as-is (matches Qt). On HDR,
            # use FP16 scRGB-linear + an EOTF to linearize.
            env_hdr = os.environ.get("SYLC_NATIVE_HDR")
            if env_hdr is not None:
                hdr = env_hdr == "1"
            else:
                hdr = bool(sdr_white) and float(sdr_white) > 1.01
            default_gamma = 2.4 if hdr else 0.0  # linearize only in HDR
            try:
                self._gamma = float(os.environ.get("SYLC_NATIVE_GAMMA", str(default_gamma)))
            except ValueError:
                self._gamma = default_gamma

            r = m.NativeRenderer()
            sz = win.size()
            logger.info(f"[NATIVE-TAP] hdr={hdr} output_gamma={self._gamma} (sdr_white={sdr_white})")
            if not r.initialize(int(win.winId()), sz.width(), sz.height(), hdr):
                logger.warning(f"[NATIVE-TAP] initialize failed: {r.last_error()}")
                self._r = False
                return False
            logger.info(f"[NATIVE-TAP] {r.backend_info()}")
            self._win = win
            self._r = r
            return True
        except Exception as e:
            logger.warning(f"[NATIVE-TAP] disabled: {e}")
            self._r = False
            return False

    def push(self, left_planes, right_planes, stereo_mode: int, sdr_white=None):
        """Upload + present one frame. left/right_planes are (Y, U, V) numpy tuples.

        sdr_white: the Qt widget's actual _sdr_white_level, so the native window
        matches its brightness/saturation exactly. SYLC_SDR_WHITE overrides it.
        """
        if not self._ensure(sdr_white):
            return
        try:
            yl, ul, vl = left_planes
            yr, ur, vr = right_planes
            if self._sdr_override is not None:
                sw = self._sdr_override
            elif sdr_white:
                sw = float(sdr_white)
            else:
                sw = self._sdr
            self._r.set_uniforms(int(stereo_mode), 0, 0.0, 0.0, 1.0, 1.0, sw, self._gamma)
            h, w = yl.shape[:2]
            if not self._r.set_yuv_frame(yl, ul, vl, yr, ur, vr, int(w), int(h)):
                if not self._fail_logged:
                    logger.warning(f"[NATIVE-TAP] set_yuv_frame: {self._r.last_error()}")
                    self._fail_logged = True
                return
            self._r.present()
        except Exception as e:
            if not self._fail_logged:
                logger.warning(f"[NATIVE-TAP] push failed: {e}")
                self._fail_logged = True

    def close(self):
        try:
            if self._r and self._r is not False:
                self._r.shutdown()
        finally:
            self._r = False
            if self._win is not None:
                try:
                    self._win.close()
                except Exception:
                    pass
                self._win = None
