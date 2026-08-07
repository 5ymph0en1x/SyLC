# -*- coding: utf-8 -*-
"""PlayerWindow coordination for SyLC's real-time 2D-to-3D pipeline."""

import glob
import logging
import math
import os
import sys
import time

from PySide6.QtCore import QSettings, QTimer

from sylc.runtime_paths import RUNTIME_DIR

from sylc.synth3d_aspect import MIN_NATIVE_WIDE_RATIO, select_installed_aspect_model
from sylc.synth3d_policy import (
    SYNTH3D_ADVISORY_PTS_CLOCK, SYNTH3D_DEPTH_PRESETS,
    SYNTH3D_DEPTH_PRESET_KEY, SYNTH3D_SETTINGS_APP, SYNTH3D_SETTINGS_ORG,
    _SOURCE_ROOT, _synth3d_models_dirs, _synth3d_seek_keeps_aspect,
    sylc_models_download_dir, synth3d_depth_preset_available,
    synth3d_depth_preset_entry, synth3d_depth_preset_stored,
    synth3d_find_model, synth3d_marker_attests,
)


logger = logging.getLogger(__name__)
NATIVE_RENDER_AVAILABLE = False


def configure_synth3d_support(native_render_available):
    """Mirror the optional native-renderer capability detected at startup."""
    global NATIVE_RENDER_AVAILABLE
    NATIVE_RENDER_AVAILABLE = bool(native_render_available)


class Synth3DCoordinationMixin:
    # Rates real media is actually authored at. A decoder-measured interval is
    # snapped to whichever of these it is within 2 % of.
    _MEDIA_STANDARD_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0,
                           59.94, 60.0)

    _SYNTH3D_PRESETS = {
        # strength is % of image width; convergence is normalized nearness.
        # Comfort keeps most content behind the screen plane, Cinema balances
        # positive/negative parallax, Immersion permits more foreground pop.
        'comfort': (0.8, 0.62),
        'cinema': (1.4, 0.52),
        'immersion': (2.2, 0.42),
    }

    def _synth3d_lookahead_thread(self):
        """Return the decoder that owns decoded-future shot information.

        H.264/MVC and native HEVC are mutually exclusive playback paths. HEVC
        must win while active: an old mvc_decoder_thread can survive briefly
        during hand-off and carries events from the wrong media timeline.
        """
        if getattr(self, '_hevc_mode_active', False):
            th = getattr(self, 'hevc_thread', None)
            if (th is not None and hasattr(th, 'set_lookahead_enabled')
                    and hasattr(th, 'lookahead_scout')):
                return th
        th = getattr(self, 'mvc_decoder_thread', None)
        if (th is not None and hasattr(th, 'set_lookahead_enabled')
                and hasattr(th, 'lookahead_scout')):
            return th
        return None

    def _forward_lookahead_cut_boundary(self, cut_pts_ms):
        """Decoded T0 -> native shot boundary before T0 presentation.

        HEVC emits directly in presentation order. H.264 emits after its
        display-order PTS repair while T0 is still in the decoded-future queue.
        Every renderer therefore records the absolute boundary before T0.
        """
        if not self._native_signal_is_current():
            return
        if not getattr(self, '_synth3d_active', False):
            return
        try:
            cut_pts_ms = float(cut_pts_ms)
            if cut_pts_ms < 0.0:
                return
            self._synth3d_note_cut_boundary(cut_pts_ms)
            for w in self._display_widgets():
                fn = getattr(w, 'set_lookahead_advisory', None)
                if fn is None:
                    continue
                shown = getattr(w, 'video_time_ms', None)
                cut_in = (cut_pts_ms - float(shown)
                          if isinstance(shown, (int, float)) and shown >= 0.0
                          else 0.0)
                try:
                    fn(cut_in, None, cut_pts_ms)
                except TypeError:
                    fn(cut_in, None)
        except Exception:
            logger.exception("[HEVC-LOOKAHEAD] cut forwarding failed")

    def _pump_lookahead_advisory(self):
        """Two-filter look-ahead pump (spec 2026-08-03): arm the decoder-side
        scout while synth3d runs, read its future events against the PRESENTED
        position and forward the delays to every display widget's renderer.
        Self-arming and self-disarming; every step best-effort."""
        owner = getattr(self, '_synth3d_lookahead_thread', None)
        if owner is not None:
            th = owner()
        elif getattr(self, '_hevc_mode_active', False):
            th = getattr(self, 'hevc_thread', None)
        else:
            # Compatibility for lightweight test/player hosts that predate the
            # owner helper but still expose the historical MVC attribute.
            th = getattr(self, 'mvc_decoder_thread', None)
        if th is None:
            return
        try:
            active = bool(getattr(self, '_synth3d_active', False))
            th.set_lookahead_enabled(active)
            if not active:
                return
            scout = th.lookahead_scout()
            if scout is None:
                return
            # Date the advisory against the pts of the frame actually ON SCREEN,
            # not the interpolated UI clock. Measured 2026-08-04: the UI clock
            # runs a steady ~110 ms BEHIND the presented pts (five consecutive
            # windows: -110.7, -108.0, -110.5, -112.5, -105.3 ms). The renderer
            # holds and ramps against the exact presented pts, so that offset
            # shifted the whole cut window: at the new shot's first frame the
            # advisory still read "cut in +110 ms", the flat hold never engaged
            # (it needs <= 0), and t2 = 110/300 left ~60% of the disparity
            # budget applied to the OLD shot's map -- the previous shot's
            # contours warped into the new one. The pump quantization fix
            # (SYLC_LOOKAHEAD_DECAY) corrected jitter, never this offset.
            # The widget carries the exact pts it last presented; fall back to
            # the UI clock only before the first frame is delivered.
            # SYLC_ADVISORY_PTS_CLOCK=0 restores the UI-clock dating (the
            # measured -110 ms offset), for A/B against this fix.
            pump_ms = None
            if SYNTH3D_ADVISORY_PTS_CLOCK:
                for _w in self._display_widgets():
                    _pts = getattr(_w, 'video_time_ms', None)
                    if isinstance(_pts, (int, float)) and _pts >= 0.0:
                        pump_ms = float(_pts)
                        break
            if pump_ms is None:
                pump_ms = float(
                    getattr(self, '_current_precise_time', 0.0) or 0.0) * 1000.0
            adv = scout.next_events(pump_ms)
            # MVC/H.264 learns the boundary from its decoded-future scout
            # through this pump (HEVC additionally has the direct ordered
            # signal above).  Record the same absolute identity for MatAnyone;
            # de-duplication makes the dual HEVC path harmless.
            self._synth3d_note_cut_boundary(adv.get('cut_pts_ms'))
            for w in self._display_widgets():
                fn = getattr(w, 'set_lookahead_advisory', None)
                if fn is not None:
                    # cut_pts_ms (04/08) : PTS média ABSOLU de la coupe — la
                    # garde cross-shot native compare cet état exact au PTS de
                    # la carte publiée ; le délai relatif reste pour les
                    # rampes. Repli 2 arguments pour un widget d'avant.
                    try:
                        fn(adv['cut_in_ms'], adv['storm_in_ms'],
                           adv.get('cut_pts_ms'))
                    except TypeError:
                        fn(adv['cut_in_ms'], adv['storm_in_ms'])
            # Diagnostic (04/08): the advisory is dated against the UI clock,
            # while the renderer holds/ramps against the EXACT pts of the frame
            # it is presenting. Any systematic offset between the two shifts
            # the whole hold window by that much -- a load-bearing assumption
            # nobody had measured. Sampled here because both clocks are in
            # scope in this one function.
            for w in self._display_widgets():
                pts = getattr(w, 'video_time_ms', None)
                if isinstance(pts, (int, float)) and pts >= 0.0:
                    bias = pump_ms - float(pts)
                    self._la_bias_n = getattr(self, '_la_bias_n', 0) + 1
                    self._la_bias_sum = getattr(self, '_la_bias_sum', 0.0) + bias
                    self._la_bias_min = min(
                        getattr(self, '_la_bias_min', 1e9), bias)
                    self._la_bias_max = max(
                        getattr(self, '_la_bias_max', -1e9), bias)
                    break
        except Exception:
            pass

    def _forward_motion_hints(self, hints):
        """Indices de mouvement du décodeur -> renderers (04/08). Best-effort:
        chaque widget encaisse ou ignore; jamais bloquant pour la lecture."""
        if not self._native_signal_is_current():
            return
        try:
            for w in self._display_widgets():
                fn = getattr(w, 'set_motion_hints', None)
                if fn is not None:
                    fn(hints)
            # Phase 3 (04/08) : les mêmes pas nourrissent l'advecteur de
            # matte — l'alpha MatAnyone2 est transporté au pts de la frame dans
            # une bande de contour source, avec rejet local intégré à l'alpha.
            # SYLC_SYNTH3D_MATTE_ADVECT=0 = rollback.
            adv = getattr(self, '_synth3d_matte_advector', None)
            matte_live = (getattr(self, '_synth3d_active', False) and
                          getattr(self, '_synth3d_matte_service', None) is not None)
            if (adv is None and matte_live and os.environ.get(
                    'SYLC_SYNTH3D_MATTE_ADVECT', '1') != '0'):
                from sylc.synth3d_matting_service import MatteAdvector
                adv = self._synth3d_matte_advector = MatteAdvector()
            if adv is not None and matte_live:
                adv.note_hints(hints)
        except Exception:
            pass

    def _synth3d_supported(self):
        # Expressed through the reason helper rather than beside it: two copies
        # of the same three conditions would eventually disagree, and the one
        # that disagrees is the one naming a cause in a tooltip.
        return self._synth3d_unsupported_reason() is None

    def _synth3d_unsupported_reason(self):
        """Which of _synth3d_supported()'s three requirements is unmet, or None.

        Every "unavailable" tooltip is an instruction in disguise, and the
        instruction differs per cause: a missing onnxruntime.dll is NOT fixed by
        downloading 3.67 GB of weights. Resolved in the same order the support
        check applies them so the two can never disagree.
        """
        if not NATIVE_RENDER_AVAILABLE:
            return 'renderer'
        model, _side = self._synth3d_model_path()
        if not os.path.exists(model):
            return 'models'
        if not os.path.exists(os.path.join(self._synth3d_ort_dir(model),
                                           'onnxruntime.dll')):
            return 'runtime'
        return None

    def _synth3d_depth_preset(self):
        """Name of the active depth preset (persisted; default Quality)."""
        return synth3d_depth_preset_stored()

    def _synth3d_set_depth_preset(self, name):
        """Persist a depth-preset choice and re-arm a running session on it.

        The inference grid is baked into the shared depth service's key and into
        the renderer's warp pipeline, so an in-flight session cannot be
        re-pointed at another preset: it goes down and comes straight back up
        through the ordinary toggle path, which re-resolves model+side and
        rebuilds the ORT session with all the usual teardown, notification and
        status-poll wiring. With synthesis idle there is nothing to re-arm and
        the new choice simply applies to the next enable.
        """
        name = str(name)
        if synth3d_depth_preset_entry(name) is None:
            logger.warning("[2D3D] ignoring unknown depth preset %r", name)
            return
        if name == self._synth3d_depth_preset():
            self._update_synth3d_menu_state()
            return
        self._synth3d_aspect_override = None
        self._synth3d_aspect_unavailable_key = None
        try:
            settings = QSettings(SYNTH3D_SETTINGS_ORG, SYNTH3D_SETTINGS_APP)
            settings.setValue(SYNTH3D_DEPTH_PRESET_KEY, name)
            settings.sync()
        except Exception:
            logger.warning("[2D3D] could not persist depth preset %s", name,
                           exc_info=True)
        model, side = self._synth3d_model_path()
        logger.info("[2D3D] depth preset %s -> %s side=%d", name,
                    os.path.basename(model), side)
        if getattr(self, '_synth3d_active', False):
            # The off leg hides the framepack window for the few microseconds
            # until the on leg re-shows it (both legs run synchronously on the
            # GUI thread, so no frame can be delivered in between). That hide
            # reaches _on_framepacking_visibility_changed, which would read it as
            # a user close and tear the session down -- including auto-stopping
            # an active cast. `_synth3d_rearming` is that handler's fourth
            # "deliberate hide" reason, set ONLY around this pair. Keeping the
            # live cast rather than restarting it also keeps the Quest attached:
            # a stop/start drops the client and forces a re-discovery.
            cast_transport = (self._cast_transport
                              if getattr(self, '_cast', None) is not None
                              else None)
            self._synth3d_rearming = True
            try:
                self.toggle_synth3d(False)
                self.toggle_synth3d(True)
            finally:
                self._synth3d_rearming = False
            # Self-heal: should the session have died down some other teardown
            # path, bring it back through the ordinary start path -- never by
            # re-wiring a CastController by hand, that wiring is its own.
            if cast_transport is not None and getattr(self, '_cast', None) is None:
                logger.info("[CAST] restarting after the depth-preset re-arm (%s)",
                            cast_transport)
                self._on_cast_requested(cast_transport)
        self.show_3d_notification(f"2D->3D depth preset: {name}")
        self._update_synth3d_menu_state()

    def _synth3d_depth_preset_tooltip(self, name):
        """What an available depth preset will actually run: the model it
        resolves to and the grid it infers on (the grid is the speed lever)."""
        entry = synth3d_depth_preset_entry(name)
        model = synth3d_find_model(entry[1]) if entry else None
        if model is None:
            return "model file not installed"
        return f"{os.path.basename(model)} at {entry[2]}x{entry[2]}"

    def _synth3d_model_path(self):
        """(absolute model path, inference grid side) for the active preset.

        Both come from ONE SYNTH3D_DEPTH_PRESETS entry and are returned
        together: see that table's header for why the grid must never be
        re-derived anywhere else.

        Round 3's preference order lives on inside the Quality chain.
        `da3_small_756.onnx` outranks `da3_small.onnx`: same DA3-SMALL weights,
        but re-exported at a FIXED grid. The older onnx-community export
        declares height/width as dynamic axes, so its position-embedding Resize
        has no DirectML kernel and silently falls back to the CPU at ~229 ms for
        that one node -- 342 ms per map versus 55 ms for the fixed-shape
        re-export, measured.
        """
        entry = synth3d_depth_preset_entry(self._synth3d_depth_preset())
        if entry is not None:
            found = synth3d_find_model(entry[1])
            if found:
                return found, entry[2]
        # The active preset has no export installed. Fall through to the Quality
        # chain, which carries QUALITY's grid -- pairing the model we actually
        # found with the grid that was asked for is the one mismatch this whole
        # design exists to prevent.
        #
        # Spelled out as a literal rather than read back from
        # SYNTH3D_DEPTH_PRESETS[0]: this is the last-resort chain, and it should
        # keep working even if the preset table is ever reordered or emptied.
        # That leaves a duplicate of Quality's candidates, which
        # test_player_gating.py::test_quality_fallback_chain_in_the_source_matches_the_table
        # pins equal to the table so the two cannot drift apart. (Round 3 had a
        # second reason -- the engine-probe cross-check parsed this literal out
        # of the source with `ast`. Round 4 retired that: test_trt_optin.py now
        # compares setup_tensorrt.py's candidates against the whole preset table
        # directly, models AND grids.)
        quality = ('da3_base_756.onnx', 'da3_small_756.onnx', 'da3_small.onnx')
        quality_side = 756
        found = synth3d_find_model(quality)
        if found:
            return found, quality_side
        # Quality's own exports are missing too (a partial models/ directory).
        # Keep walking the display order -- which is also the inter-preset
        # fallback order -- rather than reporting synthesis unsupported while a
        # usable export sits on disk. Each preset is tried with ITS OWN grid, so
        # the pair stays consistent however far down we go. Index 0 is Quality,
        # already tried just above.
        for _name, candidates, side in SYNTH3D_DEPTH_PRESETS[1:]:
            found = synth3d_find_model(candidates)
            if found:
                return found, side
        return os.path.join(_synth3d_models_dirs()[0], quality[0]), quality_side

    def _synth3d_ort_dir(self, model_path=None):
        """Directory whose onnxruntime.dll the depth engine will load.

        `model_path` is the graph the caller is about to open: the marker is
        checked for THAT graph (see synth3d_marker_attests -- TensorRT's engine
        cache is per-graph, so attesting one model says nothing about another).
        Naming no graph keeps the round-3 behaviour, presence and freshness
        only; both real call sites resolve model+side once and pass the model
        in, so a preset switch takes the runtime decision with it.
        """
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        # Opt-in TensorRT runtime (see tools_dev/setup_tensorrt.py): preferred
        # over the DirectML-only root whenever it is COMPLETE -- both
        # onnxruntime.dll and at least one nvinfer*.dll must coexist there, so
        # a half-populated staging leftover is never picked -- AND the
        # `.trt_verified` marker must be present AND FRESH. That marker is
        # written ONLY after a REAL engine build + inference actually
        # succeeds (`setup_tensorrt.py --engine-probe`), never merely on DLL
        # presence: an incomplete/incompatible TensorRT assembly can crash
        # the whole process during a real engine build (a hard native abort,
        # not a graceful error -- see task-4-report.md, "The crash"), and
        # "playback never dies for 3D" means the player must never gamble on
        # an unverified opt-in runtime. Freshness (fix round 1, F2): if any
        # *.dll in the directory is newer than the marker, the DLLs were
        # replaced (a rerun of setup_tensorrt.py, a manual SDK drop, ...)
        # WITHOUT a matching --engine-probe rerun -- the marker's verification
        # no longer describes what's actually on disk, so it must be treated
        # as absent. (A driver/GPU change invalidating an otherwise-fresh
        # marker is a narrower, less common case not handled here -- this
        # closes the realistic path at acquisition time, not every path.)
        # Checked next to the exe first, then the repo root -- the same
        # two-root order as the DirectML fallback below.
        for root in dict.fromkeys((base, _SOURCE_ROOT)):
            trt_dir = os.path.join(root, 'ort_tensorrt')
            marker = os.path.join(trt_dir, '.trt_verified')
            if (os.path.exists(os.path.join(trt_dir, 'onnxruntime.dll'))
                    and glob.glob(os.path.join(trt_dir, 'nvinfer*.dll'))
                    and os.path.exists(marker)):
                dlls = glob.glob(os.path.join(trt_dir, '*.dll'))
                newest_dll_mtime = max((os.path.getmtime(p) for p in dlls), default=0.0)
                if (os.path.getmtime(marker) >= newest_dll_mtime
                        and (model_path is None
                             or synth3d_marker_attests(marker, model_path))):
                    return trt_dir
        if os.path.exists(os.path.join(RUNTIME_DIR, 'onnxruntime.dll')):
            return RUNTIME_DIR
        if os.path.exists(os.path.join(base, 'onnxruntime.dll')):
            return base
        return _SOURCE_ROOT

    def _synth3d_trt_dir(self):
        """The `ort_tensorrt` directory the opt-in TensorRT runtime lives in.

        The CANDIDATE, not the runtime actually loaded. `_synth3d_ort_dir`
        answers a different question -- which onnxruntime.dll to load -- and
        falls back to the DirectML root, which has an onnxruntime.dll of its
        own; describing that root as a half-installed TensorRT would be a false
        statement about a perfectly good default install. Same two-root order
        as `_synth3d_ort_dir`; when neither exists the answer is the one beside
        the executable, because that is where an install would go.
        """
        roots = list(dict.fromkeys(
            (os.path.dirname(os.path.abspath(sys.argv[0])), _SOURCE_ROOT)))
        for root in roots:
            candidate = os.path.join(root, 'ort_tensorrt')
            if os.path.isdir(candidate):
                return candidate
        return os.path.join(roots[0], 'ort_tensorrt')

    def _synth3d_eligible(self):
        return ((getattr(self, 'mvc_mode_active', False)
                 or getattr(self, '_hevc_mode_active', False))
                and not self._content_is_3d() and bool(self.has_media))

    def toggle_synth3d(self, enabled, remember=True):
        if enabled and not (self._synth3d_supported() and self._synth3d_eligible()):
            self._synth3d_active = False
            self.show_3d_notification(
                "2D->3D AI needs a 2D video on the native decoder", success=False)
            self._update_synth3d_menu_state()
            return
        self._synth3d_active = bool(enabled)
        self._synth3d_pending_cut_pts = None
        self._synth3d_matte_cut_seen_ms = -math.inf
        self._synth3d_matte_floor_pts_ms = -math.inf
        # Arm synchronously. Waiting for the 100 ms timeline pump leaves a
        # reachable first-cut window immediately after the user enables 2D->3D.
        owner = getattr(self, '_synth3d_lookahead_thread', None)
        th = owner() if owner is not None else None
        if th is not None:
            try:
                th.set_lookahead_enabled(bool(enabled))
            except Exception:
                logger.exception("[LOOKAHEAD] immediate arm/disarm failed")
        # Per-file memory: 2D->3D synthesis is a per-title decision (a title
        # the viewer converts once is a title they want converted on replay) —
        # but only the viewer's EXPLICIT toggle is a decision. Two teardown
        # echoes were overwriting the remembered preference with False while
        # has_media was still True (both proven 2026-08-04, sessions that
        # verifiably ran 2D->3D and saved synth3d_enabled=false on exit):
        #   - the framepacking window being closed/hidden — its visibility
        #     handler re-enters here to dismantle the pipeline and passes
        #     remember=False, because closing the 3D output window (Alt-F4 on
        #     the fullscreen, or as part of quitting the app) says "done
        #     viewing", not "never convert this title again";
        #   - the app close itself — closeEvent sets _app_closing before its
        #     teardown, which guards every other indirect disable on the way
        #     out. (The explicit-Stop path only ever escaped by timing:
        #     has_media happened to drop before the hide event processed.)
        _rem = getattr(self, '_remember_for_file', None)
        if (remember and _rem is not None and getattr(self, 'has_media', False)
                and not getattr(self, '_app_closing', False)):
            _rem(synth3d_enabled=bool(enabled))
        # A synth3d session only ever runs on 2D content (_synth3d_eligible requires
        # `not self._content_is_3d()`), so this can never clobber a real-3D session.
        # Without it, change_stereo_mode/_set_dual_projector_enabled's `is_3d_enabled`
        # gate left every stereo-mode combo pick a silent no-op during synth3d.
        self.is_3d_enabled = bool(enabled)
        self._push_synth3d_to_widgets()
        if enabled:
            self._synth3d_restore_mono_source()
            start_matting = getattr(self, '_synth3d_start_human_matting', None)
            if start_matting is not None:
                start_matting()
            self.show_3d_notification("2D->3D AI active - depth model warming up")
            logger.info("[2D3D] enabled strength=%.1f%% convergence=%.2f",
                        self._synth3d_strength, self._synth3d_convergence)
            # Keep the selector and the actual presentation in lockstep. 'auto'
            # visually means MultiView (combo index 0); persisting a previous
            # SBS/TAB/Dual/Glasses choice is also legitimate and should be honoured.
            presentation = getattr(self, 'current_stereo_mode', 'auto')
            if presentation not in ('mvc', 'sbs', 'tab', 'dual', 'glasses'):
                presentation = 'mvc'
                self.current_stereo_mode = presentation
            self.configure_3d_output(True, presentation)
            # F2: replaces the old one-shot 2.5s _log_synth3d_status peek -- a
            # single early log said nothing about a failure hours into playback.
            self._synth3d_start_poll()
        else:
            # Attribute propagation alone reaches NativeRenderer only with the
            # next decoded frame. A paused/stopped source may never deliver it,
            # so detach every live surface synchronously (the native registry
            # hands teardown to its reaper and returns without joining ORT).
            self._synth3d_disable_renderers_now()
            if getattr(self, '_synth3d_rearming', False):
                service = getattr(self, '_synth3d_matte_service', None)
                if service is not None:
                    service.reset("depth preset re-arm")
                clear_matte = getattr(self, '_synth3d_clear_human_matte', None)
                if clear_matte is not None:
                    clear_matte()
            else:
                stop_matting = getattr(self, '_synth3d_stop_human_matting', None)
                if stop_matting is not None:
                    stop_matting()
            self.show_3d_notification("2D->3D AI off")
            self.configure_3d_output(False)
            self._synth3d_stop_poll()
        self._update_3d_button_state()
        self._update_synth3d_menu_state()

    def _synth3d_start_human_matting(self):
        """Start the optional MatAnyone 2 process without blocking playback."""
        if not getattr(self, '_synth3d_matting_requested', True):
            return False

        service = getattr(self, '_synth3d_matte_service', None)
        if service is not None and service.running:
            service.reset("2D->3D enabled")
            return True
        try:
            from sylc.synth3d_matting_service import MatAnyone2Runtime, MatAnyone2Service
            runtime = MatAnyone2Runtime.discover(_SOURCE_ROOT)
            if runtime is None:
                if not getattr(self, '_synth3d_matte_unavailable_logged', False):
                    logger.info("[MATANYONE2] offline runtime not installed; "
                                "continuing with depth-only contour protection")
                    self._synth3d_matte_unavailable_logged = True
                return False
            # Procédural (04/08) : sans env explicite, la cadence et la
            # définition sont ASSERVIES au média lu (configure_media au pont
            # par frame). Les valeurs ici ne sont que l'amorce d'avant la
            # première frame ; un env posé les ÉPINGLE définitivement.
            fps_env = os.environ.get('SYLC_MATANYONE2_FPS')
            side_env = os.environ.get('SYLC_MATANYONE2_SHORT_SIDE')
            target_fps = float(fps_env) if fps_env else 12.0
            short_side = int(side_env) if side_env else 640
            service = MatAnyone2Service(runtime, target_fps=target_fps,
                                        short_side=short_side,
                                        fps_pinned=fps_env is not None,
                                        short_side_pinned=side_env is not None)
            self._synth3d_matte_service = service
            if not service.start():
                logger.warning("[MATANYONE2] worker unavailable: %s",
                               service.status().get('error', 'start failure'))
                self._synth3d_matte_service = None
                return False
            logger.info("[MATANYONE2] async worker starting at %.1f fps, short-side=%d",
                        target_fps, short_side)
            return True
        except Exception:
            logger.exception("[MATANYONE2] optional worker initialization failed")
            self._synth3d_matte_service = None
            return False

    def _synth3d_note_cut_boundary(self, cut_pts_ms):
        """Remember a decoded-future shot boundary for the matte pipeline.

        This function deliberately does not reset anything: MVC can know T0
        twelve decoded frames early.  `_synth3d_human_matte_for_frame` consumes
        the identity against the exact presentation PTS.
        """
        try:
            cut = float(cut_pts_ms)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(cut) or cut < 0.0:
            return False
        seen = float(getattr(self, '_synth3d_matte_cut_seen_ms', -math.inf))
        if cut <= seen + 0.5:
            return False
        pending = getattr(self, '_synth3d_pending_cut_pts', None)
        if pending is None or cut < float(pending) - 0.5:
            self._synth3d_pending_cut_pts = cut
            return True
        return False

    def _synth3d_apply_matte_cut_if_due(self, video_time_ms, service):
        """Advance the matte shot epoch exactly on the first T0 presentation."""
        pending = getattr(self, '_synth3d_pending_cut_pts', None)
        if pending is None:
            return False
        try:
            pts = float(video_time_ms)
            cut = float(pending)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(pts) or pts < cut - 0.5:
            return False

        # Generation changes first.  A result already being inferred by the
        # CUDA worker may arrive before it reads the reset control, but the
        # service receiver rejects that old generation unconditionally.
        service.reset(f"shot boundary @{cut:.3f} ms")
        self._synth3d_matte_cut_seen_ms = cut
        self._synth3d_matte_floor_pts_ms = cut
        self._synth3d_pending_cut_pts = None
        clear_matte = getattr(self, '_synth3d_clear_human_matte', None)
        if clear_matte is not None:
            clear_matte()
        # service.reset() above already logs this cut's PTS *and* the new
        # generation, so an INFO line here only duplicates it -- at two lines
        # per cut, on a film whose median shot runs ~2.5 s, that was 12 % of
        # the log. Kept at DEBUG for the case where a compatibility service
        # does not log its own reset.
        logger.debug("[MATANYONE2] shot epoch advanced at %.3f ms", cut)
        return True

    def _synth3d_clear_human_matte(self):
        adv = getattr(self, '_synth3d_matte_advector', None)
        if adv is not None:
            adv.reset()
        for widget in self._display_widgets():
            try:
                widget.set_synth3d_human_matte(None)
            except (AttributeError, TypeError):
                pass

    def _synth3d_stop_human_matting(self, timeout=0.0):
        self._synth3d_clear_human_matte()
        service = getattr(self, '_synth3d_matte_service', None)
        self._synth3d_matte_service = None
        if service is not None:
            try:
                service.stop(timeout=timeout)
            except Exception:
                logger.exception("[MATANYONE2] worker shutdown failed")

    @classmethod
    def _snap_media_fps(cls, fps):
        """Snap a measured frame rate onto the rate the media was authored at.

        The HEVC thread derives ``target_frame_time`` from container PTS deltas,
        and those are integer milliseconds: 23.976 fps arrives as an alternating
        41 / 42 ms, i.e. 24.39 / 23.81 fps, changing on every single frame. Any
        consumer that compares the value against its previous reading therefore
        sees a change forever. Snapping removes the quantization instead of
        asking each consumer to tolerate it.

        A rate that matches nothing (genuinely odd or variable content) is
        returned unchanged -- this corrects a known artefact, it does not
        invent a cadence.
        """
        try:
            value = float(fps)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        for standard in cls._MEDIA_STANDARD_FPS:
            if abs(value - standard) <= 0.02 * standard:
                return standard
        return value

    def _synth3d_media_pacing(self, left_planes):
        """(fps, côté court) du média lu, ou None quand la source ne
        l'expose pas (le procédural s'abstient alors, il n'invente rien)."""
        short = None
        try:
            y = left_planes[0]
            short = int(min(int(y.shape[0]), int(y.shape[1])))
        except Exception:
            short = None
        fps = None
        try:
            threads = []
            if getattr(self, '_hevc_mode_active', False):
                threads.append(getattr(self, 'hevc_thread', None))
            threads.append(getattr(self, 'mvc_decoder_thread', None))
            for th in threads:
                tft = getattr(th, 'target_frame_time', None) if th else None
                if tft and 0.004 <= float(tft) <= 0.2:
                    fps = self._snap_media_fps(1.0 / float(tft))
                    break
        except Exception:
            fps = None
        return fps, short

    def _synth3d_human_matte_for_frame(self, left_planes, video_time_ms):
        if not getattr(self, '_synth3d_active', False):
            return None
        service = getattr(self, '_synth3d_matte_service', None)
        if service is None or not service.running:
            return None
        try:
            self._synth3d_apply_matte_cut_if_due(video_time_ms, service)
            # Contour lock (05/08): retain the luma of the exact frame being
            # displayed before the asynchronous submit.  The returned matte
            # can then be registered from its own source PTS to this PTS on
            # both H.264 and H.265, independently of codec motion metadata.
            adv = getattr(self, '_synth3d_matte_advector', None)
            if adv is None and os.environ.get(
                    'SYLC_SYNTH3D_MATTE_ADVECT', '1') != '0':
                from sylc.synth3d_matting_service import MatteAdvector
                adv = self._synth3d_matte_advector = MatteAdvector()
            if adv is not None:
                adv.note_frame(left_planes[0], float(video_time_ms))
            # Procédural (04/08) : asservir cadence/définition au média AVANT
            # la soumission (no-op quand rien ne change, env épinglés
            # intouchés). Le fps vient du thread décodeur quand il l'expose ;
            # le côté court, des plans eux-mêmes.
            cfg = getattr(service, 'configure_media', None)
            if cfg is not None:
                fps, short = self._synth3d_media_pacing(left_planes)
                cfg(fps=fps, short_side=short)
            service.submit_yuv(left_planes, float(video_time_ms))
            matte = service.latest_for_pts(float(video_time_ms))
            # Second line of defence: even a compatibility/fake service which
            # does not implement generation invalidation cannot return a mask
            # captured before the current shot epoch.
            floor = float(getattr(
                self, '_synth3d_matte_floor_pts_ms', -math.inf))
            if (matte is not None and math.isfinite(floor)
                    and float(matte.pts_ms) < floor - 0.5):
                matte = None
            # Phase 3 (04/08) : l'alpha 5 Hz est transporté au pts de CETTE
            # frame par les pas MV accumulés — les gardes de contour suivent
            # la personne au lieu de sa position d'il y a 200 ms.
            if adv is not None and matte is not None:
                matte = adv.advect(matte, float(video_time_ms))
            return matte
        except Exception:
            # Matting is guidance, never a dependency of depth synthesis.
            logger.exception("[MATANYONE2] frame bridge failed; disabling matte guidance")
            self._synth3d_stop_human_matting()
            return None

    def _synth3d_restore_mono_source(self):
        """Keep a synthesized session's input as the original unsplit 2D frame.

        Stereo-mode choices describe only how the generated L/R pair is shown.
        They are never evidence that a 2D HEVC source suddenly became packed
        SBS/TAB. Reset the display-aspect override too: the real mono plane
        dimensions are again the sole source of truth.
        """
        thread = getattr(self, 'hevc_thread', None)
        if getattr(self, '_hevc_mode_active', False) and thread is not None:
            try:
                thread.set_mode(None)
            except Exception:
                logger.exception("[2D3D] could not restore HEVC mono input mode")
        for widget in self._display_widgets():
            try:
                widget.source_aspect = 0.0
            except Exception:
                pass

    def _push_synth3d_to_widgets(self):
        # Resolved once per push, not once per widget: both do a couple of
        # os.path.exists calls, wasted work repeated identically for every
        # widget on the (potential per-slider-tick) hot path.
        model_path, side = self._synth3d_model_path()
        base_key = os.path.normcase(os.path.abspath(model_path))
        grid_width = grid_height = 0
        crop_top = crop_bottom = 0.0
        override = getattr(self, '_synth3d_aspect_override', None)
        if override is not None:
            override_model, override_side, selection = override
            if (override_model == base_key and override_side == side
                    and os.path.isfile(selection.model_path)):
                model_path = selection.model_path
                grid_width = selection.grid_width
                grid_height = selection.grid_height
                crop_top = selection.crop_top_norm
                crop_bottom = selection.crop_bottom_norm
            else:
                # A model was removed/replaced or the active preset changed.
                # Never retain a rectangular graph with the wrong base family.
                self._synth3d_aspect_override = None
        ort_dir = self._synth3d_ort_dir(model_path)
        for w in self._display_widgets():
            w.synth3d_enabled = self._synth3d_active
            w.synth3d_strength = self._synth3d_strength
            w.synth3d_convergence = self._synth3d_convergence
            w.synth3d_auto_convergence = getattr(
                self, '_synth3d_auto_convergence', False)
            w.synth3d_temporal_fill = getattr(
                self, '_synth3d_temporal_fill', False)
            # Additive only: native keeps the six v5.2.1c raw planes immutable;
            # SYLC_STEREO_LAB=0 provides an exact A/B rollback.
            w.synth3d_stereo_lab = True
            _comfort = getattr(self, '_synth3d_comfort_envelope', None)
            w.synth3d_comfort_enabled = bool(
                getattr(self, '_synth3d_comfort_enabled', False) and
                _comfort is not None)
            w.synth3d_comfort_soft_pct = (
                _comfort.soft_percent if _comfort is not None else 0.0)
            w.synth3d_comfort_hard_pct = (
                _comfort.hard_percent if _comfort is not None else 0.0)
            w.synth3d_depth_view = self._synth3d_depth_view
            w.synth3d_diagnostics = self._synth3d_diagnostics
            w.synth3d_model_path = model_path
            w.synth3d_ort_dir = ort_dir
            # Set next to the model it was resolved WITH, never anywhere else:
            # the grid is half of the (model, side) pair SYNTH3D_DEPTH_PRESETS
            # keeps together.
            w.synth3d_side = side
            w.synth3d_grid_width = grid_width
            w.synth3d_grid_height = grid_height
            w.synth3d_crop_top = crop_top
            w.synth3d_crop_bottom = crop_bottom

    def _synth3d_disable_renderers_now(self):
        """Detach live native surfaces without waiting for another video frame."""
        for w in self._display_widgets():
            # Invalidate the widget cache even when the native call succeeds.
            # A rapid off->on before another frame must re-push True rather than
            # mistake the pre-disable True tuple for the live renderer state.
            try:
                w._synth3d_pushed = None
            except Exception:
                pass
            r = getattr(w, '_r', None)
            if r is None:
                continue
            try:
                r.set_synth3d(False)
            except (AttributeError, TypeError, RuntimeError):
                pass

    @staticmethod
    def _synth3d_status_fields(status):
        fields = {}
        for token in str(status or '').split():
            if '=' in token:
                key, value = token.split('=', 1)
                fields[key] = value
        return fields

    def _maybe_apply_synth3d_aspect(self, status):
        """Select (and reselect) a rectangular graph from live full-frame evidence.

        Encoded mattes require eight agreeing depth observations.  A crop-free
        source may promote immediately because its coded dimensions are stable;
        a minimum native ratio prevents ordinary 16:9 video from doing so.
        The renderer keeps reporting an uncropped luma probe even while the
        model input is cropped, so a certified Scope/IMAX transition can return
        to square or move to another installed rectangular graph.
        """
        if not getattr(self, '_synth3d_active', False):
            return False
        override = getattr(self, '_synth3d_aspect_override', None)
        fields = self._synth3d_status_fields(status)
        if fields.get('state') != 'running':
            return False
        try:
            confidence = float(fields.get('crop_conf', '0'))
            top, bottom, source_width, source_height = (
                int(value) for value in fields.get('crop', '').split(':'))
        except (TypeError, ValueError):
            return False
        if (source_width <= 0 or source_height <= 0
                or top < 0 or bottom < 0):
            return False
        crop_ready = fields.get('crop_ready') == '1'

        def clear_override(reason):
            if override is None:
                return False
            logger.info("[2D3D] %s; returning to the square graph", reason)
            self._synth3d_aspect_override = None
            self._synth3d_aspect_unavailable_key = None
            self._push_synth3d_to_widgets()
            return True

        has_matte = top > 0 or bottom > 0
        if has_matte:
            if confidence < 0.999 or top <= 0 or bottom <= 0:
                return False
        elif source_width / float(source_height) < MIN_NATIVE_WIDE_RATIO:
            # Missing crop_ready means an older native module: preserve its
            # established override because that build cannot certify a return
            # to no-matte content. New builds clear only after eight agreeing
            # full-frame observations, never on one dark/flash frame.
            if override is not None and crop_ready:
                return clear_override(
                    "full-frame evidence now certifies %dx%d without a matte" %
                    (source_width, source_height))
            return False

        square_model, side = self._synth3d_model_path()
        base_key = os.path.normcase(os.path.abspath(square_model))
        if override is not None:
            override_model, override_side, _current_selection = override
            if override_model != base_key or override_side != side:
                return clear_override("the active depth preset changed")
        selection = select_installed_aspect_model(
            square_model, side, source_width, source_height, top, bottom)
        if selection is None:
            if override is not None:
                # While a newly attached rectangular service is accumulating
                # its first eight observations, crop=0 is intentionally not a
                # verdict. Retain the known-good selection until crop_ready=1.
                if not crop_ready:
                    return False
                return clear_override(
                    "the certified %.3f:1 content has no compatible rectangle" %
                    (source_width / float(
                        max(1, source_height - top - bottom))))
            miss_key = (os.path.normcase(os.path.abspath(square_model)), side,
                        source_width, source_height, top, bottom)
            if miss_key != getattr(self, '_synth3d_aspect_unavailable_key', None):
                if has_matte:
                    logger.info(
                        "[2D3D] stable matte %d+%d on %dx%d; no compatible "
                        "%d-wide rectangular export installed, keeping square grid",
                        top, bottom, source_width, source_height, side)
                else:
                    logger.info(
                        "[2D3D] native wide frame %dx%d (%.3f:1); no compatible "
                        "%d-wide rectangular export installed, keeping square grid",
                        source_width, source_height,
                        source_width / float(source_height), side)
                self._synth3d_aspect_unavailable_key = miss_key
            return False

        if override is not None:
            current = override[2]
            current_key = (
                os.path.normcase(os.path.abspath(current.model_path)),
                current.grid_width, current.grid_height,
                current.crop_top, current.crop_bottom,
                current.source_width, current.source_height)
            selection_key = (
                os.path.normcase(os.path.abspath(selection.model_path)),
                selection.grid_width, selection.grid_height,
                selection.crop_top, selection.crop_bottom,
                selection.source_width, selection.source_height)
            if selection_key == current_key:
                return False

        # TensorRT engines are per graph. A square service that is already on
        # TensorRT must not silently fall back to DirectML just because the
        # new rectangle has not yet been built/attested in .trt_verified.
        # Keep the faster known-good square until setup_tensorrt.py has probed
        # this exact rectangular graph.
        candidate_runtime = self._synth3d_ort_dir(selection.model_path)
        if (fields.get('provider', '').lower() == 'tensorrt'
                and os.path.basename(os.path.normpath(
                    candidate_runtime)).lower() != 'ort_tensorrt'):
            miss_key = ('trt-unattested',
                        os.path.normcase(os.path.abspath(selection.model_path)))
            if miss_key != getattr(self, '_synth3d_aspect_unavailable_key', None):
                logger.info(
                    "[2D3D] %s is not attested by the current TensorRT marker; "
                    "keeping the square TensorRT graph",
                    os.path.basename(selection.model_path))
                self._synth3d_aspect_unavailable_key = miss_key
            if override is not None:
                return clear_override(
                    "the newly matching rectangle is not TensorRT-attested")
            return False

        self._synth3d_aspect_override = (base_key, side, selection)
        self._synth3d_aspect_unavailable_key = None
        if has_matte:
            source_desc = "stable matte %d+%d on %dx%d" % (
                top, bottom, source_width, source_height)
        else:
            source_desc = "native wide frame %dx%d (%.3f:1)" % (
                source_width, source_height,
                source_width / float(source_height))
        logger.info(
            "[2D3D] %s -> %s grid=%dx%d (%d%% fewer depth pixels)%s",
            source_desc, os.path.basename(selection.model_path),
            selection.grid_width, selection.grid_height,
            round(100 * (1.0 - selection.grid_height / float(side))),
            " [ratio transition]" if override is not None else "")
        self._push_synth3d_to_widgets()
        return True

    def _log_synth3d_status(self):
        try:
            w = self.mvc_embedded_widget
            if w is not None and getattr(w, '_r', None):
                logger.info("[2D3D] %s", w._r.synth3d_status())
        except Exception:
            pass

    def _synth3d_start_poll(self):
        """F2: nothing previously observed `state=error` once synth3d was running --
        a mid-run engine failure (ORT session lost, DirectML driver reset, ...) left
        the toggle stuck "on" with nothing displaying its output and no notification.
        Poll the depth engine's status every 2s and auto-disable on state=error."""
        timer = getattr(self, '_synth3d_poll_timer', None)
        if timer is None:
            try:
                timer = QTimer(self)
            except TypeError:
                # Non-QObject host (e.g. a test stub): still usable, just unparented.
                timer = QTimer()
            timer.timeout.connect(self._synth3d_poll_status)
            self._synth3d_poll_timer = timer
        self._synth3d_poll_last_log = 0.0
        timer.start(2000)

    def _synth3d_stop_poll(self):
        timer = getattr(self, '_synth3d_poll_timer', None)
        if timer is not None:
            timer.stop()

    def _synth3d_poll_status(self):
        """Status comes from the first _display_widgets() entry with a live renderer --
        NOT hard-coded to mvc_embedded_widget (house rule, see _display_widgets), since
        the active 3D presentation may be the framepack or an eye-window widget."""
        statuses = []
        for w in self._display_widgets():
            r = getattr(w, '_r', None)
            if r is None:
                continue
            try:
                statuses.append(r.synth3d_status())
            except (AttributeError, TypeError, RuntimeError):
                continue
        if not statuses:
            return
        # A device-local renderer failure may affect only one output surface.
        # It must outrank a healthy shared-service line from another surface.
        st = next((item for item in statuses if "state=error" in item),
                  statuses[0])
        maybe_apply_aspect = getattr(self, '_maybe_apply_synth3d_aspect', None)
        if maybe_apply_aspect is not None:
            maybe_apply_aspect(st)
        self._update_synth3d_status_label(st)
        now = time.monotonic()
        last = getattr(self, '_synth3d_poll_last_log', 0.0)
        if now - last >= 10.0:
            logger.info("[2D3D] %s", st)
            # Scout coverage beside the engine line: a hard cut the scout
            # DETECTED but never surfaced produces no hold at all downstream.
            try:
                _owner = getattr(self, '_synth3d_lookahead_thread', None)
                _th = _owner() if _owner is not None else None
                _sc = _th.lookahead_scout() if _th is not None else None
                if _sc is not None:
                    _bn = getattr(self, '_la_bias_n', 0)
                    logger.info(
                        "[LOOKAHEAD] frames=%d published=%d reported=%d "
                        "skipped=%d queued=%d | clock bias pump-vs-pts "
                        "mean=%.0fms min=%.0f max=%.0f n=%d",
                        _sc.frames_analyzed, _sc.cuts_published,
                        _sc.cuts_reported, _sc.cuts_skipped,
                        _sc.cuts_published - _sc.cuts_reported
                        - _sc.cuts_skipped,
                        (getattr(self, '_la_bias_sum', 0.0) / _bn) if _bn else 0.0,
                        getattr(self, '_la_bias_min', 0.0),
                        getattr(self, '_la_bias_max', 0.0), _bn)
            except Exception:
                pass
            matte_service = getattr(self, '_synth3d_matte_service', None)
            if matte_service is not None:
                diag = dict(matte_service.status())
                matte_advector = getattr(
                    self, '_synth3d_matte_advector', None)
                if matte_advector is not None:
                    diag["contour_lock"] = matte_advector.status()
                logger.info("[MATANYONE2] %s", diag)
            self._synth3d_poll_last_log = now
        if "state=error" in st:
            err_tail = st.split("err=", 1)[-1] if "err=" in st else st
            self.toggle_synth3d(False)
            self.show_3d_notification(
                f"2D->3D AI failed - disabled ({err_tail})", success=False)

    def set_synth3d_depth_view(self, enabled):
        self._synth3d_depth_view = bool(enabled)
        self._push_synth3d_to_widgets()

    def set_synth3d_diagnostics(self, enabled):
        """Overlay depth and the continuous disocclusion confidence on the
        synthesized movie. Unlike depth-view this keeps the content visible,
        so it is useful for live tuning and qualification."""
        self._synth3d_diagnostics = bool(enabled)
        self._push_synth3d_to_widgets()

    def apply_synth3d_preset(self, name):
        name = str(name).lower()
        if name == 'custom':
            self._synth3d_preset = 'custom'
        elif name in self._SYNTH3D_PRESETS:
            self._synth3d_strength, self._synth3d_convergence = \
                self._SYNTH3D_PRESETS[name]
            self._synth3d_preset = name
        else:
            return
        self._app_settings['synth3d_preset'] = self._synth3d_preset
        self._app_settings['synth3d_strength_pct'] = self._synth3d_strength
        self._app_settings['synth3d_convergence'] = self._synth3d_convergence
        self._save_app_settings()
        self._push_synth3d_to_widgets()
        self._update_synth3d_menu_state()
        if name != 'custom':
            self.show_3d_notification(
                f"2D->3D preset: {name.title()}", success=True)

    def _mark_synth3d_custom(self):
        if getattr(self, '_synth3d_preset', 'custom') != 'custom':
            self._synth3d_preset = 'custom'
            self._app_settings['synth3d_preset'] = 'custom'

    def set_synth3d_strength(self, tenths_pct, persist=True):
        self._synth3d_strength = max(0.5, min(3.0, tenths_pct / 10.0))
        self._mark_synth3d_custom()
        if persist:
            self._app_settings['synth3d_strength_pct'] = self._synth3d_strength
            self._save_app_settings()
            # Depth strength is content-dependent tuning: also remember it for
            # THIS title (persist=False = programmatic restore, not re-saved).
            _rem = getattr(self, '_remember_for_file', None)
            if _rem is not None and getattr(self, 'has_media', False):
                _rem(synth3d_strength=self._synth3d_strength)
        self._push_synth3d_to_widgets()

    def set_synth3d_convergence(self, percent, persist=True):
        self._synth3d_convergence = max(0.0, min(1.0, percent / 100.0))
        self._mark_synth3d_custom()
        if persist:
            self._app_settings['synth3d_convergence'] = self._synth3d_convergence
            self._save_app_settings()
            _rem = getattr(self, '_remember_for_file', None)
            if _rem is not None and getattr(self, 'has_media', False):
                _rem(synth3d_convergence=self._synth3d_convergence)
        self._push_synth3d_to_widgets()

    def set_synth3d_auto_convergence(self, enabled, persist=True):
        """Per-shot zero-parallax plane from the stabilizer's tone machinery.
        The manual Convergence slider remains the fallback (and the value used
        the instant a session starts, before the first depth map lands)."""
        self._synth3d_auto_convergence = bool(enabled)
        if persist:
            self._app_settings['synth3d_auto_convergence'] = \
                self._synth3d_auto_convergence
            self._save_app_settings()
        self._push_synth3d_to_widgets()

    def set_synth3d_temporal_fill(self, enabled, persist=True):
        """Round 5a: disocclusion holes prefer flow-transported, previously
        SEEN background over the stretch fallback. Experimental (author's
        visual gate pending); off is byte-identical to the historical warp."""
        self._synth3d_temporal_fill = bool(enabled)
        if persist:
            self._app_settings['synth3d_temporal_fill'] = \
                self._synth3d_temporal_fill
            self._save_app_settings()
        self._push_synth3d_to_widgets()

    def _synth3d_on_native_path_lost(self):
        """Native decode path gone (mpv fallback, stop): synthesis cannot run."""
        if getattr(self, '_synth3d_active', False):
            self._synth3d_active = False
            # F1b: a synth3d session only ever runs on 2D content (_synth3d_eligible
            # requires `not self._content_is_3d()`), so this can never clobber a
            # real-3D session -- same invariant toggle_synth3d(True) relies on.
            # Without it, is_3d_enabled stayed True after the native path died,
            # leaving change_stereo_mode/_set_dual_projector_enabled's gate open
            # on a session with no synthesis and no real 3D content behind it.
            self.is_3d_enabled = False
            self._synth3d_stop_poll()
            self._push_synth3d_to_widgets()
            self._synth3d_disable_renderers_now()
            self.show_3d_notification("2D->3D AI off (native decoder lost)",
                                      success=False)
        stop_matting = getattr(self, '_synth3d_stop_human_matting', None)
        if stop_matting is not None:
            stop_matting()
        self._update_synth3d_menu_state()
        # The AI button's grey/white/blue cycle lives in _update_3d_button_state;
        # refresh it on every path that kills the synthesis (guarded: some
        # teardown callers run on minimal hosts without the overlay).
        try:
            self._update_3d_button_state()
        except Exception:
            pass

    def _synth3d_handle_decoder_stop(self, restarting):
        """`_stop_mvc_decoder` runs on every MVC teardown path, including a
        same-session restart (edge264 crash recovery, seek-queue decoder restart)
        where `_mvc_restarting` is already True by the time it gets here --
        `_start_mvc_decoder` sets that flag BEFORE calling `_stop_mvc_decoder()`.
        Only a genuine stop should kill synthesis; notifying "native decoder
        lost" mid-restart would be a false alarm while the path is coming right
        back. NOTE: deliberately keyed on `_mvc_restarting` alone, NOT on
        `_terminating_mpv` too (unlike the sibling mpv-video-output-restore
        guard just above this call site in `_stop_mvc_decoder`) -- stop_playback
        sets `_terminating_mpv` for a genuine, permanent stop; that flag only
        skips restoring mpv's D3D11 output (crash-avoidance), it does not mean
        the decoder is coming back, so synthesis must still be torn down then."""
        if restarting:
            return
        if hasattr(self, '_synth3d_on_native_path_lost'):
            self._synth3d_on_native_path_lost()

    def _synth3d_notify_seek_widgets(self):
        """Deterministic depth-EMA reset on seek (spec: same mechanic as a cut snap)."""
        if not getattr(self, '_synth3d_active', False):
            return
        # A seek may jump across an aspect-ratio transition (Scope/IMAX reel
        # switches are common enough to make retaining an encoded MATTE unsafe).
        # Return briefly to the warm square service and let the new position
        # earn its own eight-observation verdict -- but only for a selection a
        # seek can actually invalidate: see SYNTH3D_SEEK_KEEP_ASPECT for why a
        # coded-dimension selection is kept and what the detour costs.
        override = getattr(self, '_synth3d_aspect_override', None)
        if override is not None and not _synth3d_seek_keeps_aspect(override):
            self._synth3d_aspect_override = None
            self._synth3d_aspect_unavailable_key = None
            self._push_synth3d_to_widgets()
        matte_service = getattr(self, '_synth3d_matte_service', None)
        if matte_service is not None:
            matte_service.reset("media seek")
        self._synth3d_pending_cut_pts = None
        self._synth3d_matte_cut_seen_ms = -math.inf
        self._synth3d_matte_floor_pts_ms = -math.inf
        clear_matte = getattr(self, '_synth3d_clear_human_matte', None)
        if clear_matte is not None:
            clear_matte()
        for w in self._display_widgets():
            r = getattr(w, '_r', None)
            try:
                reset_lookahead = getattr(
                    w, 'reset_synth3d_lookahead', None)
                if reset_lookahead is not None:
                    reset_lookahead("media seek")
                if r is not None:
                    r.synth3d_notify_seek()
            except (AttributeError, TypeError):
                pass

    def _synth3d_manifest_path(self):
        """The manifest travels with the code, so it is found the same way the
        models are -- next to the executable first, then the source tree."""
        for root in dict.fromkeys(
                (os.path.dirname(os.path.abspath(sys.argv[0])), _SOURCE_ROOT)):
            candidate = os.path.join(root, 'models', 'MANIFEST.json')
            if os.path.exists(candidate):
                return candidate
        return None

    def _open_model_download_dialog(self):
        manifest = self._synth3d_manifest_path()
        if manifest is None:
            self.show_3d_notification(
                "models/MANIFEST.json is missing from this install",
                success=False)
            return
        try:
            from sylc.model_download_dialog import ModelDownloadDialog
            dialog = ModelDownloadDialog(
                manifest, sylc_models_download_dir(), self,
                ort_dir=self._synth3d_trt_dir(),
                # Downloads land in ONE directory; the player OPENS models from
                # three. TensorRT's engine cache is per graph, so the probe has
                # to build every graph the player can reach, not just the ones
                # this dialog would have written.
                models_dirs=_synth3d_models_dirs(),
                # An engine build saturates the GPU for minutes at a time and
                # the dialog's modality does not stop a decode. With a file
                # loaded the TensorRT action is refused outright rather than
                # competing with playback for the card.
                playback_active=bool(getattr(self, 'has_media', False)))
            dialog.exec()
        except Exception:
            logger.exception("[2D3D] model download dialog failed")
            self.show_3d_notification("Could not open the model downloader",
                                      success=False)
            return
        # The dialog wrote to disk; every gate re-reads it. The BUTTON's own
        # enabled state and tooltip are computed by _update_3d_button_state,
        # not by the menu refresh -- without this second call a successful
        # download leaves the button reading "unavailable" until the next
        # media event happens to recompute it.
        self._update_synth3d_menu_state()
        self._update_3d_button_state()

    def _update_synth3d_menu_state(self):
        ov = getattr(self, 'controls_overlay', None)
        if ov is None or not hasattr(ov, 'synth3d_enable_action'):
            return
        supported, eligible = self._synth3d_supported(), self._synth3d_eligible()
        act = ov.synth3d_enable_action
        act.blockSignals(True)
        act.setChecked(bool(self._synth3d_active))
        act.setEnabled(supported and (eligible or self._synth3d_active))
        if not supported:
            # Naming the WRONG missing piece here sends the user to download
            # 3.67 GB that would not help: onnxruntime.dll is part of the
            # install, not of the packs.
            reason = self._synth3d_unsupported_reason()
            if reason == 'runtime':
                act.setToolTip(
                    "onnxruntime.dll is missing from this install — "
                    "downloading depth models will not enable this")
            elif reason == 'renderer':
                act.setToolTip(
                    "The native renderer is not available in this build")
            else:
                act.setToolTip(
                    "Depth models are not installed — use “Depth models” at "
                    "the top of this menu to download them")
        elif not eligible and not self._synth3d_active:
            act.setToolTip("Available on 2D video decoded by the native pipeline")
        else:
            act.setToolTip("Synthesize stereo 3D from this 2D video")
        act.blockSignals(False)
        ov.synth3d_depth_view_action.setEnabled(bool(self._synth3d_active))
        ov.synth3d_diagnostics_action.blockSignals(True)
        ov.synth3d_diagnostics_action.setChecked(
            bool(self._synth3d_diagnostics))
        ov.synth3d_diagnostics_action.setEnabled(bool(self._synth3d_active))
        ov.synth3d_diagnostics_action.blockSignals(False)
        auto_conv = getattr(ov, 'synth3d_auto_convergence_action', None)
        if auto_conv is not None:
            auto_conv.blockSignals(True)
            auto_conv.setChecked(bool(self._synth3d_auto_convergence))
            auto_conv.blockSignals(False)
        temporal = getattr(ov, 'synth3d_temporal_fill_action', None)
        if temporal is not None:
            temporal.blockSignals(True)
            temporal.setChecked(
                bool(getattr(self, '_synth3d_temporal_fill', False)))
            temporal.blockSignals(False)
        # Programmatic synchronisation must not look like a user's manual edit:
        # otherwise merely opening the menu would turn a preset into "Custom".
        for slider, value in (
                (ov.synth3d_strength_slider,
                 int(round(self._synth3d_strength * 10))),
                (ov.synth3d_convergence_slider,
                 int(round(self._synth3d_convergence * 100)))):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        selected = getattr(self, '_synth3d_preset', 'custom')
        for name, action in ov.synth3d_preset_actions.items():
            action.blockSignals(True)
            action.setChecked(name == selected)
            action.setEnabled(bool(self._synth3d_active) or name == 'custom')
            action.blockSignals(False)
        # Depth presets, unlike the comfort ones, are pickable with synthesis
        # idle too (the choice then applies to the next enable). A preset with no
        # export installed is greyed out rather than silently resolving to
        # another grid.
        active_depth = self._synth3d_depth_preset()
        for name, action in ov.synth3d_depth_preset_actions.items():
            available = synth3d_depth_preset_available(name)
            action.blockSignals(True)
            action.setChecked(name == active_depth)
            action.setEnabled(available)
            action.setToolTip(
                self._synth3d_depth_preset_tooltip(name) if available
                else "model file not installed")
            action.blockSignals(False)
        # The two submenu rows and the gateway row state the current
        # configuration on their own faces, so it reads without opening
        # anything. The overlay stays player-blind: it is told, it never looks.
        if hasattr(ov, 'set_synth3d_selection_summary'):
            ov.set_synth3d_selection_summary(active_depth, selected)
        if hasattr(ov, 'set_synth3d_models_summary'):
            packs = self._synth3d_installed_packs()
            ov.set_synth3d_models_summary(
                " + ".join(packs) + " installed" if packs else "none installed",
                bool(packs))
        st = None
        for w in self._display_widgets():
            r = getattr(w, '_r', None)
            if r is None:
                continue
            try:
                st = r.synth3d_status()
            except (AttributeError, TypeError, RuntimeError):
                st = None
            if st is not None:
                break
        self._update_synth3d_status_label(st)

    @staticmethod
    def _synth3d_installed_packs():
        """Which depth-model packs are on disk, best first.

        The gateway row reports the LIBRARY, not the selection: naming only the
        model the active preset happens to resolve to would call an install
        "Small" while Base sits beside it. Same disk lookup the preset gating
        already does on every menu refresh.
        """
        packs = []
        for label, candidates in (
                ("Base", ("da3_base_756.onnx", "da3_base_518.onnx")),
                ("Small", ("da3_small_756.onnx", "da3_small_518.onnx",
                           "da3_small.onnx"))):
            if synth3d_find_model(candidates) is not None:
                packs.append(label)
        return tuple(packs)

    def _update_synth3d_status_label(self, status):
        ov = getattr(self, 'controls_overlay', None)
        if ov is None or not hasattr(ov, 'set_synth3d_status'):
            return
        if not status:
            ov.set_synth3d_status("Engine: off")
            return
        fields = self._synth3d_status_fields(status)
        state = fields.get('state', 'off')
        if state == 'init':
            text = "Engine: warming up · shared session"
        elif state == 'running':
            provider = fields.get('provider', 'GPU')
            fps = fields.get('fps', '0.0')
            age = fields.get('age_ms', '-')
            clients = fields.get('clients', '1')
            # Name the active preset AND the grid the engine actually reports:
            # if the preset's own export were missing, the resolution would have
            # fallen back to another grid, and that shows here rather than
            # hiding behind the preset's name.
            grid = fields.get('grid')
            side = fields.get('side')
            preset = self._synth3d_depth_preset()
            if grid and grid not in ('0x0', '-'):
                preset = f"{preset} {grid}px"
            elif side and side not in ('0', '-'):
                preset = f"{preset} {side}px"
            text = (f"{provider} · {preset} · {fps} depth fps · {age} ms · "
                    f"{clients} surface{'s' if clients != '1' else ''}")
            if fields.get('lab') == 'active':
                try:
                    lab_px = float(fields.get('lab_px', '0'))
                    lab_p95 = 100.0 * float(fields.get('lab_p95', '0'))
                    text += (f" · Lab {lab_px:.1f}% / "
                             f"p95 actif {lab_p95:.0f}%")
                    lab_edge_px = float(fields.get('lab_edge_px', '0'))
                    if lab_edge_px >= 0.05:
                        text += f" / bord protégé {lab_edge_px:.1f}%"
                    pair_grid = fields.get('pair_grid', '0x0')
                    if (fields.get('pair') == 'sparse-source' and
                            pair_grid not in ('0x0', '-')):
                        text += f" / champ source {pair_grid}"
                except (TypeError, ValueError):
                    text += " · Lab active"
            elif fields.get('lab') == 'bypass':
                text += " · Lab bypassed"
            if fields.get('comfort') == 'calibrated':
                envelope = getattr(self, '_synth3d_comfort_envelope', None)
                if envelope is not None:
                    text += (
                        f" · confort {envelope.soft_vac_diopters:.2f}→"
                        f"{envelope.hard_vac_diopters:.2f} D")
                else:
                    text += " · confort calibré"
                try:
                    comfort_hit = float(fields.get('comfort_hit_pct', '0'))
                    comfort_loss = float(
                        fields.get('comfort_loss_p95_px', '0'))
                    if comfort_hit >= 0.05:
                        text += (f" / agit {comfort_hit:.1f}% "
                                 f"(−{comfort_loss:.1f}px p95)")
                    else:
                        text += " / agit 0.0%"
                except (TypeError, ValueError):
                    pass
        elif state == 'error':
            text = "Engine error · returned to 2D"
        else:
            text = "Engine: off"
        matte_service = getattr(self, '_synth3d_matte_service', None)
        if matte_service is not None:
            matte = matte_service.status()
            matte_state = matte.get('state', 'off')
            if matte_state in ('loading', 'ready'):
                text += " · matte warming up"
            elif matte_state == 'running':
                matte_side = int(matte.get('short_side', 0) or 0)
                matte_grid = f" @{matte_side}p" if matte_side > 0 else ""
                matte_fps = float(matte.get('fps', 0.0) or 0.0)
                matte_target = float(matte.get('target_fps', 0.0) or 0.0)
                matte_ms = float(matte.get('inference_ms', 0.0) or 0.0)
                if matte_target > 0.0:
                    text += (f" · matte {matte_fps:.1f}/{matte_target:.1f} "
                             f"fps · {matte_ms:.0f} ms{matte_grid}")
                else:
                    # Old service/status dictionaries remain readable.
                    text += (f" · matte {matte_fps:.1f} fps/"
                             f"{matte_ms:.0f} ms{matte_grid}")
            elif matte_state in ('degraded', 'error'):
                text += " · matte bypassed"
        contour_lock = getattr(self, '_synth3d_matte_advector', None)
        lock_status = getattr(contour_lock, 'status', None)
        if lock_status is not None:
            try:
                lock = lock_status()
                if lock.get('kind') == 'rejected':
                    text += " · contour repli sûr"
                elif lock.get('kind') in (
                        'luma-bidirectional', 'luma-contour-sparse'):
                    text += (f" · contour {100.0 * float(lock.get('confidence', 0.0)):.0f}%/"
                             f"{float(lock.get('lock_ms', 0.0)):.1f} ms")
                    sparse_pct = float(lock.get('sparse_pct', 0.0))
                    if sparse_pct >= 0.1:
                        text += f" / sparse {sparse_pct:.1f}%"
                    local_reject = float(lock.get('local_reject_pct', 0.0))
                    if local_reject >= 0.5:
                        text += f" / rejet local {local_reject:.1f}%"
            except (TypeError, ValueError):
                pass
        ov.set_synth3d_status(text)


__all__ = ['Synth3DCoordinationMixin', 'configure_synth3d_support']
