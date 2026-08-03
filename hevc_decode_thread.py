# -*- coding: utf-8 -*-
"""Thread de decode HEVC — separe du pipeline MVC (NAL-oriented). Boucle
read_frame -> split stereo -> frameYUVReady(left, right): MEME contrat de signal
que MVCDecoderThread, tout l'aval du player (framepack, sous-titres, V60) est
inchange. Pacing par pts sur horloge monotone re-ancrable (seek/pause), avec
correction douce optionnelle via clock_offset_provider (horloge audio mpv).

Mode 'mvhevc' (MV-HEVC 2 vues, spec §5) : boucle sur read_view_pair() au lieu
de read_frame()+split -- la source assigne deja les vues left/right par
view_id (aucun split geometrique a faire ici), le pacing PTS/EOF/seek/stop
en aval reste identique (factorise dans _read_next_pair)."""
import os
import time
import logging

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


def _pct(sorted_vals, q):
    """p-quantile of an already-sorted list (nearest-rank). 0.0 on empty."""
    if not sorted_vals:
        return 0.0
    idx = int(q * (len(sorted_vals) - 1) + 0.5)
    if idx >= len(sorted_vals):
        idx = len(sorted_vals) - 1
    return sorted_vals[idx]


def split_packed_stereo(planes, mode):
    """Split zero-copy d'une frame packee (miroir de
    SyLC_3D_Player._split_packed_stereo, dtype-agnostique: uint8 ET uint16)."""
    y, u, v = planes
    if mode == 'sbs':
        wy, wc = y.shape[1] // 2, u.shape[1] // 2
        return ((y[:, :wy], u[:, :wc], v[:, :wc]),
                (y[:, wy:wy * 2], u[:, wc:wc * 2], v[:, wc:wc * 2]))
    hy, hc = y.shape[0] // 2, u.shape[0] // 2
    return ((y[:hy], u[:hc], v[:hc]),
            (y[hy:hy * 2], u[hc:hc * 2], v[hc:hc * 2]))


class HevcDecodeThread(QThread):
    frameYUVReady = Signal(object, object)
    # Exact per-frame PTS for content-temporal consumers. The legacy two-arg
    # signal remains available to callers that only need pixels.
    frameYUVTimedReady = Signal(object, object, object)  # (left, right, pts_ms)
    endOfStream = Signal()
    decodeFailed = Signal(str)
    # PGS streaming — MEME contrat de signaux que MVCDecoderThread, pour que le
    # player branche l'aval (SubtitleManager.on_pgs_data, combo pistes) a
    # l'identique. Les blocs sont collectes par la source au fil du demux (donc
    # legerement en avance sur l'horloge) — aucune extraction de fichier.
    pgsDataReady = Signal(bytes, float)          # (segments bruts, pts en s)
    subtitleTracksDetected = Signal(list)        # [{trackNumber, codecId, isPGS, ...}]
    # Position of the frame just emitted, in SECONDS, throttled to ~4 Hz.
    # This is the UI timeline's only live position source for files where the
    # mpv shell has nothing to play (no audio track): frameYUVReady carries no
    # pts, and mpv's time-pos stays dead for such media.
    positionChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._src = None
        self._mode = None            # 'sbs' | 'tab' | None (2D -> L duplique)
        self._inverted = False
        self._stop = False
        self._paused = False
        self._seek_req = None        # ms | None
        self.clock_offset_provider = None   # callable -> ms de l'horloge maitre
        # The standalone thread keeps its historical free-running defaults.  The
        # player opts into the stricter audio-master contract in configure().
        self._require_master_clock = False
        self._bounded_delivery = False
        self._presentation_pending = False
        self._av_sync_offset_ms = 0.0
        self._sync_drop_count = 0
        self._backpressure_drop_count = 0

    def configure(self, source, mode, half, inverted, swap_eyes=False,
                  start_paused=False, require_master_clock=False,
                  bounded_delivery=False, av_sync_offset_s=0.0):
        # `half` n'affecte pas le split (les moities sont ce qu'elles sont);
        # l'upscale half-res = magnification du sampler du renderer.
        self._src, self._mode = source, mode
        self._inverted = bool(inverted) ^ bool(swap_eyes)
        self._paused = bool(start_paused)
        self._require_master_clock = bool(require_master_clock)
        self._bounded_delivery = bool(bounded_delivery)
        self._presentation_pending = False
        self._av_sync_offset_ms = max(-1000.0, min(
            2000.0, float(av_sync_offset_s) * 1000.0))
        self._sync_drop_count = 0
        self._backpressure_drop_count = 0

    def set_mode(self, mode):
        """Live stereo-mode switch from the UI combo ('sbs' | 'tab' | None). A plain
        attribute store is GIL-atomic and consumed by run() on the NEXT frame — the loop
        reads self._mode once per iteration, so no lock is needed and no decoder restart.

        'mvhevc' is excluded from live switching (spec §5): the two decoded views ARE
        the stereo pair (no geometric split to toggle), and the SBS/TAB combo in mvhevc
        source files instead re-targets the RENDERER's presentation, not this source-side
        mode. So: (a) while currently in mvhevc, any incoming request is a no-op (single
        log); (b) switching INTO mvhevc live is refused too -- it is only entered via
        configure() (fresh source re-open with view_ids set), never as a live toggle."""
        if mode == 'mvhevc' and self._mode != 'mvhevc':
            logger.info("[HEVC] set_mode refuse: mvhevc n'est activable que via configure()")
            return
        if self._mode == 'mvhevc':
            logger.info("[HEVC] set_mode ignore en mvhevc (la presentation se regle au renderer)")
            return
        self._mode = mode

    def seek_to(self, ms):
        self._seek_req = float(ms)

    def set_paused(self, paused):
        self._paused = bool(paused)

    def set_master_clock_required(self, required):
        """Switch between audio-master and PTS/wall-clock pacing.

        This is used when mpv discovers that a nominally audio-backed session is
        actually video-only.  It is intentionally a single atomic attribute
        store: the decode loop samples it once per iteration.
        """
        self._require_master_clock = bool(required)

    def adjust_av_offset(self, delta_s):
        """Adjust the HEVC video trim (positive means delay video)."""
        self._av_sync_offset_ms = max(-1000.0, min(
            2000.0, self._av_sync_offset_ms + float(delta_s) * 1000.0))
        return self._av_sync_offset_ms / 1000.0

    def presentation_consumed(self):
        """Acknowledge the sole queued GUI presentation.

        With bounded delivery enabled the decoder never queues a second UHD
        frame while the GUI still owns the first one.  It continues decoding on
        the media timeline and simply discards obsolete presentations.
        """
        self._presentation_pending = False

    def sync_stats(self):
        return {
            'sync_drops': int(self._sync_drop_count),
            'backpressure_drops': int(self._backpressure_drop_count),
            'presentation_pending': bool(self._presentation_pending),
        }

    def _master_clock_ms(self):
        provider = self.clock_offset_provider
        if provider is None:
            return None
        try:
            value = provider()
            if value is None:
                return None
            value = float(value)
            if value < 0.0 or value != value:  # negative / NaN
                return None
            return value
        except Exception:
            return None

    def request_stop(self):
        self._stop = True

    def set_subtitle_track(self, track_number):
        """Route la selection de piste PGS vers la source (0/None desactive).
        Store d'attribut cote source, GIL-atomique — meme patron que set_mode."""
        src = self._src
        if src is not None and hasattr(src, 'set_subtitle_track'):
            src.set_subtitle_track(track_number)

    def _poll_subtitles(self):
        """Draine la file PGS de la source et emet pgsDataReady par bloc.
        Appele depuis la boucle de decode, apres chaque lecture (les blocs sont
        collectes au READ, contrat identique a mvc_decoder._poll_subtitles)."""
        src = self._src
        if src is None or not hasattr(src, 'has_subtitle_data'):
            return
        while src.has_subtitle_data():
            ok, block = src.read_subtitle_block()
            if not ok or block is None:
                break
            self.pgsDataReady.emit(bytes(block['data']),
                                   float(block['timestampMs']) / 1000.0)

    def _read_next_pair(self):
        """Lit la prochaine paire (left, right, pts_ms) selon le mode courant, ou None
        en EOF propre / streak d'erreurs (self._src.failed discrimine les deux, inchange
        -- le tri est fait par l'appelant exactement comme avant). Point d'unification
        mvhevc/legacy pour que le pacing/emit dans run() reste single-path (requirement
        MV-3 #1): seule la LECTURE differe, tout l'aval (pacing pts, EOF/decodeFailed,
        seek staleness, stop) est identique quel que soit le mode.

        mode == 'mvhevc': read_view_pair() -- deja apparie par pts et assigne left/right
        par view_id COTE SOURCE (spec §4/§5); AUCUN split ici. self._inverted reste
        applicable en swap optionnel par-dessus (meme convention que le split empaquete).

        sinon (sbs/tab/2D): read_frame() + split_packed_stereo, comportement inchange."""
        if self._mode == 'mvhevc':
            out = self._src.read_view_pair()
            if out is None:
                return None
            left, right, pts_ms = out
            if self._inverted:
                left, right = right, left
            return left, right, pts_ms
        out = self._src.read_frame()
        if out is None:
            return None
        planes, pts_ms = out
        if self._mode in ('sbs', 'tab'):
            left, right = split_packed_stereo(planes, self._mode)
            if self._inverted:
                left, right = right, left
        else:
            left = right = planes           # 2D: comme le MVC mono-vue
        return left, right, pts_ms

    def run(self):
        # Pistes PGS connues des l'open de la source : publiees au demarrage du
        # thread (miroir de la detection MVC), le player remplit le combo.
        try:
            src = self._src
            if src is not None and hasattr(src, 'get_subtitle_tracks'):
                tracks = src.get_subtitle_tracks()
                if tracks:
                    self.subtitleTracksDetected.emit(tracks)
        except Exception as e:
            logger.warning(f"[HEVC] detection pistes sous-titres echouee: {e}")
        anchor_wall = None           # perf_counter a l'ancre
        anchor_pts = 0.0             # pts (ms) a l'ancre
        last_interval_s = 1.0 / 24.0     # repli quand une frame n'a pas de pts
        last_pts_ms = None
        nopts_count = 0
        last_pos_emit_ms = None      # throttle de positionChanged (~4 Hz)
        # --- [HEVC-METER] instrumentation (SYLC_HEVC_DIAG=1, silent otherwise) ---
        _diag = os.environ.get("SYLC_HEVC_DIAG") == "1"
        _m_emit = []             # emit-to-emit intervals (ms) over the 5 s window
        _m_last_emit = None
        _m_reanchor = 0          # re-anchor count in window (should be RARE)
        _m_late = 0              # frames pacing found already overdue (delay<0) in window
        _m_win = time.perf_counter()
        # master-clock (mpv audio time-pos) cadence probe
        _m_master_prev = None    # last DISTINCT master value seen (ms)
        _m_master_changes = 0    # how many times the cache actually changed in window
        _m_master_lo = None
        _m_master_hi = None
        while not self._stop:
            try:
                # M2: handle a pending seek INSIDE the try so a native seek that raises
                # lands in decodeFailed below (→ mpv fallback) instead of silently killing
                # the thread. Seeks are still processed before the pause gate, so a seek
                # requested while paused re-anchors immediately (semantics unchanged).
                if self._seek_req is not None:
                    target, self._seek_req = self._seek_req, None
                    if not self._src.seek(target):
                        logger.warning(f"[HEVC] seek({target}) refuse")
                    anchor_wall, anchor_pts = None, target
                    last_pts_ms = None
                if self._paused:
                    # Re-anchor at resume: pacing is wall-anchored (due =
                    # anchor_wall + pts), so an untouched anchor turns the
                    # whole pause into a debt of past-due frames — on resume
                    # the video bursts at decode speed until it has caught up
                    # exactly the paused duration, landing that far ahead of
                    # the audio. Dropping the anchor makes the first resumed
                    # frame re-anchor at the current wall time instead.
                    anchor_wall = None
                    time.sleep(0.01)
                    continue
                # Audio-backed HEVC must not consume even one frame before mpv
                # has published a usable master clock.  This is the startup
                # barrier that prevents video from running ahead while TrueHD
                # initializes.  Video-only files leave this requirement off.
                if self._require_master_clock and self._master_clock_ms() is None:
                    anchor_wall = None
                    time.sleep(0.005)
                    continue
                out = self._read_next_pair()
                if out is None:
                    if getattr(self._src, 'failed', False):
                        self.decodeFailed.emit("streak d'erreurs decode")
                    else:
                        self.endOfStream.emit()
                    return
                left, right, pts_ms = out
                # Sous-titres PGS : drain apres CHAQUE lecture, hors du try de
                # la video ? Non — dans le try global, mais isole : un pepin de
                # sous-titre ne doit jamais casser la lecture video (meme regle
                # que mvc_decoder ligne ~2975).
                try:
                    self._poll_subtitles()
                except Exception as exc:
                    logger.warning(f"[HEVC] poll sous-titres en echec: {exc}")
                if pts_ms >= 0:
                    now = time.perf_counter()
                    if anchor_wall is None:
                        anchor_wall, anchor_pts = now, float(pts_ms)
                    if last_pts_ms is not None and pts_ms > last_pts_ms:
                        last_interval_s = min(0.5, (pts_ms - last_pts_ms) / 1000.0)
                    last_pts_ms = pts_ms
                    if _diag and self.clock_offset_provider is not None:
                        # master-clock cadence probe (diagnostic only)
                        try:
                            master = self.clock_offset_provider()
                            if master is not None:
                                if master != _m_master_prev:
                                    _m_master_changes += 1
                                    _m_master_prev = master
                                if _m_master_lo is None or master < _m_master_lo:
                                    _m_master_lo = master
                                if _m_master_hi is None or master > _m_master_hi:
                                    _m_master_hi = master
                        except Exception:
                            pass
                    # Prefer the reconstructed mpv AUDIO clock.  Unlike the old raw
                    # observer cache this clock is extrapolated at 1x between callbacks,
                    # so we never hard-snap backwards to a GIL-starved sample.  A frame
                    # ahead waits; a frame over one frame late is discarded until the
                    # video content catches up with audio.
                    master_ms = self._master_clock_ms()
                    if master_ms is not None:
                        anchor_wall = None
                        target_ms = float(pts_ms) + self._av_sync_offset_ms
                        error_ms = target_ms - master_ms
                        frame_ms = max(1.0, last_interval_s * 1000.0)
                        late_drop_ms = max(45.0, min(100.0, frame_ms * 1.25))
                        if error_ms < -late_drop_ms:
                            self._sync_drop_count += 1
                            if _diag:
                                _m_late += 1
                            continue
                        delay = min(max(0.0, error_ms / 1000.0), 0.5)
                    else:
                        # Video-only / unavailable master: preserve the proven PTS
                        # wall-clock pacing used by the standalone HEVC tests.
                        if anchor_wall is None:
                            anchor_wall, anchor_pts = now, float(pts_ms)
                        due = anchor_wall + (float(pts_ms) - anchor_pts) / 1000.0
                        delay = min(due - now, 0.5)  # borne anti-blocage (pts aberrant)
                    if _diag and master_ms is None and (due - now) < 0.0:
                        # `due` exists only in wall-clock mode.  Audio-master
                        # lateness is counted at the drop decision above.
                        _m_late += 1
                else:
                    # Pas de pts: cadence de repli (dernier intervalle valide)
                    # au lieu d'une rafale non pacee.
                    nopts_count += 1
                    if nopts_count in (1, 100):
                        logger.warning(f"[HEVC] frame sans pts (#{nopts_count}): "
                                       f"cadence de repli {last_interval_s * 1000:.0f} ms")
                    delay = last_interval_s
                while (delay > 0 and not self._stop and self._seek_req is None
                       and not self._paused):
                    step = min(delay, 0.01)
                    time.sleep(step)
                    delay -= step
                # Recheck peremption AVANT emission, quel que soit le pts:
                # une frame decodee avant un seek ne part jamais.
                if self._seek_req is not None or self._stop or self._paused:
                    continue
                if _diag:
                    _te = time.perf_counter()
                    if _m_last_emit is not None:
                        _m_emit.append((_te - _m_last_emit) * 1000.0)
                    _m_last_emit = _te
                    if (_te - _m_win) >= 5.0:
                        _s = sorted(_m_emit)
                        _win_s = _te - _m_win
                        _madv = ((_m_master_hi - _m_master_lo)
                                 if (_m_master_lo is not None and _m_master_hi is not None) else 0.0)
                        _mrate = (_madv / _win_s) if _win_s > 0 else 0.0
                        logger.info(
                            f"[HEVC-METER] thread emit ms p50={_pct(_s, 0.5):.1f} "
                            f"p99={_pct(_s, 0.99):.1f} max={(_s[-1] if _s else 0.0):.1f} "
                            f"n={len(_s)} reanchors={_m_reanchor} late={_m_late} | "
                            f"master changes={_m_master_changes} adv={_madv:.0f}ms "
                            f"rate={_mrate:.0f}ms/s")
                        _m_emit = []
                        _m_reanchor = 0
                        _m_late = 0
                        _m_win = _te
                        _m_master_changes = 0
                        _m_master_lo = None
                        _m_master_hi = None
                if self._bounded_delivery and self._presentation_pending:
                    self._backpressure_drop_count += 1
                else:
                    if self._bounded_delivery:
                        self._presentation_pending = True
                    self.frameYUVReady.emit(left, right)
                    self.frameYUVTimedReady.emit(left, right, float(pts_ms))
                # Timeline feed: throttled (250 ms of pts progress, or any
                # backward jump = seek) so the GUI never sees a per-frame storm.
                if pts_ms >= 0 and (last_pos_emit_ms is None
                                    or pts_ms - last_pos_emit_ms >= 250
                                    or pts_ms < last_pos_emit_ms):
                    last_pos_emit_ms = pts_ms
                    self.positionChanged.emit(pts_ms / 1000.0)
            except Exception as e:
                logger.error(f"[HEVC] exception inattendue du thread: {e}")
                self.decodeFailed.emit(f"exception thread: {e}")
                return
