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
    endOfStream = Signal()
    decodeFailed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._src = None
        self._mode = None            # 'sbs' | 'tab' | None (2D -> L duplique)
        self._inverted = False
        self._stop = False
        self._paused = False
        self._seek_req = None        # ms | None
        self.clock_offset_provider = None   # callable -> ms de l'horloge maitre

    def configure(self, source, mode, half, inverted, swap_eyes=False):
        # `half` n'affecte pas le split (les moities sont ce qu'elles sont);
        # l'upscale half-res = magnification du sampler du renderer.
        self._src, self._mode = source, mode
        self._inverted = bool(inverted) ^ bool(swap_eyes)

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

    def request_stop(self):
        self._stop = True

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
        anchor_wall = None           # perf_counter a l'ancre
        anchor_pts = 0.0             # pts (ms) a l'ancre
        last_interval_s = 1.0 / 24.0     # repli quand une frame n'a pas de pts
        last_pts_ms = None
        nopts_count = 0
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
                    time.sleep(0.01)
                    continue
                out = self._read_next_pair()
                if out is None:
                    if getattr(self._src, 'failed', False):
                        self.decodeFailed.emit("streak d'erreurs decode")
                    else:
                        self.endOfStream.emit()
                    return
                left, right, pts_ms = out
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
                    # FREE-RUN pacing: the video is paced PURELY by its own pts cadence, the
                    # correct 23.976 fps real-time clock. The mpv audio time-pos exposed via
                    # clock_offset_provider is a cache drained by a GIL-starved Python callback
                    # and was MEASURED advancing at only ~0.4x real time while ALWAYS trailing
                    # ([HEVC-METER] master rate ~410 ms/s; drift always positive). The former
                    # design hard-snapped the anchor to that cache whenever drift > 100 ms,
                    # which fired ~4x/s and yanked the video clock ~150 ms BACKWARD each time —
                    # THAT was the "lecture horriblement saccadee". Any drift-based re-anchor to
                    # this cache is unsafe: the ~0.4x trailing makes drift grow without bound,
                    # so every threshold eventually triggers a large backward jump. mpv's real
                    # audio device runs at 1x on its own C threads (GIL-independent) and the
                    # video pts is also 1x, so a pure free-run stays in A/V sync WITHOUT chasing
                    # the broken cache. Explicit user seeks still re-anchor via the _seek_req
                    # path above (both mpv and this thread are driven by the same seek).
                    due = anchor_wall + (float(pts_ms) - anchor_pts) / 1000.0
                    delay = min(due - now, 0.5)     # borne anti-blocage (pts aberrant)
                    if _diag and (due - now) < 0.0:
                        _m_late += 1
                else:
                    # Pas de pts: cadence de repli (dernier intervalle valide)
                    # au lieu d'une rafale non pacee.
                    nopts_count += 1
                    if nopts_count in (1, 100):
                        logger.warning(f"[HEVC] frame sans pts (#{nopts_count}): "
                                       f"cadence de repli {last_interval_s * 1000:.0f} ms")
                    delay = last_interval_s
                while delay > 0 and not self._stop and self._seek_req is None:
                    step = min(delay, 0.01)
                    time.sleep(step)
                    delay -= step
                # Recheck peremption AVANT emission, quel que soit le pts:
                # une frame decodee avant un seek ne part jamais.
                if self._seek_req is not None or self._stop:
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
                self.frameYUVReady.emit(left, right)
            except Exception as e:
                logger.error(f"[HEVC] exception inattendue du thread: {e}")
                self.decodeFailed.emit(f"exception thread: {e}")
                return
