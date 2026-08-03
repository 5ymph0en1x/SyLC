# lookahead_scout.py
"""Look-ahead à deux filtres pour synth3d (spec 2026-08-03, oracle-validé).

Le futur décodé existe déjà (presentation_queue ~12 frames) ; ce scout le
REGARDE en miniature et publie des événements datés que le stabilisateur de
profondeur consommera comme préavis. Deux filtres, chacun à SA résolution :

  - CUTS à 64 px : distance TV d'histogrammes, seuil production 0,42 —
    précision/rappel 1,0 mesurés (oracle_results_pyramid.json) ;
  - TEMPÊTES de mouvement à 192 px : flot moyen normalisé en pixels de grille
    518, seuil absolu — 192 px est LA BARRIÈRE mesurée (rappel 0,88 pour 14 %
    du coût ; en dessous le signal s'effondre : 0,51 à 96 px). L'oracle a
    prouvé qu'aucune rampe n'annonce les tempêtes, même à pleine résolution :
    le préavis vient exclusivement de l'observation directe des frames
    futures, jamais d'une prédiction.

Zéro dépendance au pipeline : on lui pousse des lumas (n'importe quelle
taille, ordre décodage toléré via un petit tampon de réordonnancement pts) et
on l'interroge avec le pts PRÉSENTÉ. Conçu pour ~1,2 ms/frame (subsample
nearest + un flot 192 sur le pool C++ existant).
"""
import bisect
import threading

import numpy as np

try:
    import mvc_demuxer_cpp as _m
    _HAVE_FLOW = hasattr(_m, '_synth3d_estimate_flow_test')
except Exception:      # pragma: no cover - flow binding absent => cuts only
    _m = None
    _HAVE_FLOW = False

# La grille de production dans laquelle tous les flots sont normalisés.
GRID_REF = 518
CUT_SIDE = 64
STORM_SIDE = 192       # la barrière : ne JAMAIS descendre en dessous
CUT_TV = 0.42          # seuil production (scene_cut_threshold)
STORM_THR_PX = 4.5     # px grille 518 (P85 Oblivion : 4.452)
CALM_FRAMES = 3        # une tempête « éclate » après >= 3 frames calmes
CALM_FACTOR = 0.6
HIST_BINS = 64
REORDER_DEPTH = 5      # profondeur de réordonnancement B-frames tolérée
# Fenêtre de MAINTIEN post-événement (ms) : un événement reste rapporté
# (délai NÉGATIF) pendant ~3 frames après son pts. Raison d'être (auteur
# 03/08, « les reliquats qui débordent sur l'image d'APRÈS la coupe ») :
# le snap du worker peut enregistrer une frame après T1 — sans maintien, la
# purge à T1 relâchait la disparité pleine pendant cette frame-là, warpée
# avec une carte potentiellement trans-plan = franges rouge/cyan. Le
# maintien garde l'aplatissement (et le plancher de mouvement des tempêtes)
# jusqu'à ce que le relais post-snap soit certain.
EVENT_HOLD_MS = 120.0


def _subsample(luma, side):
    """Miniature side x side par plus-proche-voisin (indices précalculables,
    ~0.1 ms). L'oracle a validé les seuils sur du bilinéaire ffmpeg ; le
    nearest est plus bruité mais les DEUX filtres opèrent loin de leurs
    marges (cuts : TV 1.0/1.0 ; tempêtes : événements >= 4.5 px)."""
    h, w = luma.shape
    ys = (np.arange(side) * (h / side)).astype(np.intp)
    xs = (np.arange(side) * (w / side)).astype(np.intp)
    return luma[np.ix_(ys, xs)]


def _tv_distance(a, b):
    ha, _ = np.histogram(a, bins=HIST_BINS, range=(0.0, 1.0))
    hb, _ = np.histogram(b, bins=HIST_BINS, range=(0.0, 1.0))
    return 0.5 * float(np.abs(ha / max(1, ha.sum())
                              - hb / max(1, hb.sum())).sum())


class LookAheadScout:
    """Consomme les frames décodées, publie cut/tempête datés pts."""

    def __init__(self, storm_thr_px=STORM_THR_PX, cut_tv=CUT_TV,
                 flow_threads=4):
        self.storm_thr_px = float(storm_thr_px)
        self.cut_tv = float(cut_tv)
        self.flow_threads = int(flow_threads)
        self._pending = []          # frames poussées, triées pts, pas analysées
        self._prev = None           # dernière frame ANALYSÉE (pts, mini64, mini192)
        self._calm_run = CALM_FRAMES  # frames calmes consécutives (démarre calme)
        # Les événements sont ÉCRITS par le fil décodeur (push/_analyze) et
        # LUS/purgés par le fil GUI (next_events) : verrou obligatoire.
        self._events_lock = threading.Lock()
        self._cut_events = []       # pts ms croissants
        self._storm_events = []
        self.frames_analyzed = 0
        self.last_motion_px = 0.0

    # ------------------------------------------------------------------ push
    def push(self, luma, pts_ms):
        """Une frame DÉCODÉE (luma 2D uint8/float, n'importe quelle taille).
        L'ordre décodage est toléré : l'analyse ne consomme que lorsque
        REORDER_DEPTH frames plus récentes sont arrivées (ordre pts sûr)."""
        if luma is None or getattr(luma, 'ndim', 0) != 2:
            return
        a = np.asarray(luma)
        if a.dtype != np.float32:
            a = a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 1.0)
        entry = (float(pts_ms), _subsample(a, CUT_SIDE),
                 _subsample(a, STORM_SIDE) if _HAVE_FLOW else None)
        bisect.insort(self._pending, entry, key=lambda e: e[0])
        while len(self._pending) > REORDER_DEPTH:
            self._analyze(self._pending.pop(0))

    def flush(self):
        """Fin de flux / seek : analyse ce qui reste puis oublie tout."""
        while self._pending:
            self._analyze(self._pending.pop(0))

    def reset(self):
        self._pending.clear()
        self._prev = None
        self._calm_run = CALM_FRAMES
        with self._events_lock:
            self._cut_events.clear()
            self._storm_events.clear()

    # ------------------------------------------------------------- analysis
    def _analyze(self, entry):
        pts, mini64, mini192 = entry
        prev = self._prev
        self._prev = entry
        self.frames_analyzed += 1
        if prev is None or pts <= prev[0]:
            return
        # Filtre 1 : cut (64 px, histogrammes)
        is_cut = _tv_distance(prev[1], mini64) >= self.cut_tv
        if is_cut:
            with self._events_lock:
                self._cut_events.append(pts)
        # Filtre 2 : tempête (192 px, flot moyen normalisé)
        if _HAVE_FLOW and mini192 is not None and prev[2] is not None:
            fx, fy, q = _m._synth3d_estimate_flow_test(
                prev[2].ravel(), mini192.ravel(),
                STORM_SIDE, STORM_SIDE, self.flow_threads)
            mask = q > 0.08
            if mask.sum() < STORM_SIDE * STORM_SIDE // 200:
                mask = np.ones_like(q, dtype=bool)
            motion = float(np.hypot(fx, fy)[mask].mean()) * (GRID_REF / STORM_SIDE)
            self.last_motion_px = motion
            # Un cut produit un flot géant SANS être une tempête : il ne doit
            # ni déclencher l'événement ni casser le compteur de calme.
            if is_cut:
                return
            if motion >= self.storm_thr_px:
                if self._calm_run >= CALM_FRAMES:
                    with self._events_lock:
                        self._storm_events.append(pts)   # ONSET (éclatement)
                self._calm_run = 0
            elif motion < CALM_FACTOR * self.storm_thr_px:
                self._calm_run += 1

    # ------------------------------------------------------------- advisory
    def next_events(self, presented_pts_ms):
        """{'cut_in_ms': x|None, 'storm_in_ms': y|None} — délai vers le
        prochain événement, NÉGATIF jusqu'à EVENT_HOLD_MS après son pts (le
        consommateur maintient son effet à travers la frontière), puis purgé.
        Thread-safe vis-à-vis des pushes du fil décodeur."""
        out = {}
        p = float(presented_pts_ms)
        with self._events_lock:
            for name, events in (('cut_in_ms', self._cut_events),
                                 ('storm_in_ms', self._storm_events)):
                i = bisect.bisect_right(events, p - EVENT_HOLD_MS)
                if i:
                    del events[:i]
                out[name] = (events[0] - p) if events else None
        return out
