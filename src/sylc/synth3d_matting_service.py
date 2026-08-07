"""Asynchronous MatAnyone 2 bridge for the native 2D -> 3D renderer.

The player stays on Python 3.14 while the PyTorch runtime lives in an isolated
``uv`` environment.  Frames cross the process boundary through a tiny binary
stdio protocol.  Submission is latest-only: decoder/presentation pacing is
never allowed to wait for matting.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np


log = logging.getLogger("SyLC.MatAnyone2")


def _sample_peak(plane: np.ndarray) -> int:
    """Return the stored peak for an 8/10-bit decoder plane.

    Software yuv420p10 frames carry their 10 useful bits in bits 0..9 while
    DXVA P010 carries the same code in bits 6..15.  Treating both uint16
    layouts as 0..1023 turns every hardware frame white and destroys both
    scene-cut and contour-lock evidence.
    """
    src = np.asarray(plane)
    if src.dtype == np.uint8:
        return 255
    if src.dtype != np.uint16:
        raise TypeError(f"unsupported YUV plane dtype: {src.dtype}")
    if not src.size:
        return 1023
    # A regular sample is enough to distinguish the layouts and avoids a full
    # UHD reduction in the presentation thread. Limited-range P010 black is
    # code 64 << 6 (=4096), so real P010 frames cross this threshold too.
    sy = max(1, src.shape[0] // 64)
    sx = max(1, src.shape[1] // 64)
    observed = int(np.max(src[::sy, ::sx]))
    return 65472 if observed > 4095 else 1023


def _plane_to_u8(plane: np.ndarray, sample_peak: Optional[int] = None) -> np.ndarray:
    src = np.asarray(plane)
    if src.ndim != 2:
        raise ValueError("YUV planes must be two-dimensional")
    if src.dtype == np.uint8:
        return src
    if src.dtype == np.uint16:
        peak = int(sample_peak) if sample_peak is not None else _sample_peak(src)
        return np.clip((src.astype(np.float32) * (255.0 / float(peak))) + 0.5,
                       0, 255).astype(np.uint8)
    raise TypeError(f"unsupported YUV plane dtype: {src.dtype}")


@dataclass(frozen=True)
class MatAnyone2Runtime:
    python: Path
    worker: Path
    checkpoint: Path
    seed_checkpoint: Path
    model_root: Path

    @classmethod
    def discover(cls, source_root: str | os.PathLike[str]) -> Optional["MatAnyone2Runtime"]:
        """Find a complete offline runtime; never downloads during playback."""
        source = Path(source_root).resolve()
        exe_root = Path(os.path.abspath(os.path.dirname(os.sys.argv[0])))
        module_root = Path(__file__).resolve().parent
        roots = []
        for root in (exe_root, source, module_root):
            if root not in roots:
                roots.append(root)

        worker_env = os.environ.get("SYLC_MATANYONE2_WORKER", "").strip()
        worker_candidates = ([Path(worker_env)] if worker_env else []) + [
            root / "synth3d_matanyone2_worker.py" for root in roots
        ]
        python_env = os.environ.get("SYLC_MATANYONE2_PYTHON", "").strip()
        python_candidates = ([Path(python_env)] if python_env else []) + [
            root / "models_dev_nc" / "matanyone2_runtime" / ".venv" / "Scripts" / "python.exe"
            for root in roots
        ]
        model_env = os.environ.get("SYLC_MATANYONE2_MODELS", "").strip()
        model_roots = ([Path(model_env)] if model_env else []) + [
            root / "models_dev_nc" / "matanyone2" for root in roots
        ]

        worker = next((p.resolve() for p in worker_candidates if p.is_file()), None)
        python = next((p.resolve() for p in python_candidates if p.is_file()), None)
        for model_root in model_roots:
            checkpoint = model_root / "matanyone2.pth"
            seed = model_root / "lraspp_mobilenet_v3_large-d234d4ea.pth"
            if python and worker and checkpoint.is_file() and seed.is_file():
                return cls(python, worker, checkpoint.resolve(), seed.resolve(),
                           model_root.resolve())
        return None


@dataclass(frozen=True)
class MatteFrame:
    sequence: int
    generation: int
    pts_ms: float
    alpha: np.ndarray
    inference_ms: float
    seeded: bool
    scene_cut: bool
    decode_ms: float = 0.0
    upload_ms: float = 0.0
    model_ms: float = 0.0
    readback_ms: float = 0.0
    # A network result has revision 0.  A contour-locked projection gets one
    # revision per displayed PTS so the widget really uploads the transported
    # alpha instead of keeping the first projection of the same network mask.
    transport_revision: int = 0
    tracking_confidence: float = 1.0
    transport_kind: str = "network"
    # Per-pixel registration reliability in CURRENT-frame matte coordinates.
    # None denotes a fresh network observation and is therefore fully trusted.
    # Transported mattes retain the forward/backward and photometric evidence
    # that used to be collapsed into tracking_confidence alone.
    reliability: Optional[np.ndarray] = None

    def matches(self, pts_ms: float, max_pts_delta_ms: float) -> bool:
        if not math.isfinite(pts_ms) or not math.isfinite(self.pts_ms):
            return True
        # Never project a mask from the future onto an older frame after a seek.
        delta = pts_ms - self.pts_ms
        return -40.0 <= delta <= max_pts_delta_ms


class MatteAdvector:
    """Transport and lock the asynchronous alpha to the displayed frame.

    MatAnyone2 tourne à ~5 Hz sur un film à 24 : l'alpha a jusqu'à 5 frames
    de retard.  Codec MVs cannot solve that robustly: HEVC does not expose
    them in the current player, H.264 prediction blocks are not dense optical
    flow, and the signal used to arrive after presentation.  The primary path
    therefore estimates a bidirectional dense flow between the exact luma
    submitted with the matte and the luma being displayed.  The old codec-MV
    path remains only as a compatibility fallback.

    A forward/backward plus photometric score makes the feature fail closed:
    an old or untrackable matte is removed instead of protecting a contour at
    the wrong location.  This path is codec-independent and resets with the
    existing shot/seek epoch.
    """

    MAX_STEPS = 12
    MAX_FRAMES = 64
    # The network result is already several displayed frames old when it
    # reaches the player.  Keep enough history to compensate the measured
    # 300-420 ms worker latency, but never let a mask become long-lived scene
    # memory.  Older alpha is removed, even if optical flow still returns a
    # numerically plausible field.
    DEFAULT_MAX_TRANSPORT_MS = 520.0
    # Without current-frame evidence an asynchronous observation is safe only
    # within roughly one 24 fps frame. Older data is unknown, never background.
    FRESH_WITHOUT_TRANSPORT_MS = 45.0

    def __init__(self):
        self._steps = collections.deque(maxlen=self.MAX_STEPS)
        self._frames = collections.deque(maxlen=self.MAX_FRAMES)
        self._cache_key = None
        self._cache = None
        try:
            side = int(os.environ.get("SYLC_SYNTH3D_CONTOUR_LOCK_SIDE", "288"))
        except ValueError:
            side = 288
        self._track_side = max(128, min(512, side))
        try:
            threshold = float(os.environ.get(
                "SYLC_SYNTH3D_CONTOUR_LOCK_MIN_CONFIDENCE", "0.32"))
        except ValueError:
            threshold = 0.32
        self._min_confidence = max(0.05, min(0.95, threshold))
        try:
            max_age = float(os.environ.get(
                "SYLC_SYNTH3D_CONTOUR_LOCK_MAX_AGE_MS",
                str(self.DEFAULT_MAX_TRANSPORT_MS)))
        except ValueError:
            max_age = self.DEFAULT_MAX_TRANSPORT_MS
        self._max_transport_ms = max(80.0, min(700.0, max_age))
        self._dense_enabled = os.environ.get(
            "SYLC_SYNTH3D_CONTOUR_LOCK", "1") != "0"
        self._dis = None
        self._last_lock_ms = 0.0
        self._last_confidence = 0.0
        self._last_local_reject_pct = 0.0
        self._last_sparse_pct = 0.0
        self._last_kind = "idle"
        self._last_age_ms = 0.0
        self._last_error = ""
        self._rejected = 0

    @staticmethod
    def _u8_luma(y_plane: np.ndarray, sample_peak: Optional[int] = None) -> np.ndarray:
        y = np.asarray(y_plane)
        if y.ndim != 2 or y.size == 0:
            raise ValueError("luma plane must be a non-empty 2-D array")
        return _plane_to_u8(y, sample_peak)

    def note_frame(self, y_plane: np.ndarray, pts_ms: float,
                   sample_peak: Optional[int] = None) -> None:
        """Remember the exact displayed luma on a small tracking grid.

        The resize/copy happens synchronously while the decoder-owned plane is
        valid.  At 1920x816 the default grid is 678x288; DIS then costs only a
        few milliseconds and both H.264 and H.265 use the identical path.
        """
        if not self._dense_enabled:
            return
        try:
            pts = float(pts_ms)
            if not math.isfinite(pts) or pts < 0.0:
                return
            y = np.asarray(y_plane)
            if (y.ndim != 2 or y.size == 0 or
                    y.dtype not in (np.uint8, np.uint16)):
                raise ValueError("unsupported luma plane")
            h, w = y.shape
            scale = min(1.0, self._track_side / float(min(h, w)))
            tw = max(16, int(round(w * scale / 2.0)) * 2)
            th = max(16, int(round(h * scale / 2.0)) * 2)
            small = cv2.resize(
                y, (tw, th), interpolation=(cv2.INTER_AREA
                                             if scale < 1.0
                                             else cv2.INTER_LINEAR))
            # P010 conversion happens after resize. Converting the 1920x816
            # source to float first cost several needless milliseconds on the
            # HEVC path; only the small tracking grid needs to become uint8.
            small = np.ascontiguousarray(self._u8_luma(small, sample_peak))
            if self._frames:
                previous_pts = self._frames[-1][0]
                if pts < previous_pts - 1.0 or pts > previous_pts + 1000.0:
                    self._frames.clear()
                    self._cache_key = None
                    self._cache = None
                elif abs(pts - previous_pts) <= 0.05:
                    self._frames[-1] = (pts, small)
                    return
            self._frames.append((pts, small))
        except Exception:
            # Guidance must never become a playback dependency.
            return

    def note_hints(self, hints) -> None:
        """Un pas de mouvement (dict motionHintsReady) : blocs quart-de-pel
        par frame d'affichage -> px source par frame."""
        try:
            bw = int(hints['blocks_w'])
            bh = int(hints['blocks_h'])
            mv = np.asarray(hints['mv_xy'], np.int16).reshape(bh, bw, 2)
            fx = mv[..., 0].astype(np.float32) / 4.0
            fy = mv[..., 1].astype(np.float32) / 4.0
            valid = hints.get('valid')
            if valid is not None:
                bad = np.asarray(valid, np.uint8).reshape(bh, bw) == 0
                fx[bad] = 0.0
                fy[bad] = 0.0
            self._steps.append((float(hints['pts_ms']), fx, fy,
                                int(hints['source_width']),
                                int(hints['source_height'])))
        except Exception:
            pass

    def reset(self) -> None:
        self._steps.clear()
        self._frames.clear()
        self._cache_key = None
        self._cache = None
        self._last_lock_ms = 0.0
        self._last_confidence = 0.0
        self._last_local_reject_pct = 0.0
        self._last_sparse_pct = 0.0
        self._last_kind = "reset"
        self._last_age_ms = 0.0
        self._last_error = ""

    def status(self) -> dict:
        return {
            "enabled": self._dense_enabled,
            "kind": self._last_kind,
            "confidence": self._last_confidence,
            "local_reject_pct": self._last_local_reject_pct,
            "sparse_pct": self._last_sparse_pct,
            "lock_ms": self._last_lock_ms,
            "age_ms": self._last_age_ms,
            "max_age_ms": self._max_transport_ms,
            "error": self._last_error,
            "rejected": self._rejected,
            "track_side": self._track_side,
        }

    def _frame_near(self, pts_ms: float):
        if not self._frames:
            return None
        pts, luma = min(self._frames, key=lambda item: abs(item[0] - pts_ms))
        # 30 ms covers the 23.976/24 fps half-frame rounding while rejecting
        # an unrelated decode epoch or a frame absent from the ring.
        return (pts, luma) if abs(pts - pts_ms) <= 30.0 else None

    def _ensure_dis(self):
        if self._dis is None:
            self._dis = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            self._dis.setUseSpatialPropagation(True)
        return self._dis

    @staticmethod
    def _masked_percentile(values: np.ndarray, mask: np.ndarray, q: float,
                           fallback: float) -> float:
        """Robust deterministic statistic without partitioning a whole frame."""
        # Confidence is a gate, not a published image metric.  A regular sample
        # of at most ~4096 points is stable here and saves 5-7 ms at 678x288.
        sampled = values[::2, ::2][mask[::2, ::2]]
        if not sampled.size:
            return float(fallback)
        if sampled.size > 4096:
            sampled = sampled[::int(math.ceil(sampled.size / 4096.0))]
        return float(np.percentile(sampled, q))

    @staticmethod
    def _advect_sparse_contour_alpha(
            source_alpha: np.ndarray, backward_flow: np.ndarray,
            contour_band: np.ndarray, local_confidence: np.ndarray):
        """Remap and validate only source-space contour support.

        The restored v5.2.1c warp consumes alpha but intentionally ignores the
        later experimental reliability channel.  Merely publishing a low B
        value therefore cannot remove a stale silhouette.  This function
        makes rejection effective while keeping its authority sparse: outside
        the old/current/swept contour support, network alpha is bit-exact and
        no full-matte inverse map is allocated.

        Within the band, uncertain foreground is feathered toward zero.  This
        withdraws the stale foreground-ownership claim; it does not replace
        depth or colour.  The historical depth/luma warp remains in charge at
        the conservatively eroded fringe.
        """
        alpha = np.asarray(source_alpha)
        backward = np.asarray(backward_flow, dtype=np.float32)
        band = np.asarray(contour_band)
        local = np.asarray(local_confidence, dtype=np.float32)
        if (alpha.ndim != 2 or alpha.dtype != np.uint8 or
                band.ndim != 2 or local.shape != band.shape or
                backward.shape != band.shape + (2,)):
            raise ValueError("invalid sparse contour guard inputs")

        h, w = alpha.shape
        th, tw = band.shape
        local = np.clip(local, 0.0, 1.0)
        safe_alpha = alpha.copy()
        reliability = np.full(alpha.shape, 255, dtype=np.uint8)

        # Work component-by-component in tracking space. Upsampling the band
        # and confidence over the complete 1528x640 matte still scanned a
        # million pixels to modify ~6%. A human silhouette normally produces
        # one or a few connected rings, so only their source-space rectangles
        # are resized and enumerated. Pathological noisy mattes fall back to
        # one full rectangle after 128 components to keep loop cost bounded.
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            band.astype(np.uint8), connectivity=8)
        if count <= 1:
            boxes = []
        elif count - 1 > 128:
            boxes = [(0, 0, tw, th)]
        else:
            boxes = [tuple(int(v) for v in stats[i, :4])
                     for i in range(1, count)]

        chunk_size = 30000
        for tx, ty, cw, ch in boxes:
            x0 = max(0, int(math.floor(tx * w / float(tw))))
            y0 = max(0, int(math.floor(ty * h / float(th))))
            x1 = min(w, int(math.ceil((tx + cw) * w / float(tw))))
            y1 = min(h, int(math.ceil((ty + ch) * h / float(th))))
            if x1 <= x0 or y1 <= y0:
                continue
            roi_size = (x1 - x0, y1 - y0)
            band_weight = cv2.resize(
                band[ty:ty + ch, tx:tx + cw].astype(np.float32),
                roi_size, interpolation=cv2.INTER_LINEAR)
            local_roi = cv2.resize(
                local[ty:ty + ch, tx:tx + cw], roi_size,
                interpolation=cv2.INTER_LINEAR)
            active = band_weight > 0.001
            ry, rx = np.nonzero(active)
            xx = rx + x0
            yy = ry + y0
            active_band = np.clip(band_weight[active], 0.0, 1.0)
            active_local = np.clip(local_roi[active], 0.0, 1.0)

            # Sample the inverse source-space field and alpha only at active
            # coordinates. OpenCV remap dimensions are signed-16-bit limited,
            # hence bounded 1xN chunks for long 4K silhouettes.
            for start in range(0, xx.size, chunk_size):
                stop = min(xx.size, start + chunk_size)
                xc = xx[start:stop]
                yc = yy[start:stop]
                bc = active_band[start:stop]
                lc = active_local[start:stop]
                track_x = ((xc.astype(np.float32) + 0.5) *
                           (tw / float(w)) - 0.5)
                track_y = ((yc.astype(np.float32) + 0.5) *
                           (th / float(h)) - 0.5)
                sampled_flow = cv2.remap(
                    backward, track_x[None, :], track_y[None, :],
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
                source_x = (xc.astype(np.float32) + sampled_flow[:, 0] *
                            (w / float(tw)))
                source_y = (yc.astype(np.float32) + sampled_flow[:, 1] *
                            (h / float(th)))
                transported = cv2.remap(
                    alpha, source_x[None, :], source_y[None, :],
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)[0]

                # A broad smooth knee avoids turning confidence into a new
                # binary stereo edge. Strong evidence is exactly neutral.
                alpha_confidence = np.clip(
                    (lc - 0.34) / (0.70 - 0.34), 0.0, 1.0)
                alpha_confidence *= alpha_confidence * (
                    3.0 - 2.0 * alpha_confidence)
                alpha_gate = 1.0 - bc * (1.0 - alpha_confidence)
                safe_alpha[yc, xc] = np.clip(
                    transported.astype(np.float32) * alpha_gate + 0.5,
                    0.0, 255.0).astype(np.uint8)

                # Preserve the richer channel for enhanced/debug consumers.
                reliability[yc, xc] = np.clip(
                    (1.0 - bc * (1.0 - lc)) * 255.0 + 0.5,
                    0.0, 255.0).astype(np.uint8)
        sparse_pct = 100.0 * float(np.mean(band))
        reject_pct = (100.0 * float(np.mean(local[band] < 0.50))
                      if np.any(band) else 0.0)
        return (np.ascontiguousarray(safe_alpha),
                np.ascontiguousarray(reliability),
                reject_pct, sparse_pct)

    def _advect_with_luma(self, matte: MatteFrame, video_time_ms: float):
        base_item = self._frame_near(float(matte.pts_ms))
        current_item = self._frame_near(float(video_time_ms))
        if base_item is None or current_item is None:
            return False, matte
        base_pts, base = base_item
        current_pts, current = current_item
        if base.shape != current.shape:
            return False, matte
        age_ms = current_pts - base_pts
        if age_ms < -1.0 or age_ms > self._max_transport_ms:
            return True, None
        if age_ms <= 1.0:
            return True, matte

        key = ("luma", matte.sequence, matte.generation,
               int(round(current_pts * 1000.0)))
        if key == self._cache_key:
            return True, self._cache

        dis = self._ensure_dis()
        # Backward flow answers exactly the inverse-warp question required by
        # remap: for each current pixel, where was it in the matte frame?
        backward = dis.calc(current, base, None)
        forward = dis.calc(base, current, None)
        th, tw = current.shape
        # Broadcasting avoids allocating two full meshgrids on every frame.
        map_x = backward[..., 0] + np.arange(tw, dtype=np.float32)[None, :]
        map_y = backward[..., 1] + np.arange(th, dtype=np.float32)[:, None]
        inside = ((map_x >= 0.0) & (map_x <= tw - 1.0) &
                  (map_y >= 0.0) & (map_y <= th - 1.0))

        base_at_current = cv2.remap(
            base, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        forward_at_base = cv2.remap(
            forward, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fb_error = np.hypot(backward[..., 0] + forward_at_base[..., 0],
                            backward[..., 1] + forward_at_base[..., 1])
        photo_error = cv2.absdiff(current, base_at_current).astype(
            np.float32) * (1.0 / 255.0)

        # Judge the registration primarily where it matters: in a narrow band
        # around the transported human silhouette.  A global term still catches
        # missed cuts and wholesale lighting/content changes.
        alpha_track = cv2.resize(matte.alpha, (tw, th),
                                 interpolation=cv2.INTER_LINEAR)
        warped_track = cv2.remap(
            alpha_track, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        base_foreground = (alpha_track >= 48).astype(np.uint8)
        foreground = (warped_track >= 48).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        boundary = cv2.dilate(foreground, kernel) != cv2.erode(foreground, kernel)
        base_boundary = (cv2.dilate(base_foreground, kernel) !=
                         cv2.erode(base_foreground, kernel))
        valid_boundary = boundary & inside
        valid_global = inside
        photo_global = self._masked_percentile(
            photo_error, valid_global, 65.0, 1.0)
        photo_edge = self._masked_percentile(
            photo_error, valid_boundary, 75.0, photo_global)
        fb_edge = self._masked_percentile(
            fb_error, valid_boundary, 75.0, 99.0)
        coverage = float(np.mean(inside[boundary])) if np.any(boundary) else 0.0
        # Network latency makes a 300-420 ms source age normal on the measured
        # 4K path.  Decay from 120 ms onward, then enforce the hard horizon
        # above: old masks need progressively stronger image evidence, without
        # being rejected merely because inference itself was asynchronous.
        age_penalty = math.exp(-max(0.0, age_ms - 120.0) / 420.0)
        confidence = (coverage * age_penalty *
                      math.exp(-3.6 * photo_edge - 1.8 * photo_global -
                               0.42 * fb_edge))
        confidence = max(0.0, min(1.0, confidence))

        if confidence < self._min_confidence:
            out = None
        else:
            # A scalar score can accept a globally sound registration while
            # hiding the precise failure that matters most for stereo: the
            # newly uncovered strip behind a moving silhouette has no optical-
            # flow correspondence. Preserve confidence locally in a narrow
            # contour band. Interior alpha remains trusted, while high FB or
            # photometric error makes a trailing ghost transparent to the GPU
            # ownership guards instead of protecting it as human geometry.
            local = inside.astype(np.float32)
            local *= np.exp(-0.72 * np.minimum(fb_error, 12.0))
            local *= np.exp(-5.2 * np.minimum(photo_error, 1.0))
            local = np.clip((local - 0.16) / 0.70, 0.0, 1.0)
            # The correction support is the union of the old contour, current
            # contour and their swept classification difference.  The XOR is
            # important: two thin rings alone would leave the middle of a
            # large leading/trailing strip unchanged.  A small dilation keeps
            # fractional hair/cloth alpha inside the sparse authority zone.
            motion = np.hypot(backward[..., 0], backward[..., 1])
            motion_edge = self._masked_percentile(
                motion, boundary, 90.0, 0.0)
            swept = base_foreground != foreground
            contour_seed = boundary | base_boundary | swept
            radius = 2 + min(3, int(math.ceil(motion_edge / 16.0)))
            band_kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
            contour_band = cv2.dilate(
                contour_seed.astype(np.uint8), band_kernel) != 0
            alpha, reliability, reject_pct, sparse_pct = (
                self._advect_sparse_contour_alpha(
                    matte.alpha, backward, contour_band, local))
            self._last_local_reject_pct = reject_pct
            self._last_sparse_pct = sparse_pct
            out = replace(
                matte, alpha=alpha, pts_ms=float(current_pts),
                transport_revision=int(round(current_pts * 1000.0)),
                tracking_confidence=confidence,
                transport_kind="luma-contour-sparse",
                reliability=reliability)
        self._cache_key = key
        self._cache = out
        return True, out

    def advect(self, matte: Optional[MatteFrame],
               video_time_ms: float) -> Optional[MatteFrame]:
        """Return a matte registered to this PTS, or fail closed.

        A stale matte is not an approximation of the current background.  If
        neither luma flow nor codec hints can prove its registration, remove it
        from the renderer instead of leaking the old silhouette into T+N.
        """
        if matte is None:
            return None
        # Lightweight test/compatibility services may expose an opaque marker.
        # The production protocol always returns MatteFrame.
        if not isinstance(matte, MatteFrame):
            return matte
        try:
            self._last_age_ms = float(video_time_ms) - float(matte.pts_ms)
            self._last_error = ""
            if self._dense_enabled:
                started = time.perf_counter()
                handled, dense = self._advect_with_luma(matte, video_time_ms)
                if handled:
                    self._last_lock_ms = (
                        time.perf_counter() - started) * 1000.0
                    if dense is None:
                        self._last_kind = "rejected"
                        self._last_confidence = 0.0
                        self._last_local_reject_pct = 100.0
                        self._last_sparse_pct = 0.0
                        self._last_error = "unreliable contour registration"
                        self._rejected += 1
                    elif dense is matte:
                        self._last_kind = "fresh"
                        self._last_confidence = 1.0
                        self._last_local_reject_pct = 0.0
                        self._last_sparse_pct = 0.0
                    else:
                        self._last_kind = dense.transport_kind
                        self._last_confidence = dense.tracking_confidence
                    return dense
            p = float(video_time_ms)
            steps = [s for s in self._steps
                     if matte.pts_ms + 1.0 < s[0] <= p + 21.0]
            if not steps:
                if -40.0 <= self._last_age_ms <= self.FRESH_WITHOUT_TRANSPORT_MS:
                    self._last_kind = "fresh-untransported"
                    self._last_confidence = 1.0
                    self._last_local_reject_pct = 0.0
                    self._last_sparse_pct = 0.0
                    return matte
                self._last_kind = "rejected"
                self._last_confidence = 0.0
                self._last_local_reject_pct = 100.0
                self._last_sparse_pct = 0.0
                self._last_error = "no current-frame transport evidence"
                self._rejected += 1
                return None
            key = (matte.sequence, matte.generation,
                   round(float(steps[-1][0]), 1))
            if key == self._cache_key and self._cache is not None:
                return self._cache
            h, w = matte.alpha.shape[:2]
            sw, sh = steps[-1][3], steps[-1][4]
            fx = steps[0][1].copy()
            fy = steps[0][2].copy()
            for s in steps[1:]:
                fx += s[1]
                fy += s[2]
            fx_m = cv2.resize(fx * (w / float(max(1, sw))), (w, h),
                              interpolation=cv2.INTER_LINEAR)
            fy_m = cv2.resize(fy * (h / float(max(1, sh))), (w, h),
                              interpolation=cv2.INTER_LINEAR)
            mag95 = float(np.percentile(np.hypot(fx_m, fy_m), 95))
            if mag95 < 0.7:
                out = matte          # statique : l'original est déjà juste
            else:
                xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
                adv = cv2.remap(matte.alpha, xx - fx_m, yy - fy_m,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
                projected_pts = float(steps[-1][0])
                out = replace(
                    matte, alpha=adv, pts_ms=projected_pts,
                    transport_revision=int(round(projected_pts * 1000.0)),
                    tracking_confidence=0.50,
                    transport_kind="codec-mv-fallback")
            self._cache_key = key
            self._cache = out
            self._last_kind = (getattr(out, 'transport_kind', 'static')
                               if out is not None else 'rejected')
            self._last_confidence = float(
                getattr(out, 'tracking_confidence', 0.0))
            self._last_sparse_pct = 0.0
            return out
        except Exception as exc:
            self._last_kind = "rejected"
            self._last_confidence = 0.0
            self._last_local_reject_pct = 100.0
            self._last_sparse_pct = 0.0
            self._last_error = f"transport exception: {exc}"
            self._rejected += 1
            return None


def _to_u8_plane(plane: np.ndarray,
                 sample_peak: Optional[int] = None) -> np.ndarray:
    return _plane_to_u8(plane, sample_peak)


def yuv420_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray,
                  max_short_side: int = 720,
                  sample_peak: Optional[int] = None) -> np.ndarray:
    """Convert planar limited-range BT.709 YUV420 to a capped RGB frame."""
    peak = sample_peak if sample_peak is not None else _sample_peak(y)
    y8, u8, v8 = (_to_u8_plane(p, peak) for p in (y, u, v))
    height, width = y8.shape
    if height < 2 or width < 2:
        raise ValueError("invalid luma dimensions")

    out_w, out_h = _matte_dimensions(width, height, max_short_side)
    scale = min(out_w / float(width), out_h / float(height))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    yy = cv2.resize(y8, (out_w, out_h), interpolation=interp).astype(np.float32)
    uu = cv2.resize(u8, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    vv = cv2.resize(v8, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    c = np.maximum(0.0, yy - 16.0) * 1.164383
    d = uu - 128.0
    e = vv - 128.0
    rgb = np.empty((out_h, out_w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(c + 1.792741 * e, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(c - 0.213249 * d - 0.532909 * e, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(c + 2.112402 * d, 0, 255).astype(np.uint8)
    return rgb


def _matte_dimensions(width: int, height: int,
                      max_short_side: int) -> tuple[int, int]:
    scale = min(1.0, float(max_short_side) / float(min(height, width)))
    out_w = max(16, int(round(width * scale / 2.0)) * 2)
    out_h = max(16, int(round(height * scale / 2.0)) * 2)
    return out_w, out_h


class MatAnyone2Service:
    """Latest-only asynchronous client for ``synth3d_matanyone2_worker.py``."""

    def __init__(self, runtime: MatAnyone2Runtime, *, target_fps: float = 5.0,
                 short_side: int = 640, max_pts_delta_ms: float = 650.0,
                 warmup: int = 3, fps_pinned: bool = False,
                 short_side_pinned: bool = False):
        self.runtime = runtime
        # Procédural (04/08) : la cadence suit le MÉDIA ; la définition le suit
        # jusqu'au plafond temporel mesuré (configure_media, appelé au pont par
        # frame). Un épinglage explicite par env reste intouchable.
        self.target_fps = max(1.0, min(60.0, float(target_fps)))
        self.short_side = max(256, min(2160, int(short_side)))
        self._fps_pinned = bool(fps_pinned)
        self._short_side_pinned = bool(short_side_pinned)
        # Matting needs accurate contours more than native raster resolution.
        # Reality gate (06/08): live 1710x720 measurements showed 62–118 ms at
        # 720p (typically 9–12 delivered fps for 23.976 content, with 42% of
        # latest-only submissions replaced).  640p removes 21% of model pixels
        # while the native distance field restores full-resolution contour
        # placement. The explicit SHORT_SIDE/AUTO_CAP envs remain exact opt-outs.
        try:
            auto_cap = int(os.environ.get("SYLC_MATANYONE2_AUTO_CAP", "640"))
        except ValueError:
            auto_cap = 640
        self.auto_cap = max(256, min(2160, auto_cap))
        if not self._short_side_pinned:
            self.short_side = min(self.short_side, self.auto_cap)
        transport = os.environ.get("SYLC_MATANYONE2_TRANSPORT", "yuv420")
        transport = transport.strip().lower()
        self.transport = transport if transport in ("yuv420", "rgb8", "jpeg") \
            else "yuv420"
        self.max_pts_delta_ms = max(100.0, float(max_pts_delta_ms))
        self.warmup = max(0, min(10, int(warmup)))

        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._pending = None
        self._controls = []
        self._worker_busy = False
        self._latest: Optional[MatteFrame] = None
        self._generation = 1
        self._sequence = 0
        self._last_submit_pts = -math.inf
        self._next_submit_pts = -math.inf
        self._last_sample_peak = 0
        self._stopping = False
        self._state = "off"
        self._error = ""
        self._submitted = 0
        self._dropped = 0
        self._outputs = 0
        self._started_at = 0.0
        self._last_output_at = 0.0
        self._output_fps = 0.0
        self._last_inference_ms = 0.0
        self._last_prep_ms = 0.0
        self._last_decode_ms = 0.0
        self._last_upload_ms = 0.0
        self._last_model_ms = 0.0
        self._last_readback_ms = 0.0
        self._last_input_shape = (0, 0)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            self._stopping = False
            self._state = "loading"
            self._error = ""
        command = [
            str(self.runtime.python), "-u", str(self.runtime.worker),
            "--checkpoint", str(self.runtime.checkpoint),
            "--seed-checkpoint", str(self.runtime.seed_checkpoint),
            # Épinglé : le worker reçoit exactement la consigne. Procédural :
            # le plafond mesuré (640 par défaut) s'applique à l'hôte ET au
            # modèle. Une source plus petite reste traitée à sa définition ;
            # l'env SHORT_SIDE permet toujours le plein format expérimental.
            "--short-side", str(self.short_side if self._short_side_pinned
                                else self.auto_cap),
            "--warmup", str(self.warmup),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TORCH_HOME"] = str(self.runtime.model_root / "torch-cache")
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, env=env,
                startupinfo=startupinfo, creationflags=creationflags)
        except Exception as exc:
            with self._lock:
                self._state, self._error = "error", str(exc)
            log.exception("[MATANYONE2] worker start failed")
            return False
        with self._lock:
            self._proc = proc
            self._started_at = time.monotonic()
        threading.Thread(target=self._pump, name="MatAnyone2-submit", daemon=True).start()
        threading.Thread(target=self._read_stdout, name="MatAnyone2-result", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="MatAnyone2-log", daemon=True).start()
        threading.Thread(target=self._watch_process, name="MatAnyone2-watch", daemon=True).start()
        return True

    def submit_yuv(self, planes, pts_ms: float,
                   sample_peak: Optional[int] = None) -> bool:
        if not self.running or not planes or len(planes) != 3:
            return False
        pts = float(pts_ms)
        schedule_pts = pts if math.isfinite(pts) and pts >= 0 else time.monotonic() * 1000.0
        with self._wake:
            if schedule_pts < self._last_submit_pts - 100.0:
                # Defensive local reset if a caller forgot the explicit seek hook.
                self._reset_locked("backward PTS")
            if not self._submission_due_locked(schedule_pts):
                return False
            try:
                copied = tuple(np.ascontiguousarray(np.asarray(p)).copy() for p in planes)
                peak = (int(sample_peak) if sample_peak is not None
                        else _sample_peak(copied[0]))
            except Exception as exc:
                self._error = f"plane copy: {exc}"
                return False
            if self._pending is not None:
                self._dropped += 1
            self._pending = (self._generation, pts, copied, peak)
            self._last_submit_pts = schedule_pts
            self._last_sample_peak = peak
            self._submitted += 1
            self._wake.notify()
        return True

    def _submission_due_locked(self, schedule_pts: float) -> bool:
        """Fractional media-PTS scheduler without integer-frame aliasing.

        Comparing every candidate only with the last accepted frame turns a
        requested 12 fps on 23.976 fps content into one frame out of three
        (~8 fps). A carried deadline alternates frame gaps and preserves the
        requested average without ever blocking playback.
        """
        interval = 1000.0 / self.target_fps
        if not math.isfinite(self._next_submit_pts):
            self._next_submit_pts = schedule_pts + interval
            return True
        if schedule_pts + 0.05 < self._next_submit_pts:
            return False
        missed = max(0, int(math.floor(
            (schedule_pts - self._next_submit_pts) / interval)))
        self._next_submit_pts += (missed + 1) * interval
        return True

    def latest_for_pts(self, pts_ms: float) -> Optional[MatteFrame]:
        with self._lock:
            item = self._latest
            generation = self._generation
        if item is None or item.generation != generation:
            return None
        return item if item.matches(float(pts_ms), self.max_pts_delta_ms) else None

    def configure_media(self, fps=None, short_side=None) -> None:
        """Asservit la cadence et la définition plafonnée au média LU.
        Appelé au pont par frame (donc AVANT la première soumission) ; les
        valeurs épinglées par env sont intouchables ; un changement de
        géométrie après des soumissions reseede la mémoire temporelle du
        worker (reset de génération)."""
        with self._wake:
            if fps is not None and not self._fps_pinned:
                try:
                    f = max(1.0, min(60.0, float(fps)))
                    # This method is called once per frame, and re-anchoring
                    # costs the scheduler its phase: _submission_due_locked
                    # takes the "not finite" branch and admits the frame
                    # unconditionally, so a deadline reset on every call
                    # disables rate limiting entirely rather than merely
                    # perturbing it. The threshold must therefore sit above any
                    # jitter a caller can plausibly hand us -- a rate derived
                    # from integer-millisecond PTS deltas swings ~0.6 fps at
                    # 24 fps -- while still catching a genuine 24 -> 25 -> 30
                    # cadence change, which never moves by less than 1 fps.
                    if abs(f - self.target_fps) > 0.5:
                        log.info("[MATANYONE2] cadence asservie au média: "
                                 "%.2f fps", f)
                        self.target_fps = f
                        # Re-anchor the fractional deadline at the next frame;
                        # keeping an old-period phase would create a transient
                        # burst or hole after a media-rate discovery.
                        self._next_submit_pts = -math.inf
                except (TypeError, ValueError):
                    pass
            if short_side is not None and not self._short_side_pinned:
                try:
                    source_side = max(256, min(2160, int(short_side)))
                    s = min(source_side, self.auto_cap)
                    if s != self.short_side:
                        log.info("[MATANYONE2] definition asservie au media: "
                                 "short-side=%d", s)
                        self.short_side = s
                        if self._submitted > 0:
                            self._reset_locked("media geometry change")
                            self._wake.notify()
                except (TypeError, ValueError):
                    pass

    def reset(self, reason: str = "host reset") -> None:
        with self._wake:
            self._reset_locked(reason)
            self._wake.notify()

    def _reset_locked(self, reason: str) -> None:
        self._generation += 1
        self._latest = None
        self._pending = None
        self._last_submit_pts = -math.inf
        self._next_submit_pts = -math.inf
        self._last_sample_peak = 0
        generation = self._generation
        log.info("[MATANYONE2] reset generation=%d reason=%s", generation, reason)
        # Control writes are performed by the pump thread too. A CUDA worker
        # that is temporarily not reading stdin must never block the GUI/seek
        # thread on a full pipe.
        self._controls.append(
            {"kind": "reset", "generation": generation, "reason": reason})

    def stop(self, timeout: float = 0.0) -> None:
        with self._wake:
            self._stopping = True
            proc = self._proc
            self._pending = None
            self._latest = None
            self._wake.notify_all()
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=max(0.05, timeout))
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        with self._lock:
            self._proc = None
            self._state = "off"

    def status(self) -> dict:
        with self._lock:
            latest_pts = (float(self._latest.pts_ms)
                          if self._latest is not None else -1.0)
            return {
                "state": self._state,
                "error": self._error,
                "submitted": self._submitted,
                "dropped": self._dropped,
                "outputs": self._outputs,
                "fps": self._output_fps,
                "target_fps": self.target_fps,
                "inference_ms": self._last_inference_ms,
                "prep_ms": self._last_prep_ms,
                "decode_ms": self._last_decode_ms,
                "upload_ms": self._last_upload_ms,
                "model_ms": self._last_model_ms,
                "readback_ms": self._last_readback_ms,
                "input_width": self._last_input_shape[1],
                "input_height": self._last_input_shape[0],
                "short_side": self.short_side,
                "auto_cap": self.auto_cap,
                "transport": self.transport,
                "sample_peak": self._last_sample_peak,
                "latest_pts_ms": latest_pts,
                "pending": self._pending is not None,
                "generation": self._generation,
            }

    def _pump(self) -> None:
        while True:
            with self._wake:
                while ((self._worker_busy or
                        (not self._controls and self._pending is None)) and
                       not self._stopping):
                    self._wake.wait(timeout=0.5)
                    if self._proc is None or self._proc.poll() is not None:
                        return
                if self._stopping:
                    return
                if self._state not in ("ready", "running", "degraded"):
                    # In particular, do not queue a reset behind model loading:
                    # the worker's initial "ready" is not an acknowledgement
                    # for a control it has not read yet.
                    self._wake.wait(timeout=0.1)
                    continue
                if self._controls:
                    control = self._controls.pop(0)
                    generation = pts = planes = sample_peak = None
                else:
                    control = None
                    generation, pts, planes, sample_peak = self._pending
                    self._pending = None
                # One unacknowledged message at most.  Without this gate a
                # second encoded frame sits invisibly in the OS pipe while
                # CUDA works and is stale before inference even begins.
                self._worker_busy = True
            try:
                started = time.perf_counter()
                if control is not None:
                    self._send_header(control)
                    continue
                if self.transport == "yuv420":
                    y, u, v = (np.ascontiguousarray(np.asarray(p))
                               for p in planes)
                    if (y.ndim != 2 or u.shape != (y.shape[0] // 2,
                                                   y.shape[1] // 2)
                            or v.shape != u.shape or u.dtype != y.dtype
                            or v.dtype != y.dtype
                            or y.dtype not in (np.uint8, np.uint16)):
                        raise ValueError("invalid planar YUV420 frame")
                    out_w, out_h = _matte_dimensions(
                        y.shape[1], y.shape[0], self.short_side)
                    encoding = "yuv420p16" if y.dtype == np.uint16 \
                        else "yuv420p8"
                    payload = b"".join(p.tobytes(order="C")
                                       for p in (y, u, v))
                    frame_header = {
                        "kind": "frame", "generation": generation,
                        "pts_ms": pts, "width": out_w, "height": out_h,
                        "source_width": int(y.shape[1]),
                        "source_height": int(y.shape[0]),
                        "encoding": encoding, "size": len(payload),
                        "sample_peak": int(sample_peak),
                    }
                    input_shape = (out_h, out_w)
                else:
                    rgb = yuv420_to_rgb(
                        *planes, max_short_side=self.short_side,
                        sample_peak=sample_peak)
                    input_shape = tuple(rgb.shape[:2])
                    frame_header = {
                        "kind": "frame", "generation": generation,
                        "pts_ms": pts, "width": int(rgb.shape[1]),
                        "height": int(rgb.shape[0]),
                        "encoding": self.transport,
                    }
                if self.transport == "jpeg":
                    ok, encoded = cv2.imencode(
                        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if not ok:
                        raise RuntimeError("JPEG encoder refused frame")
                    payload = encoded.tobytes()
                elif self.transport == "rgb8":
                    payload = rgb.tobytes(order="C")
                frame_header["size"] = len(payload)
                self._send_header(frame_header, payload)
                with self._lock:
                    self._last_prep_ms = (time.perf_counter() - started) * 1000.0
                    self._last_input_shape = input_shape
            except Exception as exc:
                with self._wake:
                    self._worker_busy = False
                    self._error = f"submit: {exc}"
                    self._wake.notify()
                log.warning("[MATANYONE2] frame submission failed: %s", exc)

    def _send_header(self, header: dict, payload: bytes = b"", *,
                     tolerate_failure: bool = False) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            if tolerate_failure:
                return
            raise BrokenPipeError("MatAnyone 2 worker is not running")
        data = (json.dumps(header, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                proc.stdin.write(data)
                if payload:
                    proc.stdin.write(payload)
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            if not tolerate_failure:
                raise

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            block = stream.read(remaining)
            if not block:
                raise EOFError("worker closed in payload")
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while not self._stopping:
                line = proc.stdout.readline()
                if not line:
                    return
                message = json.loads(line.decode("utf-8"))
                kind = message.get("kind")
                if kind == "status":
                    with self._wake:
                        self._state = str(message.get("state", self._state))
                        self._error = str(message.get("error", ""))
                        if self._state in ("ready", "error"):
                            self._worker_busy = False
                        self._wake.notify()
                    continue
                if kind == "error":
                    with self._wake:
                        self._worker_busy = False
                        self._state = "degraded"
                        self._error = str(message.get("error", "worker error"))
                        self._wake.notify()
                    log.warning("[MATANYONE2] %s", self._error)
                    continue
                if kind != "matte":
                    continue
                width, height = int(message["width"]), int(message["height"])
                size = int(message["size"])
                if width <= 0 or height <= 0 or size != width * height:
                    raise ValueError("invalid matte payload dimensions")
                payload = self._read_exact(proc.stdout, size)
                alpha = np.frombuffer(payload, dtype=np.uint8).reshape(height, width).copy()
                generation = int(message["generation"])
                with self._wake:
                    self._worker_busy = False
                    self._wake.notify()
                    if generation != self._generation:
                        continue
                    self._sequence += 1
                    self._last_inference_ms = float(message.get("inference_ms", 0.0))
                    self._last_decode_ms = float(message.get("decode_ms", 0.0))
                    self._last_upload_ms = float(message.get("upload_ms", 0.0))
                    self._last_model_ms = float(message.get("model_ms", 0.0))
                    self._last_readback_ms = float(message.get("readback_ms", 0.0))
                    self._latest = MatteFrame(
                        self._sequence, generation, float(message["pts_ms"]), alpha,
                        self._last_inference_ms, bool(message.get("seeded", False)),
                        bool(message.get("scene_cut", False)),
                        self._last_decode_ms, self._last_upload_ms,
                        self._last_model_ms, self._last_readback_ms)
                    self._outputs += 1
                    now = time.monotonic()
                    if self._last_output_at > 0.0:
                        instantaneous = 1.0 / max(1e-3, now - self._last_output_at)
                        self._output_fps = (instantaneous if self._output_fps <= 0.0
                                            else 0.20 * instantaneous
                                            + 0.80 * self._output_fps)
                    self._last_output_at = now
                    self._state = "running"
                    self._error = ""
        except Exception as exc:
            if not self._stopping:
                with self._lock:
                    self._state, self._error = "error", f"protocol: {exc}"
                log.exception("[MATANYONE2] result protocol failed")

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while not self._stopping:
            line = proc.stderr.readline()
            if not line:
                return
            log.info("[worker] %s", line.decode("utf-8", "replace").rstrip())

    def _watch_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        if not self._stopping and code != 0:
            with self._lock:
                self._state = "error"
                if not self._error:
                    self._error = f"worker exited with code {code}"
            log.error("[MATANYONE2] worker exited with code %d", code)
