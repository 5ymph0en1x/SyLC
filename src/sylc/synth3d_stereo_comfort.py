"""Deterministic comfort geometry for SyLC's binocular coupling field.

The calibrated envelope in this module is consumed by the live native warp.
The array helpers remain the executable reference used by tests and future
coarse-field optimizers.  All temporal operators take an explicit
``scene_cut`` flag: no state is mathematically allowed across a cut.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StereoDisplayGeometry:
    """Physical geometry needed to turn pixels into an ocular demand."""

    screen_width_m: float
    horizontal_pixels: int
    viewing_distance_m: float
    interpupillary_distance_m: float = 0.064

    def __post_init__(self) -> None:
        values = (self.screen_width_m, self.viewing_distance_m,
                  self.interpupillary_distance_m)
        if any(not math.isfinite(v) or v <= 0.0 for v in values):
            raise ValueError("display dimensions and IPD must be positive")
        if self.horizontal_pixels <= 0:
            raise ValueError("horizontal_pixels must be positive")

    @property
    def metres_per_pixel(self) -> float:
        return self.screen_width_m / float(self.horizontal_pixels)

    def physical_disparity_m(self, disparity_px):
        return np.asarray(disparity_px, dtype=np.float64) * self.metres_per_pixel

    def angular_disparity_deg(self, disparity_px):
        physical = self.physical_disparity_m(disparity_px)
        return np.degrees(2.0 * np.arctan2(
            physical, 2.0 * self.viewing_distance_m))

    def vac_diopters(self, disparity_px):
        """Vergence minus accommodation demand for centered stereo geometry.

        If ``p`` is physical on-screen disparity, ``I`` the IPD and ``Zs`` the
        screen distance, the virtual distance is ``Zv = Zs/(1+p/I)``.  Hence
        the vergence-accommodation conflict is exactly ``p/(I*Zs)`` diopters
        under this symmetric pinhole model. Positive disparity is crossed.
        """
        physical = self.physical_disparity_m(disparity_px)
        return physical / (self.interpupillary_distance_m *
                           self.viewing_distance_m)

    def pixels_for_vac_diopters(self, conflict_diopters):
        physical = (np.asarray(conflict_diopters, dtype=np.float64) *
                    self.interpupillary_distance_m * self.viewing_distance_m)
        return physical / self.metres_per_pixel


@dataclass(frozen=True)
class StereoComfortEnvelope:
    """A calibrated, sign-preserving soft disparity envelope.

    Disparities below ``soft_vac_diopters`` retain the artistic intent exactly.
    Above it, a rational proximal knee approaches ``hard_vac_diopters`` without
    a hard clipping shelf. The same closed form is implemented in native HLSL.
    """

    geometry: StereoDisplayGeometry
    soft_vac_diopters: float = 0.18
    hard_vac_diopters: float = 0.30

    def __post_init__(self) -> None:
        soft = self.soft_vac_diopters
        hard = self.hard_vac_diopters
        if (not math.isfinite(soft) or not math.isfinite(hard) or
                soft < 0.0 or hard <= soft):
            raise ValueError("comfort VAC knee must satisfy 0 <= soft < hard")

    @property
    def soft_pixels(self) -> float:
        return float(self.geometry.pixels_for_vac_diopters(
            self.soft_vac_diopters))

    @property
    def hard_pixels(self) -> float:
        return float(self.geometry.pixels_for_vac_diopters(
            self.hard_vac_diopters))

    @property
    def soft_percent(self) -> float:
        return 100.0 * self.soft_pixels / self.geometry.horizontal_pixels

    @property
    def hard_percent(self) -> float:
        return 100.0 * self.hard_pixels / self.geometry.horizontal_pixels

    def project(self, disparity_px: np.ndarray) -> np.ndarray:
        """Project pixels through the production soft knee.

        The map is continuous with unit derivative at the knee, monotonic,
        sign preserving, and asymptotically bounded by the hard envelope.
        """
        d = np.asarray(disparity_px, dtype=np.float64)
        if not np.all(np.isfinite(d)):
            raise ValueError("disparity must be finite")
        magnitude = np.abs(d)
        soft = self.soft_pixels
        span = self.hard_pixels - soft
        over = np.maximum(0.0, magnitude - soft)
        projected = np.where(
            magnitude <= soft,
            magnitude,
            soft + over / (1.0 + over / span))
        return np.copysign(projected, d).astype(np.float32)


@dataclass(frozen=True)
class ComfortScales:
    """Normalization scales for the first transparent energy, not medical limits."""

    vac_diopters: float = 0.60
    disparity_gradient_px_per_px: float = 0.75
    disparity_velocity_px_per_s: float = 90.0
    vac_weight: float = 1.0
    gradient_weight: float = 0.35
    velocity_weight: float = 0.45


@dataclass(frozen=True)
class StereoComfortMetrics:
    energy: float
    vac_rms_diopters: float
    vac_p95_diopters: float
    gradient_p95_px_per_px: float
    velocity_p95_px_per_s: float
    temporal_samples_used: int


def _p95_abs(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), 95.0)) if values.size else 0.0


def _charbonnier(x: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    return np.sqrt(x * x + epsilon * epsilon) - epsilon


def evaluate_comfort_energy(
        disparity_px: np.ndarray,
        geometry: StereoDisplayGeometry,
        *,
        previous_disparity_px: Optional[np.ndarray] = None,
        delta_time_s: Optional[float] = None,
        saliency: Optional[np.ndarray] = None,
        scene_cut: bool = False,
        scales: ComfortScales = ComfortScales()) -> StereoComfortMetrics:
    """Measure an interpretable first comfort energy on a disparity field.

    The field is cyclopean: one scalar disparity owns the paired L/R samples.
    ``scene_cut=True`` unconditionally removes the temporal term.
    """
    d = np.asarray(disparity_px, dtype=np.float64)
    if d.ndim != 2 or not np.all(np.isfinite(d)):
        raise ValueError("disparity_px must be a finite 2-D field")
    if saliency is None:
        weight = np.ones_like(d)
    else:
        weight = np.asarray(saliency, dtype=np.float64)
        if weight.shape != d.shape or not np.all(np.isfinite(weight)):
            raise ValueError("saliency must be finite and match disparity")
        weight = np.clip(weight, 0.0, None)
    weight /= max(float(weight.mean()), 1e-9)

    vac = geometry.vac_diopters(d)
    gx = np.diff(d, axis=1)
    gy = np.diff(d, axis=0)
    gradient = np.concatenate((gx.ravel(), gy.ravel()))
    e_vac = np.mean(weight * _charbonnier(
        vac / max(scales.vac_diopters, 1e-9)))
    e_grad = (float(np.mean(_charbonnier(
        gradient / max(scales.disparity_gradient_px_per_px, 1e-9))))
        if gradient.size else 0.0)

    velocity = np.empty(0, dtype=np.float64)
    if not scene_cut and previous_disparity_px is not None:
        previous = np.asarray(previous_disparity_px, dtype=np.float64)
        if previous.shape != d.shape or not np.all(np.isfinite(previous)):
            raise ValueError("previous disparity must be finite and shape-matched")
        if delta_time_s is None or not math.isfinite(delta_time_s) or delta_time_s <= 0:
            raise ValueError("a positive delta_time_s is required temporally")
        velocity = (d - previous) / float(delta_time_s)
    e_velocity = (float(np.mean(weight * _charbonnier(
        velocity / max(scales.disparity_velocity_px_per_s, 1e-9))))
        if velocity.size else 0.0)

    energy = (scales.vac_weight * float(e_vac) +
              scales.gradient_weight * e_grad +
              scales.velocity_weight * e_velocity)
    return StereoComfortMetrics(
        energy=energy,
        vac_rms_diopters=float(np.sqrt(np.mean(vac * vac))),
        vac_p95_diopters=_p95_abs(vac),
        gradient_p95_px_per_px=_p95_abs(gradient),
        velocity_p95_px_per_s=_p95_abs(velocity),
        temporal_samples_used=int(velocity.size))


def project_to_comfort_budget(
        disparity_px: np.ndarray,
        geometry: StereoDisplayGeometry,
        *,
        max_abs_vac_diopters: float,
        previous_disparity_px: Optional[np.ndarray] = None,
        delta_time_s: Optional[float] = None,
        max_velocity_px_per_s: Optional[float] = None,
        scene_cut: bool = False) -> np.ndarray:
    """First proximal projection: physical VAC bound plus temporal slew.

    It is deliberately order-preserving and has no hidden memory. At a cut the
    slew constraint is absent, so the previous shot cannot influence T0.
    """
    if not math.isfinite(max_abs_vac_diopters) or max_abs_vac_diopters <= 0:
        raise ValueError("max_abs_vac_diopters must be positive")
    d = np.asarray(disparity_px, dtype=np.float64)
    limit = float(geometry.pixels_for_vac_diopters(max_abs_vac_diopters))
    projected = np.clip(d, -limit, limit)
    if (not scene_cut and previous_disparity_px is not None and
            max_velocity_px_per_s is not None):
        if (delta_time_s is None or not math.isfinite(delta_time_s) or
                delta_time_s <= 0.0 or not math.isfinite(max_velocity_px_per_s)
                or max_velocity_px_per_s <= 0.0):
            raise ValueError("valid dt and velocity are required for slew limiting")
        previous = np.asarray(previous_disparity_px, dtype=np.float64)
        if previous.shape != projected.shape:
            raise ValueError("previous disparity must match")
        step = max_velocity_px_per_s * delta_time_s
        projected = np.clip(projected, previous - step, previous + step)
    return projected.astype(np.float32)
