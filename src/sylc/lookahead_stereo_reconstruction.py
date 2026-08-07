"""Temporal look-ahead disocclusion reconstruction for 2D -> 3D.

This module is the CPU reference for the renderer implementation.  It turns a
future video frame into *evidence* for pixels exposed by a stereo warp; it does
not interpolate time and it never replaces a source-supported stereo pixel.

The important convention is the same one used by ``synth3d_flow`` in the
native worker: ``future_to_current`` is sampled on the current grid and maps a
current coordinate to its future source with ``future = current - flow``.
At a stereo hole that flow is unknown, so the field is extrapolated from the
nearest visible background donor selected by the ordinary spatial fill.  The
future candidate is accepted only when it agrees with that background in
colour and, when available, depth.

The implementation deliberately operates on a compact analysis grid.  Its
outputs and metrics make an A/B hypothesis test cheap; the proven operations
map one-for-one to a D3D11 compute/pixel shader and NVOFA can later replace the
flow provider without changing the reconstruction contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowField:
    """Future->current displacement sampled at current-grid coordinates."""

    x: np.ndarray
    y: np.ndarray
    reliability: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.x).shape
        if len(shape) != 2:
            raise ValueError("flow fields must be 2-D")
        if np.asarray(self.y).shape != shape or np.asarray(
                self.reliability).shape != shape:
            raise ValueError("flow x/y/reliability shapes must match")


@dataclass(frozen=True)
class EyeReconstruction:
    """One reconstructed eye and the masks needed for objective A/B."""

    baseline: np.ndarray
    lookahead: np.ndarray
    supported: np.ndarray
    holes: np.ndarray
    accepted: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class StereoReconstruction:
    left: EyeReconstruction
    right: EyeReconstruction


def normalize_luma(luma: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 luma image in [0, 1]."""

    data = np.asarray(luma)
    if data.ndim != 2:
        raise ValueError("luma must be a 2-D array")
    if data.dtype == np.uint8:
        scale = 255.0
    elif data.dtype == np.uint16:
        peak = float(data.max(initial=0))
        scale = 65535.0 if peak > 4095.0 else 1023.0
    elif np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)
    else:
        scale = 1.0
    result = data.astype(np.float32, copy=False)
    if scale != 1.0:
        result = result / scale
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0))


def resize_plane(plane: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one scalar or interleaved plane to the analysis grid."""

    data = np.asarray(plane)
    if data.ndim not in (2, 3):
        raise ValueError("plane must be HxW or HxWxC")
    if data.shape[:2] == (height, width):
        return np.ascontiguousarray(data)
    return cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)


def bgr_to_yuv(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """BGR frame -> full-resolution normalized YUV analysis image."""

    frame = resize_plane(np.asarray(bgr), width, height)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("bgr must be HxWx3")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    return np.ascontiguousarray(yuv.astype(np.float32) / 255.0)


def estimate_future_to_current_flow(
    current_luma: np.ndarray,
    future_luma: np.ndarray,
    width: int,
    height: int,
    *,
    native_module=None,
    threads: int = 4,
    backend: str = "cpu",
    nvof_perf: str = "medium",
    nvof_grid: int = 4,
    nvof_session=None,
) -> FlowField:
    """Estimate future->current flow on ``width`` x ``height``.

    The native production estimator is preferred.  OpenCV Farneback is a
    portable fallback for tests/developer machines; its forward convention is
    converted to the destination-grid convention by remapping the result.
    """

    if backend not in ("auto", "cpu", "nvof"):
        raise ValueError("backend must be auto, cpu or nvof")
    current = resize_plane(normalize_luma(current_luma), width, height)
    future = resize_plane(normalize_luma(future_luma), width, height)
    have_nvof = nvof_session is not None or (
        native_module is not None and hasattr(native_module, "NvofFlow"))
    if backend == "nvof" and not have_nvof:
        raise RuntimeError("NVOFA backend was requested but is unavailable")
    if have_nvof and backend in ("auto", "nvof"):
        session = nvof_session
        if session is None:
            session = native_module.NvofFlow(
                width, height, nvof_perf, int(nvof_grid))
        elif session.width != width or session.height != height:
            raise ValueError("persistent NVOFA session has the wrong grid")
        # NVOFA reports input->reference at input coordinates.  Calling it
        # current->future gives the exact field needed at visible background
        # donors; negate it to preserve this module's future=current-flow
        # convention used by the shader/reference implementation.
        fx, fy, reliability = session.estimate(
            np.clip(current * 255.0 + 0.5, 0, 255).astype(np.uint8),
            np.clip(future * 255.0 + 0.5, 0, 255).astype(np.uint8))
        fx = np.asarray(fx, dtype=np.float32)
        fy = np.asarray(fy, dtype=np.float32)
        reliability = np.asarray(reliability, dtype=np.float32)
        if fx.shape != (height, width):
            fx = cv2.resize(fx, (width, height), interpolation=cv2.INTER_LINEAR)
            fy = cv2.resize(fy, (width, height), interpolation=cv2.INTER_LINEAR)
            reliability = cv2.resize(
                reliability, (width, height), interpolation=cv2.INTER_LINEAR)
        return FlowField(-fx, -fy, np.clip(reliability, 0.0, 1.0))
    if native_module is not None and hasattr(
            native_module, "_synth3d_estimate_flow_test"):
        fx, fy, quality = native_module._synth3d_estimate_flow_test(
            future.ravel(), current.ravel(), width, height, max(1, int(threads)))
        return FlowField(
            np.asarray(fx, dtype=np.float32).reshape(height, width),
            np.asarray(fy, dtype=np.float32).reshape(height, width),
            np.clip(np.asarray(quality, dtype=np.float32).reshape(
                height, width), 0.0, 1.0),
        )

    # calcOpticalFlowFarneback returns source->destination vectors at source
    # coordinates.  Estimate current->future, then splat onto the future grid
    # and negate it so future=current-flow follows the native convention.
    forward = cv2.calcOpticalFlowFarneback(
        current, future, None, 0.5, 4, 19, 4, 7, 1.5, 0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dst_x = np.rint(xx + forward[..., 0]).astype(np.int32)
    dst_y = np.rint(yy + forward[..., 1]).astype(np.int32)
    valid = ((dst_x >= 0) & (dst_x < width) &
             (dst_y >= 0) & (dst_y < height))
    fx = np.zeros((height, width), np.float32)
    fy = np.zeros_like(fx)
    hits = np.zeros_like(fx)
    np.add.at(fx, (dst_y[valid], dst_x[valid]), -forward[..., 0][valid])
    np.add.at(fy, (dst_y[valid], dst_x[valid]), -forward[..., 1][valid])
    np.add.at(hits, (dst_y[valid], dst_x[valid]), 1.0)
    denom = np.maximum(hits, 1.0)
    fx /= denom
    fy /= denom
    reliability = np.clip(hits, 0.0, 1.0)
    # Fill sparse splat gaps without pretending they are measured: the vector
    # is useful as a local extrapolation, while reliability retains the veto.
    for channel in (fx, fy):
        missing = reliability <= 0.0
        if np.any(missing):
            filled = cv2.inpaint(
                channel, missing.astype(np.uint8), 3.0, cv2.INPAINT_NS)
            channel[missing] = filled[missing]
    return FlowField(fx, fy, reliability)


def _bilinear_sample(image: np.ndarray, x: np.ndarray,
                     y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    valid = ((x >= 0.0) & (x <= width - 1.0) &
             (y >= 0.0) & (y <= height - 1.0))
    sampled = cv2.remap(
        image.astype(np.float32, copy=False), x.astype(np.float32),
        y.astype(np.float32), interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)
    return sampled, valid


def _forward_warp(
    current_yuv: np.ndarray,
    depth: np.ndarray,
    disparity: np.ndarray,
    eye_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-layer forward DIBR and source ownership for one eye."""

    height, width = depth.shape
    output = np.zeros_like(current_yuv, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int32)
    source_x = np.broadcast_to(np.arange(width, dtype=np.float32),
                               (height, width))
    destination_x = np.rint(
        source_x + eye_sign * 0.5 * disparity).astype(np.int32)
    valid = (destination_x >= 0) & (destination_x < width)
    yy = np.broadcast_to(np.arange(height, dtype=np.int64)[:, None],
                         (height, width))
    source_linear = np.arange(height * width, dtype=np.int64).reshape(
        height, width)
    destination_linear = yy * width + np.clip(destination_x, 0, width - 1)
    valid_source = source_linear[valid]
    valid_destination = destination_linear[valid]
    valid_depth = depth[valid]

    # Vectorized z-buffer.  maximum.at finds the nearest layer for each eye
    # destination; a second pass copies only winners.  Equal-depth collisions
    # are harmless (they belong to the same surface and assignment is stable).
    owner_depth = np.full(height * width, -np.inf, dtype=np.float32)
    np.maximum.at(owner_depth, valid_destination, valid_depth)
    winners = valid_depth >= owner_depth[valid_destination] - 1.0e-7
    winning_source = valid_source[winners]
    winning_destination = valid_destination[winners]
    output.reshape(-1, 3)[winning_destination] = current_yuv.reshape(
        -1, 3)[winning_source]
    owner.reshape(-1)[winning_destination] = winning_source % width
    supported = owner >= 0
    return output, supported, owner


def _nearest_background_fill(
    warped: np.ndarray,
    supported: np.ndarray,
    owner: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill holes from the nearest supported, locally farther source."""

    height, width = supported.shape
    filled = warped.copy()
    x_grid = np.broadcast_to(np.arange(width, dtype=np.int32),
                             (height, width))
    left = np.maximum.accumulate(
        np.where(supported, x_grid, -1), axis=1)
    right = np.minimum.accumulate(
        np.where(supported, x_grid, width)[:, ::-1], axis=1)[:, ::-1]
    left_valid = left >= 0
    right_valid = right < width
    left_safe = np.clip(left, 0, width - 1)
    right_safe = np.clip(right, 0, width - 1)
    left_owner = np.take_along_axis(owner, left_safe, axis=1)
    right_owner = np.take_along_axis(owner, right_safe, axis=1)
    left_depth = np.take_along_axis(
        depth, np.clip(left_owner, 0, width - 1), axis=1)
    right_depth = np.take_along_axis(
        depth, np.clip(right_owner, 0, width - 1), axis=1)
    left_distance = x_grid - left
    right_distance = right - x_grid
    # Nearness is larger for the foreground.  Prefer the farther supported
    # layer; use destination distance only when both sides have the same depth.
    choose_left = left_valid & (~right_valid | (left_depth < right_depth - 1.0e-5) |
                                ((np.abs(left_depth - right_depth) <= 1.0e-5) &
                                 (left_distance <= right_distance)))
    donors = np.where(choose_left, left, right).astype(np.int32)
    donors[~(left_valid | right_valid)] = -1
    donor_safe = np.clip(donors, 0, width - 1)
    donor_colour = np.take_along_axis(
        warped, donor_safe[..., None].repeat(3, axis=2), axis=1)
    holes = ~supported
    filled[holes] = donor_colour[holes]
    return filled, donors


def _reconstruct_eye(
    current_yuv: np.ndarray,
    future_yuv: np.ndarray,
    current_depth: np.ndarray,
    future_depth: Optional[np.ndarray],
    disparity: np.ndarray,
    flow: FlowField,
    eye_sign: float,
    *,
    confidence_threshold: float,
    color_sigma: float,
    depth_sigma: float,
) -> EyeReconstruction:
    warped, supported, owner = _forward_warp(
        current_yuv, current_depth, disparity, eye_sign)
    baseline, donors = _nearest_background_fill(
        warped, supported, owner, current_depth)
    holes = ~supported
    height, width = holes.shape
    confidence = np.zeros((height, width), dtype=np.float32)
    accepted = np.zeros_like(holes)
    result = baseline.copy()
    if not np.any(holes):
        return EyeReconstruction(
            baseline, result, supported, holes, accepted, confidence)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    donor_x = np.clip(donors, 0, width - 1)
    donor_valid = donors >= 0
    donor_owner_x = np.take_along_axis(owner, donor_x, axis=1)
    donor_owner_x = np.clip(donor_owner_x, 0, width - 1)

    # Flow is measured on visible current background.  Extrapolate that local
    # motion into the adjacent stereo hole, then ask the future frame whether
    # the same background really became visible there.
    flow_x = flow.x[yy.astype(np.intp), donor_owner_x]
    flow_y = flow.y[yy.astype(np.intp), donor_owner_x]
    flow_rel = flow.reliability[yy.astype(np.intp), donor_owner_x]
    future_x = xx - flow_x
    future_y = yy - flow_y
    candidate, in_bounds = _bilinear_sample(future_yuv, future_x, future_y)

    donor_colour = current_yuv[
        yy.astype(np.intp), donor_owner_x]
    luma_delta = np.abs(candidate[..., 0] - donor_colour[..., 0])
    chroma_delta = np.linalg.norm(
        candidate[..., 1:3] - donor_colour[..., 1:3], axis=2)
    colour_conf = np.exp(-(
        luma_delta + 0.45 * chroma_delta) / max(color_sigma, 1.0e-4))
    evidence = flow_rel * colour_conf * in_bounds.astype(np.float32)

    if future_depth is not None:
        candidate_depth, depth_bounds = _bilinear_sample(
            future_depth, future_x, future_y)
        donor_depth = current_depth[
            yy.astype(np.intp), donor_owner_x]
        # A legitimate reveal belongs to the same rear layer as the spatial
        # donor.  This rejects a foreground that still covers the target in
        # the future frame without requiring semantic segmentation.
        depth_delta = np.abs(candidate_depth - donor_depth)
        depth_conf = np.exp(-depth_delta / max(depth_sigma, 1.0e-4))
        evidence *= depth_conf * depth_bounds.astype(np.float32)

    confidence[holes & donor_valid] = evidence[holes & donor_valid]
    accepted = holes & donor_valid & (confidence >= confidence_threshold)
    # A narrow soft knee avoids a binary temporal seam.  Pixels below the
    # threshold remain exactly the established spatial fallback.
    blend = np.clip(
        (confidence - confidence_threshold) /
        max(1.0 - confidence_threshold, 1.0e-4), 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    result = baseline * (1.0 - blend[..., None]) + candidate * blend[..., None]
    result[~holes] = warped[~holes]
    return EyeReconstruction(
        baseline, result, supported, holes, accepted, confidence)


def reconstruct_stereo(
    current_yuv: np.ndarray,
    future_yuv: np.ndarray,
    current_depth: np.ndarray,
    flow: FlowField,
    *,
    future_depth: Optional[np.ndarray] = None,
    strength_pct: float = 1.5,
    convergence: float = 0.5,
    confidence_threshold: float = 0.24,
    color_sigma: float = 0.12,
    depth_sigma: float = 0.10,
) -> StereoReconstruction:
    """Reconstruct a stereo pair using a future frame only inside DIBR holes."""

    current = np.asarray(current_yuv, dtype=np.float32)
    future = np.asarray(future_yuv, dtype=np.float32)
    depth = np.asarray(current_depth, dtype=np.float32)
    if current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("current_yuv must be HxWx3")
    if future.shape != current.shape or depth.shape != current.shape[:2]:
        raise ValueError("future_yuv/depth must match current_yuv")
    if flow.x.shape != depth.shape:
        raise ValueError("flow must match the analysis grid")
    future_d = None if future_depth is None else np.asarray(
        future_depth, dtype=np.float32)
    if future_d is not None and future_d.shape != depth.shape:
        raise ValueError("future_depth must match current_depth")
    if not math.isfinite(strength_pct) or strength_pct < 0.0:
        raise ValueError("strength_pct must be finite and non-negative")
    convergence = float(np.clip(convergence, 0.0, 1.0))
    # Match the live renderer's width-relative disparity budget.  Nearness is
    # already normalized; using symmetric convergence spans keeps the maximum
    # on each side bounded by strength_pct * image width.
    delta = depth - convergence
    span = np.where(delta >= 0.0, max(0.08, 1.0 - convergence),
                    max(0.08, convergence))
    disparity = (strength_pct / 100.0) * depth.shape[1] * delta / span
    left = _reconstruct_eye(
        current, future, depth, future_d, disparity, flow, +1.0,
        confidence_threshold=confidence_threshold,
        color_sigma=color_sigma, depth_sigma=depth_sigma)
    right = _reconstruct_eye(
        current, future, depth, future_d, disparity, flow, -1.0,
        confidence_threshold=confidence_threshold,
        color_sigma=color_sigma, depth_sigma=depth_sigma)
    return StereoReconstruction(left, right)


def reconstruction_metrics(result: StereoReconstruction) -> dict[str, float]:
    """Coverage/confidence metrics that remain meaningful without GT stereo."""

    holes = np.concatenate((result.left.holes.ravel(), result.right.holes.ravel()))
    accepted = np.concatenate((
        result.left.accepted.ravel(), result.right.accepted.ravel()))
    confidence = np.concatenate((
        result.left.confidence.ravel(), result.right.confidence.ravel()))
    hole_count = int(holes.sum())
    accepted_count = int(accepted.sum())
    return {
        "hole_pixels": float(hole_count),
        "hole_pct": 100.0 * hole_count / max(1, holes.size),
        "lookahead_pixels": float(accepted_count),
        "lookahead_coverage_pct": 100.0 * accepted_count / max(1, hole_count),
        "lookahead_confidence_mean": (
            float(confidence[accepted].mean()) if accepted_count else 0.0),
    }


def yuv_to_bgr(yuv: np.ndarray) -> np.ndarray:
    """Normalized full-resolution YUV analysis image -> display BGR8."""

    data = np.clip(np.asarray(yuv) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_YUV2BGR)
