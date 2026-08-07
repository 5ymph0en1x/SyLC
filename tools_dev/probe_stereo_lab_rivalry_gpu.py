"""D3D11 regression probe for subpixel binocular rivalry.

The scene deliberately puts high-contrast source edges next to sloped and
piecewise depth.  Both eyes are rendered through the production Synth3D warp,
first with the immutable raw path and then with Stereo Lab.  The score follows
source points into both projected eyes and keeps only points that survive a
small z-buffer visibility test in both views.

This is a developer probe, not a portable pytest: it requires Windows, D3D11
and a freshly built ``mvc_demuxer_cpp`` native module.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build_py314" / "python" / "Release"))
sys.path.insert(1, str(ROOT / "src"))
_DLL_HANDLES = [
    os.add_dll_directory(str(path))
    for path in (ROOT / "runtime", ROOT / "ort_tensorrt")
    if path.exists()
]

import mvc_demuxer_cpp as native


def _hidden_hwnd() -> int:
    user32 = ctypes.windll.user32
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    hwnd = user32.CreateWindowExW(
        0, "STATIC", "SyLC Stereo Lab rivalry probe", 0,
        0, 0, 16, 16, wintypes.HWND(-3), None, None, None)
    if not hwnd:
        raise RuntimeError("could not create the hidden D3D11 window")
    return int(hwnd)


def _scene(width: int, height: int, seed: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    # Non-periodic texture prevents a wrong epipolar match from looking valid.
    y = 38.0 + 0.11 * xx + 13.0 * np.sin(0.017 * xx + 0.031 * yy)
    y += 8.0 * np.cos(0.043 * xx - 0.019 * yy)
    for edge, delta in ((0.17, 92.0), (0.39, -74.0),
                        (0.61, 106.0), (0.82, -83.0)):
        phase = edge * 5.0 + rng.uniform(-0.8, 0.8)
        boundary = edge * width + 7.0 * np.sin(yy * 0.037 + phase)
        y += delta * (xx >= boundary)
    # Independent broad bands make the source path cross adjacent contrast
    # while depth ownership changes elsewhere.  Widths stay well above the
    # fine-structure veto's one-pixel regime.
    for edge in np.sort(rng.uniform(0.08, 0.92, 7)):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        boundary = edge * width + rng.uniform(2.0, 9.0) * np.sin(
            yy * rng.uniform(0.018, 0.052) + phase)
        level = rng.uniform(34.0, 218.0)
        y = np.where(xx >= boundary, 0.72 * y + 0.28 * level, y)
    # A one-pixel ridge remains present to catch destructive "fixes".
    ridge = np.rint(0.52 * width + 0.12 * np.arange(height)).astype(np.int32)
    ridge = np.clip(ridge, 0, width - 1)
    y[np.arange(height), ridge] = 232.0
    y = np.clip(y, 16.0, 235.0).astype(np.uint8)

    n = 0.18 + 0.34 * (xx / max(width - 1, 1))
    n += 0.055 * np.sin(0.021 * xx + 0.014 * yy)
    slab_a = (xx > 0.235 * width + 0.10 * yy) & (xx < 0.43 * width + 0.10 * yy)
    slab_b = (xx > 0.665 * width - 0.08 * yy) & (xx < 0.79 * width - 0.08 * yy)
    n = np.where(slab_a, 0.86, n)
    n = np.where(slab_b, 0.73, n)
    for edge in np.sort(rng.uniform(0.10, 0.90, 6)):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        boundary = edge * width + rng.uniform(3.0, 12.0) * np.sin(
            yy * rng.uniform(0.014, 0.046) + phase)
        n = np.where(xx >= boundary, rng.uniform(0.10, 0.92), n)
    depth = np.rint(np.clip(n, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cy, cx = np.mgrid[0:height // 2, 0:width // 2]
    u = 128.0 + 34.0 * np.sin(0.031 * cx + 0.017 * cy)
    v = 128.0 + 29.0 * np.cos(0.023 * cx - 0.027 * cy)
    for edge in np.sort(rng.uniform(0.09, 0.91, 6)):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        boundary = edge * (width // 2) + rng.uniform(1.0, 5.0) * np.sin(
            cy * rng.uniform(0.021, 0.061) + phase)
        u = np.where(cx >= boundary, rng.uniform(54.0, 205.0), u)
        v = np.where(cx >= boundary, rng.uniform(48.0, 211.0), v)
    return (np.ascontiguousarray(y),
            np.ascontiguousarray(np.clip(u, 16.0, 240.0).astype(np.uint8)),
            np.ascontiguousarray(np.clip(v, 16.0, 240.0).astype(np.uint8)),
            np.ascontiguousarray(depth))


def _comfortable_offset(nearness: np.ndarray, convergence: float) -> np.ndarray:
    d = nearness - convergence
    span = np.where(d >= 0.0, max(0.08, 1.0 - convergence),
                    max(0.08, convergence))
    z = np.clip(np.abs(d) / span, 0.0, 1.0)
    knee = np.where(d >= 0.0, 0.68, 0.76)
    over = np.maximum(0.0, z - knee)
    softened = np.where(z <= knee, z, knee + over / (1.0 + 1.60 * over))
    return np.where(d < 0.0, -1.0, 1.0) * softened * span


def _render(hwnd: int, y: np.ndarray, u: np.ndarray, v: np.ndarray,
            depth: np.ndarray,
            strength_pct: float, convergence: float,
            stereo_lab: bool) -> tuple[
                np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray, str]:
    height, width = y.shape
    renderer = native.NativeRenderer()
    try:
        if not renderer.initialize(hwnd, width, height, False):
            raise RuntimeError(renderer.last_error())
        if not renderer.set_synth3d(
                True, strength_pct, convergence, False,
                side=width, grid_width=width, grid_height=height,
                stereo_lab=stereo_lab):
            raise RuntimeError(renderer.last_error())
        renderer.synth3d_set_test_depth(depth)
        if not renderer.set_yuv_frame(y, u, v):
            raise RuntimeError(renderer.last_error())
        if not renderer.present(0):
            raise RuntimeError(renderer.last_error())
        planes = tuple(
            np.asarray(renderer.synth3d_read_plane(slot)).copy()
            for slot in range(6))
        return (*planes, renderer.synth3d_status())
    finally:
        renderer.shutdown()


def _bilinear_rows(image: np.ndarray, x: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    x = np.clip(x, 0.0, width - 1.0)
    x0 = np.floor(x).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    t = x - x0
    rows = np.arange(image.shape[0])[:, None]
    return (1.0 - t) * image[rows, x0] + t * image[rows, x1]


def _visible_in_eye(nearness: np.ndarray, destination_x: np.ndarray) -> np.ndarray:
    height, width = nearness.shape
    target = np.rint(destination_x).astype(np.int32)
    inside = (target >= 0) & (target < width)
    target = np.clip(target, 0, width - 1)
    rows = np.arange(height)[:, None]
    flat_target = (rows * width + target).ravel()
    front = np.full(height * width, -np.inf, np.float32)
    np.maximum.at(front, flat_target[inside.ravel()], nearness.ravel()[inside.ravel()])
    return inside & (nearness >= front[flat_target].reshape(height, width) - 1e-4)


def _score(left: np.ndarray, right: np.ndarray, source: np.ndarray,
           depth: np.ndarray, strength_pct: float,
           convergence: float) -> dict[str, float]:
    height, width = source.shape
    nearness = depth.astype(np.float32) / 65535.0
    disparity_px = (strength_pct / 100.0) * width * _comfortable_offset(
        nearness, convergence)
    source_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    left_x = source_x + 0.5 * disparity_px
    right_x = source_x - 0.5 * disparity_px
    visible = _visible_in_eye(nearness, left_x) & _visible_in_eye(nearness, right_x)
    visible &= (left_x >= 0.0) & (left_x <= width - 1.0)
    visible &= (right_x >= 0.0) & (right_x <= width - 1.0)
    visible &= np.abs(disparity_px) >= 1.5

    source_f = source.astype(np.float32)
    gx = np.zeros_like(source_f)
    gx[:, 1:-1] = np.abs(source_f[:, 2:] - source_f[:, :-2])
    rivalry_zone = visible & (gx >= 12.0)
    paired_error = np.abs(
        _bilinear_rows(left.astype(np.float32), left_x) -
        _bilinear_rows(right.astype(np.float32), right_x))
    values = paired_error[rivalry_zone]
    if values.size == 0:
        raise RuntimeError("synthetic scene produced no visible rivalry samples")
    return {
        "sample_count": int(values.size),
        "mean_code": float(values.mean()),
        "p95_code": float(np.percentile(values, 95.0)),
        "p99_code": float(np.percentile(values, 99.0)),
        "over_8_code_pct": float(100.0 * np.mean(values > 8.0)),
        "over_16_code_pct": float(100.0 * np.mean(values > 16.0)),
    }


def _score_chroma(left_u: np.ndarray, left_v: np.ndarray,
                  right_u: np.ndarray, right_v: np.ndarray,
                  source_u: np.ndarray, source_v: np.ndarray,
                  depth: np.ndarray, strength_pct: float,
                  convergence: float) -> dict[str, float]:
    nearness = depth.astype(np.float32).reshape(
        source_u.shape[0], 2, source_u.shape[1], 2).mean(axis=(1, 3)) / 65535.0
    height, width = source_u.shape
    disparity_px = (strength_pct / 100.0) * width * _comfortable_offset(
        nearness, convergence)
    source_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    left_x = source_x + 0.5 * disparity_px
    right_x = source_x - 0.5 * disparity_px
    visible = _visible_in_eye(nearness, left_x) & _visible_in_eye(nearness, right_x)
    visible &= (left_x >= 0.0) & (left_x <= width - 1.0)
    visible &= (right_x >= 0.0) & (right_x <= width - 1.0)
    visible &= np.abs(disparity_px) >= 0.75

    su = source_u.astype(np.float32)
    sv = source_v.astype(np.float32)
    gradient = np.zeros_like(su)
    gradient[:, 1:-1] = np.maximum(
        np.abs(su[:, 2:] - su[:, :-2]),
        np.abs(sv[:, 2:] - sv[:, :-2]))
    rivalry_zone = visible & (gradient >= 8.0)
    du = _bilinear_rows(left_u.astype(np.float32), left_x) - \
         _bilinear_rows(right_u.astype(np.float32), right_x)
    dv = _bilinear_rows(left_v.astype(np.float32), left_x) - \
         _bilinear_rows(right_v.astype(np.float32), right_x)
    values = np.sqrt(du * du + dv * dv)[rivalry_zone]
    if values.size == 0:
        raise RuntimeError("synthetic scene produced no visible chroma samples")
    return {
        "sample_count": int(values.size),
        "mean_code": float(values.mean()),
        "p95_code": float(np.percentile(values, 95.0)),
        "p99_code": float(np.percentile(values, 99.0)),
        "over_8_code_pct": float(100.0 * np.mean(values > 8.0)),
        "over_16_code_pct": float(100.0 * np.mean(values > 16.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--strength", type=float, default=2.4)
    parser.add_argument("--convergence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--capture", type=Path,
                        help="optional NPZ containing source/depth and raw/Lab eyes")
    args = parser.parse_args()
    if args.width < 320 or args.height < 180 or args.width % 2 or args.height % 2:
        parser.error("width/height must be even and at least 320x180")

    source, source_u, source_v, depth = _scene(
        args.width, args.height, args.seed)
    hwnd = _hidden_hwnd()
    try:
        raw_planes = _render(
            hwnd, source, source_u, source_v, depth,
            args.strength, args.convergence, False)
        lab_planes = _render(
            hwnd, source, source_u, source_v, depth,
            args.strength, args.convergence, True)
    finally:
        ctypes.windll.user32.DestroyWindow(wintypes.HWND(hwnd))

    raw_l, raw_u, raw_v, raw_r, raw_ur, raw_vr, raw_status = raw_planes
    lab_l, lab_u, lab_v, lab_r, lab_ur, lab_vr, lab_status = lab_planes
    raw = _score(raw_l, raw_r, source, depth, args.strength, args.convergence)
    lab = _score(lab_l, lab_r, source, depth, args.strength, args.convergence)
    raw_chroma = _score_chroma(
        raw_u, raw_v, raw_ur, raw_vr, source_u, source_v,
        depth, args.strength, args.convergence)
    lab_chroma = _score_chroma(
        lab_u, lab_v, lab_ur, lab_vr, source_u, source_v,
        depth, args.strength, args.convergence)
    changed = np.maximum(np.abs(lab_l.astype(np.int16) - raw_l.astype(np.int16)),
                         np.abs(lab_r.astype(np.int16) - raw_r.astype(np.int16)))
    result = {
        "raw": raw,
        "lab": lab,
        "raw_chroma": raw_chroma,
        "lab_chroma": lab_chroma,
        "p95_improvement_pct": 100.0 * (raw["p95_code"] - lab["p95_code"]) /
                               max(raw["p95_code"], 1e-6),
        "changed_pixel_pct": float(100.0 * np.mean(changed > 0)),
        "max_change_code": int(changed.max()),
        "raw_status": raw_status,
        "lab_status": lab_status,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.capture:
        np.savez_compressed(
            args.capture, source=source, source_u=source_u, source_v=source_v,
            depth=depth,
            raw_left=raw_l, raw_right=raw_r,
            raw_ul=raw_u, raw_vl=raw_v, raw_ur=raw_ur, raw_vr=raw_vr,
            lab_left=lab_l, lab_right=lab_r,
            lab_ul=lab_u, lab_vl=lab_v, lab_ur=lab_ur, lab_vr=lab_vr)

    # This threshold is intentionally modest: the probe guards against a
    # disconnected/no-op Lab while leaving calibration to real-film captures.
    if lab["p95_code"] > raw["p95_code"] + 0.25:
        raise SystemExit("Stereo Lab increased paired p95 rivalry")
    if lab_chroma["p95_code"] > raw_chroma["p95_code"] + 0.50:
        raise SystemExit("Stereo Lab increased paired chroma p95 rivalry")
    if result["changed_pixel_pct"] <= 0.0:
        raise SystemExit("Stereo Lab made no correction in the rivalry scene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
