"""Offline A/B harness for SyLC's precomputed-human-matte prototype.

Example:
    python tools/synth3d_matte_prototype.py \
        --frame frame.png --depth depth.npy --matte alpha.png \
        --output matte_ab --benchmark 120

`depth.npy` may be uint16 nearness or float nearness in [0, 1].  The matte is
an ordinary grayscale PNG produced by MatAnyone/MatAnyone 2.  The tool renders
off/guard/contour through the real D3D11 shader and writes both eyes, SBS and a
red/cyan anaglyph for each policy.  It never starts the ONNX depth service.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import time

import cv2
import numpy as np

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))
    sys.path.insert(0, str(_SOURCE_ROOT / "runtime"))
import mvc_demuxer_cpp


def _hidden_hwnd() -> int:
    user32 = ctypes.windll.user32
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    hwnd = user32.CreateWindowExW(
        0, "STATIC", "SyLC matte prototype", 0,
        0, 0, 16, 16, wintypes.HWND(-3), None, None, None)
    if not hwnd:
        raise RuntimeError("could not create the hidden D3D11 window")
    return int(hwnd)


def _bgr_to_yuv420(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BT.709 limited-range YUV420 matching synth3d's plane convention."""
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    yf = 0.2126 * r + 0.7152 * g + 0.0722 * b
    y = np.rint(16.0 + 219.0 * yf).clip(0, 255).astype(np.uint8)
    u444 = 128.0 + 224.0 * (b - yf) / 1.8556
    v444 = 128.0 + 224.0 * (r - yf) / 1.5748
    h, w = y.shape
    if h % 2 or w % 2:
        raise ValueError("frame dimensions must be even for YUV420")
    u = u444.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    v = v444.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return y, np.rint(u).clip(0, 255).astype(np.uint8), \
        np.rint(v).clip(0, 255).astype(np.uint8)


def _yuv420_to_bgr(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u444 = cv2.resize(u, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    v444 = cv2.resize(v, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    yf = (y.astype(np.float32) - 16.0) / 219.0
    uf = (u444.astype(np.float32) - 128.0) / 224.0
    vf = (v444.astype(np.float32) - 128.0) / 224.0
    r = yf + 1.5748 * vf
    b = yf + 1.8556 * uf
    g = yf - 0.18732 * uf - 0.46812 * vf
    rgb = np.stack([r, g, b], axis=-1)
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[..., ::-1]


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"cannot read depth map: {path}")
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2-D map")
    if depth.dtype == np.uint16:
        return np.ascontiguousarray(depth)
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.all():
        raise ValueError("depth contains NaN or infinity")
    if depth.min() < 0.0 or depth.max() > 1.0:
        lo, hi = np.percentile(depth, (1.0, 99.0))
        depth = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return np.rint(depth * 65535.0).astype(np.uint16)


def _render(renderer, planes, depth, matte, mode):
    renderer.synth3d_set_test_depth(depth)
    renderer.synth3d_set_test_matte(matte, mode or "guard")
    renderer.set_yuv_frame(*planes)
    renderer.present(0)
    left = _yuv420_to_bgr(
        renderer.synth3d_read_plane(0),
        renderer.synth3d_read_plane(1),
        renderer.synth3d_read_plane(2))
    right = _yuv420_to_bgr(
        renderer.synth3d_read_plane(3),
        renderer.synth3d_read_plane(4),
        renderer.synth3d_read_plane(5))
    return left, right


def _write_views(output: Path, name: str, left: np.ndarray, right: np.ndarray):
    cv2.imwrite(str(output / f"{name}_left.png"), left)
    cv2.imwrite(str(output / f"{name}_right.png"), right)
    cv2.imwrite(str(output / f"{name}_sbs.png"), np.hstack([left, right]))
    anaglyph = np.empty_like(left)
    anaglyph[..., 0] = right[..., 0]  # B/G from right, R from left
    anaglyph[..., 1] = right[..., 1]
    anaglyph[..., 2] = left[..., 2]
    cv2.imwrite(str(output / f"{name}_anaglyph.png"), anaglyph)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--matte", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("matte_ab"))
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("--convergence", type=float, default=0.5)
    parser.add_argument("--benchmark", type=int, default=0,
                        help="timed frames per policy; readback forces GPU completion")
    args = parser.parse_args()

    if not hasattr(mvc_demuxer_cpp.NativeRenderer, "synth3d_set_test_matte"):
        raise RuntimeError("loaded mvc_demuxer_cpp predates the matte prototype")
    frame = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
    matte = cv2.imread(str(args.matte), cv2.IMREAD_GRAYSCALE)
    if frame is None or matte is None:
        raise ValueError("frame or matte image could not be decoded")
    depth = _load_depth(args.depth)
    h, w = frame.shape[:2]
    if abs((matte.shape[1] / matte.shape[0]) - (w / h)) > 0.01:
        raise ValueError("matte and frame aspect ratios differ")
    planes = _bgr_to_yuv420(frame)
    matte = np.ascontiguousarray(matte)
    args.output.mkdir(parents=True, exist_ok=True)

    hwnd = _hidden_hwnd()
    renderer = mvc_demuxer_cpp.NativeRenderer()
    try:
        if not renderer.initialize(hwnd, w, h, False):
            raise RuntimeError(renderer.last_error())
        if not renderer.set_synth3d(
                True, args.strength, args.convergence, False,
                side=int(depth.shape[1]), grid_width=int(depth.shape[1]),
                grid_height=int(depth.shape[0])):
            raise RuntimeError(renderer.last_error())

        policies = (
                ("off", None, None),
                ("guard", "guard", matte),
                ("contour", "contour", matte))
        for name, mode, alpha in policies:
            left, right = _render(renderer, planes, depth, alpha, mode)
            _write_views(args.output, name, left, right)
        if args.benchmark > 0:
            samples = {name: [] for name, _, _ in policies}
            orders = (policies, tuple(reversed(policies)),
                      policies[1:] + policies[:1])
            warmup = max(12, min(32, args.benchmark // 4))

            def timed_frame():
                renderer.set_yuv_frame(*planes)
                renderer.present(0)
                # A blocking readback forces completion of both warp draws;
                # keep colour conversion and PNG work outside the timing.
                renderer.synth3d_read_plane(0)

            for order in orders:
                for name, mode, alpha in order:
                    renderer.synth3d_set_test_matte(alpha, mode or "guard")
                    for _ in range(warmup):
                        timed_frame()
                    start = time.perf_counter()
                    for _ in range(args.benchmark):
                        timed_frame()
                    samples[name].append(
                        1000.0 * (time.perf_counter() - start) /
                        args.benchmark)
            for name, values in samples.items():
                milliseconds = float(np.median(values))
                print(f"{name:7s}: {milliseconds:7.3f} ms/frame "
                      f"(range {min(values):.3f}..{max(values):.3f}, "
                      "GPU-forced)")
        print(f"wrote A/B views to {args.output.resolve()}")
    finally:
        renderer.shutdown()
        ctypes.windll.user32.DestroyWindow(wintypes.HWND(hwnd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
