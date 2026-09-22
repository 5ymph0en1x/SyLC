"""Regression tests for the accepted multiscale depth + safety pipeline."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
for _candidate in reversed((_ROOT / "build_py314" / "python" / "Release",
                            _ROOT / "runtime")):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
_DLL_HANDLES = [
    os.add_dll_directory(str(path))
    for path in (_ROOT / "runtime", _ROOT / "ort_tensorrt")
    if path.is_dir() and hasattr(os, "add_dll_directory")
]

native = pytest.importorskip("mvc_demuxer_cpp")
if not hasattr(native, "DepthMultiscaleFilter"):
    pytest.skip("module built without Synth3D temporal filters",
                allow_module_level=True)


REF_DT = 1000.0 * 1001.0 / 24000.0


def _q16(value):
    return np.round(np.asarray(value) * 65535.0).astype(np.uint16)


def test_safety_attack_is_immediate_and_release_is_conservative():
    filt = native.SafetyTemporalFilter(8, 8)
    low = np.full((8, 8), _q16(0.20), dtype=np.uint16)
    high = np.full((8, 8), _q16(0.90), dtype=np.uint16)
    unsafe = np.full((8, 8), _q16(0.10), dtype=np.uint16)

    np.testing.assert_array_equal(filt.apply(low), low)
    released = filt.apply(high)
    assert np.all(released <= high)
    assert np.max(np.abs(released.astype(np.int32) - _q16(0.212))) <= 2

    attacked = filt.apply(unsafe)
    np.testing.assert_array_equal(attacked, unsafe)

    filt.reset()
    np.testing.assert_array_equal(filt.apply(high), high)


def test_safety_release_is_video_time_invariant():
    low = np.full((8, 8), _q16(0.20), dtype=np.uint16)
    high = np.full((8, 8), _q16(0.90), dtype=np.uint16)
    one = native.SafetyTemporalFilter(8, 8)
    two = native.SafetyTemporalFilter(8, 8)
    one.apply(low)
    two.apply(low)

    one_result = one.apply(high, source_dt_ms=2.0 * REF_DT)
    two.apply(high, source_dt_ms=REF_DT)
    two_result = two.apply(high, source_dt_ms=REF_DT)
    assert np.max(np.abs(one_result.astype(np.int32) -
                         two_result.astype(np.int32))) <= 1


def test_multiscale_removes_global_affine_gauge_and_reset_reprime():
    height, width = 16, 24
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    base = 0.20 + 0.48 * xx / (width - 1) + 0.10 * yy / (height - 1)
    changed = 1.08 * base - 0.03
    base_q16 = _q16(base)
    changed_q16 = _q16(changed)
    ones = np.ones((height, width), dtype=np.float32)
    zeros = np.zeros((height, width), dtype=np.float32)

    filt = native.DepthMultiscaleFilter(width, height)
    np.testing.assert_array_equal(
        filt.apply(base_q16, reliability=ones,
                   confidence=ones, boundary=zeros),
        base_q16)
    filtered = filt.apply(changed_q16, reliability=ones,
                          confidence=ones, boundary=zeros)
    raw_change = np.mean(np.abs(changed_q16.astype(np.int32) -
                                base_q16.astype(np.int32)))
    filtered_change = np.mean(np.abs(filtered.astype(np.int32) -
                                     base_q16.astype(np.int32)))
    assert filtered_change < 0.25 * raw_change

    filt.reset()
    np.testing.assert_array_equal(
        filt.apply(changed_q16, reliability=ones,
                   confidence=ones, boundary=zeros),
        changed_q16)


def test_multiscale_output_is_bit_exact_across_worker_counts():
    height, width = 16, 24
    rng = np.random.default_rng(20260808)
    frames = [_q16(rng.uniform(0.15, 0.85, (height, width)))
              for _ in range(3)]
    flow_x = rng.uniform(-0.4, 0.4, (height, width)).astype(np.float32)
    flow_y = rng.uniform(-0.4, 0.4, (height, width)).astype(np.float32)
    reliability = rng.uniform(0.65, 1.0, (height, width)).astype(np.float32)
    confidence = np.ones((height, width), dtype=np.float32)
    boundary = np.zeros((height, width), dtype=np.float32)
    sequential = native.DepthMultiscaleFilter(width, height)
    parallel = native.DepthMultiscaleFilter(width, height)
    sequential.worker_threads = 1
    parallel.worker_threads = 4

    for frame in frames:
        a = sequential.apply(frame, flow_x, flow_y, reliability,
                             confidence, boundary)
        b = parallel.apply(frame, flow_x, flow_y, reliability,
                           confidence, boundary)
        np.testing.assert_array_equal(a, b)
