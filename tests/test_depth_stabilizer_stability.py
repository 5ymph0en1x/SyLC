"""Regression tests for the production DepthStabilizer stability memory.

The flow fusion's motion signal includes a transport-uncertainty penalty.  It
must not make ordinary aligned film motion erase the persistent surface state
twice (once in reproject and once again in step), otherwise the robust history
and local jitter deadband never arm.
"""

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
if not hasattr(native, "DepthStabilizer"):
    pytest.skip("module built without DepthStabilizer", allow_module_level=True)


def _arrays(confidence=0.75, motion=0.065, boundary=0.20):
    """Inputs in production units for an 8x8 aligned moving surface."""
    n = 64
    source_dt_ms = 41.7
    video_time_scale = source_dt_ms / native.DepthStabilizer.kReferenceDtMs
    raw = np.linspace(0.1, 0.9, n, dtype=np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    # step() converts the per-observation signal back to reference-interval
    # velocity.  0.065 is the measured status-line value on normal film.
    motion_map = np.full(n, motion * video_time_scale, dtype=np.float32)
    confidence_map = np.full(n, confidence, dtype=np.float32)
    boundary_map = np.full(n, boundary, dtype=np.float32)
    reliability = np.full(n, 0.70, dtype=np.float32)
    return (source_dt_ms, raw, zeros, motion_map, confidence_map,
            boundary_map, reliability)


def test_aligned_film_motion_accumulates_surface_stability():
    (source_dt_ms, raw, zeros, motion, confidence,
     boundary, reliability) = _arrays()
    stabilizer = native.DepthStabilizer(raw.size)
    stabilizer.set_source_dt_ms(source_dt_ms)

    _, was_cut = stabilizer.step(
        raw, None, 0.0, confidence, boundary)
    assert not was_cut

    for _ in range(20):
        stabilizer.reproject(zeros, zeros, reliability, 8, 8)
        _, was_cut = stabilizer.step(
            raw, motion, 0.0, confidence, boundary)
        assert not was_cut

    # Before the fix this exact sequence reads ~1.3e-5 (and crosses 0.002
    # around map ten), so neither the 0.45 history threshold nor the 0.62
    # local-deadband threshold is ever reachable.
    assert stabilizer.last_stability > 0.62
    assert stabilizer.last_history_support >= 2.5


def test_moving_depth_boundary_still_collapses_history():
    (source_dt_ms, raw, zeros, motion, confidence,
     _boundary, reliability) = _arrays(boundary=0.60)
    boundary = np.full(raw.size, 0.60, dtype=np.float32)
    stabilizer = native.DepthStabilizer(raw.size)
    stabilizer.set_source_dt_ms(source_dt_ms)
    stabilizer.step(raw, None, 0.0, confidence, boundary)

    for _ in range(20):
        stabilizer.reproject(zeros, zeros, reliability, 8, 8)
        stabilizer.step(raw, motion, 0.0, confidence, boundary)

    assert stabilizer.last_stability < 0.20
    assert stabilizer.last_history_support == pytest.approx(1.0)
def test_fine_structure_rejects_moderate_background_motion_bypass():
    """A fixed wire over moving fog keeps its temporal majority armed."""
    (source_dt_ms, raw, zeros, motion, confidence,
     _boundary, reliability) = _arrays(boundary=0.60)
    boundary = np.full(raw.size, 0.60, dtype=np.float32)
    fine_structure = np.ones(raw.size, dtype=np.float32)
    stabilizer = native.DepthStabilizer(raw.size)
    stabilizer.set_source_dt_ms(source_dt_ms)
    stabilizer.step(
        raw, None, 0.0, confidence, boundary, fine_structure)

    for _ in range(20):
        stabilizer.reproject(zeros, zeros, reliability, 8, 8)
        stabilizer.step(
            raw, motion, 0.0, confidence, boundary, fine_structure)

    assert stabilizer.last_stability > 0.62
    assert stabilizer.last_history_support >= 2.5


def test_genuinely_fast_fine_structure_still_takes_fast_path():
    """The filament guard is a discount, not an unconditional freeze."""
    (source_dt_ms, raw, zeros, motion, confidence,
     _boundary, reliability) = _arrays(motion=0.35, boundary=0.60)
    boundary = np.full(raw.size, 0.60, dtype=np.float32)
    fine_structure = np.ones(raw.size, dtype=np.float32)
    stabilizer = native.DepthStabilizer(raw.size)
    stabilizer.set_source_dt_ms(source_dt_ms)
    stabilizer.step(
        raw, None, 0.0, confidence, boundary, fine_structure)

    for _ in range(20):
        stabilizer.reproject(zeros, zeros, reliability, 8, 8)
        stabilizer.step(
            raw, motion, 0.0, confidence, boundary, fine_structure)

    assert stabilizer.last_stability < 0.20
    assert stabilizer.last_history_support == pytest.approx(1.0)


def test_production_detector_separates_wire_from_broad_edge():
    """A one-texel spoke is a ridge; an ordinary silhouette is not."""
    wire = np.full((9, 9), 0.75, dtype=np.float32)
    wire[:, 4] = 0.10
    wire_score = native._synth3d_fine_structure_test(wire)

    broad_edge = np.full((9, 9), 0.75, dtype=np.float32)
    broad_edge[:, :4] = 0.10
    edge_score = native._synth3d_fine_structure_test(broad_edge)

    assert wire_score[4, 4] > 0.95
    assert np.max(edge_score[1:-1, 1:-1]) < 0.05
