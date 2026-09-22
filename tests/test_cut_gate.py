"""CutGate: a hard cut must confirm, a flash must not.

The shipped rule required two consecutive exceedances and therefore confirmed
zero hard cuts on real content -- 28 of 28 missed over 4500 frames of 2160p,
and `cut_src=0` for a whole live session.  These tests pin the discriminator
that actually separates the two shapes, so the leg cannot silently die again.

Drives the C++ class through its pybind11 binding, so a build that regresses
the rule fails here rather than in a film.
"""

import os
import sys
from pathlib import Path

import pytest

# Same import prelude as tools_dev/profile_synth3d_e2e.py: the build output
# wins over runtime/ so a rebuild is tested before it is promoted, and the DLL
# directory must be registered explicitly -- PATH is ignored for extension
# modules on Python 3.8+.
_ROOT = Path(__file__).resolve().parents[1]
# Reversed on purpose: successive insert(0, ...) calls would put the LAST
# entry first and silently test the promoted binary instead of the build.
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
if not hasattr(native, "CutGate"):
    pytest.skip("module built without CutGate", allow_module_level=True)


def _gate(threshold=0.42):
    gate = native.CutGate()
    gate.histogram_threshold = threshold
    return gate


def _run(distances, threshold=0.42):
    """Return the frame indices on which the gate confirms."""
    gate = _gate(threshold)
    return [i for i, d in enumerate(distances)
            if gate.update(False, float(d))]


def test_hard_cut_confirms_one_frame_late():
    # The measured shape: quiet, one spike, quiet again.  Every one of the 28
    # real cuts traced looked like this, with the post-cut distance in the
    # 0.002-0.03 range because it is the same new shot.
    assert _run([0.01, 0.02, 0.68, 0.005, 0.01]) == [3]


def test_flash_is_refused():
    # A flash is TWO exceedances -- into the bright frame and back out -- and
    # the content after it is the content from before.  "Spike then calm"
    # alone confirms on the way out, so the second exceedance must be refused
    # the right to arm.  This case was found by this test, not in a film.
    assert _run([0.01, 0.02, 0.68, 0.64, 0.01]) == []


def test_flash_then_a_real_cut_still_confirms():
    # Refusing the flash must not desensitise the gate for what follows.
    assert _run([0.01, 0.68, 0.64, 0.01, 0.02, 0.71, 0.006]) == [6]


def test_two_frame_shot_is_refused_by_design():
    # Fail-closed: ambiguous with a flash on the scalar distance alone, and a
    # false cut costs more than a late one (2026-08-06 depth-pumping work).
    assert _run([0.01, 0.60, 0.60, 0.01]) == []


def test_real_transitions_from_the_film_all_confirm():
    # Verbatim (distance, next distance) pairs from tools_dev/cut_trace.py on
    # Oblivion 2160p, frames 40000+.
    for spike, after in ((0.679, 0.005), (0.574, 0.003), (0.515, 0.005),
                         (0.508, 0.002), (0.624, 0.174), (0.876, 0.002),
                         (0.647, 0.016), (0.647, 0.012), (0.606, 0.030),
                         (0.598, 0.007), (0.637, 0.012)):
        assert _run([0.01, spike, after]) == [2], (spike, after)


def test_below_threshold_never_confirms():
    assert _run([0.01, 0.30, 0.41, 0.02, 0.39, 0.01]) == []


def test_boundary_cut_confirms_immediately_and_disarms():
    # An authoritative boundary must not also produce a late second cut.
    gate = _gate()
    assert gate.update(False, 0.68) is False   # arms
    assert gate.update(True, 0.01) is True     # boundary confirms now
    assert gate.update(False, 0.01) is False   # and did not leave it armed


def test_dissolve_keeps_the_blend_path():
    # A dissolve is sustained elevation, never spike-then-quiet: refusing it
    # is what keeps the temporal blend instead of hard-resetting mid-fade.
    assert _run([0.01, 0.45, 0.47, 0.44, 0.46, 0.43, 0.30, 0.02]) == []


def test_pending_reports_the_undecided_frame():
    gate = _gate()
    assert gate.pending() is False
    gate.update(False, 0.68)
    assert gate.pending() is True
    gate.update(False, 0.005)
    assert gate.pending() is False
