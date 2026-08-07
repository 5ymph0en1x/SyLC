# -*- coding: utf-8 -*-
"""Pure model, preset and seek policies for SyLC's 2D-to-3D pipeline."""

import logging
import os
import sys

from sylc.runtime_paths import PROJECT_ROOT

from PySide6.QtCore import QSettings


logger = logging.getLogger(__name__)


# =============================================================================
# 2D->3D AI DEPTH PRESETS — (model candidates, inference grid) pairs
# =============================================================================
# Display order in the AI menu AND the inter-preset fallback order.
#
# Each entry pairs its candidate models with the square inference grid those
# graphs were exported for, and the two ALWAYS travel together from here to
# NativeRenderer.set_synth3d(..., side=): the grid is part of the shared depth
# service's cache key and of the ONNX graph's own fixed input shape, so a model
# opened against a service built for another grid is rejected by DepthEngine —
# after having already paid for a full ORT session build. Nothing downstream
# re-derives the grid from a filename, a resolution or a default, which is what
# makes that mismatch unreachable rather than merely unlikely.
#
# Candidates within one preset are therefore SAME-SIDE only. `da3_small.onnx` is
# the round-2 dynamic-axes export (no grid in its name because it accepts any),
# kept last in the Quality chain so an install directory holding only that file
# still works. A preset with no candidate on disk is greyed out in the menu, and
# if it is somehow the active one the resolution walks this order from the top —
# each preset with ITS OWN grid, so however far down it lands the pair is still
# one entry's.
SYNTH3D_DEPTH_PRESETS = (
    ("Quality",     ("da3_base_756.onnx", "da3_small_756.onnx",
                     "da3_small.onnx"),                          756),
    ("Balanced",    ("da3_base_518.onnx", "da3_small_518.onnx"),  518),
    ("Performance", ("da3_small_518.onnx",),                      518),
)
SYNTH3D_DEPTH_PRESET_DEFAULT = SYNTH3D_DEPTH_PRESETS[0][0]
# Fixed adaptive graphs shipped outside the user-facing square preset table.
# The selector itself discovers compatible HxW files generically; this literal
# is the offline TensorRT engine-probe/packaging contract and is pinned in tests.
SYNTH3D_ADAPTIVE_MODEL_GRIDS = (
    ("da3_base_756x406.onnx", 756, 406),
    ("da3_base_756x378.onnx", 756, 378),
    ("da3_base_756x350.onnx", 756, 350),
    ("da3_base_756x322.onnx", 756, 322),
    ("da3_small_756x406.onnx", 756, 406),
    ("da3_small_756x378.onnx", 756, 378),
    ("da3_small_756x350.onnx", 756, 350),
    ("da3_small_756x322.onnx", 756, 322),
    ("da3_base_518x280.onnx", 518, 280),
    ("da3_base_518x266.onnx", 518, 266),
    ("da3_base_518x238.onnx", 518, 238),
    ("da3_base_518x210.onnx", 518, 210),
    ("da3_small_518x280.onnx", 518, 280),
    ("da3_small_518x266.onnx", 518, 266),
    ("da3_small_518x238.onnx", 518, 238),
    ("da3_small_518x210.onnx", 518, 210),
)

# A seek used to bounce EVERY adaptive selection back through the square graph
# so the new position could re-earn its eight-observation verdict. Measured
# 2026-08-03 on a native 1920x808 master: that detour costs ~1-3 s of
# non-nominal depth per seek -- flat 2D, then depth computed on the SQUARE
# 756x756 grid over 2.376:1 content, then flat again, then the 300 ms ramp --
# and re-primes EMA/plate/tone/history TWICE instead of once.
# It only ever guarded against a matte that changes mid-title (a Scope/IMAX
# reel switch). A selection earned from the CODED frame dimensions cannot be
# invalidated by a seek: those dimensions belong to the file, not to the
# position. So only matte-derived overrides still take the detour.
# SYLC_SYNTH3D_SEEK_KEEP_ASPECT=0 restores the unconditional detour.
SYNTH3D_SEEK_KEEP_ASPECT = os.environ.get(
    'SYLC_SYNTH3D_SEEK_KEEP_ASPECT', '1').strip().lower() not in {
        '0', 'false', 'off', 'disabled', 'no'}

# The look-ahead advisory is dated against the pts of the frame ON SCREEN, not
# the interpolated UI clock. Measured 2026-08-04 over five consecutive windows:
# the UI clock runs a steady 105-113 ms BEHIND the presented pts. The renderer
# holds and ramps against the exact presented pts, so that offset shifted the
# entire cut window — at a new shot's first frame the advisory still read
# "cut ahead", the flat hold never engaged, and ~60% of the disparity budget
# was applied to the OLD shot's map. SYLC_ADVISORY_PTS_CLOCK=0 restores the
# UI-clock dating so the two can be compared directly.
SYNTH3D_ADVISORY_PTS_CLOCK = os.environ.get(
    'SYLC_ADVISORY_PTS_CLOCK', '1').strip().lower() not in {
        '0', 'false', 'off', 'disabled', 'no'}


def _synth3d_seek_keeps_aspect(override):
    """True when a seek must NOT bounce this aspect override through the square.

    The probe is deliberately getattr-based (house idiom): a foreign, fake or
    legacy selection object answers None, never 0, and therefore keeps the
    historical clear-everything behaviour. Only a real AspectSelection whose
    matte is empty on BOTH edges -- i.e. one derived from the coded dimensions
    -- is retained across the seek.
    """
    if not SYNTH3D_SEEK_KEEP_ASPECT:
        return False
    selection = (override[2]
                 if isinstance(override, tuple) and len(override) >= 3
                 else None)
    return (getattr(selection, 'crop_top', None) == 0
            and getattr(selection, 'crop_bottom', None) == 0)

# The preset choice is a per-user preference that must survive a restart, so it
# lives in QSettings rather than in the JSON app-settings file (which holds the
# per-session playback tuning). Organization/application are passed explicitly
# so the lookup works without a configured QCoreApplication.
SYNTH3D_SETTINGS_ORG = "SyLC"
SYNTH3D_SETTINGS_APP = "SyLC3DPlayer"
SYNTH3D_DEPTH_PRESET_KEY = "synth3d/depth_preset"


# The directory this module was loaded from. Under Nuitka standalone it is the
# dist directory, which is also where sys.argv[0] lives -- the two collapse and
# the de-duplication below drops the repeat. In a dev checkout run from another
# working directory they differ, and this is the one that finds models/.
_SOURCE_ROOT = PROJECT_ROOT


def sylc_user_data_dir():
    """Per-user writable directory. The install directory is the preferred home
    for models, but a copy unzipped under Program Files is not writable, and a
    4.6 GB download must not fail on that."""
    base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    return os.path.join(base, 'SyLC')


def sylc_models_download_dir():
    """Where the in-app downloader writes: next to the executable when that is
    writable, otherwise the per-user directory. Both are searched by
    _synth3d_models_dirs(), so either choice resolves afterwards."""
    beside_exe = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[0])), 'models')
    try:
        os.makedirs(beside_exe, exist_ok=True)
        probe = os.path.join(beside_exe, '.write_probe')
        with open(probe, 'wb'):
            pass
        os.remove(probe)
        return beside_exe
    except OSError:
        return os.path.join(sylc_user_data_dir(), 'models')


def _synth3d_models_dirs():
    """The directories searched for depth models, in order: next to the running
    executable, the source tree this module came from (a dev checkout run from
    elsewhere), then the per-user directory the in-app downloader falls back to
    when the install directory is read-only.

    Order matters: an install-directory copy must win over a stale per-user one
    left behind by an earlier version.
    """
    base = os.path.dirname(os.path.abspath(sys.argv[0]))
    candidates = (os.path.join(base, 'models'),
                  os.path.join(_SOURCE_ROOT, 'models'),
                  os.path.join(sylc_user_data_dir(), 'models'))
    seen, ordered = set(), []
    for directory in candidates:
        key = os.path.normcase(os.path.abspath(directory))
        if key not in seen:
            seen.add(key)
            ordered.append(directory)
    return tuple(ordered)


def synth3d_find_model(candidates):
    """First candidate present on disk, or None. Preference order is the
    caller's; each candidate is looked for in both model directories."""
    for models_dir in _synth3d_models_dirs():
        for name in candidates:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                return path
    return None


def synth3d_depth_preset_entry(name):
    """The (name, candidates, side) entry called `name`, or None."""
    for entry in SYNTH3D_DEPTH_PRESETS:
        if entry[0] == name:
            return entry
    return None


def synth3d_depth_preset_available(name):
    """True iff at least one of this preset's own models is installed. Drives
    the menu's greying: selecting a preset with nothing behind it would silently
    resolve to another grid."""
    entry = synth3d_depth_preset_entry(name)
    return bool(entry) and synth3d_find_model(entry[1]) is not None


def synth3d_depth_preset_stored():
    """The persisted preset name, validated against the table.

    Read on every resolution rather than cached: the value is a single QSettings
    lookup, and a cache would have to be invalidated from the setter, from a
    second window and (in tests) between cases.
    """
    try:
        stored = QSettings(SYNTH3D_SETTINGS_ORG, SYNTH3D_SETTINGS_APP).value(
            SYNTH3D_DEPTH_PRESET_KEY, SYNTH3D_DEPTH_PRESET_DEFAULT)
    except Exception:
        logger.warning("[2D3D] could not read the persisted depth preset",
                       exc_info=True)
        return SYNTH3D_DEPTH_PRESET_DEFAULT
    name = str(stored) if stored is not None else ''
    # An unknown name (hand-edited setting, a preset dropped by a later
    # version) must not leave synthesis pointing at nothing.
    return name if synth3d_depth_preset_entry(name) else SYNTH3D_DEPTH_PRESET_DEFAULT


def synth3d_marker_attests(marker, model_path):
    """True iff a `.trt_verified` marker covers the graph about to be opened.

    TensorRT's engine cache is keyed PER GRAPH, so a marker written for
    `da3_base_756.onnx` says nothing about a 518 preset: opening one against
    that runtime would pay a full cold engine compile on the first in-playback
    enable — minutes — which is exactly the wait the offline probe exists to
    absorb, and this feature never builds an engine during playback.

    A marker that names NO graph (unreadable, or written by a probe from before
    model names were recorded) attests the directory as a whole, as it did in
    round 3: its presence still proves a real engine build succeeded against
    these DLLs, which is what guards against the hard native abort an
    incomplete TensorRT assembly causes. Absent information is not negative
    information — the per-graph refinement applies only once the probe has said
    which graphs it built.
    """
    try:
        with open(marker, 'r', encoding='utf-8', errors='replace') as f:
            probed = [line.split('=', 1)[1].strip() for line in f
                      if line.startswith('probe_model=')]
    except OSError:
        return True
    if not probed:
        return True
    return os.path.basename(model_path).lower() in {n.lower() for n in probed}


__all__ = [
    'SYNTH3D_DEPTH_PRESETS', 'SYNTH3D_DEPTH_PRESET_DEFAULT',
    'SYNTH3D_ADAPTIVE_MODEL_GRIDS', 'SYNTH3D_SEEK_KEEP_ASPECT',
    'SYNTH3D_ADVISORY_PTS_CLOCK', 'SYNTH3D_SETTINGS_ORG',
    'SYNTH3D_SETTINGS_APP', 'SYNTH3D_DEPTH_PRESET_KEY',
    '_SOURCE_ROOT',
    '_synth3d_seek_keeps_aspect', 'sylc_user_data_dir',
    'sylc_models_download_dir', '_synth3d_models_dirs',
    'synth3d_find_model', 'synth3d_depth_preset_entry',
    'synth3d_depth_preset_available', 'synth3d_depth_preset_stored',
    'synth3d_marker_attests',
]
