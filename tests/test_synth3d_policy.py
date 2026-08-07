"""Tests for the standalone Synth3D model and seek policies."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from sylc import synth3d_policy as policy


def test_every_depth_preset_keeps_candidates_and_grid_together():
    assert policy.SYNTH3D_DEPTH_PRESETS
    for name, candidates, side in policy.SYNTH3D_DEPTH_PRESETS:
        assert name
        assert candidates
        assert side > 0
        assert policy.synth3d_depth_preset_entry(name) == (
            name, candidates, side)


def test_model_lookup_respects_directory_then_candidate_order():
    with tempfile.TemporaryDirectory() as root:
        first = Path(root) / 'first'
        second = Path(root) / 'second'
        first.mkdir()
        second.mkdir()
        (first / 'fallback.onnx').touch()
        (second / 'preferred.onnx').touch()
        original = policy._synth3d_models_dirs
        policy._synth3d_models_dirs = lambda: (str(first), str(second))
        try:
            found = policy.synth3d_find_model(
                ('preferred.onnx', 'fallback.onnx'))
        finally:
            policy._synth3d_models_dirs = original

        # Directory precedence is intentional: installed models win over a
        # stale per-user copy, even when the latter has the preferred filename.
        assert found == str(first / 'fallback.onnx')


def test_seek_keeps_only_crop_free_aspect_selection():
    original = policy.SYNTH3D_SEEK_KEEP_ASPECT
    policy.SYNTH3D_SEEK_KEEP_ASPECT = True
    try:
        crop_free = ('key', 756, SimpleNamespace(crop_top=0, crop_bottom=0))
        matte_derived = ('key', 756, SimpleNamespace(crop_top=72, crop_bottom=72))
        assert policy._synth3d_seek_keeps_aspect(crop_free) is True
        assert policy._synth3d_seek_keeps_aspect(matte_derived) is False
        assert policy._synth3d_seek_keeps_aspect(None) is False
    finally:
        policy.SYNTH3D_SEEK_KEEP_ASPECT = original


def test_unconditional_seek_reset_switch_overrides_crop_free_selection():
    original = policy.SYNTH3D_SEEK_KEEP_ASPECT
    policy.SYNTH3D_SEEK_KEEP_ASPECT = False
    try:
        selection = ('key', 756, SimpleNamespace(crop_top=0, crop_bottom=0))
        assert policy._synth3d_seek_keeps_aspect(selection) is False
    finally:
        policy.SYNTH3D_SEEK_KEEP_ASPECT = original


def test_tensorrt_marker_must_attest_the_selected_graph():
    with tempfile.TemporaryDirectory() as root:
        marker = Path(root) / '.trt_verified'
        marker.write_text(
            'probe_model=da3_base_756.onnx\nprobe_model=da3_small_518.onnx\n',
            encoding='utf-8')
        assert policy.synth3d_marker_attests(
            str(marker), r'I:\\models\\DA3_BASE_756.ONNX') is True
        assert policy.synth3d_marker_attests(
            str(marker), r'I:\\models\\da3_base_518.onnx') is False


def test_legacy_marker_without_graph_names_still_attests_runtime():
    with tempfile.TemporaryDirectory() as root:
        marker = Path(root) / '.trt_verified'
        marker.write_text('verified=1\n', encoding='utf-8')
        assert policy.synth3d_marker_attests(
            str(marker), 'any-model.onnx') is True


if __name__ == '__main__':
    import unittest

    unittest.main()
