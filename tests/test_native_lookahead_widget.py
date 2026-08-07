import sys
from types import SimpleNamespace

import numpy as np

from sylc.native_renderer import native_framepack_widget as widget_module


class _Renderer:
    def synth3d_set_lookahead_frame(self, *args):
        return True

    def synth3d_clear_lookahead(self):
        pass


def _widget():
    # The staging contract is plain Python and deliberately testable without a
    # QWidget/D3D11 device. __init__ would require a GUI application.
    w = widget_module.NativeFramepackWidget.__new__(
        widget_module.NativeFramepackWidget)
    w.video_time_ms = 1000.0
    w.plane_scale = 65535.0 / 1023.0
    w.synth3d_enabled = True
    w.synth3d_grid_width = 8
    w.synth3d_grid_height = 4
    w.synth3d_side = 0
    w._have_synth3d = True
    w._synth3d_human_matte = None
    w._synth3d_human_matte_key = None
    w._lookahead_requested = True
    w._lookahead_native_supported = True
    w._lookahead_failure_logged = False
    w._lookahead_held = None
    w._lookahead_native_armed = False
    w._lookahead_resets = 0
    w._lookahead_flow_ms = []
    w._r = _Renderer()
    return w


def _planes(value):
    return (np.full((4, 8), value, np.uint8),
            np.full((2, 4), 128, np.uint8),
            np.full((2, 4), 128, np.uint8))


def test_live_lookahead_holds_exactly_one_frame(monkeypatch):
    monkeypatch.setitem(
        sys.modules, 'mvc_demuxer_cpp', SimpleNamespace(NvofFlow=object))
    monkeypatch.setattr(
        widget_module, '_estimate_shared_lookahead_flow',
        lambda *args: (np.zeros((1, 2), np.float32),
                       np.zeros((1, 2), np.float32),
                       np.ones((1, 2), np.float32), 0.8))
    w = _widget()
    first = _planes(32)
    second = _planes(48)

    staged = w._prepare_synth3d_lookahead(*first)
    assert staged[0] is True

    w.video_time_ms = 1041.708
    resolved = w._prepare_synth3d_lookahead(*second)
    assert resolved[0] is False
    assert resolved[1] is first[0]
    assert resolved[4] == 1000.0
    upload = resolved[7]
    assert upload[0] is second[0]
    assert upload[6:8] == (1000.0, 1041.708)


def test_live_lookahead_pts_discontinuity_rolls_back(monkeypatch):
    monkeypatch.setitem(
        sys.modules, 'mvc_demuxer_cpp', SimpleNamespace(NvofFlow=object))
    w = _widget()
    first = _planes(32)
    second = _planes(48)
    assert w._prepare_synth3d_lookahead(*first)[0] is True

    w.video_time_ms = 1800.0
    result = w._prepare_synth3d_lookahead(*second)
    assert result[0] is False
    assert result[1] is second[0]
    assert result[7] is None
    assert w._lookahead_held is None
    assert w._lookahead_resets == 1
