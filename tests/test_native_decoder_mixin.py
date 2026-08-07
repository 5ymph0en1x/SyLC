"""Characterization tests for the extracted native decoder lifecycle."""

from types import SimpleNamespace

import numpy as np

from sylc import native_decoder_mixin as native_module
from sylc.native_decoder_mixin import NativeDecoderMixin


class _Harness(NativeDecoderMixin):
    def __init__(self):
        self._mpv_time_pos_cache = None
        self._mpv_time_pos_cache_mono = None
        self._mpv_pause_cache = True
        self.hevc_thread = None
        self.hevc_source = None
        self._hevc_mode_active = False
        self._hevc_leaked = []
        self._hevc_shutdown_blocked = False
        self._hevc_clocked = False
        self._hevc_start_request = None
        self.hevc_media_info = object()
        self._hevc_half = True
        self._glasses_eye_plane_dims = (1920, 1080)
        self.current_video_fps = None
        self.video_3d_info = None
        self.player = None
        self.controls_overlay = SimpleNamespace(
            time_slider=SimpleNamespace(value=lambda: 12_345))
        self.widgets = []
        self.synth_stop_values = []

    def _display_widgets(self):
        return self.widgets

    def _reap_hevc_leaked(self):
        return []

    def _synth3d_handle_decoder_stop(self, restarting):
        self.synth_stop_values.append(restarting)


class _PresentationHarness(_Harness):
    def __init__(self):
        super().__init__()
        self.hevc_thread = SimpleNamespace(
            consumed=0,
            presentation_consumed=lambda: None,
        )
        self.forwarded = []

    def _native_signal_is_current(self):
        return True

    def _on_mvc_frame_yuv_timed_ready(self, left, right, pts):
        self.forwarded.append((left, right, pts))
        return 'presented'


class _Source:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_sbs_plane_split_returns_left_and_right_zero_copy_views():
    y = np.arange(4 * 8, dtype=np.uint8).reshape(4, 8)
    u = np.arange(2 * 4, dtype=np.uint8).reshape(2, 4)
    v = u + 40

    left, right = NativeDecoderMixin._split_packed_stereo((y, u, v), 'sbs')

    assert np.array_equal(left[0], y[:, :4])
    assert np.array_equal(right[0], y[:, 4:])
    assert np.shares_memory(left[0], y)
    assert np.shares_memory(right[1], u)


def test_tab_plane_split_uses_top_as_base_eye():
    y = np.arange(8 * 4, dtype=np.uint8).reshape(8, 4)
    u = np.arange(4 * 2, dtype=np.uint8).reshape(4, 2)
    v = u + 20

    left, right = NativeDecoderMixin._split_packed_stereo((y, u, v), 'tab')

    assert np.array_equal(left[0], y[:4])
    assert np.array_equal(right[0], y[4:])
    assert np.array_equal(left[1], u[:2])
    assert np.array_equal(right[1], u[2:])


def test_dual_projector_target_receives_only_its_own_eye_in_first_slot():
    left = object()
    right = object()

    assert NativeDecoderMixin._planes_for_target(
        SimpleNamespace(eye_view='left'), left, right) == (left, None)
    assert NativeDecoderMixin._planes_for_target(
        SimpleNamespace(eye_view='right'), left, right) == (right, None)
    assert NativeDecoderMixin._planes_for_target(
        SimpleNamespace(eye_view=None), left, right) == (left, right)


def test_audio_clock_uses_cache_and_extrapolates_only_while_playing():
    player = _Harness()
    assert player._mpv_time_pos_ms() is None

    player._mpv_time_pos_cache = 10.0
    player._mpv_time_pos_cache_mono = 100.0
    original = native_module.time.monotonic
    native_module.time.monotonic = lambda: 100.25
    try:
        player._mpv_pause_cache = True
        assert player._mpv_time_pos_ms() == 10_000.0
        player._mpv_pause_cache = False
        assert player._mpv_time_pos_ms() == 10_250.0
    finally:
        native_module.time.monotonic = original


def test_hevc_wrapper_acknowledges_presentation_after_forwarding():
    player = _PresentationHarness()
    owner = SimpleNamespace(consumed=0)
    owner.presentation_consumed = lambda: setattr(
        owner, 'consumed', owner.consumed + 1)
    player.sender = lambda: owner

    result = player._on_hevc_frame_yuv_timed_ready('L', 'R', 456.0)

    assert result == 'presented'
    assert player.forwarded == [('L', 'R', 456.0)]
    assert owner.consumed == 1


def test_hevc_wrapper_acknowledges_even_when_presentation_raises():
    player = _PresentationHarness()
    owner = SimpleNamespace(consumed=0)
    owner.presentation_consumed = lambda: setattr(
        owner, 'consumed', owner.consumed + 1)
    player.sender = lambda: owner
    player._on_mvc_frame_yuv_timed_ready = lambda *_args: (_ for _ in ()).throw(
        RuntimeError('upload failed'))

    try:
        player._on_hevc_frame_yuv_timed_ready('L', 'R', 456.0)
    except RuntimeError as exc:
        assert str(exc) == 'upload failed'
    else:
        raise AssertionError('presentation error was unexpectedly swallowed')

    assert owner.consumed == 1


def test_hevc_teardown_without_thread_closes_source_and_resets_widget_state():
    player = _Harness()
    source = _Source()
    widget = SimpleNamespace(
        plane_scale=64.0,
        source_aspect=2.39,
        yuv_matrix_sel=9,
        transfer_sel=16,
    )
    player.hevc_source = source
    player._hevc_mode_active = True
    player.widgets = [widget]

    player._stop_hevc_decoder(restarting=True)

    assert source.closed == 1
    assert player.hevc_source is None
    assert player._hevc_mode_active is False
    assert player.hevc_media_info is None
    assert player._hevc_half is False
    assert player._glasses_eye_plane_dims is None
    assert (widget.plane_scale, widget.source_aspect) == (1.0, 0.0)
    assert (widget.yuv_matrix_sel, widget.transfer_sel) == (0, 0)
    assert player.synth_stop_values == [True]


def test_effective_fps_prefers_metadata_and_repairs_suspicious_low_rate():
    player = _Harness()
    player.video_3d_info = {'fps': 25.0}
    assert player._get_effective_video_fps() == 25.0

    player.current_video_fps = 7.5
    assert player._get_effective_video_fps() == 23.976

    player.current_video_fps = 240.0
    assert player._get_effective_video_fps() == 120.0


def test_current_time_falls_back_to_slider_when_mpv_has_no_position():
    player = _Harness()
    assert player._current_mpv_time() == 12.345

    player.player = SimpleNamespace(time_pos=42.25)
    assert player._current_mpv_time() == 42.25


def test_native_capabilities_are_injected_without_importing_main_module():
    original = (
        native_module.MVC_SUPPORT_AVAILABLE,
        native_module.NATIVE_RENDER_AVAILABLE,
        native_module.MVCDecoderThread,
        native_module.Framepacking3DWindow,
        native_module.EDGE264_CONTAINERS,
    )
    mvc_class = object()
    window_class = object()
    try:
        native_module.configure_native_decoder_support(
            True, True, mvc_class, window_class, ('.mkv', '.mp4'))
        assert native_module.MVC_SUPPORT_AVAILABLE is True
        assert native_module.NATIVE_RENDER_AVAILABLE is True
        assert native_module.MVCDecoderThread is mvc_class
        assert native_module.Framepacking3DWindow is window_class
        assert native_module.EDGE264_CONTAINERS == ('.mkv', '.mp4')
    finally:
        native_module.configure_native_decoder_support(*original)


if __name__ == '__main__':
    import unittest

    unittest.main()
