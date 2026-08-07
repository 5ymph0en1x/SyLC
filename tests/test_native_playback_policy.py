"""Tests for native decoder scheduling and presentation policies."""

from types import SimpleNamespace

from sylc import native_playback_policy as policy


class _Widget:
    def __init__(self, visible=True):
        self._visible = visible

    def isVisible(self):
        return self._visible


def test_edge264_startup_timeout_is_longer_for_3d_sources():
    assert policy._edge264_startup_timeout_ms({'is_3d': True}) == 25_000
    assert policy._edge264_startup_timeout_ms({'is_3d': False}) == 12_000
    assert policy._edge264_startup_timeout_ms(None) == 12_000


def test_worker_budget_reserves_cpu_and_stops_at_saturation():
    original_physical = policy._physical_core_count
    original_logical = policy.multiprocessing.cpu_count
    try:
        policy._physical_core_count = lambda: 16
        policy.multiprocessing.cpu_count = lambda: 32
        assert policy._recommended_edge264_threads() == 6

        policy._physical_core_count = lambda: 4
        policy.multiprocessing.cpu_count = lambda: 8
        assert policy._recommended_edge264_threads() == 3

        policy._physical_core_count = lambda: 2
        policy.multiprocessing.cpu_count = lambda: 4
        assert policy._recommended_edge264_threads() == 2
    finally:
        policy._physical_core_count = original_physical
        policy.multiprocessing.cpu_count = original_logical


def test_framepack_is_vsync_authority_when_visible():
    embedded = _Widget(True)
    framepack_widget = _Widget(True)
    window = SimpleNamespace(
        isVisible=lambda: True,
        display_widget=framepack_widget,
    )

    assert policy._select_stereo_presentation_targets(
        embedded, window, framepack_widget) == [
            (framepack_widget, True),
            (embedded, False),
        ]


def test_dual_projector_replaces_framepack_and_left_eye_owns_vsync():
    embedded = _Widget(True)
    framepack_widget = _Widget(True)
    window = SimpleNamespace(
        isVisible=lambda: True,
        display_widget=framepack_widget,
    )
    left = _Widget(True)
    right = _Widget(True)
    left_window = SimpleNamespace(
        isVisible=lambda: True, display_widget=left)
    right_window = SimpleNamespace(
        isVisible=lambda: True, display_widget=right)

    assert policy._select_stereo_presentation_targets(
        embedded, window, framepack_widget, (left_window, right_window)) == [
            (left, True),
            (right, False),
            (embedded, False),
        ]


def test_hidden_targets_are_not_selected_or_duplicated():
    embedded = _Widget(False)
    framepack_widget = _Widget(True)
    window = SimpleNamespace(
        isVisible=lambda: True,
        display_widget=framepack_widget,
    )

    assert policy._select_stereo_presentation_targets(
        embedded, window, framepack_widget) == [(framepack_widget, True)]


if __name__ == '__main__':
    import unittest

    unittest.main()
