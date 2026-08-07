"""Characterization tests for window-presentation coordination."""

from types import SimpleNamespace

from PySide6.QtCore import QPoint, Qt

from sylc import window_presentation_mixin as presentation_module
from sylc.window_presentation_mixin import WindowPresentationMixin


class _Effect:
    def __init__(self, opacity=1.0):
        self._opacity = opacity

    def opacity(self):
        return self._opacity


class _Overlay:
    def __init__(self, visible=True, opacity=1.0, effect=None):
        self._visible = visible
        self._opacity = opacity
        self._effect = effect

    def isVisible(self):
        return self._visible

    def graphicsEffect(self):
        return self._effect

    def windowOpacity(self):
        return self._opacity


class _EventBase:
    def eventFilter(self, watched, event):
        return ('base-filter', watched, event)


class _Harness(WindowPresentationMixin, _EventBase):
    def __init__(self):
        self.controls_overlay = _Overlay()
        self.stereo_hud = None
        self._is_fake_fullscreen = False
        self._qt_fullscreen = False
        self._hidden = False
        self._minimized = False
        self.has_media = False
        self.geometry_updates = 0
        self.activity_marks = 0
        self.play_toggles = 0
        self.fullscreen_toggles = 0
        self.played_paths = []

    def isFullScreen(self):
        return self._qt_fullscreen

    def isHidden(self):
        return self._hidden

    def isMinimized(self):
        return self._minimized

    def _update_overlays_geometry(self):
        self.geometry_updates += 1

    def _mark_activity(self):
        self.activity_marks += 1

    def toggle_play(self):
        self.play_toggles += 1

    def toggle_fullscreen(self):
        self.fullscreen_toggles += 1

    def play_file(self, path):
        self.played_paths.append(path)


class _KeyEvent:
    def __init__(self, key):
        self._key = key
        self.accepted = False

    def key(self):
        return self._key

    def accept(self):
        self.accepted = True


class _DropEvent:
    def __init__(self, path):
        url = SimpleNamespace(toLocalFile=lambda: path)
        self._mime = SimpleNamespace(
            hasUrls=lambda: True,
            urls=lambda: [url],
        )
        self.accepted = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True


class _GeometryWidget:
    def __init__(self, width=0, height=0, visible=True):
        self._width = width
        self._height = height
        self._visible = visible
        self.moves = []
        self.resizes = []
        self.minimum_widths = []
        self.maximum_widths = []
        self.raised = 0

    def width(self):
        return self._width

    def height(self):
        return self._height

    def move(self, *position):
        self.moves.append(position)

    def resize(self, width, height):
        self.resizes.append((width, height))

    def isVisible(self):
        return self._visible

    def raise_(self):
        self.raised += 1

    def setMinimumWidth(self, width):
        self.minimum_widths.append(width)

    def setMaximumWidth(self, width):
        self.maximum_widths.append(width)

    def sizeHint(self):
        return SimpleNamespace(height=lambda: self._height)


def test_controls_shown_respects_hud_requested_visibility():
    player = _Harness()
    player.controls_overlay = _Overlay(visible=False, opacity=0.0)
    player.stereo_hud = SimpleNamespace(active=True, desired_visible=True)

    assert player._controls_shown() is True

    player.stereo_hud.desired_visible = False
    assert player._controls_shown() is False


def test_controls_shown_rejects_transparent_native_overlay():
    player = _Harness()
    player.controls_overlay = _Overlay(effect=_Effect(0.0))

    assert player._controls_shown() is False

    player.controls_overlay = _Overlay(opacity=0.0)
    assert player._controls_shown() is False


def test_fake_fullscreen_counts_as_effective_fullscreen():
    player = _Harness()
    assert player._nav_is_fullscreen() is False

    player._is_fake_fullscreen = True
    assert player._nav_is_fullscreen() is True

    player._is_fake_fullscreen = False
    player._qt_fullscreen = True
    assert player._nav_is_fullscreen() is True


def test_refresh_after_window_transition_reanchors_and_marks_activity():
    player = _Harness()
    player.has_media = True

    player._refresh_nav_after_window_transition()

    assert player.geometry_updates == 1
    assert player.activity_marks == 1

    player._minimized = True
    player._refresh_nav_after_window_transition()
    assert player.geometry_updates == 1


def test_overlay_geometry_clamps_controls_to_narrow_video_area():
    player = _Harness()
    player.isVisible = lambda: True
    player.video_container = _GeometryWidget(width=500, height=400)
    player.video_container.mapToGlobal = lambda _point: QPoint(100, 200)
    player.info_overlay = _GeometryWidget(visible=True)
    player.loading_overlay = _GeometryWidget()
    player.controls_overlay = _GeometryWidget(height=80, visible=True)
    player.stereo_hud = None
    player._update_monitoring_overlay_geometry = lambda: None
    player._update_metrics_overlay_geometry = lambda: None

    WindowPresentationMixin._update_overlays_geometry(player)

    assert player.controls_overlay.minimum_widths == [0]
    assert player.controls_overlay.maximum_widths == [500]
    assert player.controls_overlay.moves == [(100, 500)]
    assert player.controls_overlay.resizes == [(500, 80)]
    assert player.info_overlay.resizes == [(500, 400)]


def test_space_key_uses_normal_play_pause_path():
    player = _Harness()
    event = _KeyEvent(Qt.Key.Key_Space)

    player.keyPressEvent(event)

    assert player.play_toggles == 1
    assert event.accepted is True


def test_escape_only_exits_fake_fullscreen():
    player = _Harness()
    event = _KeyEvent(Qt.Key.Key_Escape)

    player.keyPressEvent(event)
    assert player.fullscreen_toggles == 0
    assert event.accepted is True

    player._is_fake_fullscreen = True
    player.keyPressEvent(_KeyEvent(Qt.Key.Key_Escape))
    assert player.fullscreen_toggles == 1


def test_drop_event_accepts_and_opens_first_local_path():
    player = _Harness()
    event = _DropEvent(r'I:\\films\\feature.mkv')

    player.dropEvent(event)

    assert event.accepted is True
    assert player.played_paths == [r'I:\\films\\feature.mkv']


def test_windowed_settings_schedule_hdr_refresh():
    player = _Harness()
    # mpv's proxy is truthy while alive; mirror that contract with a seeded map.
    player.player = {'alive': True}
    player._refresh_windows_hdr_brightness = object()
    scheduled = []
    original_timer = presentation_module.QTimer
    presentation_module.QTimer = SimpleNamespace(
        singleShot=lambda delay, callback: scheduled.append((delay, callback)))
    try:
        player._apply_windowed_video_settings()
    finally:
        presentation_module.QTimer = original_timer

    assert player.player['video-sync'] == 'display-resample'
    assert scheduled == [(200, player._refresh_windows_hdr_brightness)]


if __name__ == '__main__':
    import unittest

    unittest.main()
