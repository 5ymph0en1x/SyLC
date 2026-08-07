"""Behavioural tests for seek, timeline and pause orchestration."""

from types import SimpleNamespace

from sylc.playback_timeline_mixin import PlaybackTimelineMixin


class _Slider:
    def __init__(self, value=0, snapped=None):
        self._value = value
        self._snapped = snapped

    def value(self):
        return self._value

    def snap_to_vignette(self, value):
        return value if self._snapped is None else self._snapped


class _Overlay:
    def __init__(self, slider=None):
        self.time_slider = slider or _Slider()
        self.times = []

    def set_time(self, value):
        self.times.append(value)


class _SeekQueue:
    def __init__(self):
        self.requests = []
        self.resume_after_seek = False

    def request_seek(self, target, *, is_mvc, resume_after):
        self.requests.append((target, is_mvc, resume_after))


class _Core:
    def __init__(self, paused=False):
        self.pause = paused
        self.commands = []

    def command_async(self, *args):
        self.commands.append(args)


class _Harness(PlaybackTimelineMixin):
    def __init__(self):
        self.player = _Core(False)
        self.controls_overlay = _Overlay()
        self.controls_hide_timer = SimpleNamespace(stop=lambda: None)
        self._seek_queue = _SeekQueue()
        self._is_scrubbing = False
        self._is_seeking = False
        self._was_playing_before_scrub = False
        self._was_playing_before_seek = False
        self.mvc_mode_active = False
        self._last_ui_time = 10.0
        self._subtitle_manager = None
        self._active_pgs_track_index = None
        self.has_media = True
        self.is_playing = True
        self.current_file_path = 'film.mkv'
        self._playback_ended = False
        self._is_loading_file = False
        self.pause_changes = []
        self.play_requests = []

    def _mark_activity(self):
        pass

    def _ensure_controls_timer_initialized(self):
        pass

    def _handle_pause_change(self, paused, from_observer=False):
        self.pause_changes.append((paused, from_observer))
        self.is_playing = not paused

    def play_file(self, path):
        self.play_requests.append(path)


def test_scrub_preserves_playing_state_and_queues_resuming_seek():
    player = _Harness()
    player.controls_overlay = _Overlay(_Slider(12_500, snapped=12.0))

    player._on_slider_pressed()
    assert player.player.pause is True
    assert player._was_playing_before_scrub is True

    player._on_slider_released()
    assert player._seek_queue.requests == [(12.0, False, True)]


def test_scrub_started_while_paused_stays_paused_after_seek_request():
    player = _Harness()
    player.player.pause = True
    player.controls_overlay = _Overlay(_Slider(3_000))

    player._on_slider_pressed()
    player._on_slider_released()

    assert player._was_playing_before_scrub is False
    assert player._seek_queue.requests == [(3.0, False, False)]


def test_seek_completion_restores_exact_previous_transport_state():
    player = _Harness()
    requested = []
    player._on_seek_queue_pause_request = requested.append

    player._was_playing_before_seek = True
    player._on_seek_completed_logic()
    player._was_playing_before_seek = False
    player._on_seek_completed_logic()

    assert requested == [False, True]
    assert player._is_seeking is False


def test_small_backward_clock_jitter_is_ignored_but_large_seek_is_allowed():
    player = _Harness()

    player._set_ui_time(9.7)
    assert player._last_ui_time == 10.0
    player._set_ui_time(8.5)
    assert player._last_ui_time == 8.5


def test_scrubbing_blocks_late_clock_updates_unless_forced():
    player = _Harness()
    player._is_scrubbing = True

    player._set_ui_time(20.0)
    assert player._last_ui_time == 10.0
    player._set_ui_time(20.0, force=True)
    assert player._last_ui_time == 20.0


def test_safe_mpv_commands_are_non_blocking_and_encode_pause_state():
    player = _Harness()

    assert player._safe_mpv_set_pause(True) is True
    assert player._safe_mpv_set_pause(False) is True
    assert player._safe_mpv_seek(12.5) is True
    assert player.player.commands == [
        ('set', 'pause', 'yes'),
        ('set', 'pause', 'no'),
        ('seek', '12.5', 'absolute'),
    ]


def test_toggle_play_uses_player_transport_state_not_mpv_pause_property():
    player = _Harness()
    player.player.pause = True  # MVC deliberately keeps mpv paused.
    player.is_playing = True

    player.toggle_play()

    assert player.player.commands == [('set', 'pause', 'yes')]
    assert player.pause_changes == [(True, False)]


def test_toggle_play_reloads_last_title_after_stop():
    player = _Harness()
    player.has_media = False

    player.toggle_play()

    assert player.play_requests == ['film.mkv']

