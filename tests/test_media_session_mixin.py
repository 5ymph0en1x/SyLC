"""Behavioural tests for media-session and mpv-core ownership."""

import threading
from types import SimpleNamespace

from sylc import media_session_mixin as session_module
from sylc.media_session_mixin import MediaSessionMixin


class _Signal:
    def __init__(self):
        self.payloads = []

    def emit(self, payload):
        self.payloads.append(payload)


class _SeekQueue:
    def __init__(self):
        self.invalidations = 0

    def invalidate_session(self):
        self.invalidations += 1


class _Core:
    def __init__(self):
        self.observed = []
        self.unobserved = []
        self.commands = []
        self.terminated = False

    def observe_property(self, name, handler):
        self.observed.append((name, handler))

    def unobserve_property(self, name, handler):
        self.unobserved.append((name, handler))

    def command(self, command):
        self.commands.append(command)

    def terminate(self):
        self.terminated = True


class _Harness(MediaSessionMixin):
    def __init__(self):
        self._app_closing = False
        self._media_session_id = 4
        self._media_cancel_event = threading.Event()
        self._media_workers = set()
        self._media_workers_lock = threading.Lock()
        self._seek_queue = _SeekQueue()
        self._mpv_media_observers = []
        self._mpv_subtext_observer = None
        self._mpv_subtext_observer_registered = False
        self._is_loading_file = True
        self._loading_session_id = 4
        self._pgs_startup_pending_session = 4
        self._mpv_transition_in_progress = True
        self.current_file_path = 'film.mkv'
        self.player = _Core()
        self.mpv_eof_event = _Signal()
        self.notifications = []
        self.stop_calls = 0
        self._sender = None

    def sender(self):
        return self._sender

    def show_3d_notification(self, message, success=False):
        self.notifications.append((message, success))

    def stop_playback(self):
        self.stop_calls += 1

    def on_time_update(self, *_args):
        pass

    def on_duration_change(self, *_args):
        pass

    def on_pause_state_change(self, *_args):
        pass


def test_session_ownership_checks_token_core_and_shutdown_state():
    player = _Harness()

    assert player._session_is_current(4, core=player.player)
    assert not player._session_is_current(3)
    assert not player._session_is_current(4, core=_Core())

    player._app_closing = True
    assert not player._session_is_current(4)


def test_begin_media_session_cancels_previous_generation():
    player = _Harness()
    previous_cancel = player._media_cancel_event

    session_id = player._begin_media_session('next.mkv')

    assert session_id == 5
    assert previous_cancel.is_set()
    assert not player._media_cancel_event.is_set()
    assert player._seek_queue.invalidations == 1
    assert player._loading_session_id == 5
    assert player._mpv_transition_in_progress is True


def test_invalidate_media_session_leaves_no_active_owner():
    player = _Harness()

    session_id = player._invalidate_media_session('stop')

    assert session_id == 5
    assert player._media_cancel_event.is_set()
    assert player._loading_session_id is None
    assert player._pgs_startup_pending_session is None


def test_abort_media_load_ignores_stale_owner():
    player = _Harness()

    player._abort_media_load(3, 'stale failure')

    assert player._is_loading_file is True
    assert player.notifications == []


def test_media_observers_are_owned_by_exact_session_and_core():
    player = _Harness()
    core = player.player

    assert player._install_mpv_media_observers(4, core) is True
    assert [name for name, _handler in core.observed] == [
        'time-pos', 'duration', 'pause', 'eof-reached']
    assert len(player._mpv_media_observers) == 4
    assert player._install_mpv_media_observers(3, core) is False


def test_native_signal_rejects_decoder_from_previous_session():
    player = _Harness()
    current = SimpleNamespace(_sylc_session_id=4)
    stale = SimpleNamespace(_sylc_session_id=3)
    player.mvc_decoder_thread = current
    player.hevc_thread = None

    player._sender = current
    assert player._native_signal_is_current()
    player._sender = stale
    assert not player._native_signal_is_current()


def test_mpv_eof_dispatch_rejects_stale_and_transition_events():
    player = _Harness()
    core = player.player
    player._mpv_transition_in_progress = False

    player._dispatch_mpv_eof((3, core, True))
    assert player.stop_calls == 0

    player._dispatch_mpv_eof((4, core, True))
    assert player.stop_calls == 1

    player._mpv_transition_in_progress = True
    player._dispatch_mpv_eof((4, core, True))
    assert player.stop_calls == 1


def test_release_mpv_core_detaches_before_deferred_termination():
    player = _Harness()
    core = player.player
    handler = lambda *_args: None
    player._mpv_media_observers = [(core, 'time-pos', handler)]
    scheduled = []
    original_timer = session_module.QTimer
    session_module.QTimer = SimpleNamespace(
        singleShot=lambda delay, callback: scheduled.append((delay, callback)))
    try:
        player._release_mpv_core(core)
    finally:
        session_module.QTimer = original_timer

    assert core.commands == ['stop']
    assert core.unobserved == [('time-pos', handler)]
    assert player._mpv_dying is core
    assert scheduled[0][0] == session_module._MPV_RELEASE_SETTLE_MS

