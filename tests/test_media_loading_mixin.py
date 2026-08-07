"""Behavioural tests for transactional media loading ownership."""

import threading
from types import SimpleNamespace

from sylc import media_loading_mixin as loading_module
from sylc.media_loading_mixin import MediaLoadingMixin


class _Overlay:
    def __init__(self):
        self.progress = []
        self.hidden = 0

    def set_progress(self, value):
        self.progress.append(value)

    def hide_loading(self):
        self.hidden += 1


class _Harness(MediaLoadingMixin):
    def __init__(self):
        self._media_session_id = 7
        self._pending_play_request_id = 0
        self._loading_session_id = 7
        self._pgs_startup_pending_session = 7
        self._media_cancel_event = threading.Event()
        self._feature_edl_uri = None
        self.current_file_path = 'film.mkv'
        self.loading_overlay = _Overlay()
        self.player = object()
        self.aborts = []
        self.configured = []
        self.played = []
        self.pending_dismounts = 0

    def _session_is_current(self, session_id):
        return session_id == self._media_session_id

    def _abort_media_load(self, session_id, message=None):
        self.aborts.append((session_id, message))

    def _dismount_pending_iso(self):
        self.pending_dismounts += 1

    def _configure_and_start_playback(self, path, session_id=None):
        self.configured.append((path, session_id))


def test_loading_capabilities_are_injected_without_entrypoint_import():
    loading_module.configure_media_loading_support(True, False, True, {'.mkv'})

    assert loading_module.MVC_SUPPORT_AVAILABLE is True
    assert loading_module.NATIVE_RENDER_AVAILABLE is False
    assert loading_module.PGS_SUBTITLE_AVAILABLE is True
    assert loading_module.EDGE264_CONTAINERS == ('.mkv',)


def test_mpv_source_prefers_continuous_feature_edl():
    player = _Harness()

    assert player._mpv_source_for('segment.ssif') == 'segment.ssif'
    player._feature_edl_uri = 'edl://feature'
    assert player._mpv_source_for('segment.ssif') == 'edl://feature'


def test_play_file_contains_unhandled_load_failures():
    player = _Harness()
    player._play_file_impl = lambda *_args: (_ for _ in ()).throw(
        RuntimeError('broken source'))

    assert player.play_file('broken.mkv') is None
    assert player._pending_play_request_id == 1
    assert player.aborts == [(7, 'Could not load this source: broken source')]
    assert player.pending_dismounts == 1
    assert player.loading_overlay.hidden == 1


def test_retry_request_cannot_outlive_a_newer_request():
    player = _Harness()
    player._pending_play_request_id = 3
    player.play_file = player.played.append

    player._retry_play_file_when_ready('old.mkv', 2, 0)
    player._retry_play_file_when_ready('current.mkv', 3, 0)

    assert player.played == ['current.mkv']


def test_deferred_load_rejects_stale_session():
    player = _Harness()
    owned = []
    player._continue_play_file_owned = lambda *args: owned.append(args)

    player._continue_play_file('old.mkv', 6)
    player._continue_play_file('current.mkv', 7)

    assert owned == [('current.mkv', 7)]


def test_extraction_progress_rejects_stale_worker_payload():
    player = _Harness()

    player._on_extraction_progress({'session': 6, 'progress': 25})
    player._on_extraction_progress({'session': 7, 'progress': 50})

    assert player.loading_overlay.progress == [50.0]


def test_pgs_completion_requires_session_path_and_pending_token():
    player = _Harness()
    player._subtitle_manager = SimpleNamespace(
        install_parser=lambda *_args: True)

    player._on_pgs_extraction_complete({
        'session': 6, 'file_path': 'film.mkv', 'tracks': [1]})
    player._on_pgs_extraction_complete({
        'session': 7, 'file_path': 'other.mkv', 'tracks': [1]})
    assert player.configured == []

    player._on_pgs_extraction_complete({
        'session': 7, 'file_path': 'film.mkv', 'tracks': [1]})
    assert player.configured == [('film.mkv', 7)]
    assert player._pgs_startup_pending_session is None


def test_pgs_timeout_cancels_only_current_pending_startup():
    player = _Harness()

    player._on_pgs_startup_timeout('film.mkv', 6)
    assert not player._media_cancel_event.is_set()
    player._on_pgs_startup_timeout('film.mkv', 7)

    assert player._media_cancel_event.is_set()
    assert player.aborts[-1][0] == 7
    assert player.loading_overlay.hidden == 1

