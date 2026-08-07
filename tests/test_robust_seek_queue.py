import unittest

from PySide6.QtCore import QCoreApplication, QObject

from sylc.robust_seek_queue import (
    RobustSeekQueue, SeekState, should_resume_after_sync,
)


class _FakePlayer:
    def __init__(self):
        self.pause = False
        self.options = {}

    def __setitem__(self, key, value):
        self.options[key] = value


class _FakeParent(QObject):
    def __init__(self):
        super().__init__()
        self._media_session_id = 7
        self.mvc_mode_active = False
        self.mvc_decoder_thread = None
        self.player = _FakePlayer()
        self.is_playing = False
        self.pauses = []
        self.mpv_seeks = []
        self.decoder_seeks = []
        self.scheduled = []

    def _on_seek_queue_pause_request(self, paused):
        self.pauses.append(paused)

    def _on_seek_queue_mpv_seek(self, target):
        self.mpv_seeks.append(target)

    def _on_seek_queue_decoder_seek(self, target):
        self.decoder_seeks.append(target)

    def _media_single_shot(self, delay_ms, callback, session_id=None):
        owner = self._media_session_id if session_id is None else session_id
        self.scheduled.append((delay_ms, callback, owner))


def _make_queue():
    app = QCoreApplication.instance() or QCoreApplication([])
    parent = _FakeParent()
    queue = RobustSeekQueue(parent)
    return app, parent, queue


def test_simple_seek_runs_on_current_session_and_completes():
    _app, parent, queue = _make_queue()
    completion_intent = []
    queue.seek_completed.connect(
        lambda: completion_intent.append(queue.resume_after_seek))

    # The slider has already applied its technical pause at this point, so
    # parent.is_playing is false; the explicit pre-scrub intent must win.
    queue.request_seek(12.5, is_mvc=False, resume_after=True)
    assert queue._state is SeekState.SEEKING

    queue._on_debounce_expired()
    assert parent.mpv_seeks == [12.5]
    assert [(delay, session) for delay, _callback, session in parent.scheduled] == [
        (300, 7)
    ]

    parent.scheduled.pop()[1]()
    assert completion_intent == [True]
    assert queue._state is SeekState.IDLE
    assert not queue.is_busy()


def test_seek_started_while_paused_preserves_pause_intent():
    _app, parent, queue = _make_queue()
    completion_intent = []
    queue.seek_completed.connect(
        lambda: completion_intent.append(queue.resume_after_seek))

    queue.request_seek(15.0, is_mvc=False, resume_after=False)
    queue._on_debounce_expired()
    parent.scheduled.pop()[1]()

    assert completion_intent == [False]


def test_debounce_cannot_execute_a_seek_from_an_old_session():
    _app, parent, queue = _make_queue()

    queue.request_seek(20.0, is_mvc=False)
    parent._media_session_id = 8
    queue._on_debounce_expired()

    assert parent.mpv_seeks == []
    assert queue._seeks_executed == 0
    assert queue._state is SeekState.IDLE


def test_session_invalidation_stops_timers_and_drops_pending_work():
    _app, _parent, queue = _make_queue()

    queue.request_seek(30.0, is_mvc=False)
    assert queue._debounce_timer.isActive()

    queue.invalidate_session()

    assert queue._pending_request is None
    assert queue._state is SeekState.IDLE
    assert not queue._debounce_timer.isActive()


def test_mvc_seek_pauses_mpv_and_seeks_only_the_decoder_first():
    _app, parent, queue = _make_queue()
    parent.mvc_mode_active = True

    queue.request_seek(42.0, is_mvc=True)
    queue._on_debounce_expired()

    assert parent.pauses == [True]
    assert parent.mpv_seeks == []
    assert parent.scheduled[0][0] == 50

    parent.scheduled.pop(0)[1]()
    assert parent.player.pause is True
    assert parent.decoder_seeks == [42.0]
    assert parent.player.options == {
        'demuxer-readahead-secs': 1,
        'demuxer-max-bytes': '8MiB',
        'demuxer-max-back-bytes': '4MiB',
    }


def test_stale_decoder_completion_does_not_mutate_current_queue():
    _app, _parent, queue = _make_queue()
    queue._state = SeekState.SEEKING

    queue.notify_seek_finished(session_id=6)

    assert queue._state is SeekState.SEEKING


def test_initial_framepack_handshake_always_starts_playback():
    assert should_resume_after_sync(False, False) is True


def test_user_seek_restores_the_exact_previous_transport_state():
    assert should_resume_after_sync(True, True) is True
    assert should_resume_after_sync(True, False) is False


if __name__ == '__main__':
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    ]
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
