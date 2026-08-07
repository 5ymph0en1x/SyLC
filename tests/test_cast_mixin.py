"""Behavioural tests for the PlayerWindow Cast extraction."""

from types import SimpleNamespace

from sylc.cast_mixin import CastMixin


class _Overlay:
    def __init__(self):
        self.states = []

    def set_cast_transport_state(self, *state):
        self.states.append(state)


class _Harness(CastMixin):
    def __init__(self):
        self.current_file_path = None
        self.framepacking_window = None
        self.mvc_embedded_widget = None
        self.mvc_decoder_thread = None
        self._cast = None
        self._cast_connected = False
        self._cast_transport = None
        self.controls_overlay = _Overlay()
        self.notifications = []
        self.seeks = []
        self.pause_calls = []

    def show_3d_notification(self, message, success=False):
        self.notifications.append((message, success))

    def on_seek(self, seconds):
        self.seeks.append(seconds)

    def _safe_mpv_set_pause(self, paused):
        self.pause_calls.append(('mpv', paused))

    def _handle_pause_change(self, paused):
        self.pause_calls.append(('player', paused))


def test_cast_renderer_prefers_framepacking_renderer():
    player = _Harness()
    preferred = object()
    fallback = object()
    player.framepacking_window = SimpleNamespace(
        display_widget=SimpleNamespace(_r=preferred))
    player.mvc_embedded_widget = SimpleNamespace(_r=fallback)

    assert player._cast_renderer() is preferred


def test_cast_renderer_falls_back_to_embedded_renderer():
    player = _Harness()
    fallback = object()
    player.framepacking_window = SimpleNamespace(
        display_widget=SimpleNamespace(_r=False))
    player.mvc_embedded_widget = SimpleNamespace(_r=fallback)

    assert player._cast_renderer() is fallback


def test_cast_media_path_returns_regular_file_unchanged():
    player = _Harness()
    player.current_file_path = r'I:\films\feature.mkv'
    player._is_optical_class_source = lambda _path: False

    assert player._cast_media_path() == player.current_file_path


def test_cast_media_path_never_reopens_untappable_optical_source():
    player = _Harness()
    player.current_file_path = r'Z:\BDMV\STREAM\00000.ssif'
    player._is_optical_class_source = lambda _path: True

    assert player._cast_media_path() is None


def test_cast_seek_converts_milliseconds_to_seconds():
    player = _Harness()

    player._on_cast_seek(12_345)

    assert player.seeks == [12.345]


def test_cast_pause_updates_mpv_and_player_state():
    player = _Harness()

    player._on_cast_pause(True)
    player._on_cast_pause(False)

    assert player.pause_calls == [
        ('mpv', True), ('player', True),
        ('mpv', False), ('player', False),
    ]


def test_cast_error_turns_transport_indicator_off():
    player = _Harness()
    player._cast_transport = 'wifi'

    player._on_cast_status({'error': 'connection lost'})

    assert player._cast_transport is None
    assert player.controls_overlay.states[-1] == (None,)
    assert player.notifications[-1] == (
        'Streaming to Quest: connection lost', False)


def test_cast_connected_notification_is_emitted_only_once():
    player = _Harness()
    player._cast_transport = 'usb'
    player._cast = SimpleNamespace(is_active=True)

    player._on_cast_status({'connected': True})
    player._on_cast_status({'connected': True})

    assert player.notifications == [('Quest connected — streaming.', True)]
    assert player.controls_overlay.states[-1] == ('usb', True)


def test_demuxer_stream_tap_source_reads_and_closes_its_native_tap():
    class Demuxer:
        def __init__(self):
            self.requests = []
            self.closed = False

        def read_stream_tap(self, size):
            self.requests.append(size)
            return b'audio'

        def disable_stream_tap(self):
            self.closed = True

    demuxer = Demuxer()
    source = CastMixin._DemuxerStreamTapSource(demuxer, 'native-audio')

    assert source.name == 'native-audio'
    assert source.read(4096) == b'audio'
    assert demuxer.requests == [4096]

    source.close()
    assert demuxer.closed is True


def test_demuxer_stream_tap_source_degrades_to_empty_bytes():
    class BrokenDemuxer:
        def read_stream_tap(self, _size):
            raise RuntimeError('tap unavailable')

        def disable_stream_tap(self):
            raise RuntimeError('already closed')

    source = CastMixin._DemuxerStreamTapSource(BrokenDemuxer(), 'broken')

    assert source.read(512) == b''
    source.close()


if __name__ == '__main__':
    import unittest

    unittest.main()
