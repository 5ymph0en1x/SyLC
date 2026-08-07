import tempfile
import unittest
from pathlib import Path

from sylc.subtitle_track_mixin import SubtitleTrackMixin
from sylc.track_metadata import _find_clpi_for_media, _friendly_track_label


class _Renderer:
    def __init__(self):
        self.cleared = 0
        self.disparities = []

    def clear(self):
        self.cleared += 1

    def set_disparity(self, value):
        self.disparities.append(value)


class _Player:
    def __init__(self):
        self.commands = []
        self.properties = {}
        self.track_list = []

    def command(self, *args):
        self.commands.append(args)

    def __setitem__(self, name, value):
        self.properties[name] = value


class _Combo:
    pass


class _Overlay:
    def __init__(self):
        self.subtitle_track_combo = _Combo()
        self.subtitle_updates = []
        self.streaming_updates = []

    def update_subtitle_tracks(self, tracks):
        self.subtitle_updates.append(tracks)

    def update_subtitle_tracks_streaming(self, tracks):
        self.streaming_updates.append(tracks)


class _SubtitleManager:
    def __init__(self):
        self.enabled = []

    def set_enabled(self, enabled):
        self.enabled.append(enabled)


class _NativeThread:
    def __init__(self):
        self.subtitle_tracks = []

    def set_subtitle_track(self, track):
        self.subtitle_tracks.append(track)


class _Harness(SubtitleTrackMixin):
    def __init__(self):
        self._media_session_id = 4
        self.notifications = []
        self._text_sub_active = True
        self._text_subtitle_renderer = _Renderer()
        self._active_text_sub_depth_key = ('film.mkv', 2)
        self._sub_depth_cache = {}
        self.has_media = True
        self.player = _Player()
        self.remembered = []
        self.controls_overlay = _Overlay()
        self.current_file_path = None
        self.mvc_mode_active = False
        self.mvc_decoder_thread = None
        self.hevc_thread = None
        self._hevc_mode_active = False
        self._subtitle_manager = None
        self._subtitle_extractor = None
        self._pgs_subtitle_tracks = []
        self._streaming_subtitle_tracks = []
        self._active_streaming_track = None
        self._active_pgs_track_index = None
        self._file_memory = {}

    def _session_is_current(self, session_id, **_kwargs):
        return session_id == self._media_session_id

    def show_3d_notification(self, message, success=False):
        self.notifications.append((message, success))

    def _remember_for_file(self, **fields):
        self.remembered.append(fields)

    def _native_signal_is_current(self):
        return True


def test_track_label_rejects_placeholder_titles_and_humanizes_audio():
    label = _friendly_track_label({
        'id': 3,
        'lang': 'fra',
        'codec': 'truehd',
        'demux-channel-count': 8,
        'title': 'TRACK_3',
    }, 'audio')
    assert label == 'French · Dolby TrueHD · 7.1'


def test_matching_clpi_is_found_beside_the_stream_directory():
    with tempfile.TemporaryDirectory() as root:
        bdmv = Path(root) / 'BDMV'
        stream = bdmv / 'STREAM'
        clipinf = bdmv / 'CLIPINF'
        stream.mkdir(parents=True)
        clipinf.mkdir()
        media = stream / '00042.m2ts'
        clpi = clipinf / '00042.clpi'
        media.touch()
        clpi.touch()
        assert _find_clpi_for_media(str(media)) == str(clpi)


def test_stale_pgs_notification_cannot_reach_the_current_ui():
    player = _Harness()
    player._on_pgs_notification({
        'session': 3, 'message': 'stale', 'success': False,
    })
    player._on_pgs_notification({
        'session': 4, 'message': 'current', 'success': True,
    })
    assert player.notifications == [('current', True)]


def test_text_depth_is_applied_only_to_the_active_track_and_session():
    player = _Harness()
    player._on_text_sub_depth({
        'session': 3, 'file_path': 'film.mkv', 'sub_index': 2,
        'disparity': 0.03, 'pairs': 2,
    })
    assert player._sub_depth_cache == {}

    player._on_text_sub_depth({
        'session': 4, 'file_path': 'film.mkv', 'sub_index': 2,
        'disparity': 0.03, 'pairs': 2,
    })
    assert player._sub_depth_cache == {('film.mkv', 2): 0.03}
    assert player._text_subtitle_renderer.disparities == [0.03]


def test_disabling_text_subtitles_clears_renderer_and_depth_owner():
    player = _Harness()
    player._disable_text_subtitles()
    assert player._text_sub_active is False
    assert player._active_text_sub_depth_key is None
    assert player._text_subtitle_renderer.cleared == 1


def test_audio_track_change_uses_mpv_queue_and_updates_file_memory():
    player = _Harness()
    player.change_audio_track(7)
    assert player.player.commands == [('set', 'aid', '7')]
    assert player.remembered == [{'audio_track': 7}]


def test_plain_track_population_enforces_visible_none_in_mpv():
    player = _Harness()
    player.player.track_list = [
        {'type': 'sub', 'id': 3, 'lang': 'eng', 'codec': 'subrip'},
    ]

    player._fetch_subtitle_tracks(player._media_session_id)

    assert player.controls_overlay.subtitle_updates
    assert player.player.properties['sid'] == 'no'
    assert player._text_sub_active is False
    # A neutral load state must not overwrite the per-title preference.
    assert player.remembered == []


def test_streaming_track_population_disables_every_backend_when_none_shown():
    player = _Harness()
    native = _NativeThread()
    manager = _SubtitleManager()
    player.mvc_mode_active = True
    player.mvc_decoder_thread = native
    player._subtitle_manager = manager

    player._on_subtitle_tracks_detected([
        {
            'trackNumber': 12,
            'codecId': 'S_HDMV/PGS',
            'language': 'fra',
            'name': 'French',
            'isPGS': True,
        },
    ])

    assert player.controls_overlay.streaming_updates
    assert native.subtitle_tracks == [0]
    assert manager.enabled == [False]
    assert player.player.properties['sid'] == 'no'
    assert player._active_streaming_track is None
    assert player._active_pgs_track_index is None


def test_explicit_none_selection_is_persisted_and_disables_native_track():
    player = _Harness()
    native = _NativeThread()
    manager = _SubtitleManager()
    player.mvc_mode_active = True
    player.mvc_decoder_thread = native
    player._subtitle_manager = manager
    player._streaming_subtitle_tracks = [
        {'trackNumber': 12, 'isPGS': True},
    ]

    player.change_subtitle_track(0)

    assert player.remembered == [{
        'subtitle_track': 0,
        'subtitle_track_kind': 'streaming',
    }]
    assert native.subtitle_tracks == [0]
    assert manager.enabled == [False]
    assert player.player.properties['sid'] == 'no'


def test_remembered_plain_subtitle_still_overrides_neutral_default():
    player = _Harness()
    player.player.track_list = [
        {'type': 'sub', 'id': 7, 'lang': 'fra', 'codec': 'subrip'},
    ]
    player._file_memory = {
        'subtitle_track_kind': 'mpv',
        'subtitle_track': 7,
    }
    player._apply_remembered_track = lambda _combo, _field: 7

    player._fetch_subtitle_tracks(player._media_session_id)

    assert player.player.sid == 7
    assert player.player.properties.get('sid') != 'no'


if __name__ == '__main__':
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    ]
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
