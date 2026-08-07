import unittest

from sylc.player_memory_mixin import PlayerMemoryMixin


class _MemoryStore:
    def __init__(self):
        self.calls = []

    def remember(self, path, **fields):
        self.calls.append((path, fields))


class _Combo:
    def __init__(self, values):
        self.values = values
        self.index = -1
        self.blocked = []

    def findData(self, value):
        return self.values.index(value) if value in self.values else -1

    def blockSignals(self, blocked):
        self.blocked.append(blocked)

    def setCurrentIndex(self, index):
        self.index = index


class _PlayerMemoryHarness(PlayerMemoryMixin):
    def __init__(self):
        self.current_file_path = r'D:\BDMV\STREAM\00001.ssif'
        self._active_iso_mount = (r'I:\Images\film.iso', 'D:\\')
        self._pending_iso_mount = None
        self._playback_memory = _MemoryStore()
        self._file_memory = {}
        self._file_memory_applied = set()


def test_mounted_iso_path_is_the_stable_memory_identity():
    player = _PlayerMemoryHarness()
    assert player._memory_key_path() == r'I:\Images\film.iso'


def test_valid_remembered_presentation_wins_over_detection():
    player = _PlayerMemoryHarness()
    player._file_memory = {'stereo_mode': 'dual'}
    assert player._choose_initial_stereo_mode('mvc') == 'dual'


def test_invalid_remembered_presentation_cannot_escape_the_known_set():
    player = _PlayerMemoryHarness()
    player._file_memory = {'stereo_mode': 'sideways-on-a-tuesday'}
    assert player._choose_initial_stereo_mode('mvc') == 'mvc'


def test_track_restore_is_idempotent_and_blocks_combo_signals():
    player = _PlayerMemoryHarness()
    player._file_memory = {'audio_track': 4}
    combo = _Combo([1, 4, 8])

    assert player._apply_remembered_track(combo, 'audio_track') == 4
    assert combo.index == 1
    assert combo.blocked == [True, False]
    assert player._apply_remembered_track(combo, 'audio_track') is None


if __name__ == '__main__':
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    ]
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
