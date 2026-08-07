"""Behavioural tests for archive/export/ISO coordination extraction."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sylc.archive_export_mixin import ArchiveExportMixin
from sylc.stereo_eye_order import LEFT_FIRST, RIGHT_FIRST, UNKNOWN


class _Action:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _Harness(ArchiveExportMixin):
    def __init__(self):
        self._active_iso_mount = None
        self._pending_iso_mount = None
        self._deferred_iso_dismounts = []
        self._export_job = None
        self.video_3d_info = {}
        self.controls_overlay = SimpleNamespace(
            export_eye_left_action=_Action(),
            export_eye_right_action=_Action(),
        )


class _Job:
    def __init__(self, running=True, wait_result=True, source_desc=None):
        self.running = running
        self.wait_result = wait_result
        self.source_desc = source_desc or {}
        self.cancelled = False
        self.waited = []
        self.deleted = False

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancelled = True

    def wait(self, timeout_ms):
        self.waited.append(timeout_ms)
        return self.wait_result

    def deleteLater(self):
        self.deleted = True


def test_mounted_iso_letters_are_normalised_and_deduplicated():
    player = _Harness()
    player._active_iso_mount = ('one.iso', 'q:\\')
    player._pending_iso_mount = ('two.iso', 'Q:')

    assert player._mounted_iso_letters() == {'Q'}


def test_export_eye_order_user_override_wins_and_keeps_duration():
    player = _Harness()
    player.controls_overlay.export_eye_right_action = _Action(True)
    player.video_3d_info = {'duration': '123.5'}

    desc = player._export_desc_with_eye_order(
        {'path': 'feature.mkv'}, LEFT_FIRST, 'container metadata')

    assert desc['eye_order'] == RIGHT_FIRST
    assert desc['eye_order_source'] == 'user override'
    assert desc['duration_s'] == 123.5


def test_export_eye_order_preserves_unknown_as_not_signalled():
    player = _Harness()

    desc = player._export_desc_with_eye_order({}, UNKNOWN, 'ignored')

    assert desc == {
        'eye_order': UNKNOWN,
        'eye_order_source': 'not signalled',
    }


def test_half_packed_detection_uses_layout_dimensions():
    player = _Harness()
    player.video_3d_info = {'stereo_mode': 'sbs', 'width': 1920, 'height': 1080}
    assert player._is_half_packed_source() is True

    player.video_3d_info['width'] = 3840
    assert player._is_half_packed_source() is False

    player.video_3d_info = {'stereo_mode': 'tab', 'width': 1920, 'height': 1080}
    assert player._is_half_packed_source() is True


def test_export_output_defaults_next_to_writable_source():
    player = _Harness()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'My Feature.mkv'

        result = player._resolve_export_out_path(str(source))

        assert result == os.path.join(directory, 'My Feature_MVHEVC.mov')


def test_running_export_reports_all_source_drive_letters():
    player = _Harness()
    player._export_job = _Job(source_desc={
        'path': r'R:\\BDMV\\STREAM\\00000.m2ts',
        'dep_path': r'S:\\BDMV\\STREAM\\00001.m2ts',
    })

    assert player._export_job_source_drives() == {'R', 'S'}


def test_iso_dismount_is_deferred_while_export_reads_drive():
    player = _Harness()
    player._export_job = _Job(source_desc={
        'path': r'R:\\BDMV\\STREAM\\00000.m2ts',
    })
    mount = (r'I:\\images\\movie.iso', 'R:')

    assert player._dismount_iso_or_defer(mount) is False
    assert player._deferred_iso_dismounts == [mount]

    # Repeating the request must not duplicate the deferred operation.
    assert player._dismount_iso_or_defer(mount) is False
    assert player._deferred_iso_dismounts == [mount]


def test_stop_export_cancels_and_joins_before_dropping_reference():
    player = _Harness()
    job = _Job(running=True, wait_result=True)
    player._export_job = job

    assert player._stop_export_job(timeout_ms=2500) is True
    assert job.cancelled is True
    assert job.waited == [2500]
    assert job.deleted is True
    assert player._export_job is None


def test_stop_export_keeps_running_job_owned_after_timeout():
    player = _Harness()
    job = _Job(running=True, wait_result=False)
    player._export_job = job

    assert player._stop_export_job(timeout_ms=10) is False
    assert player._export_job is job
    assert job.deleted is False


if __name__ == '__main__':
    import unittest

    unittest.main()
