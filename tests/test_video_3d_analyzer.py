import subprocess
import unittest
from unittest.mock import patch

from sylc import video_3d_analyzer as analyzer


def test_ssif_is_classified_without_starting_ffprobe():
    with patch.object(analyzer.subprocess, 'run') as run:
        result = analyzer.Video3DAnalyzer.analyze_file('feature.ssif')

    run.assert_not_called()
    assert result['is_3d'] is True
    assert result['stereo_mode'] == 'mvc'
    assert result['has_mvc_track'] is True


def test_ffprobe_timeout_returns_a_safe_filename_hint():
    timeout = subprocess.TimeoutExpired(['ffprobe'], 30)
    with (
        patch.object(analyzer, '_resolve_external_tool', return_value='ffprobe'),
        patch.object(analyzer, '_check_ffmpeg_runtime', return_value=None),
        patch.object(analyzer.subprocess, 'run', side_effect=timeout),
    ):
        result = analyzer.Video3DAnalyzer.analyze_file('film.3d.htab.mkv')

    assert result['is_3d'] is True
    assert result['stereo_mode'] == 'tab'
    assert 'timed out' in result['analysis_error']


def test_permission_failure_never_invents_an_mvc_stream():
    with (
        patch.object(analyzer, '_resolve_external_tool', return_value='ffprobe'),
        patch.object(analyzer, '_check_ffmpeg_runtime', return_value=None),
        patch.object(
            analyzer.subprocess, 'run',
            side_effect=PermissionError('blocked')),
    ):
        result = analyzer.Video3DAnalyzer.analyze_file('ordinary-film.mkv')

    assert result['is_3d'] is False
    assert result['stereo_mode'] == 'none'
    assert result['has_mvc_track'] is False


def test_fractional_frame_rate_parser_is_bounded_and_exact():
    assert analyzer._parse_ffprobe_fps('24000/1001') == 24000 / 1001
    assert analyzer._parse_ffprobe_fps('25') == 25.0
    assert analyzer._parse_ffprobe_fps('1/0') is None
    assert analyzer._parse_ffprobe_fps('not-a-rate') is None


if __name__ == '__main__':
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    ]
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
