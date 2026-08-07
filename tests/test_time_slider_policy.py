import unittest

from sylc.time_slider import _decide_thumbs_mode


def test_physical_optical_media_disables_competing_thumbnail_reads():
    assert _decide_thumbs_mode(
        r'D:\BDMV\STREAM\00001.ssif', set(), {'D'}, 'h264') == ('off', True)


def test_mounted_iso_keeps_thumbnails_with_optical_pacing():
    assert _decide_thumbs_mode(
        r'D:\BDMV\STREAM\00001.m2ts', {'D'}, {'D'}, 'h264') == (
            'edge264', True)


def test_hevc_in_matroska_never_routes_to_the_h264_thumbnail_decoder():
    assert _decide_thumbs_mode(
        r'I:\Films\stereo.mkv', set(), set(), 'hevc') == ('avcodec', False)


def test_explicit_h264_codec_can_use_edge264_in_an_mp4_container():
    assert _decide_thumbs_mode(
        r'I:\Films\movie.mp4', set(), set(), 'h264') == ('edge264', False)


if __name__ == '__main__':
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    ]
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
