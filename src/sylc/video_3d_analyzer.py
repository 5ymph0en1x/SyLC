# -*- coding: utf-8 -*-
"""Media probing and stereoscopic-format classification for SyLC."""

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache

from sylc.stereo_eye_order import (
    UNKNOWN, classify_stereo_value, eye_order_from_stereo_value,
    normalise_eye_order,
)


logger = logging.getLogger(__name__)
APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def configure_app_base_dir(path):
    """Use the runtime directory selected by the application's boot logic."""
    global APP_BASE_DIR
    APP_BASE_DIR = os.path.abspath(path)
    _resolve_external_tool.cache_clear()

@lru_cache(maxsize=None)
def _resolve_external_tool(executable_name):
    """Return an absolute path to an external tool (ffmpeg/ffprobe) if available."""
    # Use APP_BASE_DIR for Nuitka compatibility
    base_dir = APP_BASE_DIR

    candidates = []
    if sys.platform == 'win32' and not executable_name.lower().endswith('.exe'):
        candidates.append(f"{executable_name}.exe")
    candidates.append(executable_name)

    # PRIORITY 1: Check local directory first (for bundled executables)
    for candidate in candidates:
        local_candidate = os.path.join(base_dir, candidate)
        if os.path.exists(local_candidate):
            return local_candidate

    # PRIORITY 2: Check system PATH
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


def _describe_windows_returncode(returncode):
    """Return a human readable explanation for common Windows subprocess errors."""
    if returncode in (3221225781, -1073741515):  # 0xC0000135
        return (
            "Failed to start the executable (code 0xC0000135). "
            "This usually indicates that DLLs for ffmpeg/ffprobe are missing. "
            "Download a static build of ffmpeg from https://www.gyan.dev/ffmpeg/builds/ "
            "and place ffmpeg.exe/ffprobe.exe and their DLLs in the application's folder, "
            "or add the ffmpeg /bin folder to your PATH."
        )
    if returncode in (3221225501, -1073741795):  # 0xC0000025 or similar
        return (
            "The system prevented ffmpeg/ffprobe from running (code 0xC0000025). "
            "Check your antivirus or try running the application with sufficient privileges."
        )
    return None


def _check_ffmpeg_runtime(executable_path):
    """
    Checks if essential DLLs for ffmpeg/ffprobe are present (Windows).

    Returns:
        str | None: error message if a dependency is missing.
    """
    if sys.platform != 'win32' or not executable_path:
        return None

    # Check in multiple locations: ffprobe's folder AND APP_BASE_DIR
    folders_to_check = [os.path.dirname(executable_path)]
    if APP_BASE_DIR and APP_BASE_DIR not in folders_to_check:
        folders_to_check.append(APP_BASE_DIR)

    required_bases = ['avcodec', 'avformat', 'avutil']
    missing = []

    for base in required_bases:
        found = False
        for folder in folders_to_check:
            pattern = os.path.join(folder, f"{base}-*.dll")
            if glob.glob(pattern):
                found = True
                break
        if not found:
            missing.append(base)

    if missing:
        return (
            f"ffmpeg/ffprobe found but the following DLLs are missing in the same folder: "
            f"{', '.join(missing)}. Copy all DLLs provided with ffmpeg (from the /bin directory of the archive) "
            "next to the executables, or install a full static build."
        )

    return None


_STEREO_PRIORITY = {
    'none': 0,
    'tab': 1,
    'sbs': 2,
    'mvc': 3,
    'anaglyph': 1,
}


def _classify_stereo_mode(mode_str):
    """Normalizes a stereo_mode value to sbs/tab/mvc/anaglyph."""
    if not mode_str:
        return None

    # Covers the normative numeric Matroska StereoMode values too (notably
    # 11=right-first SBS and 13/14=MVC left/right first).
    classified, _order = classify_stereo_value(mode_str)
    if classified:
        return classified

    mode = str(mode_str).strip().lower()
    mode = mode.replace('-', '_').replace(' ', '_')

    if mode in ('mono', 'left', 'right', 'both', '2d'):
        return None

    if any(keyword in mode for keyword in ('anaglyph', 'cyan', 'magenta', 'red_cyan', 'cyan_red')):
        return 'anaglyph'

    if any(keyword in mode for keyword in (
            'frame_altern', 'framealternate', 'frame_packing', 'frame_sequential',
            'frame_packed', 'view_packed', 'mvc', 'framepacking', 'frameinterleaved',
            'block_lr', 'block_rl', 'packed'
    )):
        return 'mvc'

    if any(keyword in mode for keyword in (
            'top_bottom', 'bottom_top', 'tab', 'over_under', 'under_over',
            'block_tb', 'block_bt', 'topbottom', 'bt', 'tb'
    )):
        return 'tab'

    if any(keyword in mode for keyword in (
            'side_by_side', 'sbs', 'left_right', 'right_left',
            'row_interleaved', 'column_interleaved'
    )):
        return 'sbs'

    return None


def _promote_stereo_mode(result_dict, mode, mark_mvc=False,
                         eye_order=UNKNOWN, eye_order_source=None):
    """Updates the 3D detection result with priority."""
    if not mode:
        return

    priority = _STEREO_PRIORITY.get(mode, 0)
    current_priority = _STEREO_PRIORITY.get(result_dict.get('stereo_mode', 'none'), 0)

    if priority >= current_priority:
        result_dict['stereo_mode'] = mode
        order = normalise_eye_order(eye_order)
        if order != UNKNOWN:
            result_dict['eye_order'] = order
            result_dict['eye_order_source'] = eye_order_source

    result_dict['is_3d'] = True

    if mark_mvc or mode == 'mvc':
        result_dict['has_mvc_track'] = True


def _parse_ffprobe_fps(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split('/')
    if len(parts) == 2:
        try:
            num = float(parts[0])
            den = float(parts[1])
            if den > 0:
                fps_val = num / den
                return fps_val
        except (ValueError, ZeroDivisionError):
            return None
    else:
        try:
            fps_val = float(value)
            return fps_val
        except ValueError:
            return None
    return None


_FFPROBE_ANALYSIS_TIMEOUT_S = 30.0


def _apply_filename_3d_hint(result, file_path):
    """Apply the deliberately weak filename fallback after a failed probe.

    TAB must be tested before the generic ``3d`` hint: the old ordering made
    ``movie.3d.htab.mkv`` match the SBS branch first, so the TAB branch could
    never win for the very filenames it was meant to recognize.
    """
    filename = os.path.basename(str(file_path)).lower()
    if 'tab' in filename or 'htab' in filename:
        result['is_3d'] = True
        result['stereo_mode'] = 'tab'
    elif 'sbs' in filename or 'hsbs' in filename or '3d' in filename:
        result['is_3d'] = True
        result['stereo_mode'] = 'sbs'
    return result


def _probe_mvc_extension(file_path, cap=8 * 1024 * 1024):
    """Detect a genuine MVC substream in a single-track H.264 Matroska/TS file.

    The classic "MVC-in-Matroska" case (a BD3D remuxed by mkvmerge to ONE H.264
    track carrying base + dependent views) probes through ffprobe as a plain
    High-profile 1920x1080 h264 stream: no second track, no stereo3d side-data,
    no Matroska StereoMode element, no `dependent` disposition. Every heuristic in
    analyze_file therefore leaves it 2D, and the D1 governor then (correctly, for a
    genuinely-2D file) greys the 3D controls — which is exactly why a real MVC MKV
    stopped being usable as 3D once the old blanket `enable_3d_controls(True)`
    override was removed.

    The definitive marker is the `mvcC` extension box (holding the MVC SubsetSPS)
    inside the track's CodecPrivate/extradata — the SAME signature the native C++
    matroska demuxer keys on (mvc_matroska_demuxer.cpp: mvcC_sig = "mvcC"). We scan
    the file HEADER for that 4-byte signature (or the `V_MPEG4/ISO/MVC` CodecID used
    when the dependent view is a separate track). CodecPrivate always precedes the
    first Matroska Cluster, so we stop as soon as a Cluster ID (0x1F43B675) appears —
    bounding the read on large 2D files to just their header. A plain 2D H.264 MKV
    has no mvcC box (verified), so this never fires for real 2D content and the D1
    2D-lockout is preserved.

    Returns True iff the MVC marker is found.
    """
    SIG = b'mvcC'
    CODECID = b'V_MPEG4/ISO/MVC'
    CLUSTER = b'\x1f\x43\xb6\x75'  # Matroska Cluster element ID — media starts here
    overlap = max(len(SIG), len(CODECID), len(CLUSTER)) - 1
    try:
        read_total = 0
        prev_tail = b''
        with open(file_path, 'rb') as fh:
            while read_total < cap:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                window = prev_tail + chunk
                if SIG in window or CODECID in window:
                    return True
                # CodecPrivate precedes the first Cluster; once media data begins
                # there is no more track header to inspect — stop scanning frames.
                if CLUSTER in window:
                    return False
                prev_tail = window[-overlap:] if len(window) >= overlap else window
                read_total += len(chunk)
    except Exception:
        return False
    return False


class Video3DAnalyzer:
    """
    Analyzes video files to detect 3D content.
    Uses ffprobe to extract metadata.
    """

    @staticmethod
    def analyze_file(file_path):
        """
        Analyzes a video file and returns its 3D properties.
        """
        result = {
            'is_3d': False,
            'stereo_mode': 'none',
            'has_mvc_track': False,
            # Never silently collapse "not signalled" into "left first".
            # The Apple exporter prompts for an explicit choice when this stays
            # unknown; playback remains backward-compatible.
            'eye_order': UNKNOWN,
            'eye_order_source': None,
            'width': 0,
            'height': 0,
            'analysis_error': None,
            'duration': None,
            'fps': None,
            'codec_name': None,    # H.264 ('h264') eligible for edge264 path
            'container_ext': None, # File extension (.mkv, .mp4, .ssif, ...)
            'has_audio': None,     # None = probe unavailable, bool after ffprobe
        }
        # Capture extension early — analyzer uses it for codec routing decisions.
        try:
            result['container_ext'] = os.path.splitext(file_path)[1].lower()
        except Exception:
            pass

        # Force MVC for SSIF files (Blu-ray 3D)
        # ffprobe often misidentifies them or hangs on large files
        if file_path.lower().endswith('.ssif'):
            result['is_3d'] = True
            result['stereo_mode'] = 'mvc'
            result['has_mvc_track'] = True
            # Default values, will be refined by decoder/demuxer
            result['width'] = 1920
            result['height'] = 1080
            result['fps'] = 23.976
            return result

        try:
            ffprobe_path = _resolve_external_tool('ffprobe')
            if not ffprobe_path:
                raise FileNotFoundError(
                    "ffprobe not found. Add ffprobe to the PATH or place ffprobe.exe "
                    "in SyLC's runtime directory."
                )

            runtime_issue = _check_ffmpeg_runtime(ffprobe_path)
            if runtime_issue:
                print(runtime_issue)
                result['analysis_error'] = runtime_issue
                raise FileNotFoundError(runtime_issue)

            cmd = [
                ffprobe_path,
                '-v', 'error',
                '-print_format', 'json',
                '-show_streams',
                '-show_format',
                file_path
            ]

            creationflags = 0
            if sys.platform == 'win32':
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=_FFPROBE_ANALYSIS_TIMEOUT_S,
                    creationflags=creationflags
                )
            except subprocess.TimeoutExpired:
                result['analysis_error'] = (
                    f"ffprobe timed out after {_FFPROBE_ANALYSIS_TIMEOUT_S:.0f}s")
                logger.warning("[ANALYZE] %s for %s",
                               result['analysis_error'], file_path)
                return _apply_filename_3d_hint(result, file_path)
            except PermissionError as e:
                # A process-launch failure says nothing about the media codec.  The
                # previous code forced MVC here, which could route an arbitrary flat
                # or HEVC file into the native MVC decoder.
                result['analysis_error'] = f"ffprobe permission error: {e}"
                return _apply_filename_3d_hint(result, file_path)
            except Exception as e:
                result['analysis_error'] = f"ffprobe failed: {e}"
                return _apply_filename_3d_hint(result, file_path)

            data = json.loads(completed.stdout or "{}")
            streams = data.get('streams', [])
            result['has_audio'] = any(
                stream.get('codec_type') == 'audio' for stream in streams)

            format_info = data.get('format', {})
            duration_str = format_info.get('duration')
            if duration_str:
                try:
                    result['duration'] = float(duration_str)
                except ValueError:
                    pass

            for stream_index, stream in enumerate(streams):
                if stream.get('codec_type') == 'video':
                    result['width'] = stream.get('width', 0)
                    result['height'] = stream.get('height', 0)
                    fps_value = _parse_ffprobe_fps(stream.get('avg_frame_rate')) or \
                                _parse_ffprobe_fps(stream.get('r_frame_rate'))
                    if fps_value:
                        result['fps'] = fps_value
                    width = result['width']
                    height = result['height']

                    codec_name = (stream.get('codec_name') or '').lower()
                    profile = (stream.get('profile') or '').lower()
                    # Remember the video codec so the player can decide whether
                    # to use the edge264 path (H.264) or fall back to MPV native.
                    # C1: this MUST be read BEFORE the framepack heuristic below so the
                    # heuristic can be gated on the codec.
                    if codec_name and not result['codec_name']:
                        result['codec_name'] = codec_name

                    # C1: the framepack-dimension heuristic (1920x2205/2160, 3840x4320)
                    # forces stereo_mode='mvc', routing the file to the MVC (edge264)
                    # decoder. That is correct for H.264/MVC packed streams, but it MUST
                    # NOT fire for an FTAB *HEVC* clip (same 1920x2160 / 3840x4320 dims):
                    # HEVC has its own avcodec path (_try_start_hevc) and, once marked
                    # 'mvc', would be sent to the MVC decoder and never reach it. So only
                    # apply the heuristic for h264/mvc or an unknown/empty codec — never
                    # for hevc. Behaviour for h264/mvc/no-codec is unchanged.
                    is_framepacked = (
                        ((width == 1920 and height in [2205, 2160])
                         or (width == 3840 and height == 4320))
                        and codec_name in ('h264', 'mvc', '')
                    )

                    if is_framepacked:
                        result['is_3d'] = True
                        result['has_mvc_track'] = True
                        result['stereo_mode'] = 'mvc'

                    if codec_name in ('mvc', 'h264'):
                        if 'stereo' in profile or 'mvc' in profile:
                            _promote_stereo_mode(result, 'mvc', mark_mvc=True)

                    # Some ffprobe builds expose Matroska StereoMode directly on
                    # AVStream instead of under tags/side_data_list.
                    for field_name in ('stereo_mode', 'stereo_mode_name'):
                        raw_stereo = stream.get(field_name)
                        classified = _classify_stereo_mode(raw_stereo)
                        if classified:
                            _promote_stereo_mode(
                                result, classified,
                                mark_mvc=(classified == 'mvc'),
                                eye_order=eye_order_from_stereo_value(raw_stereo),
                                eye_order_source=f'ffprobe stream {stream_index}.{field_name}')

                    disposition = stream.get('disposition') or {}
                    if isinstance(disposition, dict) and disposition.get('dependent'):
                        _promote_stereo_mode(result, 'mvc', mark_mvc=True)

                    if not is_framepacked:
                        for side_data in stream.get('side_data_list', []):
                            side_type = (
                                    side_data.get('side_data_type')
                                    or side_data.get('type')
                                    or ''
                            ).lower()
                            if ('stereo3d' in side_type or 'stereo_3d' in side_type
                                    or 'stereo 3d' in side_type):
                                detected = (
                                        side_data.get('stereo_mode')
                                        or side_data.get('type')
                                        or side_data.get('layout')
                                        or side_data.get('view')
                                        or ''
                                )
                                classified = _classify_stereo_mode(detected)
                                inverted_flag = side_data.get('inverted')
                                order = eye_order_from_stereo_value(
                                    detected, inverted=inverted_flag)
                                if classified == 'mvc':
                                    _promote_stereo_mode(
                                        result, 'mvc', mark_mvc=True,
                                        eye_order=order,
                                        eye_order_source=(
                                            f'ffprobe stream {stream_index} Stereo3D'))
                                elif classified:
                                    _promote_stereo_mode(
                                        result, classified,
                                        eye_order=order,
                                        eye_order_source=(
                                            f'ffprobe stream {stream_index} Stereo3D'))

                        tags = stream.get('tags') or {}
                        for key, value in tags.items():
                            if key.lower().startswith('stereo'):
                                classified = _classify_stereo_mode(value)
                                if classified:
                                    _promote_stereo_mode(
                                        result, classified,
                                        mark_mvc=(classified == 'mvc'),
                                        eye_order=eye_order_from_stereo_value(value),
                                        eye_order_source=(
                                            f'ffprobe stream {stream_index} tag {key}'))

            if not result['has_mvc_track']:
                for stream in data.get('streams', []):
                    if stream.get('codec_name') == 'mvc':
                        result['is_3d'] = True
                        result['has_mvc_track'] = True
                        result['stereo_mode'] = 'mvc'
                        break

            # Genuine MVC-in-Matroska recovery: a single H.264 track carrying the
            # base + dependent views (mkvmerge BD3D remux) has no ffprobe-visible
            # stereo marker, so all the checks above leave it 2D and the D1 governor
            # would grey its 3D controls. The mvcC SubsetSPS box in the track's
            # extradata is the definitive signal (see _probe_mvc_extension). Probe
            # ONLY when we are otherwise about to call an h264/native-demux-container
            # file 2D — plain 2D H.264 MKVs have no mvcC box, so real 2D content is
            # unaffected (D1 2D-lockout intact). Any later decode failure still
            # degrades via _fallback_from_edge264 (which now demotes is_3d→False).
            if (not result['is_3d']
                    and (result.get('codec_name') or '') == 'h264'
                    and (result.get('container_ext') or '') in
                        ('.mkv', '.mk3d', '.m2ts', '.ts')):
                if _probe_mvc_extension(file_path):
                    result['is_3d'] = True
                    result['stereo_mode'] = 'mvc'
                    result['has_mvc_track'] = True

            if not result['duration']:
                for stream in data.get('streams', []):
                    dur = stream.get('duration')
                    if dur:
                        try:
                            result['duration'] = float(dur)
                            break
                        except ValueError:
                            continue

        except subprocess.CalledProcessError as e:
            error_output = (e.stderr or e.stdout or '').strip()
            message = error_output if error_output else str(e)
            print(f"Error during 3D analysis (ffprobe): {message}")
            hint = _describe_windows_returncode(e.returncode)
            if hint:
                print(hint)
                result['analysis_error'] = hint
            else:
                result['analysis_error'] = message
            _apply_filename_3d_hint(result, file_path)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error during 3D analysis: {e}")
            result['analysis_error'] = str(e)
            _apply_filename_3d_hint(result, file_path)

        return result

__all__ = [
    'Video3DAnalyzer', 'configure_app_base_dir',
    '_FFPROBE_ANALYSIS_TIMEOUT_S', '_STEREO_PRIORITY',
    '_apply_filename_3d_hint', '_check_ffmpeg_runtime',
    '_classify_stereo_mode', '_describe_windows_returncode',
    '_parse_ffprobe_fps', '_probe_mvc_extension',
    '_promote_stereo_mode', '_resolve_external_tool',
]
