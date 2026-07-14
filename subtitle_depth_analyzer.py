"""
SubtitleDepthAnalyzer - Recovers the authored stereoscopic depth of 3D text tracks.

3D SBS/TAB releases author text subtitles as per-eye DUPLICATED events: each cue
exists twice, one copy confined to each eye's half of the frame (via ASS margins
— e.g. MarginR=1920 / MarginL=1920 on a 3840-wide PlayRes — or {\\pos} overrides).
The two copies are usually NOT at the same in-eye position: a small opposite bias
encodes the intended parallax (crossed = the text floats in front of the screen).

SyLC renders text subtitles itself (one overlay per eye), so it needs that depth
as a number: this module samples the first cues of the track with ffprobe, pairs
the duplicated events, and measures the authored disparity, normalized to EYE
width (>0 = in front of the screen) — the exact unit the native renderer's
subtitle_disparity uniform expects.
"""

import binascii
import json
import logging
import os
import re
import statistics
import subprocess
import sys

logger = logging.getLogger(__name__)

# Sanity bounds: real subtitle depth is a few percent of eye width. Anything
# larger is a parsing artifact or broken authoring — fall back to screen depth.
MAX_ABS_DISPARITY = 0.15
# Minimum duplicated pairs measured before we trust the estimate.
MIN_PAIRS = 2

_POS_RE = re.compile(r'\{[^}]*\\pos\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)[^}]*\}')


def _resolve_ffprobe():
    """ffprobe.exe next to the app first (bundled), then PATH."""
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffprobe.exe')
    if os.path.isfile(local):
        return local
    return 'ffprobe'


def _run_ffprobe(args):
    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0
    out = subprocess.run([_resolve_ffprobe()] + args, capture_output=True, text=True,
                         check=True, creationflags=creationflags)
    return json.loads(out.stdout or '{}')


def _decode_hexdump(dump):
    """ffprobe -show_data emits 'OFFSET: xxxx xxxx ...  ascii' lines -> raw bytes.

    The hex column is single-space separated; a double space delimits the ASCII
    column — split there instead of slicing at a fixed width (line width varies
    with payload length, a fixed slice bleeds ASCII into the hex and corrupts
    the decode).
    """
    raw = b''
    for line in (dump or '').splitlines():
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        hexcol = parts[1].split('  ', 1)[0].replace(' ', '')
        if len(hexcol) % 2:
            hexcol = hexcol[:-1]
        try:
            raw += binascii.unhexlify(hexcol)
        except (binascii.Error, ValueError):
            pass
    return raw


def _parse_ass_header(filepath, sub_index):
    """Extract PlayResX/PlayResY + per-style (MarginL, MarginR) from CodecPrivate."""
    data = _run_ffprobe(['-v', 'error', '-select_streams', f's:{sub_index}',
                         '-show_streams', '-show_data', '-print_format', 'json', filepath])
    streams = data.get('streams') or []
    if not streams:
        return None
    header = _decode_hexdump(streams[0].get('extradata', '')).decode('utf-8', 'replace')

    play_res_x = 0
    styles = {}
    style_format = []
    for line in header.splitlines():
        line = line.strip()
        if line.lower().startswith('playresx:'):
            try:
                play_res_x = int(float(line.split(':', 1)[1].strip()))
            except ValueError:
                pass
        elif line.lower().startswith('format:'):
            style_format = [f.strip().lower() for f in line.split(':', 1)[1].split(',')]
        elif line.lower().startswith('style:') and style_format:
            fields = [f.strip() for f in line.split(':', 1)[1].split(',')]
            if len(fields) >= len(style_format):
                entry = dict(zip(style_format, fields))
                try:
                    styles[entry.get('name', '')] = (
                        int(float(entry.get('marginl', 0) or 0)),
                        int(float(entry.get('marginr', 0) or 0)),
                    )
                except ValueError:
                    pass
    return {'play_res_x': play_res_x, 'styles': styles}


def _event_center_x(margin_l, margin_r, style_margins, play_res_x, text):
    """Horizontal center of one ASS event on the PlayRes canvas.

    {\\pos(x,y)} wins outright; otherwise libass centers the line between the
    effective margins (an event margin of 0 means 'inherit from the style').
    """
    mpos = _POS_RE.search(text or '')
    if mpos:
        return float(mpos.group(1))
    sl, sr = style_margins
    l = margin_l if margin_l > 0 else sl
    r = margin_r if margin_r > 0 else sr
    return (l + play_res_x - r) / 2.0


def analyze_text_track_depth(filepath, sub_index, stereo_layout='sbs', probe_seconds=600):
    """Measure the authored stereo depth of text subtitle track s:<sub_index>.

    Returns (disparity_normalized_to_eye_width, pairs_measured).
    disparity > 0 = crossed = the subtitle floats in FRONT of the screen.
    (0.0, 0) when the track is not per-eye duplicated (plain 2D authoring).
    """
    try:
        hdr = _parse_ass_header(filepath, sub_index)
        if not hdr or hdr['play_res_x'] <= 0:
            logger.info("[SUB-DEPTH] no usable ASS header (SRT or no PlayRes) -> flat")
            return 0.0, 0
        play_w = hdr['play_res_x']

        data = _run_ffprobe(['-v', 'error', '-select_streams', f's:{sub_index}',
                             '-show_packets', '-show_data',
                             '-read_intervals', f'%+{int(probe_seconds)}',
                             '-print_format', 'json', filepath])
        packets = data.get('packets') or []

        # Bucket events by presentation time, then pair identical texts.
        by_pts = {}
        for p in packets:
            payload = _decode_hexdump(p.get('data', '')).decode('utf-8', 'replace')
            # MKV S_TEXT/ASS block: ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            fields = payload.split(',', 8)
            if len(fields) < 9:
                continue
            try:
                margin_l, margin_r = int(fields[4] or 0), int(fields[5] or 0)
            except ValueError:
                continue
            style = fields[2].strip()
            text = fields[8]
            by_pts.setdefault((p.get('pts_time'), text), []).append(
                (margin_l, margin_r, style, text))

        disparities = []
        half = play_w / 2.0
        for (_pts, _text), events in by_pts.items():
            if len(events) != 2:
                continue
            centers = []
            for ml, mr, style, text in events:
                sm = hdr['styles'].get(style, (0, 0))
                centers.append(_event_center_x(ml, mr, sm, play_w, text))
            if stereo_layout == 'tab':
                # TAB: both copies span the full width; depth only via \pos x offset.
                c_left_eye, c_right_eye = centers[0], centers[1]
                eye_w = float(play_w)
            else:
                # SBS: one copy per horizontal half -> map to in-eye coordinates.
                left = [c for c in centers if c < half]
                right = [c - half for c in centers if c >= half]
                if len(left) != 1 or len(right) != 1:
                    continue
                c_left_eye, c_right_eye = left[0], right[0]
                eye_w = half
            disparities.append((c_left_eye - c_right_eye) / eye_w)

        if len(disparities) < MIN_PAIRS:
            logger.info(f"[SUB-DEPTH] {len(disparities)} duplicated pair(s) in first "
                        f"{probe_seconds}s -> not a 3D-authored track, flat")
            return 0.0, len(disparities)

        disparity = statistics.median(disparities)
        spread = max(disparities) - min(disparities)
        if abs(disparity) > MAX_ABS_DISPARITY or spread > 0.02:
            logger.warning(f"[SUB-DEPTH] implausible depth (median={disparity:.4f}, "
                           f"spread={spread:.4f}) -> flat")
            return 0.0, len(disparities)

        logger.info(f"[SUB-DEPTH] authored disparity {disparity:+.4f} of eye width "
                    f"({disparity * half:+.0f}px on {play_w}px canvas, "
                    f"{len(disparities)} pairs)")
        return float(disparity), len(disparities)
    except Exception as e:
        logger.warning(f"[SUB-DEPTH] analysis failed ({e}) -> flat")
        return 0.0, 0
