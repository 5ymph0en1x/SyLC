# -*- coding: utf-8 -*-
"""Human-readable stream labels and Blu-ray CLPI language metadata."""

import os
import re
import struct

_TRACK_LANG_NAMES = {
    'eng': 'English', 'en': 'English', 'fre': 'French', 'fra': 'French', 'fr': 'French',
    'spa': 'Spanish', 'es': 'Spanish', 'ger': 'German', 'deu': 'German', 'de': 'German',
    'ita': 'Italian', 'it': 'Italian', 'jpn': 'Japanese', 'ja': 'Japanese',
    'chi': 'Chinese', 'zho': 'Chinese', 'zh': 'Chinese', 'rus': 'Russian', 'ru': 'Russian',
    'por': 'Portuguese', 'pt': 'Portuguese', 'dut': 'Dutch', 'nld': 'Dutch', 'nl': 'Dutch',
    'kor': 'Korean', 'ko': 'Korean', 'ara': 'Arabic', 'ar': 'Arabic', 'pol': 'Polish', 'pl': 'Polish',
    'swe': 'Swedish', 'dan': 'Danish', 'nor': 'Norwegian', 'fin': 'Finnish', 'cze': 'Czech', 'ces': 'Czech',
    'hun': 'Hungarian', 'tur': 'Turkish', 'tha': 'Thai', 'hin': 'Hindi', 'heb': 'Hebrew',
    'ell': 'Greek', 'gre': 'Greek', 'ukr': 'Ukrainian', 'vie': 'Vietnamese', 'ind': 'Indonesian',
}


_TRACK_CODEC_NAMES = {
    'eac3': 'Dolby Digital+', 'ac3': 'Dolby Digital', 'ac-3': 'Dolby Digital',
    'truehd': 'Dolby TrueHD', 'mlp': 'Dolby TrueHD',
    'dts': 'DTS', 'dca': 'DTS', 'dts-hd': 'DTS-HD', 'aac': 'AAC', 'flac': 'FLAC',
    'mp3': 'MP3', 'mp2': 'MP2', 'opus': 'Opus', 'vorbis': 'Vorbis',
    'pcm_bluray': 'LPCM', 'pcm_dvd': 'LPCM', 'pcm_s16le': 'PCM', 'pcm_s24le': 'PCM', 'pcm': 'PCM',
    # subtitles
    'hdmv_pgs_subtitle': 'PGS', 'pgssub': 'PGS', 'pgs': 'PGS',
    'subrip': 'SRT', 'srt': 'SRT', 'ass': 'ASS', 'ssa': 'SSA',
    'dvd_subtitle': 'VobSub', 'dvdsub': 'VobSub', 'mov_text': 'TX3G',
}


_TRACK_CHANNELS = {1: 'Mono', 2: 'Stereo', 3: '2.1', 6: '5.1', 7: '6.1', 8: '7.1'}


def _humanize_lang(code):
    if not code:
        return ''
    return _TRACK_LANG_NAMES.get(str(code).strip().lower(), str(code).upper())


def _humanize_codec(codec, profile=''):
    base = _TRACK_CODEC_NAMES.get(str(codec).strip().lower(), str(codec).strip().upper()) if codec else ''
    p = str(profile or '').strip()
    if not p or p.lower() in ('unknown', 'none'):
        return base
    # DTS family: ffmpeg/mpv profile strings are already canonical ("DTS-HD MA", "DTS-HD HRA", "DTS-ES")
    label = p if p.lower().startswith('dts') else base
    if 'atmos' in p.lower() and 'atmos' not in label.lower():
        label += ' Atmos'
    return label


def _track_int(track, *keys):
    for k in keys:
        v = track.get(k)
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def _friendly_track_label(track, kind='audio', lang_map=None):
    """Build a human-readable label for an mpv track-list entry (audio/sub).

    lang_map: optional {PID: 'iso639'} from a Blu-ray .clpi, used when the
    container itself carries no language tag (raw M2TS/SSIF case).
    """
    tid = track.get('id', '?')
    parts = []
    # language: prefer the container tag; fall back to the Blu-ray .clpi by PID (src-id)
    lang_code = (track.get('lang') or '').strip()
    if not lang_code and lang_map:
        lang_code = lang_map.get(track.get('src-id'), '') or ''
    lang = _humanize_lang(lang_code)
    if lang:
        parts.append(lang)
    codec = _humanize_codec(track.get('codec', ''), track.get('codec-profile', ''))
    if codec:
        parts.append(codec)
    if kind == 'audio':
        ch = _track_int(track, 'demux-channel-count', 'demux_channel_count',
                        'audio-channels', 'audio_channels')
        if ch:
            parts.append(_TRACK_CHANNELS.get(ch, f'{ch}.0'))
    # keep a real, non-placeholder title (skip MakeMKV "TRACK_1" style)
    title = (track.get('title') or '').strip()
    if title and not re.match(r'(?i)^track[\s_]*\d+$', title):
        parts.append(f'“{title}”')
    if not parts:
        parts.append(f'{"Audio" if kind == "audio" else "Subtitle"} {tid}')
    if track.get('forced'):
        parts.append('(forced)')
    elif track.get('default'):
        parts.append('(default)')
    return ' · '.join(parts)


_BD_AUDIO_CT = {0x03, 0x04, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0xA1, 0xA2}


_BD_PGIG_CT = {0x90, 0x91}   # Presentation Graphics / Interactive Graphics (PGS)


_BD_TEXT_CT = {0x92}         # Text subtitle


def _parse_clpi_languages(path):
    """Parse a Blu-ray .clpi ProgramInfo and return {PID: 'iso639'} (lowercase)."""
    out = {}
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if data[0:4] != b'HDMV':
            return out
        proginfo_addr = struct.unpack('>I', data[12:16])[0]
        q = proginfo_addr + 4   # skip ProgramInfo length
        q += 1                  # reserved (8 bits)
        num_prog = data[q]; q += 1
        for _ in range(num_prog):
            q += 6              # spn_program_sequence_start(4) + program_map_pid(2)
            num_streams = data[q]; q += 1
            q += 1              # num_groups
            for _ in range(num_streams):
                pid = struct.unpack('>H', data[q:q+2])[0]; q += 2
                sci_len = data[q]; q += 1
                sci = data[q:q+sci_len]; q += sci_len
                if not sci:
                    continue
                ct = sci[0]
                lang = ''
                if ct in _BD_AUDIO_CT and len(sci) >= 5:
                    lang = sci[2:5].decode('ascii', 'replace')   # after coding_type + format byte
                elif ct in _BD_PGIG_CT and len(sci) >= 4:
                    lang = sci[1:4].decode('ascii', 'replace')   # right after coding_type
                elif ct in _BD_TEXT_CT and len(sci) >= 5:
                    lang = sci[2:5].decode('ascii', 'replace')
                lang = lang.strip('\x00').strip()
                if lang:
                    out[pid] = lang.lower()
    except Exception:
        pass
    return out


def _find_clpi_for_media(media_path):
    """Locate the matching CLIPINF/<stem>.clpi for a BDMV STREAM file, or None."""
    try:
        stem = os.path.splitext(os.path.basename(media_path))[0]
        d = os.path.dirname(os.path.abspath(media_path))
        for _ in range(4):   # .../STREAM/SSIF -> STREAM -> BDMV (CLIPINF sits beside STREAM)
            cand = os.path.join(d, 'CLIPINF', stem + '.clpi')
            if os.path.isfile(cand):
                return cand
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    except Exception:
        pass
    return None

__all__ = [
    '_find_clpi_for_media', '_friendly_track_label', '_humanize_codec',
    '_humanize_lang', '_parse_clpi_languages', '_track_int',
]

