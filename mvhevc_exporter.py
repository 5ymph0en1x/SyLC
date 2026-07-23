# -*- coding: utf-8 -*-
r"""MV-HEVC QuickTime export pipeline engine (Task EX-2, remux fast-path EX-2b).

One background, cancellable job (`MVHEVCExporter`, a QThread) turns any of SyLC's
four 3D source families into an Apple-spatial `.mov` (hvc1 + hvcC + lhvC + colr
nclx, 2 views), following the empirical verdicts of EX-1
(`.superpowers\sdd\briefs\ex-task-1-report.md`):

    [source adapter] -> raw 8-bit yuv420p PACKED frames (SBS canvas, or native TAB)
       -> x265.exe --multiview-config (STDIN pipe) -> out.hevc
       -> MP4Box mux (video + colr nclx + audio) -> out.mov

EX-2b amendment: when the source is ALREADY MV-HEVC (`kind == 'mvhevc'`), never
re-encode when a copy suffices. Three tiers (spec 2026-07-22 amendment §2):

    Tier 1: source container already hvc1+hvcC+lhvC-conformant (probed by a
            bounded box-tree walk, `probe_mv_hevc_container` -- header-only
            reads, NEVER touches mdat/moof payload) -> temp-copy the file
            verbatim + an audio-compat pass (table below) if the source audio
            needs it -> DONE, no x265, no MP4Box.
    Tier 2: not Tier-1-conformant -> extract the video ES with
            `-bsf:v hevc_mp4toannexb` (layer NALs travel with a stream copy) ->
            verify dependent-view NAL survival (`_count_layer_nals`, must be
            roughly half the VCL NALs) -> MP4Box mux (hvcC+lhvC, same command
            as the reencode path) + audio.
    Tier 3: any Tier-1/2 failure (mux error, layer-NAL check, or the existing
            `_validate` ffprobe+dogfood gate) -> log
            "[EXPORT] remux impossible (<reason>) -> reencodage" and fall
            through to the UNCHANGED reencode path (MVHEVCAdapter -> x265).

    EX-2b finding (refines EX-1 §3): EX-1 found ffmpeg CANNOT WRITE lhvC when
    building an hvc1 sample entry FROM A BARE ELEMENTARY STREAM. That is a
    different code path from a `-c:v copy` REMUX of a container that ALREADY
    carries a conformant hvc1+hvcC+lhvC sample entry -- empirically verified
    (see ex-task-2b-report.md) to copy hvcC/lhvC byte-for-byte AND leave the
    VCL NAL bytes bit-exact. Tier 1's audio-compat pass therefore uses a plain
    ffmpeg `-c:v copy` remux (re-verified against the box probe afterward,
    belt-and-braces), NOT Tier-2's extract+MP4Box mechanics.

Four adapters, one uniform packed-frame stream to x265 (§2 of the design spec):
  * PackedAdapter  (F-SBS / F-TAB, HEVC or H.264): decoded OUT-OF-PROCESS by the
    installed ffmpeg CLI to rawvideo on a pipe. F-SBS -> x265 format=1 (side-by-
    side); F-TAB -> x265 format=2 (top-bottom, fed NATIVELY, no re-pack — EX-1
    verdict 2c2). 10-bit sources are dithered to 8-bit by ffmpeg (`-pix_fmt
    yuv420p`).
  * MVHEVCAdapter  (MV-HEVC): OUR `LavfHevcSource.read_view_pair` (SW, allow_hw=
    False) -> hstack(L,R) -> SBS feed (format=1). 10-bit views ordered-dithered
    to 8-bit here.
  * MVCAdapter     (MVC MKV / SSIF / dual-file): a DEDICATED demuxer
    (`mvc_demuxer_cpp`) + a PRIVATE synchronous edge264 session (never the live
    player's). Base + dependent AUs are fed to one edge264 session; each decoded
    Edge264Frame carries BOTH views (base in `samples[]`, dependent in
    `samples_mvc[]`). Views are RE-PAIRED by PictureOrderCnt (edge264 can glue
    base[X] with dep[X-1] on B-frame GOPs — [[mvc-view-pairing-saccade]]) before
    hstack -> SBS feed (format=1). Nothing else on earth decodes MVC; this is the
    crown jewel.

The `--input-res` passed to x265 is the PER-VIEW resolution, NOT the packed
canvas (EX-1's single most important gotcha): x265 doubles it internally
(width for format=1, height for format=2).

Audio (§2 step 4): remuxed from the ORIGINAL source by ffmpeg. AAC/AC3-family
(QuickTime-safe) are stream-copied; TrueHD/DTS/other are transcoded to AAC 384k.
The audio elementary is then muxed by MP4Box alongside the video so the single
muxer that writes `lhvC` (MP4Box — ffmpeg cannot, EX-1 §3) stays the sole muxer
and the layered stream is never re-wrapped by a tool that would drop the
dependent-view parameter sets.

Lifecycle: a per-job temp workdir under %TEMP% holds every intermediate; a
try/finally purges it on ANY exit. `cancel()` terminates the child processes
(x265/ffmpeg/mp4box) without zombies, purges, and emits failed('annule'). The
job NEVER touches the live player's decode instances (dedicated instances only).

NB: the `vexu` box injection is Task EX-3 (`vexu_injector.py`), out of scope
here — the MP4Box output already carries hvc1+hvcC+lhvC+colr (ffprobe
view_ids_available=0,1), which is what EX-2 validates.

Windows-only; ctypes + PySide6.
"""
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))
X265 = os.path.join(_ROOT, 'tools', 'x265', 'x265.exe')
MP4BOX = os.path.join(_ROOT, 'tools', 'gpac', 'mp4box.exe')

# CREATE_NO_WINDOW so spawned console tools never flash a window when the GUI runs.
_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0

REORDER_DEPTH = 4          # matches MVCDecoderThread — buffers the base/dep POC drift


# ---------------------------------------------------------------------------
# tool discovery
# ---------------------------------------------------------------------------
def _ffmpeg_path():
    return shutil.which('ffmpeg') or 'ffmpeg'


def _ffprobe_path():
    return shutil.which('ffprobe') or 'ffprobe'


def tools_available():
    """True iff every external tool the pipeline needs is present."""
    return (os.path.isfile(X265) and os.path.isfile(MP4BOX)
            and shutil.which('ffmpeg') and shutil.which('ffprobe'))


# ---------------------------------------------------------------------------
# ffprobe metadata + colour mapping
# ---------------------------------------------------------------------------
# ffprobe colour token -> x265 token. Most names are shared verbatim; the map is
# just the guard/normaliser. Unknown/unspecified fall back to bt709 (8-bit SDR).
_X265_PRIM = {'bt709', 'bt470m', 'bt470bg', 'smpte170m', 'smpte240m', 'film',
              'bt2020', 'smpte428', 'smpte431', 'smpte432'}
_X265_TRC = {'bt709', 'bt470m', 'bt470bg', 'smpte170m', 'smpte240m', 'linear',
             'log100', 'log316', 'iec61966-2-4', 'bt1361e', 'iec61966-2-1',
             'bt2020-10', 'bt2020-12', 'smpte2084', 'smpte428', 'arib-std-b67'}
_X265_MATRIX = {'gbr', 'bt709', 'fcc', 'bt470bg', 'smpte170m', 'smpte240m',
                'ycgco', 'bt2020nc', 'bt2020c', 'smpte2085',
                'chroma-derived-nc', 'chroma-derived-c', 'ictcp'}
# LavfHevcSource heuristic names ('709'/'601') and a few aliases -> x265 tokens.
_ALIAS = {'709': 'bt709', '601': 'smpte170m', 'bt601': 'smpte170m',
          'bt2020': 'bt2020', 'bt2020nc': 'bt2020nc', 'unknown': None,
          'unspecified': None, 'reserved': None, 'n/a': None, '': None, None: None}


def _map_prim(v):
    v = _ALIAS.get(v, v)
    return v if v in _X265_PRIM else 'bt709'


def _map_trc(v):
    v = _ALIAS.get(v, v)
    return v if v in _X265_TRC else 'bt709'


def _map_matrix(v):
    v = _ALIAS.get(v, v)
    return v if v in _X265_MATRIX else 'bt709'


def probe_metadata(path):
    """ffprobe the first video + audio stream. Returns a dict with the packed-
    canvas geometry-agnostic facts (per-view derivation is the adapter's job):
    width/height (as stored), pix_fmt, fps_str/fps_float, colour tokens (x265),
    range_full, nb_frames, duration_s, audio_codec (or None)."""
    exe = _ffprobe_path()
    v = subprocess.run(
        [exe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,pix_fmt,r_frame_rate,avg_frame_rate,'
         'color_primaries,color_transfer,color_space,color_range,nb_frames,duration',
         '-of', 'json', path], capture_output=True, text=True,
        creationflags=_NO_WINDOW)
    st = (json.loads(v.stdout or '{}').get('streams') or [{}])[0]
    a = subprocess.run(
        [exe, '-v', 'error', '-select_streams', 'a:0', '-show_entries',
         'stream=codec_name', '-of', 'json', path],
        capture_output=True, text=True, creationflags=_NO_WINDOW)
    ast = (json.loads(a.stdout or '{}').get('streams') or [{}])
    audio_codec = ast[0].get('codec_name') if ast else None

    def _rate(s):
        try:
            n, d = s.split('/')
            n, d = float(n), float(d)
            return (s, n / d) if d else (None, 0.0)
        except Exception:
            return (None, 0.0)

    fps_str, fps_float = _rate(st.get('r_frame_rate') or '')
    if not fps_float:
        fps_str, fps_float = _rate(st.get('avg_frame_rate') or '')
    if not fps_float:
        fps_str, fps_float = '24/1', 24.0
    try:
        dur = float(st.get('duration')) if st.get('duration') not in (None, 'N/A') else 0.0
    except Exception:
        dur = 0.0
    try:
        nbf = int(st.get('nb_frames')) if st.get('nb_frames') not in (None, 'N/A') else 0
    except Exception:
        nbf = 0
    return {
        'width': int(st.get('width') or 0),
        'height': int(st.get('height') or 0),
        'pix_fmt': st.get('pix_fmt') or '',
        'fps_str': fps_str or '24/1',
        'fps_float': fps_float or 24.0,
        'colorprim': _map_prim(st.get('color_primaries')),
        'transfer': _map_trc(st.get('color_transfer')),
        'colormatrix': _map_matrix(st.get('color_space')),
        'range_full': (st.get('color_range') == 'pc'),
        'nb_frames': nbf,
        'duration_s': dur,
        'audio_codec': audio_codec,
    }


# ---------------------------------------------------------------------------
# EX-2b: bounded ISO-BMFF box-tree probe (Tier-1 conformance verdict)
# ---------------------------------------------------------------------------
# Header-only box walk: NEVER reads a box's payload except the (small, KB-range)
# moov subtree we actually need to inspect. mdat/moof (which can be GB-sized on
# a real movie) are skipped by seeking straight to their end offset -- this is
# what makes the probe "bounded" regardless of source file size.
_VSE_SKIP = 78            # VisualSampleEntry: fixed fields after the 8B box header
_STSD_SKIP = 8             # stsd FullBox: version/flags(4) + entry_count(4)


def _iter_boxes(f, start, end):
    """Yield (fourcc, offset, size, header_len, box_end) for each top-level box
    in [start, end), reading only 8 (or 16, for a 64-bit size) header bytes per
    box -- payload is never read here."""
    off = start
    while off + 8 <= end:
        f.seek(off)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        size = struct.unpack('>I', hdr[:4])[0]
        typ = hdr[4:8].decode('latin1', 'replace')
        header = 8
        if size == 1:
            sz64 = f.read(8)
            if len(sz64) < 8:
                break
            size = struct.unpack('>Q', sz64)[0]
            header = 16
        elif size == 0:
            size = end - off
        box_end = off + size
        if size < header or box_end > end:
            break
        yield typ, off, size, header, box_end
        off = box_end


def _descend_boxes(f, start, end, path, chain=None):
    """Descend through a fixed path of fourccs (honouring the stsd/sample-entry
    header quirks), returning the (start, end) byte range of the LAST matched
    box's children, or None if any hop in `path` isn't found. When `chain` is
    a list, every matched box's (fourcc, offset, size, header_len, box_end)
    5-tuple is appended to it in descent order -- reused (not reimplemented)
    by vexu_injector.py (EX-3) to get the exact ancestor offsets it needs to
    patch/splice, on top of this same walker (DRY)."""
    cur_start, cur_end = start, end
    for want in path:
        found = None
        for typ, off, size, header, box_end in _iter_boxes(f, cur_start, cur_end):
            if typ == want:
                found = (typ, off, size, header, box_end)
                break
        if found is None:
            return None
        if chain is not None:
            chain.append(found)
        typ, off, size, header, box_end = found
        if want == 'stsd':
            cur_start = off + header + _STSD_SKIP
        elif want in ('hvc1', 'hev1'):
            cur_start = off + header + _VSE_SKIP
        else:
            cur_start = off + header
        cur_end = box_end
    return cur_start, cur_end


def probe_mv_hevc_container(path):
    """Bounded box-tree probe: does `path`'s video sample entry already carry
    hvc1 + hvcC + lhvC (Tier-1 conformant, per EX-1's reference dump of
    docs\\hevc_4k24P_main_multiview_1.mp4: hvc1 -> hvcC + lhvC + vexu)?

    Returns {'conformant': bool, 'sample_entry': 'hvc1'|'hev1'|None,
             'has_hvcC': bool, 'has_lhvC': bool, 'has_vexu': bool,
             'moov_found': bool}. Never raises for a well-formed file; a
    missing/unreadable moov just yields conformant=False."""
    result = {'conformant': False, 'sample_entry': None, 'has_hvcC': False,
              'has_lhvC': False, 'has_vexu': False, 'moov_found': False}
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            fsize = f.tell()
            moov = None
            for typ, off, size, header, box_end in _iter_boxes(f, 0, fsize):
                if typ == 'moov':
                    moov = (off, size, header, box_end)
                    break
            if moov is None:
                return result
            _off, _size, header, box_end = moov
            result['moov_found'] = True
            moov_start, moov_end = _off + header, box_end
            for typ, off, size, header, box_end in _iter_boxes(f, moov_start, moov_end):
                if typ != 'trak':
                    continue
                stsd_region = _descend_boxes(f, off + header, box_end,
                                              ['mdia', 'minf', 'stbl', 'stsd'])
                if stsd_region is None:
                    continue
                s_start, s_end = stsd_region
                entry = None
                for etyp, eoff, esize, ehdr, eend in _iter_boxes(f, s_start, s_end):
                    if etyp in ('hvc1', 'hev1'):
                        entry = (etyp, eoff, esize, ehdr, eend)
                        break
                if entry is None:
                    continue
                etyp, eoff, esize, ehdr, eend = entry
                result['sample_entry'] = etyp
                child_start = eoff + ehdr + _VSE_SKIP
                for ctyp, coff, csize, chdr, cend in _iter_boxes(f, child_start, eend):
                    if ctyp == 'hvcC':
                        result['has_hvcC'] = True
                    elif ctyp == 'lhvC':
                        result['has_lhvC'] = True
                    elif ctyp == 'vexu':
                        result['has_vexu'] = True
                break     # first video sample entry found is enough
    except OSError:
        return result
    result['conformant'] = (result['sample_entry'] == 'hvc1'
                             and result['has_hvcC'] and result['has_lhvC'])
    return result


# ---------------------------------------------------------------------------
# EX-2b: Annex-B NAL scanning (layer-NAL survival check for Tier 2)
# ---------------------------------------------------------------------------
def _iter_annexb_nals(data):
    """Yield (nal_type, layer_id, nal_bytes) for every NAL unit in an Annex-B
    byte string. Uses 3-byte start-code scanning: a 4-byte start code
    `00 00 00 01` always contains the 3-byte pattern `00 00 01` as its suffix,
    so searching for the 3-byte pattern alone finds every NAL boundary (2- and
    4-byte-prefixed alike) with a single `bytes.find` loop.
    `nal_bytes` is the 2-byte NAL header + RBSP (start code stripped).
    HEVC NAL header (7.3.1.2): byte0 = forbidden(1)+type(6)+layer_id_hi(1);
    byte1 = layer_id_lo(5)+temporal_id_plus1(3) ->
    layer_id = ((byte0 & 0x01) << 5) | (byte1 >> 3)."""
    starts = []
    j = 0
    while True:
        j = data.find(b'\x00\x00\x01', j)
        if j < 0:
            break
        starts.append(j)
        j += 3
    n = len(starts)
    for i, s in enumerate(starts):
        nal_start = s + 3
        nal_end = starts[i + 1] if i + 1 < n else len(data)
        if nal_end - nal_start < 2:
            continue
        chunk = data[nal_start:nal_end]
        b0, b1 = chunk[0], chunk[1]
        nal_type = (b0 >> 1) & 0x3F
        layer_id = ((b0 & 0x01) << 5) | (b1 >> 3)
        yield nal_type, layer_id, chunk


def _count_layer_nals(hevc_path, max_bytes=256 * 1024 * 1024):
    """Parse an Annex-B HEVC ES and count VCL NALs (nal_unit_type <= 31) whose
    nuh_layer_id > 0 (dependent-view slices). Bounded to `max_bytes` (default
    256 MiB -- easily thousands of frames, plenty for a statistical survival
    verdict without loading a multi-GB ES fully into memory). Returns
    (vcl_total, vcl_layer_gt0)."""
    with open(hevc_path, 'rb') as f:
        data = f.read(max_bytes)
    vcl_total = vcl_layer1 = 0
    for nal_type, layer_id, _chunk in _iter_annexb_nals(data):
        if nal_type <= 31:
            vcl_total += 1
            if layer_id > 0:
                vcl_layer1 += 1
    return vcl_total, vcl_layer1


# ---------------------------------------------------------------------------
# 10-bit -> 8-bit ordered dither (MVHEVCAdapter; ffmpeg does its own for packed)
# ---------------------------------------------------------------------------
_BAYER2 = np.array([[0, 2], [3, 1]], dtype=np.uint16)


def _down10to8(p):
    """yuv420p10le plane (uint16, 0..1023) -> 8-bit with a light 2x2 ordered
    dither (spec §3: '>>2 + dither ordonne leger'). Any shape."""
    if p.dtype == np.uint8:
        return np.ascontiguousarray(p)
    h, w = p.shape
    dith = np.tile(_BAYER2, ((h + 1) // 2, (w + 1) // 2))[:h, :w]
    return np.clip((p.astype(np.uint16) + dith) >> 2, 0, 255).astype(np.uint8)


def _as_u8(p):
    return p if p.dtype == np.uint8 else _down10to8(p)


def _pack_i420(yl, ul, vl, yr, ur, vr):
    """hstack a left+right I420 view pair into one packed-SBS 8-bit frame's bytes
    (Y row-plane || then U || then V, all left|right hstacked)."""
    Y = np.hstack([_as_u8(yl), _as_u8(yr)])
    U = np.hstack([_as_u8(ul), _as_u8(ur)])
    V = np.hstack([_as_u8(vl), _as_u8(vr)])
    return Y.tobytes() + U.tobytes() + V.tobytes()


# ===========================================================================
# Source adapters — each yields packed 8-bit yuv420p frames + metadata.
# Contract: metadata() (cheap, ffprobe) -> open() (acquire decode resources) ->
# frames() (generator of `frame_bytes`-sized bytes) -> close() (release).
# ===========================================================================
class _BaseAdapter:
    def __init__(self, desc, opts):
        self.desc = desc
        self.opts = opts
        self.path = desc['path']
        self._cancel = threading.Event()
        self._procs = []            # child Popen handles for cancel()
        self._meta = None

    # subclasses fill self._meta with the packed-canvas facts
    def metadata(self):
        raise NotImplementedError

    def open(self):
        pass

    def frames(self):
        raise NotImplementedError

    def close(self):
        for p in list(self._procs):
            _kill(p)
        self._procs.clear()

    def cancel(self):
        self._cancel.set()
        for p in list(self._procs):
            _kill(p)

    # ---- shared per-view/canvas geometry from base ffprobe facts ----
    def _finalise_meta(self, m, per_view_w, per_view_h, fmt):
        canvas_w = per_view_w * (2 if fmt == 1 else 1)
        canvas_h = per_view_h * (2 if fmt == 2 else 1)
        m.update({
            'per_view_w': per_view_w, 'per_view_h': per_view_h,
            'canvas_w': canvas_w, 'canvas_h': canvas_h,
            'frame_bytes': canvas_w * canvas_h * 3 // 2,
            'format': fmt,
        })
        return m


class PackedAdapter(_BaseAdapter):
    """F-SBS / F-TAB (HEVC or H.264). ffmpeg CLI decodes to rawvideo on a pipe —
    out-of-process, fast, and packed sources need no pairing logic. F-TAB is fed
    to x265 as native top-bottom (format=2, no re-pack)."""

    def metadata(self):
        if self._meta:
            return self._meta
        m = probe_metadata(self.path)
        packing = self.desc.get('packing')
        if packing not in ('sbs', 'tab'):
            # auto-detect from the packed aspect / filename (hevc_stereo_detect)
            packing = _detect_packing(self.path, m)
        self._packing = packing
        if packing == 'tab':
            per_w, per_h, fmt = m['width'], m['height'] // 2, 2
        else:
            per_w, per_h, fmt = m['width'] // 2, m['height'], 1
        self._meta = self._finalise_meta(m, per_w, per_h, fmt)
        return self._meta

    def frames(self):
        m = self.metadata()
        fb = m['frame_bytes']
        max_frames = self.opts.get('max_frames')
        cmd = [_ffmpeg_path(), '-v', 'error', '-nostdin', '-i', self.path, '-map', '0:v:0']
        if max_frames:
            cmd += ['-frames:v', str(int(max_frames))]  # OUTPUT option — must follow -i
        cmd += ['-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0,
                                creationflags=_NO_WINDOW)
        self._procs.append(proc)
        try:
            while not self._cancel.is_set():
                data = _read_exact(proc.stdout, fb)
                if len(data) < fb:
                    break
                yield data
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            _kill(proc)


class MVHEVCAdapter(_BaseAdapter):
    """Native MV-HEVC: OUR LavfHevcSource.read_view_pair (SW) -> hstack(L,R)."""

    def __init__(self, desc, opts):
        super().__init__(desc, opts)
        self._src = None

    def metadata(self):
        if self._meta:
            return self._meta
        m = probe_metadata(self.path)
        # MV-HEVC stores each view natively: ffprobe dims ARE the per-view dims.
        self._meta = self._finalise_meta(m, m['width'], m['height'], 1)
        return self._meta

    def open(self):
        from lavf_hevc_source import LavfHevcSource
        self._src = LavfHevcSource()
        mi = self._src.open(self.path, allow_hw=False)
        if mi is None or not mi.multiview:
            raise RuntimeError(f'LavfHevcSource open/multiview failed: {mi}')
        # Prefer the decoder's own colour verdict when ffprobe said 'unknown'.
        m = self.metadata()
        if mi.color_space:
            cand = _map_matrix(mi.color_space)
            if cand != 'bt709' or m['colormatrix'] == 'bt709':
                m['colormatrix'] = cand
                m['colorprim'] = _map_prim(mi.color_space)
        if mi.color_trc:
            m['transfer'] = _map_trc(mi.color_trc)

    def frames(self):
        max_frames = self.opts.get('max_frames')
        n = 0
        while not self._cancel.is_set():
            o = self._src.read_view_pair()
            if o is None:
                break
            (yl, ul, vl), (yr, ur, vr), _pts = o
            yield _pack_i420(yl, ul, vl, yr, ur, vr)
            n += 1
            if max_frames and n >= max_frames:
                break

    def close(self):
        super().close()
        if self._src is not None:
            try:
                self._src.close()
            except Exception:
                pass
            self._src = None


class MVCAdapter(_BaseAdapter):
    """MVC (MKV / SSIF / dual-file). A DEDICATED demuxer + a PRIVATE synchronous
    edge264 session (never the live player's). One Edge264Frame carries base
    (`samples[]`) + dependent (`samples_mvc[]`); views are re-paired by
    PictureOrderCnt (edge264 glues base[X] with dep[X-1] on B-frame GOPs) via a
    small sliding POC window before hstack. Output order = ascending base POC
    (display order)."""

    def __init__(self, desc, opts):
        super().__init__(desc, opts)
        self._dmx = None
        self._dec = None            # ctypes.c_void_p edge264 session (ours)
        self._edge = None
        self._lock = None
        self._Frame = None
        self._find_nals = None
        self._keep = []             # NAL buffers kept alive across a drain

    def metadata(self):
        if self._meta:
            return self._meta
        # ffprobe reads the BASE H.264 view (full frame = per-view dims), fps,
        # colour, audio — it just can't DECODE the MVC dependent view.
        m = probe_metadata(self.path)
        if not m['width'] or not m['height']:
            m['width'], m['height'] = 1920, 1080     # BD3D default
        self._meta = self._finalise_meta(m, m['width'], m['height'], 1)
        return self._meta

    def open(self):
        import ctypes  # noqa
        from mvc_decoder import (edge264, Edge264Frame, find_nal_units,
                                  create_demuxer, convert_avcc_to_annexb,
                                  _apply_bd_seek_tables, edge264_session_lock)
        import mvc_demuxer_cpp
        self._ctypes = ctypes
        self._edge = edge264
        self._Frame = Edge264Frame
        self._find_nals = find_nal_units
        self._convert = convert_avcc_to_annexb
        self._lock = edge264_session_lock
        if edge264 is None:
            raise RuntimeError('edge264.dll not loaded — MVC decode unavailable')

        kind = self.desc.get('mvc_container')
        dep = self.desc.get('dep_path')
        if kind == 'dual' or (dep and os.path.isfile(dep)):
            dmx = mvc_demuxer_cpp.MVCSSIFDemuxer()
            if not dmx.open_dual(self.path, dep):
                raise RuntimeError('MVCSSIFDemuxer.open_dual failed')
            eff = self.path
        else:
            dmx, eff = create_demuxer(self.path)
            if not hasattr(dmx, 'read_next_frame_pair') or not dmx.open(eff):
                raise RuntimeError(f'demuxer open failed: {eff}')
        try:
            _apply_bd_seek_tables(dmx, eff)
        except Exception:
            pass
        self._dmx = dmx
        with self._lock:
            ptr = self._edge.edge264_alloc(0, None, None, 0, None, None, None)  # n_threads=0
        if not ptr:
            raise RuntimeError('edge264_alloc failed')
        self._dec = ctypes.c_void_p(ptr)
        # headers ONCE (MKV carries them only in codec-private; SSIF repeats them
        # in-stream too — feeding once is the universal safe path).
        cp = b''
        try:
            cp = bytes(dmx.get_codec_private() or b'')
        except Exception:
            cp = b''
        if cp[:4] == b'\x00\x00\x00\x01' or cp[:3] == b'\x00\x00\x01':
            hdr = cp
        else:
            hdr = self._convert(cp)
        if hdr:
            self._feed_au(hdr)

    def _feed_nal(self, nal):
        """Feed one start-code-prefixed NAL: strip start code, 2x0xFF head pad +
        64-byte tail pad, blocking decode (n_threads=0). Mirrors
        ThumbnailService._feed exactly, INCLUDING the edge264_session_lock around
        the decode call: alloc/free in another session (e.g. a live-playback
        seek re-alloc under the same lock) racing an unlocked decode wedges the
        DLL — see mvc_decoder.py:496-501."""
        ctypes = self._ctypes
        off = 4 if nal[:4] == b'\x00\x00\x00\x01' else 3
        content = nal[off:]
        n = len(content)
        if n == 0:
            return
        ntype = content[0] & 0x1F
        if ntype == 24:            # STAP-A aggregator — skip (matches _process_au_data)
            return
        buf = ctypes.create_string_buffer(2 + n + 64)
        buf[0] = b'\xFF'
        buf[1] = b'\xFF'
        ctypes.memmove(ctypes.addressof(buf) + 2, bytes(content), n)
        start = ctypes.cast(ctypes.addressof(buf) + 2, ctypes.POINTER(ctypes.c_uint8))
        end = ctypes.cast(ctypes.addressof(buf) + 2 + n, ctypes.POINTER(ctypes.c_uint8))
        self._keep.append(buf)     # keep alive until this AU is drained
        try:
            with self._lock:
                self._edge.edge264_decode_NAL(self._dec, start, end, 0, None, None, None)
        except (OSError, RuntimeError):
            pass

    def _feed_au(self, au_bytes):
        for nal in self._find_nals(au_bytes):
            self._feed_nal(nal)

    def _copy_plane(self, ptr, w, h, stride):
        arr = np.ctypeslib.as_array(ptr, shape=(h, stride))
        return np.array(arr[:, :w], copy=True)

    def _drain(self, base_planes, dep_planes):
        """bump once, then borrow-drain every ready frame; pool base by POC and
        dep by POC_mvc; return frame after copying."""
        ctypes = self._ctypes
        frame = self._Frame()
        with self._lock:
            self._edge.edge264_bump_frames(self._dec)
            while True:
                ret = self._edge.edge264_get_frame(self._dec, ctypes.byref(frame), 1)  # borrow
                if ret != 0 or not frame.samples[0]:
                    break
                w, h = frame.width_Y, frame.height_Y
                cw, ch = frame.width_C, frame.height_C
                if cw <= 0 or ch <= 0:
                    cw, ch = w // 2, h // 2
                sy, sc = frame.stride_Y, frame.stride_C
                base = (self._copy_plane(frame.samples[0], w, h, sy),
                        self._copy_plane(frame.samples[1], cw, ch, sc),
                        self._copy_plane(frame.samples[2], cw, ch, sc))
                base_planes.append((frame.PictureOrderCnt, base))
                if frame.samples_mvc[0]:
                    dep = (self._copy_plane(frame.samples_mvc[0], w, h, sy),
                           self._copy_plane(frame.samples_mvc[1], cw, ch, sc),
                           self._copy_plane(frame.samples_mvc[2], cw, ch, sc))
                    dep_planes[frame.PictureOrderCnt_mvc] = dep
                if frame.return_arg:
                    try:
                        self._edge.edge264_return_frame(self._dec, frame.return_arg)
                    except (OSError, RuntimeError):
                        pass

    def frames(self):
        max_frames = self.opts.get('max_frames')
        pending = []                # [(poc_base, (yl,ul,vl))] sorted lazily
        dep_pool = {}               # poc_mvc -> (yr,ur,vr)
        emitted = 0

        def _emit(poc, left):
            yl, ul, vl = left
            right = dep_pool.pop(poc, None)
            if right is None:
                right = left        # 2D / missing dep -> duplicate left (matches player)
            # prune stale deps that can no longer pair (older than what we emit)
            for k in [k for k in dep_pool if k < poc - 8]:
                dep_pool.pop(k, None)
            return _pack_i420(yl, ul, vl, right[0], right[1], right[2])

        while not self._cancel.is_set():
            ok, base, dep = self._dmx.read_next_frame_pair()
            if not ok:
                break
            self._keep.clear()      # previous AU consumed (synchronous decode)
            au = bytearray()
            if base and 'data' in base:
                au += bytes(base['data'])
            if dep and 'data' in dep:
                au += bytes(dep['data'])
            if au:
                self._feed_au(bytes(au))
            new_bases = []
            self._drain(new_bases, dep_pool)
            pending.extend(new_bases)
            if len(pending) > REORDER_DEPTH:
                pending.sort(key=lambda x: x[0])
                while len(pending) > REORDER_DEPTH:
                    poc, left = pending.pop(0)
                    yield _emit(poc, left)
                    emitted += 1
                    if max_frames and emitted >= max_frames:
                        return
        # flush the DPB tail, then emit everything still pending in display order
        if not self._cancel.is_set():
            new_bases = []
            for _ in range(REORDER_DEPTH + 2):
                self._drain(new_bases, dep_pool)
            pending.extend(new_bases)
            pending.sort(key=lambda x: x[0])
            for poc, left in pending:
                if self._cancel.is_set():
                    break
                yield _emit(poc, left)
                emitted += 1
                if max_frames and emitted >= max_frames:
                    return

    def close(self):
        super().close()
        if self._dec is not None and self._edge is not None:
            try:
                with self._lock:
                    self._edge.edge264_free(self._ctypes.byref(self._dec))
            except (OSError, RuntimeError):
                pass
            self._dec = None
        if self._dmx is not None:
            try:
                self._dmx.close()
            except Exception:
                pass
            self._dmx = None
        self._keep = []


def _detect_packing(path, meta):
    """Fallback packing detection for a PackedAdapter with no explicit 'packing'."""
    try:
        import hevc_stereo_detect
        mode, _half, _inv = hevc_stereo_detect.detect(path, None)
        if mode in ('sbs', 'tab'):
            return mode
    except Exception:
        pass
    # geometry heuristic: very wide -> sbs, very tall -> tab
    w, h = meta['width'], meta['height']
    return 'tab' if (h and w / h < 1.4) else 'sbs'


def build_adapter(desc, opts):
    kind = desc.get('kind')
    if kind == 'packed':
        return PackedAdapter(desc, opts)
    if kind == 'mvhevc':
        return MVHEVCAdapter(desc, opts)
    if kind == 'mvc':
        return MVCAdapter(desc, opts)
    raise ValueError(f'unknown source kind: {kind!r}')


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------
def _read_exact(pipe, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _kill(proc):
    """Terminate a child process without leaving a zombie."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
    except Exception:
        pass


# Audio codecs QuickTime accepts natively -> stream-copy; everything else -> AAC.
_AUDIO_COPY = {'aac', 'ac3', 'eac3', 'alac'}
_AUDIO_ELEMENTARY = {'aac': ('adts', 'aac'), 'ac3': ('ac3', 'ac3'),
                     'eac3': ('eac3', 'eac3'), 'alac': ('ipod', 'm4a')}


# ===========================================================================
# The export job
# ===========================================================================
class MVHEVCExporter(QThread):
    """Background, cancellable MV-HEVC export job.

    Signals:
      progress(dict):       {step, mode, [frames_done, total_frames, fps, eta_s]}
                             — the reencode path's dicts carry frames/fps; the
                             EX-2b remux path's dicts are step-based only
                             (step in {probe, copy, audio, extracting, muxing,
                             validating, done} — no frames/fps, since x265
                             never runs on that path).
      exportFinished(str):  the output .mov path
      failed(str):          a short reason ('annule' on cancel)

    Attributes (readable after completion):
      mode: 'remux-tier1' | 'remux-tier2' | 'reencode' — which path was
            actually taken (EX-2b §2: Tier 1 container-copy, Tier 2
            extract+MP4Box, or Tier 3 = the unchanged reencode pipeline,
            reached directly for packed/mvc sources or as the remux
            fallback).
      layer_nal_counts: (vcl_total, vcl_layer_gt0) from Tier 2's dependent-
            view NAL-survival check, or None (Tier 1 / reencode don't count).

    NB: named `exportFinished` (not `finished`) — QThread already defines a
    `finished` signal (emitted when run() returns); shadowing it with a
    same-named custom Signal is a landmine for any future connect() (EX-4 and
    beyond) that means "the QThread itself finished" vs "the export finished".
    """
    progress = Signal(object)
    exportFinished = Signal(str)
    failed = Signal(str)

    def __init__(self, source_desc, out_path, opts=None, parent=None):
        super().__init__(parent)
        self.source_desc = dict(source_desc)
        self.out_path = out_path
        self.opts = dict(opts or {})
        self._cancelled = threading.Event()
        self._workdir = None
        self._adapter = None
        self._procs = []            # x265 / mp4box handles for cancel()
        self._proc_lock = threading.Lock()
        # EX-2b: reported outcome ('remux-tier1'|'remux-tier2'|'reencode') and,
        # for Tier 2, the layer-NAL survival counts (vcl_total, vcl_layer_gt0).
        self.mode = None
        self.layer_nal_counts = None

    # ---- public API ----
    def cancel(self):
        self._cancelled.set()
        if self._adapter is not None:
            try:
                self._adapter.cancel()
            except Exception:
                pass
        with self._proc_lock:
            for p in list(self._procs):
                _kill(p)

    # ---- preset resolution ----
    def _preset_crf(self):
        preset = self.opts.get('preset')
        crf = self.opts.get('crf')
        if preset and crf is not None:
            return str(preset), str(crf)
        q = self.opts.get('quality', 'quality')
        if q == 'fast':
            return 'medium', '23'
        return 'slow', '20'          # default = quality (EX-1 slow/crf20)

    def _register(self, proc):
        with self._proc_lock:
            self._procs.append(proc)
        return proc

    # ---- QThread entry ----
    def run(self):
        try:
            self._run_pipeline()
        except _Cancelled:
            self._purge()
            self.failed.emit('annule')
        except Exception as e:
            logger.exception('[EXPORT] pipeline failed')
            self._purge()
            self.failed.emit(str(e))

    def _ck(self):
        if self._cancelled.is_set():
            raise _Cancelled()

    def _run_pipeline(self):
        if not tools_available():
            raise RuntimeError('required tool missing (x265/mp4box/ffmpeg/ffprobe)')
        self._workdir = tempfile.mkdtemp(prefix='sylc_mvhevc_')
        try:
            if (self.source_desc.get('kind') == 'mvhevc'
                    and not self.opts.get('force_reencode')):
                self._ck()
                if self._attempt_remux():
                    return
            self._run_reencode()
        finally:
            if self._adapter is not None:
                try:
                    self._adapter.close()
                except Exception:
                    pass
                self._adapter = None
            self._purge()

    # ---- EX-2b remux fast-path: Tier 1 (container copy) / Tier 2 (ES extract
    # + MP4Box) — never re-encode when a copy suffices. Returns True iff a
    # remux tier completed successfully (exportFinished already emitted);
    # False means "fall through to the unchanged reencode path" (Tier 3), with
    # the reason logged per spec §2: "[EXPORT] remux impossible (<reason>) ->
    # reencodage". A cancellation is NOT swallowed here — it propagates so
    # run() reports failed('annule') instead of silently reencoding. ----
    def _attempt_remux(self):
        path = self.source_desc['path']
        try:
            probe = probe_mv_hevc_container(path)
        except Exception as e:
            logger.warning('[EXPORT] remux impossible (box-probe failed: %s) -> reencodage', e)
            return False
        self._ck()
        mode = 'remux-tier1' if probe['conformant'] else 'remux-tier2'
        self.mode = mode
        logger.info('[EXPORT] mv-hevc source probe: %s -> attempting %s', probe, mode)
        try:
            if probe['conformant']:
                tmp_mov = self._remux_tier1(probe)
            else:
                tmp_mov = self._remux_tier2(probe)
            self._finalize(tmp_mov, mode)
            return True
        except _Cancelled:
            raise
        except Exception as e:
            logger.warning('[EXPORT] remux impossible (%s) -> reencodage', e)
            self.mode = None
            self.layer_nal_counts = None
            return False

    def _emit_step(self, step, mode):
        self.progress.emit({'step': step, 'mode': mode})

    def _copy_cancelable(self, src, dst):
        """Chunked, cancellation-aware file copy (Tier 1's "temp-then-atomic-
        move" copy stage): a single blocking shutil.copy2() can't be
        interrupted by cancel() on a multi-GB source, so this checks
        self._cancelled between chunks instead. `_copy_chunk_bytes`/
        `_copy_delay_s` are test-only knobs (default 8 MiB / no delay —
        negligible overhead vs. shutil.copy2 in production)."""
        chunk_bytes = int(self.opts.get('_copy_chunk_bytes') or (8 * 1024 * 1024))
        delay_s = float(self.opts.get('_copy_delay_s') or 0.0)
        with open(src, 'rb') as fin, open(dst, 'wb') as fout:
            while True:
                if self._cancelled.is_set():
                    raise _Cancelled()
                buf = fin.read(chunk_bytes)
                if not buf:
                    break
                fout.write(buf)
                if delay_s:
                    time.sleep(delay_s)
        try:
            shutil.copystat(src, dst)
        except Exception:
            pass

    # ---- Tier 1: source container already hvc1+hvcC+lhvC-conformant ----
    def _remux_tier1(self, probe):
        src = self.source_desc['path']
        self._emit_step('copy', 'remux-tier1')
        ext = os.path.splitext(src)[1] or '.mp4'
        tmp_copy = os.path.join(self._workdir, 'tier1_copy' + ext)
        self._copy_cancelable(src, tmp_copy)
        self._ck()

        meta = probe_metadata(src)
        codec = meta.get('audio_codec')
        if not codec or codec in _AUDIO_COPY:
            # No audio, or already QuickTime-safe: the verbatim copy IS the
            # Tier-1 output — untouched bytes, hvcC/lhvC guaranteed intact.
            return tmp_copy

        # Audio needs transcoding (TrueHD/DTS/...). A `-c:v copy` ffmpeg remux
        # of an ALREADY-conformant hvc1+hvcC+lhvC container preserves hvcC/
        # lhvC byte-for-byte (EX-2b finding — see module docstring); this is
        # NOT the EX-1 "ffmpeg can't write lhvC" case (that was building hvc1
        # from a bare ES). Re-probed below regardless, belt-and-braces.
        self._emit_step('audio', 'remux-tier1')
        out_mov = os.path.join(self._workdir, 'tier1_remux.mov')
        cmd = [_ffmpeg_path(), '-v', 'error', '-nostdin', '-y', '-i', tmp_copy,
               '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'copy',
               '-c:a', 'aac', '-b:a', '384k', '-f', 'mp4', '-tag:v', 'hvc1', out_mov]
        logger.info('[EXPORT] tier1 audio remux: %s', ' '.join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                creationflags=_NO_WINDOW)
        self._register(proc)
        _, err = proc.communicate()
        if self._cancelled.is_set():
            raise _Cancelled()
        if proc.returncode != 0 or not (os.path.isfile(out_mov) and os.path.getsize(out_mov) > 0):
            raise RuntimeError(f'tier-1 audio remux failed (rc={proc.returncode}): '
                               f'{(err or b"").decode("utf-8", "replace")[-400:]}')
        p2 = probe_mv_hevc_container(out_mov)
        if not (p2['has_hvcC'] and p2['has_lhvC']):
            raise RuntimeError('tier-1 audio remux dropped hvcC/lhvC')
        return out_mov

    # ---- Tier 2: extract the video ES (layer NALs intact) -> MP4Box mux ----
    def _remux_tier2(self, probe):
        src = self.source_desc['path']
        meta = probe_metadata(src)

        self._emit_step('extracting', 'remux-tier2')
        hevc_path = os.path.join(self._workdir, 'tier2_video.hevc')
        cmd = [_ffmpeg_path(), '-v', 'error', '-nostdin', '-y', '-i', src,
               '-map', '0:v:0', '-c:v', 'copy', '-bsf:v', 'hevc_mp4toannexb',
               '-f', 'hevc', hevc_path]
        logger.info('[EXPORT] tier2 ES extraction: %s', ' '.join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                creationflags=_NO_WINDOW)
        self._register(proc)
        _, err = proc.communicate()
        if self._cancelled.is_set():
            raise _Cancelled()
        if proc.returncode != 0 or not (os.path.isfile(hevc_path) and os.path.getsize(hevc_path) > 0):
            raise RuntimeError(f'tier-2 ES extraction failed (rc={proc.returncode}): '
                               f'{(err or b"").decode("utf-8", "replace")[-400:]}')

        vcl_total, vcl_layer1 = _count_layer_nals(hevc_path)
        self.layer_nal_counts = (vcl_total, vcl_layer1)
        if vcl_total <= 0:
            raise RuntimeError('tier-2 extraction produced no VCL NALs')
        ratio = vcl_layer1 / vcl_total
        if ratio < 0.35:      # expect ~0.5 (base+dependent view); generous floor
            raise RuntimeError(f'tier-2 layer-NAL survival check failed: '
                               f'{vcl_layer1}/{vcl_total} ({ratio:.2f}) — dependent view lost')

        audio = self._extract_audio(meta, mode='remux-tier2')
        self._ck()

        self._emit_step('muxing', 'remux-tier2')
        tmp_mov = os.path.join(self._workdir, 'tier2_mux.mov')
        rng = 'on' if meta['range_full'] else 'off'
        colr = 'colr=nclx,%s,%s,%s,%s' % (meta['colorprim'], meta['transfer'],
                                          meta['colormatrix'], rng)
        vspec = '%s:fps=%s:%s' % (hevc_path.replace('\\', '/'), meta['fps_str'], colr)
        cmd = [MP4BOX, '-add', vspec]
        if audio:
            cmd += ['-add', audio.replace('\\', '/')]
        cmd += ['-new', tmp_mov]
        logger.info('[EXPORT] tier2 mp4box: %s', ' '.join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=_NO_WINDOW)
        self._register(proc)
        out, _ = proc.communicate()
        if self._cancelled.is_set():
            raise _Cancelled()
        log = (out or b'').decode('utf-8', 'replace')
        if proc.returncode != 0 or not (os.path.isfile(tmp_mov) and os.path.getsize(tmp_mov) > 0):
            raise RuntimeError(f'tier-2 MP4Box mux failed (rc={proc.returncode})\n{log[-800:]}')
        return tmp_mov

    # ---- shared tail: inject vexu -> validate -> atomic move -> emit ----
    def _finalize(self, tmp_mov, mode, done_extra=None):
        # Task EX-3: lazy import breaks the circular reference (vexu_injector
        # imports box-walker helpers FROM this module) -- by the time
        # _finalize() actually runs, mvhevc_exporter is already fully loaded.
        # Runs for EVERY tier (reencode, remux-tier1, remux-tier2): this is
        # the single shared tail all three funnel through. Idempotent if the
        # source already carried vexu (Tier 1 copying an already-conformant
        # file with a vexu box, e.g. the reference sample -- no-op).
        from vexu_injector import inject_vexu
        inject_vexu(tmp_mov)
        self._validate(tmp_mov)
        self._ck()
        self.mode = mode
        dst_dir = os.path.dirname(os.path.abspath(self.out_path)) or '.'
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(self.out_path):
            os.remove(self.out_path)
        shutil.move(tmp_mov, self.out_path)
        logger.info('[EXPORT] mode=%s output=%s', mode, self.out_path)
        payload = {'step': 'done', 'mode': mode}
        if done_extra:
            payload.update(done_extra)
        self.progress.emit(payload)
        self.exportFinished.emit(self.out_path)

    # ---- Tier 3 / default: the UNCHANGED reencode pipeline (adapter -> x265
    # -> audio -> MP4Box), reached directly for packed/mvc sources, or as the
    # fallback when the remux attempt above failed or its validation did ----
    def _run_reencode(self):
        self.mode = 'reencode'
        self._adapter = build_adapter(self.source_desc, self.opts)
        meta = self._adapter.metadata()
        self._ck()
        self._adapter.open()
        self._ck()

        hevc = os.path.join(self._workdir, 'video.hevc')
        self._encode(meta, hevc)
        self._ck()

        audio = self._extract_audio(meta)
        self._ck()

        tmp_mov = os.path.join(self._workdir, 'out.mov')
        self._mux(meta, hevc, audio, tmp_mov)
        self._ck()

        self._finalize(tmp_mov, 'reencode',
                       {'frames_done': self._frames_done, 'total_frames': self._total,
                        'fps': 0.0, 'eta_s': 0})

    # ---- stage 2: x265 multiview encode over a STDIN pipe ----
    def _encode(self, meta, out_hevc):
        preset, crf = self._preset_crf()
        cfg = os.path.join(self._workdir, 'mv.cfg')
        with open(cfg, 'w') as f:
            f.write('--num-views 2\n--format %d\n--input "-"\n' % meta['format'])
        cmd = [X265, '--multiview-config', cfg, '--num-views', '2',
               '--input-res', f"{meta['per_view_w']}x{meta['per_view_h']}",
               '--fps', meta['fps_str'], '--input-csp', 'i420', '--input-depth', '8',
               '--profile', 'main',
               '--colorprim', meta['colorprim'], '--transfer', meta['transfer'],
               '--colormatrix', meta['colormatrix'],
               '--range', 'full' if meta['range_full'] else 'limited',
               '--preset', preset, '--crf', crf, '--output', out_hevc]
        logger.info('[EXPORT] x265: %s', ' '.join(cmd))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, bufsize=0, creationflags=_NO_WINDOW)
        self._register(proc)

        # total frames (best-effort, for ETA); capped by max_frames
        max_frames = self.opts.get('max_frames')
        total = meta.get('nb_frames') or 0
        if not total and meta.get('duration_s') and meta.get('fps_float'):
            total = int(round(meta['duration_s'] * meta['fps_float']))
        if max_frames:
            total = min(total, max_frames) if total else max_frames
        self._total = total or 0

        # drain x265 stderr on a side thread (avoids pipe-buffer deadlock + captures log)
        self._x265_log = []
        self._x265_fps = 0.0
        rd = threading.Thread(target=self._read_x265_stderr, args=(proc,), daemon=True)
        rd.start()

        self._frames_done = 0
        fb = meta['frame_bytes']
        t0 = time.monotonic()
        last_emit = 0.0
        broken = False
        try:
            for frame in self._adapter.frames():
                if self._cancelled.is_set():
                    break
                if len(frame) != fb:
                    raise RuntimeError(f'adapter frame size {len(frame)} != {fb}')
                try:
                    proc.stdin.write(frame)
                except (BrokenPipeError, OSError):
                    broken = True
                    break
                self._frames_done += 1
                now = time.monotonic()
                if now - last_emit >= 0.2:
                    last_emit = now
                    el = now - t0
                    fps = self._x265_fps or (self._frames_done / el if el > 0 else 0.0)
                    eta = ((self._total - self._frames_done) / fps
                           if fps > 0 and self._total else -1)
                    self.progress.emit({'step': 'encoding', 'mode': self.mode,
                                        'frames_done': self._frames_done,
                                        'total_frames': self._total, 'fps': round(fps, 2),
                                        'eta_s': int(eta) if eta >= 0 else -1})
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
        rc = self._wait_bounded(proc)
        rd.join(timeout=2)
        if self._cancelled.is_set():
            raise _Cancelled()
        log = ''.join(self._x265_log)
        if broken or rc != 0 or not (os.path.isfile(out_hevc) and os.path.getsize(out_hevc) > 0):
            raise RuntimeError(f'x265 failed (rc={rc}, frames={self._frames_done})\n{log[-800:]}')
        if self._frames_done <= 0:
            raise RuntimeError('adapter produced no frames')
        self.progress.emit({'step': 'encoded', 'mode': self.mode, 'frames_done': self._frames_done,
                            'total_frames': self._total, 'fps': round(self._x265_fps, 2),
                            'eta_s': 0})

    def _wait_bounded(self, proc, poll_s=0.5, max_wait_s=3600):
        """Cancel-aware bounded replacement for a raw `proc.wait()`: polls with
        a short timeout so cancel() (which already terminates registered procs)
        is noticed promptly instead of blocking forever, and a generous overall
        deadline force-kills a wedged/hung process rather than hanging the
        export thread indefinitely. Returns the process' returncode (or -1 if
        it had to be force-killed after the deadline)."""
        deadline = time.monotonic() + max_wait_s
        while True:
            try:
                return proc.wait(timeout=poll_s)
            except subprocess.TimeoutExpired:
                if self._cancelled.is_set():
                    _kill(proc)
                    try:
                        return proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        return -1
                if time.monotonic() >= deadline:
                    logger.error('[EXPORT] child process %s did not exit within %ss — '
                                 'force-killing', getattr(proc, 'pid', '?'), max_wait_s)
                    _kill(proc)
                    return -1

    def _read_x265_stderr(self, proc):
        import re
        rx_fps = re.compile(rb'([0-9]+(?:\.[0-9]+)?)\s*fps')
        buf = b''
        try:
            while True:
                chunk = proc.stderr.read(256)
                if not chunk:
                    break
                buf += chunk
                self._x265_log.append(chunk.decode('utf-8', 'replace'))
                tail = buf[-400:]
                mm = rx_fps.findall(tail)
                if mm:
                    try:
                        self._x265_fps = float(mm[-1])
                    except ValueError:
                        pass
                buf = buf[-400:]
        except Exception:
            pass

    # ---- stage 4: audio extraction/transcode from the ORIGINAL source ----
    def _extract_audio(self, meta, mode=None):
        codec = meta.get('audio_codec')
        if not codec:
            return None            # source has no audio (e.g. the MV-HEVC sample)
        if mode:
            # remux mode: step-based progress only (no frames/fps — self._frames_done
            # / self._total are never set on this path, since x265 never ran).
            self._emit_step('audio', mode)
        else:
            self.progress.emit({'step': 'audio', 'frames_done': self._frames_done,
                                'total_frames': self._total, 'fps': 0.0, 'eta_s': -1})
        max_frames = self.opts.get('max_frames')
        dur = None
        if max_frames and meta.get('fps_float'):
            dur = max_frames / meta['fps_float']
        src_path = self.source_desc['path']
        cmd = [_ffmpeg_path(), '-v', 'error', '-nostdin', '-y']
        if dur:
            cmd += ['-t', f'{dur:.6f}']
        cmd += ['-i', src_path, '-map', '0:a:0', '-vn']
        if codec in _AUDIO_COPY and codec in _AUDIO_ELEMENTARY:
            fmt, ext = _AUDIO_ELEMENTARY[codec]
            out = os.path.join(self._workdir, f'audio.{ext}')
            cmd += ['-c:a', 'copy', '-f', fmt, out]
            action = 'copy'
        else:
            out = os.path.join(self._workdir, 'audio.aac')
            cmd += ['-c:a', 'aac', '-b:a', '384k', '-f', 'adts', out]
            action = 'transcode->aac'
        logger.info('[EXPORT] audio %s (%s): %s', action, codec, ' '.join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                creationflags=_NO_WINDOW)
        self._register(proc)
        _, err = proc.communicate()
        if self._cancelled.is_set():
            raise _Cancelled()
        if proc.returncode != 0 or not (os.path.isfile(out) and os.path.getsize(out) > 0):
            # audio is best-effort: log and continue video-only rather than abort
            logger.warning('[EXPORT] audio extraction failed (%s) — muxing video-only: %s',
                           codec, (err or b'').decode('utf-8', 'replace')[-300:])
            return None
        return out

    # ---- stage 3: MP4Box mux (video + colr nclx + optional audio) ----
    def _mux(self, meta, hevc, audio, out_mov):
        self.progress.emit({'step': 'muxing', 'mode': self.mode, 'frames_done': self._frames_done,
                            'total_frames': self._total, 'fps': 0.0, 'eta_s': -1})
        rng = 'on' if meta['range_full'] else 'off'
        colr = 'colr=nclx,%s,%s,%s,%s' % (meta['colorprim'], meta['transfer'],
                                          meta['colormatrix'], rng)
        vspec = '%s:fps=%s:%s' % (hevc.replace('\\', '/'), meta['fps_str'], colr)
        cmd = [MP4BOX, '-add', vspec]
        if audio:
            cmd += ['-add', audio.replace('\\', '/')]
        cmd += ['-new', out_mov]
        logger.info('[EXPORT] mp4box: %s', ' '.join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=_NO_WINDOW)
        self._register(proc)
        out, _ = proc.communicate()
        if self._cancelled.is_set():
            raise _Cancelled()
        log = (out or b'').decode('utf-8', 'replace')
        if proc.returncode != 0 or not (os.path.isfile(out_mov) and os.path.getsize(out_mov) > 0):
            raise RuntimeError(f'MP4Box mux failed (rc={proc.returncode})\n{log[-800:]}')

    # ---- post-export validation (spec §6): ffprobe view_ids_available + a
    # dogfood re-read with OUR OWN reader (LavfHevcSource), the same one a real
    # playback session would use to open the exported file ----
    def _validate(self, mov):
        if self.opts.get('validate') is False:
            return
        j = subprocess.run(
            [_ffprobe_path(), '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=view_ids_available', '-of', 'json', mov],
            capture_output=True, text=True, creationflags=_NO_WINDOW)
        st = (json.loads(j.stdout or '{}').get('streams') or [{}])[0]
        if st.get('view_ids_available') != '0,1':
            raise RuntimeError(f"post-export validation: view_ids_available="
                               f"{st.get('view_ids_available')!r} (expected '0,1')")
        self._validate_dogfood(mov)

    def _validate_dogfood(self, mov, max_pairs=60):
        """Re-open the freshly-muxed .mov with LavfHevcSource (allow_hw=False,
        exactly as MVHEVCAdapter does) and read up to `max_pairs` view pairs (or
        until EOF for shorter exports). Requires >=1 pair and no exception —
        this is our own decoder proving the file it just wrote is actually
        readable, not just well-formed to ffprobe."""
        try:
            from lavf_hevc_source import LavfHevcSource
        except Exception as e:
            raise RuntimeError(f'post-export dogfood: LavfHevcSource unavailable: {e}')
        s = LavfHevcSource()
        try:
            try:
                mi = s.open(mov, allow_hw=False)
            except Exception as e:
                raise RuntimeError(f'post-export dogfood: open() raised: {e}')
            if mi is None or not mi.multiview:
                raise RuntimeError(f'post-export dogfood: open/multiview failed: {mi}')
            n = 0
            for _ in range(max_pairs):
                try:
                    o = s.read_view_pair()
                except Exception as e:
                    raise RuntimeError(f'post-export dogfood: read_view_pair() raised '
                                       f'after {n} pair(s): {e}')
                if o is None:
                    break
                n += 1
            if n < 1:
                raise RuntimeError('post-export dogfood: read 0 view pairs')
        finally:
            try:
                s.close()
            except Exception:
                pass

    def _purge(self):
        wd = self._workdir
        if wd and os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
        self._workdir = None


class _Cancelled(Exception):
    pass
