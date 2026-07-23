# -*- coding: utf-8 -*-
r"""Task EX-3 -- the home-made `vexu` box injector.

Writes the Apple-spatial `vexu` (Video Extended Usage) box into an already-
muxed MV-HEVC `.mov`/`.mp4`, matching EX-1's byte-per-byte reference dump of
the real sample (`docs\hevc_4k24P_main_multiview_1.mp4`, vexu @ absolute
offset 824, size 78) EXACTLY:

    vexu (78B)
      must (16B)              FullBox v0/f0, requires=['eyes']
      eyes (54B)
        must (20B)             FullBox v0/f0, requires=['stri','hero']
        stri (13B) = 0x03      FullBox v0/f0, 1 data byte (L+R views present)
        hero (13B) = 0x01      FullBox v0/f0, 1 data byte (left eye is hero)

NO cams/blin/hfov/proj -- the spec's amended v1 decision (post-EX-1) is to
reproduce this MINIMAL, PROVEN structure rather than speculate boxes that
have no reference-validated home (spec Sec.3 "vexu maison -- AMENDE").

## Parent box (determined from EX-1's dump)

The vexu box in the reference sample lives at absolute offset 824, which
EX-1's dump situates as a DIRECT CHILD of the `hvc1` VisualSampleEntry
(hvc1 @429 -> hvcC @515 (230B) -> lhvC @745 (79B) -> vexu @824 (78B), all
inside `moov/trak/mdia/minf/stbl/stsd`). It is NOT trak-level (no `udta`
involvement) and NOT inside `hvcC`/`lhvC` -- it is a sibling of those two
boxes, appended after them in the same VisualSampleEntry. `mvhevc_exporter`'s
own `probe_mv_hevc_container` already treats vexu exactly this way (a hvc1
child alongside hvcC/lhvC), which this module's own chain-walker confirms
independently. Injection therefore appends the 78-byte vexu box as the LAST
child of the trak's `hvc1` sample entry (box order among sample-entry
children is not semantically significant per ISO/IEC 14496-12; the reference
happens to have no other children after lhvC, so "last child" and "right
after lhvC" coincide for that file -- our own MP4Box outputs may additionally
carry a `pasp` box, which we leave untouched, vexu simply goes after it).

## Injection strategy (empirically determined, see ex-task-3-report.md)

`moov` is small (a few KB) -- read it whole, splice + patch it in memory,
then rewrite the file via a temp-then-atomic-move (`os.replace`, same
directory as the target so the move is atomic on the same volume):

    [0 .. moov_start)      copied verbatim, byte-for-byte, unchanged
    [moov_start .. moov_end)  REPLACED by the patched+grown moov bytes
    [moov_end .. EOF)       copied verbatim, byte-for-byte, unchanged
                            (this is the ONLY part of the file whose
                            ABSOLUTE FILE POSITION shifts, by +len(vexu);
                            its BYTES are never touched)

Growing `moov` in place means every byte physically located at or after the
ORIGINAL moov's end offset moves forward by `len(vexu)` once rewritten. Any
`stco`/`co64` chunk-offset table entry (an ABSOLUTE file byte offset,
normally pointing into `mdat`) that lives at or after that boundary is
therefore stale and rebased by +delta; an entry that lives BEFORE it (a
foreign "mdat-before-moov" layout) needs no change at all -- both cases fall
out of one comparison (`offset >= original_moov_end`), so the code does not
need to special-case which layout it is looking at.

**Empirically determined (script dump, ex-task-3-report.md): our own
pipeline's MP4Box output (`mvhevc_exporter.py`'s Tier-2/Tier-3 `_mux`) is
ALWAYS `ftyp, moov, mdat, [free]` -- moov BEFORE mdat, single non-fragmented
`mdat`, 32-bit `stco` (not `co64`) chunk offsets, across every trak
(video+audio).** That is the case this module fully supports and the one
that matters for our pipeline: the stco/co64 rebase above always fires for
real on these outputs (verified, not just theoretically correct).

The reference sample and a verbatim Tier-1 copy of it, by contrast, are
FRAGMENTED MP4 (`moov` + repeated `moof`/`mdat` pairs, sample locations
described by `trun`/`tfhd`, not `stco`/`co64`). This module does not attempt
to patch fragmented-MP4 sample addressing (`tfhd` base-data-offset, `trun`
data-offsets) -- injecting into a fragmented file whose sample entry does
NOT already carry vexu is refused with a clear error (see `_ensure_not_
fragmented`). In practice this never bites our own pipeline (Tier 1 only
ever COPIES the reference, which already carries the reference vexu, so
idempotence -- not injection -- is what Tier-1 outputs actually exercise);
it is a documented limitation for a hypothetical foreign fragmented source
that lacks vexu.

## Idempotence (chosen + documented, per the brief)

CHOSEN: **no-op**. `inject_vexu` checks `has_vexu` first (via the existing
`probe_mv_hevc_container` box-probe -- reused, not reimplemented) and returns
immediately, untouched, if a vexu box is already present. Since this
injector only ever writes ONE fixed, reference-minimal payload
(stri=0x03/hero=0x01 -- there is no source-dependent data to reconcile), a
"replace" policy would be strictly more code for zero observable behavioural
difference in any file this injector itself would ever produce or accept;
no-op is simplest and is what Tier-1's "copy of an already-conformant
source" case needs (EX-2b finding #4).

## Removal (`remove_vexu`, added post-interruption)

The exact inverse of the insertion above (same stco/co64 rebase + ancestor-
size patch, delta sign flipped, splice removes bytes instead of adding
them). Unlike `inject_vexu`, this is NOT idempotent -- it REFUSES if no
vexu box is present (removing something that isn't there is a caller bug,
not a valid steady state). Added to let test fixtures normalize themselves
back to a known vexu-less state regardless of what an earlier run may have
left on disk, rather than assuming a checkpointed .mov has never been
touched by `inject_vexu`.
"""
import io
import logging
import os
import struct
import tempfile

from mvhevc_exporter import (
    _iter_boxes, _descend_boxes, _VSE_SKIP, probe_mv_hevc_container,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed reference-minimal vexu parameters (spec Sec.3 amendment; EX-1 dump).
# ---------------------------------------------------------------------------
STRI_VALUE = 0x03   # bit0 has_left_eye_view | bit1 has_right_eye_view
HERO_VALUE = 0x01   # hero_eye = left


# ---------------------------------------------------------------------------
# vexu box construction (byte-exact vs. the EX-1 reference dump)
# ---------------------------------------------------------------------------
def _box(fourcc, payload):
    return struct.pack('>I', 8 + len(payload)) + fourcc + payload


def _fullbox(fourcc, payload, version_flags=b'\x00\x00\x00\x00'):
    return _box(fourcc, version_flags + payload)


def _build_vexu_bytes(stri=STRI_VALUE, hero=HERO_VALUE):
    """Build the vexu box byte-for-byte: vexu{must[eyes], eyes{must[stri,hero],
    stri, hero}} -- reference-minimal, no cams/blin/hfov."""
    stri_box = _fullbox(b'stri', bytes([stri]))
    hero_box = _fullbox(b'hero', bytes([hero]))
    eyes_must = _fullbox(b'must', b'stri' + b'hero')
    eyes_box = _box(b'eyes', eyes_must + stri_box + hero_box)
    vexu_must = _fullbox(b'must', b'eyes')
    return _box(b'vexu', vexu_must + eyes_box)


_VEXU_BYTES = _build_vexu_bytes()   # fixed, built once (78 bytes, matches EX-1 dump)


# ---------------------------------------------------------------------------
# box-tree navigation -- REUSES mvhevc_exporter's _iter_boxes/_descend_boxes
# (EX-2b's box-probe walker) rather than writing a third walker (DRY). Only
# the bits that walker doesn't need (returning the full ancestor CHAIN with
# absolute offsets, not just a verdict) are added here, on top of it.
# ---------------------------------------------------------------------------
def _locate_hvc1_chain(f, fsize):
    """Return [moov, trak, mdia, minf, stbl, stsd, hvc1-or-hev1] as
    (fourcc, off, size, header_len, box_end) tuples in ABSOLUTE file
    coordinates (the same box-header 5-tuple `_iter_boxes` yields), for the
    first trak whose stsd carries a hvc1/hev1 sample entry -- or None."""
    moov = None
    for typ, off, size, header, box_end in _iter_boxes(f, 0, fsize):
        if typ == 'moov':
            moov = (typ, off, size, header, box_end)
            break
    if moov is None:
        return None
    _typ, moff, msize, mheader, mend = moov
    moov_start = moff + mheader
    for typ, toff, tsize, theader, tbox_end in _iter_boxes(f, moov_start, mend):
        if typ != 'trak':
            continue
        trak_box = (typ, toff, tsize, theader, tbox_end)
        chain = []
        region = _descend_boxes(f, toff + theader, tbox_end,
                                 ['mdia', 'minf', 'stbl', 'stsd'], chain=chain)
        if region is None:
            continue
        s_start, s_end = region
        entry = None
        for etyp, eoff, esize, eheader, ebox_end in _iter_boxes(f, s_start, s_end):
            if etyp in ('hvc1', 'hev1'):
                entry = (etyp, eoff, esize, eheader, ebox_end)
                break
        if entry is None:
            continue
        return [moov, trak_box] + chain + [entry]
    return None


def _find_vexu_range(f, hvc1_box):
    """Return (offset, size) of the vexu box among hvc1_box's children (the
    5-tuple format `_locate_hvc1_chain` returns for its last entry), or None
    if absent. Shared by read_vexu and remove_vexu (DRY)."""
    _etyp, hoff, hsize, hheader, hend = hvc1_box
    child_start = hoff + hheader + _VSE_SKIP
    for typ, off, size, header, box_end in _iter_boxes(f, child_start, hend):
        if typ == 'vexu':
            return (off, size)
    return None


def _has_moof(path, fsize):
    """Cheap top-level-only scan: does this file carry fragmented-MP4 `moof`
    boxes (sample addressing via trun/tfhd, not stco/co64)? Header-only."""
    with open(path, 'rb') as f:
        for typ, off, size, header, box_end in _iter_boxes(f, 0, fsize):
            if typ == 'moof':
                return True
    return False


# ---------------------------------------------------------------------------
# in-memory moov surgery
# ---------------------------------------------------------------------------
def _patch_box_size(buf, rel_off, old_size, header_len, delta):
    new_size = old_size + delta
    if header_len == 8:
        struct.pack_into('>I', buf, rel_off, new_size)
    else:
        struct.pack_into('>Q', buf, rel_off + 8, new_size)


_MOOV_RECURSE = {'moov', 'trak', 'mdia', 'minf', 'stbl', 'edts', 'dinf', 'udta'}


def _patch_offset_table(buf, off, header, typ, moov_end_abs, delta):
    payload = off + header
    count = struct.unpack('>I', buf[payload + 4:payload + 8])[0]
    entry_size = 4 if typ == 'stco' else 8
    fmt = '>I' if typ == 'stco' else '>Q'
    p = payload + 8
    for _ in range(count):
        val = struct.unpack(fmt, buf[p:p + entry_size])[0]
        if val >= moov_end_abs:
            struct.pack_into(fmt, buf, p, val + delta)
        p += entry_size


def _rebase_chunk_offsets(moov_buf, moov_end_abs, delta):
    """Walk EVERY trak inside moov_buf and rebase any stco/co64 entry that
    points at or after the file's original moov end (see module docstring:
    moov-before-mdat is our own pipeline's actual layout, verified) -- an
    entry pointing before that boundary (a foreign mdat-before-moov layout)
    is left untouched, which falls out of the same comparison."""
    def _walk(start, end):
        bio = io.BytesIO(moov_buf)
        for typ, off, size, header, box_end in _iter_boxes(bio, start, end):
            if typ in ('stco', 'co64'):
                _patch_offset_table(moov_buf, off, header, typ, moov_end_abs, delta)
            elif typ in _MOOV_RECURSE:
                _walk(off + header, box_end)
    _walk(0, len(moov_buf))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def has_vexu(mov_path):
    """True iff the file's (first) hvc1/hev1 sample entry already carries a
    vexu box. Reuses mvhevc_exporter.probe_mv_hevc_container verbatim."""
    return bool(probe_mv_hevc_container(mov_path).get('has_vexu'))


def read_vexu(mov_path):
    """Parse the vexu box's stri/hero values. Raises RuntimeError if no vexu
    box (or no hvc1/hev1 sample entry at all) is present."""
    with open(mov_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        fsize = f.tell()
        chain = _locate_hvc1_chain(f, fsize)
        if chain is None:
            raise RuntimeError(f'read_vexu: no hvc1/hev1 sample entry found in {mov_path!r}')
        vexu_range = _find_vexu_range(f, chain[-1])
        if vexu_range is None:
            raise RuntimeError(f'read_vexu: no vexu box present in {mov_path!r}')
        voff, vsize = vexu_range
        f.seek(voff)
        raw = f.read(vsize)
    return _parse_vexu(raw)


def _parse_vexu(raw):
    """Parse a vexu box's raw bytes (its own 8-byte header included) into
    {'stri': int, 'hero': int} by walking must/eyes/stri/hero."""
    result = {}
    bio = io.BytesIO(raw)
    for typ, off, size, header, box_end in _iter_boxes(bio, 8, len(raw)):
        if typ == 'eyes':
            for etyp, eoff, esize, eheader, ebox_end in _iter_boxes(bio, off + header, box_end):
                if etyp == 'stri':
                    result['stri'] = raw[eoff + eheader + 4]   # FullBox: skip v/flags(4)
                elif etyp == 'hero':
                    result['hero'] = raw[eoff + eheader + 4]
    if 'stri' not in result or 'hero' not in result:
        raise RuntimeError('read_vexu: could not parse stri/hero out of the vexu box')
    return result


def inject_vexu(mov_path):
    """Inject the fixed reference-minimal vexu box (stri=0x03, hero=0x01)
    into `mov_path`'s video trak's hvc1 sample entry, in place.

    Raises RuntimeError (clean, descriptive message) if:
      - no moov / no hvc1|hev1 sample entry can be located;
      - the source has no lhvC (2D / non-MV-HEVC file -- refused per spec);
      - the sample entry isn't `hvc1` (vexu's proven home per EX-1 -- a bare
        `hev1` file, which never has lhvC anyway in this pipeline, is
        already caught by the check above);
      - the file is fragmented MP4 (`moof` present) and lacks vexu -- sample
        addressing there is not stco/co64-based and this injector does not
        patch trun/tfhd (documented limitation for foreign files).

    Idempotent: a no-op (file left byte-identical) if vexu is already
    present -- see module docstring "Idempotence".
    """
    probe = probe_mv_hevc_container(mov_path)
    if not probe['moov_found']:
        raise RuntimeError(f'vexu injection: no moov box found in {mov_path!r}')
    if not probe['has_lhvC']:
        raise RuntimeError(
            f'vexu injection refused: {mov_path!r} has no lhvC box -- not an '
            f'MV-HEVC file (2D / single-view source)')
    if probe['has_vexu']:
        logger.info('[VEXU] %s already carries a vexu box -- idempotent no-op', mov_path)
        return

    fsize = os.path.getsize(mov_path)
    if _has_moof(mov_path, fsize):
        raise RuntimeError(
            f'vexu injection refused: {mov_path!r} is a fragmented MP4 (moof '
            f'present) without a vexu box -- sample addressing there is '
            f'trun/tfhd-based, not stco/co64, which this injector does not '
            f'patch (documented limitation for foreign files; our own '
            f'pipeline never produces fragmented output)')

    with open(mov_path, 'rb') as f:
        chain = _locate_hvc1_chain(f, fsize)
    if chain is None:
        raise RuntimeError(f'vexu injection: could not locate hvc1 sample entry in {mov_path!r}')
    if chain[-1][0] != 'hvc1':
        raise RuntimeError(
            f'vexu injection: sample entry is {chain[-1][0]!r}, only hvc1 is '
            f'supported (EX-1 reference: vexu lives in hvc1)')

    moov_box, hvc1_box = chain[0], chain[-1]
    _mtyp, moov_off, moov_size, _mheader, moov_end = moov_box
    _htyp, _hoff, _hsize, _hheader, hvc1_end = hvc1_box

    delta = len(_VEXU_BYTES)

    with open(mov_path, 'rb') as f:
        f.seek(moov_off)
        moov_buf = bytearray(f.read(moov_size))

    # 1) rebase stco/co64 (any absolute offset located at/after the ORIGINAL
    #    moov's end will move forward by `delta` once moov grows in place).
    _rebase_chunk_offsets(moov_buf, moov_end, delta)

    # 2) grow every ancestor box's size field by delta (all are strictly
    #    BEFORE the insertion point, so this is safe before step 3 as well).
    for (typ, off, size, header, box_end) in chain:
        _patch_box_size(moov_buf, off - moov_off, size, header, delta)

    # 3) splice the vexu bytes in as hvc1's last child (length-changing --
    #    done last; insertion point uses the ORIGINAL hvc1_end, still valid
    #    since steps 1/2 never change moov_buf's length).
    insert_rel = hvc1_end - moov_off
    moov_buf[insert_rel:insert_rel] = _VEXU_BYTES

    # temp-then-atomic-move rewrite: [0, moov_off) + patched moov_buf +
    # [moov_end, EOF), all verbatim except moov_buf itself.
    target_dir = os.path.dirname(os.path.abspath(mov_path)) or '.'
    fd, tmp_path = tempfile.mkstemp(prefix='.vexu_tmp_', suffix='.mov', dir=target_dir)
    try:
        with os.fdopen(fd, 'wb') as fout, open(mov_path, 'rb') as fin:
            fout.write(fin.read(moov_off))
            fin.seek(moov_end)
            fout.write(bytes(moov_buf))
            _copy_rest(fin, fout)
        os.replace(tmp_path, mov_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    post = probe_mv_hevc_container(mov_path)
    if not post['has_vexu']:
        raise RuntimeError(
            f'vexu injection: post-write verification failed -- vexu not '
            f'found in {mov_path!r} after rewrite (internal error)')
    logger.info('[VEXU] injected stri=0x%02x hero=0x%02x into %s', STRI_VALUE, HERO_VALUE, mov_path)


def remove_vexu(mov_path):
    """Inverse surgery of `inject_vexu`: locate and splice the vexu box OUT
    of `mov_path`'s hvc1 sample entry, in place -- shrinking ancestor box
    sizes and rebasing stco/co64 chunk offsets by -len(vexu), the exact
    mirror image of inject_vexu's insertion (same temp-then-atomic-move
    rewrite; see module docstring "Injection strategy", which applies
    symmetrically here with the sign of delta flipped and the splice
    removing bytes instead of adding them).

    Raises RuntimeError (clean, descriptive message) if:
      - no moov / no hvc1|hev1 sample entry can be located;
      - the sample entry has no vexu box to remove -- refused, NOT an
        idempotent no-op like inject_vexu: removing something that was
        never there is a caller error (e.g. a fixture-normalization bug),
        not a valid steady state this function should silently accept;
      - the sample entry isn't `hvc1`;
      - the file is fragmented MP4 (`moof` present) -- sample addressing
        there is not stco/co64-based (same documented limitation as
        inject_vexu -- this module does not patch trun/tfhd).

    Added post-interruption (stabilizing EX-3's test fixtures): lets a
    caller (notably test_vexu.py) normalize a checkpointed .mov fixture back
    to a known vexu-less state before exercising inject_vexu's False->True
    precondition, instead of depending on upstream fixture/checkpoint
    history never having already run inject_vexu on it.
    """
    probe = probe_mv_hevc_container(mov_path)
    if not probe['moov_found']:
        raise RuntimeError(f'vexu removal: no moov box found in {mov_path!r}')
    if not probe['has_vexu']:
        raise RuntimeError(
            f'vexu removal refused: {mov_path!r} has no vexu box to remove')

    fsize = os.path.getsize(mov_path)
    if _has_moof(mov_path, fsize):
        raise RuntimeError(
            f'vexu removal refused: {mov_path!r} is a fragmented MP4 (moof '
            f'present) -- sample addressing there is trun/tfhd-based, not '
            f'stco/co64, which this module does not patch (same documented '
            f'limitation as inject_vexu; our own pipeline never produces '
            f'fragmented output)')

    with open(mov_path, 'rb') as f:
        chain = _locate_hvc1_chain(f, fsize)
        if chain is None:
            raise RuntimeError(f'vexu removal: could not locate hvc1 sample entry in {mov_path!r}')
        if chain[-1][0] != 'hvc1':
            raise RuntimeError(
                f'vexu removal: sample entry is {chain[-1][0]!r}, only hvc1 is '
                f'supported (EX-1 reference: vexu lives in hvc1)')
        vexu_range = _find_vexu_range(f, chain[-1])
        if vexu_range is None:
            raise RuntimeError(
                f'vexu removal: no vexu box found under hvc1 in {mov_path!r} '
                f'(internal error -- probe said has_vexu=True)')

    moov_box = chain[0]
    _mtyp, moov_off, moov_size, _mheader, moov_end = moov_box
    voff, vsize = vexu_range

    with open(mov_path, 'rb') as f:
        f.seek(moov_off)
        moov_buf = bytearray(f.read(moov_size))

    delta = -vsize

    # 1) rebase stco/co64 (uses the CURRENT, pre-removal moov_end as the
    #    boundary -- same convention/comparison as inject_vexu; delta just
    #    carries the opposite sign, so entries shift backward).
    _rebase_chunk_offsets(moov_buf, moov_end, delta)

    # 2) shrink every ancestor box's size field by vsize (all strictly
    #    BEFORE the splice point, safe before step 3 as well).
    for (typ, off, size, header, box_end) in chain:
        _patch_box_size(moov_buf, off - moov_off, size, header, delta)

    # 3) splice the vexu bytes OUT (length-changing -- done last, mirroring
    #    inject_vexu's insertion-last ordering).
    rel_off = voff - moov_off
    del moov_buf[rel_off:rel_off + vsize]

    # temp-then-atomic-move rewrite: [0, moov_off) + patched moov_buf +
    # [moov_end, EOF), all verbatim except moov_buf itself (moov_end here is
    # still the ORIGINAL, pre-removal absolute end -- everything from there
    # on is copied unchanged and simply lands `vsize` bytes earlier).
    target_dir = os.path.dirname(os.path.abspath(mov_path)) or '.'
    fd, tmp_path = tempfile.mkstemp(prefix='.vexu_tmp_', suffix='.mov', dir=target_dir)
    try:
        with os.fdopen(fd, 'wb') as fout, open(mov_path, 'rb') as fin:
            fout.write(fin.read(moov_off))
            fin.seek(moov_end)
            fout.write(bytes(moov_buf))
            _copy_rest(fin, fout)
        os.replace(tmp_path, mov_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    post = probe_mv_hevc_container(mov_path)
    if post['has_vexu']:
        raise RuntimeError(
            f'vexu removal: post-write verification failed -- vexu still '
            f'present in {mov_path!r} after rewrite (internal error)')
    logger.info('[VEXU] removed vexu box from %s', mov_path)


def _copy_rest(fin, fout, chunk=4 * 1024 * 1024):
    while True:
        buf = fin.read(chunk)
        if not buf:
            break
        fout.write(buf)
