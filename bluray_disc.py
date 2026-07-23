# -*- coding: utf-8 -*-
"""
Blu-ray 3D disc / folder support: locate the feature ("main title") 3D SSIF.

Point the player at a drive letter (J:\\), a BDMV folder, an index.bdmv, or any
folder containing a BDMV, and this finds the 3D feature film automatically using
the proper Blu-ray "main title" heuristic:

  1. Parse every BDMV/PLAYLIST/*.mpls and compute its play duration from the
     PlayItems (Sum of OUT-IN, 45 kHz clock) -- robust against the decoy/obfuscated
     playlists some discs ship (dozens of similar-sized files; only DURATION tells
     the real feature apart).
  2. Keep only the 3D playlists -- those whose clips have a STREAM/SSIF/<clip>.ssif.
  3. The longest one is the feature; return its main clip's SSIF.
  4. Fallback: if no playlist resolves, the largest SSIF on the disc.

Pure standard-library; safe to import anywhere.
"""
import os
import struct
import glob
import time
import subprocess

__all__ = ["is_bluray_path", "resolve_bdmv_root", "find_feature_3d_ssif", "find_feature",
           "parse_mpls", "is_iso", "mount_iso", "dismount_iso", "get_iso_mount_drive"]


def resolve_bdmv_root(path):
    """Return the BDMV directory for a drive root / BDMV folder / index.bdmv / parent, or None."""
    try:
        if not path:
            return None
        path = os.path.abspath(path)
        # index.bdmv / MovieObject.bdmv -> its directory is BDMV
        if os.path.isfile(path):
            base = os.path.basename(path).lower()
            if base in ("index.bdmv", "movieobject.bdmv"):
                d = os.path.dirname(path)
                return d if os.path.isdir(os.path.join(d, "PLAYLIST")) else None
            return None
        if not os.path.isdir(path):
            return None
        # the folder IS BDMV
        if os.path.basename(path).upper() == "BDMV" and os.path.isdir(os.path.join(path, "PLAYLIST")):
            return path
        # the folder CONTAINS BDMV (drive root or disc folder)
        cand = os.path.join(path, "BDMV")
        if os.path.isdir(os.path.join(cand, "PLAYLIST")):
            return cand
        # a STREAM / STREAM/SSIF / PLAYLIST subfolder -> walk up to BDMV
        d = path
        for _ in range(3):
            if os.path.basename(d).upper() == "BDMV" and os.path.isdir(os.path.join(d, "PLAYLIST")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    except Exception:
        pass
    return None


def is_bluray_path(path):
    """True if path points at (or into) a BDMV structure."""
    return resolve_bdmv_root(path) is not None


# A real feature — even a seamless-branched one — is built from DISTINCT segments, so
# its "replay ratio" (total runtime / runtime of the distinct segments) is ~1. An anti-rip
# "loop" decoy inflates its runtime by replaying the SAME (clip,in,out) segment many times
# (e.g. an 81 s clip x200 = a fake 4.5 h playlist), so its ratio is large. That is the
# signal used to discard decoys before ranking playlists by duration.
_DECOY_REPLAY_RATIO = 1.5


def _parse_mpls_full(path):
    """Parse a .mpls PlayList. Returns a dict with 'duration_s', the ordered 'segments'
    [(clip, in_45k, out_45k), ...] (repeats preserved) and 'clips' [names], or None."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if data[0:4] != b"MPLS":
            return None
        playlist_start = struct.unpack(">I", data[8:12])[0]
        p = playlist_start
        # PlayList(): length(4), reserved(2), number_of_PlayItems(2), number_of_SubPaths(2)
        num_items = struct.unpack(">H", data[p + 6:p + 8])[0]
        q = p + 10
        segments = []
        for _ in range(num_items):
            item_len = struct.unpack(">H", data[q:q + 2])[0]
            body = data[q + 2:q + 2 + item_len]
            if len(body) >= 20:
                clip = body[0:5].decode("ascii", "replace")
                in_t = struct.unpack(">I", body[12:16])[0]
                out_t = struct.unpack(">I", body[16:20])[0]
                if out_t > in_t:
                    segments.append((clip, in_t, out_t))
            q += 2 + item_len
        total_45k = sum(o - i for _, i, o in segments)
        return {"duration_s": total_45k / 45000.0,
                "segments": segments,
                "clips": [c for c, _, _ in segments]}
    except Exception:
        return None


def parse_mpls(path):
    """Parse a .mpls PlayList. Returns (duration_seconds, [clip_names]) or (0.0, [])."""
    r = _parse_mpls_full(path)
    return (r["duration_s"], r["clips"]) if r else (0.0, [])


def _is_decoy_playlist(segments):
    """True if a playlist looks like an anti-rip 'loop' decoy: its runtime is mostly
    REPEATED footage (the same (clip,in,out) segment replayed). Real features — including
    seamless-branched ones — are built from distinct segments, so they are never flagged."""
    if len(segments) <= 1:
        return False
    total = sum(o - i for _, i, o in segments)
    unique = sum(o - i for (_, i, o) in set(segments))
    return unique > 0 and total > unique * _DECOY_REPLAY_RATIO


def _safe_size(p):
    """os.path.getsize that never raises (a single unreadable clip must not abort detection)."""
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


# ============================================================================
# BD3D dual-file pairing (DF-1/DF-3): some MakeMKV-style backups (GITS S.A.C. Solid
# State Society 3D) store the base and MVC-dependent views as SEPARATE m2ts files
# and never materialize the interleaved .ssif (see ssif_interleave_missing above).
# On such discs the base<->dependent clip pairing lives in the MPLS ExtensionData
# block (ID1=2/ID2=1, STN_table_SS-shaped) -- DF-1 found it empirically on a real
# disc; there is no public spec for this vendor extension, so parsing here is
# deliberately bounded/defensive (every offset is range-checked against the actual
# file size; a malformed field aborts parsing instead of raising or over-reading).
# ============================================================================

_EXT_MAX_ENTRIES = 64      # sane cap on an untrusted "number of entries" field
_EXT_MAX_BLOCK = 4096      # sane cap on an untrusted "entry length" field


def _playlist_has_3d_extension(playlist_dir, name, cache=None):
    """True if the .mpls at playlist_dir/name carries an ExtensionData entry with
    ID1=2/ID2=1 (the STN_table_SS-shaped BD3D dependent-view pairing block). Only
    reads the small ExtensionData header + entry table (never the whole file);
    `cache` (a dict keyed by playlist name) avoids re-parsing the same file when
    called repeatedly from the tie-break scan below."""
    if cache is not None and name in cache:
        return cache[name]
    result = False
    try:
        path = os.path.join(playlist_dir, name)
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(20)
            if len(head) == 20 and head[0:4] == b"MPLS":
                ext_start = struct.unpack(">I", head[16:20])[0]
                if 0 < ext_start < size:
                    f.seek(ext_start)
                    hdr = f.read(12)
                    if len(hdr) == 12:
                        n_entries = struct.unpack(">I", hdr[8:12])[0] & 0xFFFF
                        n_entries = min(n_entries, _EXT_MAX_ENTRIES)
                        entries = f.read(n_entries * 12)
                        for k in range(0, len(entries) - 3, 12):
                            id1, id2 = struct.unpack(">HH", entries[k:k + 4])
                            if id1 == 2 and id2 == 1:
                                result = True
                                break
    except (OSError, struct.error):
        result = False
    if cache is not None:
        cache[name] = result
    return result


def _apply_3d_preference(group, playlist_dir, cache):
    """Re-rank playlists that are 'equivalent' (same resolved clip sequence, or
    duration within 1%) so one carrying a BD3D ExtensionData block goes first.

    Tie-breaker ONLY: a playlist that is not equivalent to another is never
    reordered relative to it, so this cannot promote a shorter/unrelated playlist
    over the real feature. The caller applies this separately to the non-decoy and
    decoy groups (never mixing them), so it also cannot resurrect a rejected decoy.

    Fixes GITS: 00001.mpls (no extension) and 00002.mpls (has it) parse to the
    exact same single PlayItem/duration (byte-identical PlayList section) -- with
    this, 00002.mpls wins so its base->dependent pairing becomes available.
    """
    if len(group) < 2:
        return group
    out = list(group)
    n = len(out)
    for i in range(n):
        dur_i, seg_i, name_i = out[i]
        if _playlist_has_3d_extension(playlist_dir, name_i, cache):
            continue  # already carries 3D data; nothing to prefer over it
        for j in range(i + 1, n):
            dur_j, seg_j, name_j = out[j]
            equivalent = (seg_j == seg_i) or (dur_i > 0 and abs(dur_j - dur_i) <= 0.01 * dur_i)
            if not equivalent:
                continue
            if _playlist_has_3d_extension(playlist_dir, name_j, cache):
                out[i], out[j] = out[j], out[i]
                break
    return out


def _parse_ss_dependent_clips(playlist_dir, name):
    """Parse the playlist's ExtensionData (ID1=2/ID2=1, STN_table_SS-shaped) and
    return the dependent-view clip ids it references, in PlayItem order (best
    effort; [] if the extension is absent, malformed, or doesn't parse).

    Byte layout (empirically confirmed against a real BD3D MakeMKV backup -- GITS
    S.A.C. Solid State Society 3D, DF-1, 2026-07-22 -- there is no public spec for
    this vendor extension):
        ExtensionData_start_address (u32 @ header offset 16) ->
          length(u32) data_block_start_address(u32) number_of_entries(u32)
          entries[n] { ID1(u16) ID2(u16) ext_data_start(u32) ext_data_length(u32) }
          ... data blocks, addressed relative to (ext_start + 4 + data_block_start) ...
    Within an ID1=2/ID2=1 block, each PlayItem's dependent clip is referenced by the
    same fixed anchor a normal PlayItem/SubPlayItem clip reference uses: 5 ASCII
    digit bytes (clip id) immediately followed by the literal 4-byte codec tag
    "M2TS". Scanning for that anchor (rather than trusting exact reserved-field
    offsets, unverified across other discs/authoring tools) is self-describing and
    bounds-safe against untrusted disc data.
    """
    try:
        path = os.path.join(playlist_dir, name)
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 20 or data[0:4] != b"MPLS":
            return []
        ext_start = struct.unpack(">I", data[16:20])[0]
        if not (0 < ext_start < len(data) - 12):
            return []
        data_block_start = struct.unpack(">I", data[ext_start + 4:ext_start + 8])[0]
        n_entries = struct.unpack(">I", data[ext_start + 8:ext_start + 12])[0] & 0xFFFF
        n_entries = min(n_entries, _EXT_MAX_ENTRIES)
        base = ext_start + 4 + data_block_start
        if not (0 <= base <= len(data)):
            return []
        off = ext_start + 12
        clips = []
        for _ in range(n_entries):
            if off + 12 > len(data):
                break
            id1, id2 = struct.unpack(">HH", data[off:off + 4])
            e_start, e_len = struct.unpack(">II", data[off + 4:off + 12])
            off += 12
            if id1 != 2 or id2 != 1:
                continue
            abs_off = base + e_start
            if not (0 <= abs_off <= len(data)):
                continue
            e_len = min(e_len, len(data) - abs_off, _EXT_MAX_BLOCK)
            block = data[abs_off:abs_off + e_len]
            # scan for every "<5 ascii digits>M2TS" anchor inside this block, in order
            k = 0
            while True:
                m = block.find(b"M2TS", k)
                if m < 0 or m < 5:
                    break
                cand = block[m - 5:m]
                if cand.isdigit():
                    clips.append(cand.decode("ascii"))
                k = m + 4
        return clips
    except (OSError, struct.error):
        return []


def _maybe_set_dual_file_pair(info, playlist_dir, m2ts_set):
    """When the disc's SSIF interleave is missing (MakeMKV-style base+dependent
    split), populate info['dual_file_pair'] = (abs base .m2ts, abs dependent .m2ts)
    so the player can route to the dual-file MVC demuxer instead of the 2D
    fallback. Only set when BOTH files are confirmed present on disk (size > 0) --
    a failed/partial lookup must fall through to the existing 2D path, never hang
    or crash.

    v1 scope (spec 2026-07-22 sec 4): single-segment features only -- uses the
    FIRST PlayItem's pairing. A multi-segment feature (>1 PlayItem) is logged, not
    handled, and still gets the first pair.
    """
    if not info.get("ssif_interleave_missing") or info.get("kind") != "m2ts":
        return
    base_clip = info.get("clip")
    if not base_clip:
        return
    playlist = info.get("playlist")
    dep_clip = None
    if playlist:
        dep_clips = _parse_ss_dependent_clips(playlist_dir, playlist)
        if dep_clips:
            dep_clip = dep_clips[0]
            num_segments = len(info.get("segments") or [])
            if num_segments > 1 or len(dep_clips) > 1:
                n = max(num_segments, len(dep_clips))
                print(f"[BD3D] dual-file multi-segment non gere ({n} segments)")
    if not dep_clip:
        # ExtensionData absent/unparseable -- fall back to the id+1 BD authoring
        # convention (only reached because ssif_interleave_missing is already set).
        try:
            dep_clip = "%05d" % (int(base_clip) + 1)
        except (TypeError, ValueError):
            return
        print(f"[BD3D] no ExtensionData pairing for playlist {playlist!r} -- "
              f"falling back to id+1 convention: {base_clip} -> {dep_clip}")
    base_path = m2ts_set.get(base_clip)
    dep_path = m2ts_set.get(dep_clip)
    if base_path and dep_path and _safe_size(dep_path) > 0:
        info["dual_file_pair"] = (os.path.abspath(base_path), os.path.abspath(dep_path))


def _index_clips(directory, exts):
    """Map clip-stem -> full path for files in `directory` whose extension is in `exts`
    (lowercase, leading dot). Case-insensitive on the extension."""
    out = {}
    try:
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                stem, ext = os.path.splitext(f)
                if ext.lower() in exts:
                    out[stem] = os.path.join(directory, f)
    except Exception:
        pass
    return out


def find_feature(path):
    """Locate the main feature ("main title") on a Blu-ray disc/folder.

    Prefers a 3D SSIF (stereoscopic); falls back to the 2D M2TS feature when the disc
    has no SSIF (a plain 2D Blu-ray). Returns (feature_path, info); info['kind'] is
    'ssif' (3D) or 'm2ts' (2D), or (None, info) when nothing playable is found.

    Main-title detection ranks playlists by duration (sum of PlayItem OUT-IN) but first
    discards anti-rip "loop" decoys (playlists whose runtime is mostly replayed footage),
    so the obfuscated/decoy playlists many discs ship don't win over the real feature.
    """
    info = {"bdmv": None, "method": None, "duration_s": 0.0, "playlist": None,
            "clip": None, "kind": None, "candidates_ssif": 0, "candidates_m2ts": 0,
            "candidates_playlists": 0, "decoys_filtered": 0, "feature_clips": [],
            "segments": [], "ssif_interleave_missing": False, "dual_file_pair": None}
    bdmv = resolve_bdmv_root(path)
    info["bdmv"] = bdmv
    if not bdmv:
        return None, info

    stream_dir = os.path.join(bdmv, "STREAM")
    ssif_dir = os.path.join(stream_dir, "SSIF")
    playlist_dir = os.path.join(bdmv, "PLAYLIST")

    ssif_set = _index_clips(ssif_dir, (".ssif",))
    m2ts_set = _index_clips(stream_dir, (".m2ts", ".mts"))
    info["candidates_ssif"] = len(ssif_set)
    info["candidates_m2ts"] = len(m2ts_set)

    # MakeMKV-style Blu-ray 3D backup: the disc IS authored 3D (STREAM/SSIF holds the
    # per-clip interleaving maps) but the interleaved .ssif files themselves were never
    # written — only their `.ssif.smap` sidecars. The whole MVC pipeline reads BOTH views
    # from ONE source (a real interleaved .ssif, or a single dual-PID m2ts); with the
    # interleave gone the base and dependent views live in SEPARATE m2ts files it cannot
    # pair, so feeding the base m2ts to the MVC demuxer makes it buffer base frames forever
    # (isDualPID heuristic) and emit nothing — a black-screen hang. Flag the case so the
    # player can present the base view in reliable 2D instead. Signature is exact: a plain
    # 2D Blu-ray has no STREAM/SSIF folder at all, and a real 3D disc (e.g. the Avatar
    # reference) ships the actual .ssif, so this never fires for either.
    info["ssif_interleave_missing"] = bool(
        not ssif_set and _index_clips(ssif_dir, (".smap",))
    )

    if not ssif_set and not m2ts_set:
        return None, info  # not a Blu-ray STREAM layout

    # Parse every playlist once. We rank by duration but FIRST drop anti-rip "loop"
    # decoys (playlists whose runtime is mostly replayed footage) — many discs ship a
    # giant decoy (e.g. an 81 s clip looped x200 = a fake 4.5 h playlist) that a naive
    # "longest playlist" pick mistakes for the feature. Non-decoys are tried first
    # (longest-first); decoys are kept only as a last resort so detection never fails.
    parsed = []  # (duration_s, segments, name)
    seen_names = set()
    if os.path.isdir(playlist_dir):
        for mpls in glob.glob(os.path.join(playlist_dir, "*.mpls")) + \
                    glob.glob(os.path.join(playlist_dir, "*.MPLS")):
            key = os.path.basename(mpls).lower()
            if key in seen_names:
                continue  # case-insensitive FS: *.mpls and *.MPLS match the same file
            seen_names.add(key)
            r = _parse_mpls_full(mpls)
            if r and r["duration_s"] > 0 and r["segments"]:
                parsed.append((r["duration_s"], r["segments"], os.path.basename(mpls)))
    parsed.sort(key=lambda t: t[0], reverse=True)

    real = [pl for pl in parsed if not _is_decoy_playlist(pl[1])]
    decoys = [pl for pl in parsed if _is_decoy_playlist(pl[1])]
    info["candidates_playlists"] = len(parsed)
    info["decoys_filtered"] = len(decoys)
    # 3D-extension tie-break (DF-3): applied separately to each group so it can
    # never cross the decoy/non-decoy boundary (see _apply_3d_preference docstring).
    _ext_cache = {}
    real = _apply_3d_preference(real, playlist_dir, _ext_cache)
    decoys = _apply_3d_preference(decoys, playlist_dir, _ext_cache)
    ranked = real + decoys  # prefer non-decoys; both groups stay longest-first

    def _resolvable_sequence(segments, clip_set):
        """The feature's playable segment sequence: the playlist's PlayItems (IN ORDER)
        whose clip resolves in clip_set, as [{'clip','path','m2ts','duration_s'}, ...].
        This is what multi-segment (seamless-branching) playback walks; the base .m2ts
        carries the audio (for an mpv EDL), the resolved path carries the video. Empty
        if nothing resolves."""
        seq = []
        for clip, in_t, out_t in segments:
            if clip in clip_set:
                seq.append({"clip": clip, "path": clip_set[clip],
                            "m2ts": m2ts_set.get(clip),
                            "duration_s": (out_t - in_t) / 45000.0})
        return seq

    # Phase 1 (3D): longest non-decoy playlist whose clips resolve to SSIF.
    for dur, segments, name in ranked:
        seq = _resolvable_sequence(segments, ssif_set)
        if seq:
            info.update(method="mpls", duration_s=dur, playlist=name, clip=seq[0]["clip"],
                        kind="ssif", feature_clips=[s["clip"] for s in seq], segments=seq)
            return seq[0]["path"], info
    # Phase 2 (2D): longest non-decoy playlist whose clips resolve to M2TS.
    for dur, segments, name in ranked:
        seq = _resolvable_sequence(segments, m2ts_set)
        if seq:
            info.update(method="mpls", duration_s=dur, playlist=name, clip=seq[0]["clip"],
                        kind="m2ts", feature_clips=[s["clip"] for s in seq], segments=seq)
            _maybe_set_dual_file_pair(info, playlist_dir, m2ts_set)
            return seq[0]["path"], info

    # Phase 3: no usable playlist -> largest SSIF, else largest M2TS (single segment).
    if ssif_set:
        p = max(ssif_set.values(), key=_safe_size)
        stem = os.path.splitext(os.path.basename(p))[0]
        info.update(method="largest", kind="ssif", clip=stem, feature_clips=[stem],
                    segments=[{"clip": stem, "path": p, "m2ts": m2ts_set.get(stem),
                               "duration_s": 0.0}])
        return p, info
    if m2ts_set:
        p = max(m2ts_set.values(), key=_safe_size)
        stem = os.path.splitext(os.path.basename(p))[0]
        info.update(method="largest", kind="m2ts", clip=stem, feature_clips=[stem],
                    segments=[{"clip": stem, "path": p, "m2ts": p, "duration_s": 0.0}])
        _maybe_set_dual_file_pair(info, playlist_dir, m2ts_set)
        return p, info
    return None, info


def find_feature_3d_ssif(path):
    """Back-compat: locate ONLY a 3D SSIF feature. Returns (ssif_path, info) or (None, info)."""
    feat, info = find_feature(path)
    if feat and info.get("kind") == "ssif":
        return feat, info
    return None, info


def build_feature_edl(segments):
    """Build an mpv `edl://` URI concatenating the base .m2ts of each feature segment into
    ONE continuous, SEEKABLE timeline (the whole film's audio + master clock), while
    edge264 renders the matching .ssif sequence. Returns "" for a single segment (no EDL
    needed) — the caller then plays the clip directly.

    Uses the edl:// PROTOCOL, not a `.edl` file: the bundled mpv does not recognise the
    `.edl` extension ("Failed to recognize file format"), but the protocol works. Each
    segment is length-prefixed (`%<bytes>%<path>`, safely quoting any path) with an
    explicit `length=` so the total duration is known up front and seeking is reliable.
    Validated against the bundled mpv: duration = sum of segments, absolute seek lands
    across segment boundaries."""
    segs = [s for s in (segments or []) if s.get("m2ts")]
    if len(segs) < 2:
        return ""  # single segment (or none): no EDL needed
    parts = []
    for s in segs:
        p = s["m2ts"].replace("\\", "/")          # mpv accepts forward slashes on Windows
        seg = f"%{len(p.encode('utf-8'))}%{p}"    # %<bytelen>% quotes paths with any char
        dur = float(s.get("duration_s") or 0.0)
        if dur > 0:
            seg += f",length={dur:.3f}"
        parts.append(seg)
    return "edl://" + ";".join(parts)


# ============================================================================
# Blu-ray ISO support — mount the disc image (no admin needed for an ISO, same
# mechanism as Explorer's "Mount" verb) so the BDMV detection above works on the
# resulting drive letter. The feature then streams straight off the mounted
# volume (no extraction, no extra disk space).
# ============================================================================
_PS_NOWINDOW = 0x08000000  # CREATE_NO_WINDOW — keep the GUI free of console flashes


def _ps_quote(s):
    """Quote a string as a PowerShell single-quoted literal (handles spaces/quotes)."""
    return "'" + str(s).replace("'", "''") + "'"


def _run_ps(command):
    """Run a PowerShell command; return stripped stdout on success, else None."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=120, creationflags=_PS_NOWINDOW,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip()
    except Exception:
        pass
    return None


def is_iso(path):
    """True if path is an existing .iso file."""
    return bool(path) and os.path.isfile(path) and path.lower().endswith(".iso")


def _drive_from_letter(letter):
    letter = (letter or "").strip().rstrip(":")
    if len(letter) == 1 and letter.isalpha():
        return letter.upper() + ":\\"
    return None


def get_iso_mount_drive(iso_path):
    """If the ISO is already mounted, return its drive root (e.g. 'E:\\'), else None."""
    out = _run_ps(
        "$ErrorActionPreference='SilentlyContinue';"
        f"$d = Get-DiskImage -ImagePath {_ps_quote(iso_path)};"
        "if ($d -and $d.Attached) { ($d | Get-Volume).DriveLetter }"
    )
    if out:
        return _drive_from_letter(out.splitlines()[-1])
    return None


def _wait_volume_ready(drive, timeout_s=20.0):
    """Poll until the mounted volume's BDMV/PLAYLIST is listable.

    Timeout raised 8s -> 20s: a large UDF image was MEASURED taking 8.2s to
    become browsable after mount churn — the old deadline lost that race by a
    hair, the feature scan saw an empty drive and the ISO was dismounted as
    featureless. The poll returns as soon as the volume is ready, so the
    higher ceiling costs nothing on the happy path.

    A UDF volume's drive letter appears a moment before the filesystem is actually
    ready for I/O; querying too early yields WinError 87 (invalid parameter) or empty
    listings, which would silently drop us to the size-based fallback. Returns True
    once ready, False on timeout (caller proceeds regardless; detection is defensive).
    """
    deadline = time.monotonic() + timeout_s
    pl = os.path.join(drive, "BDMV", "PLAYLIST")
    while time.monotonic() < deadline:
        try:
            if os.path.isdir(pl) and os.listdir(pl):
                return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


def mount_iso(iso_path):
    """Mount an ISO (no admin needed) and return its drive root 'E:\\', or None on failure.

    Reuses the existing mount if the image is already attached, and waits for the
    volume to become I/O-ready before returning so detection doesn't race the mount.
    """
    if not is_iso(iso_path):
        return None
    drv = get_iso_mount_drive(iso_path)
    if not drv:
        out = _run_ps(
            "$ErrorActionPreference='Stop';"
            f"$img = Mount-DiskImage -ImagePath {_ps_quote(iso_path)} -PassThru;"
            "Start-Sleep -Milliseconds 500;"
            "($img | Get-Volume).DriveLetter"
        )
        if out:
            drv = _drive_from_letter(out.splitlines()[-1])
        if not drv:
            # The volume may not be ready in time — re-query once.
            drv = get_iso_mount_drive(iso_path)
    if drv:
        if not _wait_volume_ready(drv):
            print(f"[bluray_disc] WARNING: {drv} still not browsable after wait — "
                  "feature detection may scan an empty drive")
    return drv


def dismount_iso(iso_path):
    """Dismount a previously-mounted ISO. Best-effort; returns True if the command ran."""
    if not iso_path:
        return False
    _run_ps(f"Dismount-DiskImage -ImagePath {_ps_quote(iso_path)} | Out-Null")
    return True
