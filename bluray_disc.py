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


def parse_mpls(path):
    """Parse a .mpls PlayList. Returns (duration_seconds, [clip_names]) or (0.0, [])."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if data[0:4] != b"MPLS":
            return 0.0, []
        playlist_start = struct.unpack(">I", data[8:12])[0]
        p = playlist_start
        # PlayList(): length(4), reserved(2), number_of_PlayItems(2), number_of_SubPaths(2)
        num_items = struct.unpack(">H", data[p + 6:p + 8])[0]
        q = p + 10
        total_45k = 0
        clips = []
        for _ in range(num_items):
            item_len = struct.unpack(">H", data[q:q + 2])[0]
            body = data[q + 2:q + 2 + item_len]
            if len(body) >= 20:
                clip = body[0:5].decode("ascii", "replace")
                in_t = struct.unpack(">I", body[12:16])[0]
                out_t = struct.unpack(">I", body[16:20])[0]
                if out_t > in_t:
                    total_45k += (out_t - in_t)
                    clips.append(clip)
            q += 2 + item_len
        return total_45k / 45000.0, clips
    except Exception:
        return 0.0, []


def _safe_size(p):
    """os.path.getsize that never raises (a single unreadable clip must not abort detection)."""
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


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

    Main-title detection is duration-based (sum of PlayItem OUT-IN across each .mpls),
    which is robust against the decoy/obfuscated playlists many discs ship.
    """
    info = {"bdmv": None, "method": None, "duration_s": 0.0, "playlist": None,
            "clip": None, "kind": None, "candidates_ssif": 0, "candidates_m2ts": 0}
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
    if not ssif_set and not m2ts_set:
        return None, info  # not a Blu-ray STREAM layout

    # Parse every playlist once; consider the longest first.
    playlists = []
    if os.path.isdir(playlist_dir):
        for mpls in glob.glob(os.path.join(playlist_dir, "*.mpls")) + \
                    glob.glob(os.path.join(playlist_dir, "*.MPLS")):
            dur, clips = parse_mpls(mpls)
            if dur > 0 and clips:
                playlists.append((dur, clips, os.path.basename(mpls)))
    playlists.sort(key=lambda t: t[0], reverse=True)

    # Phase 1 (3D): longest playlist whose clip resolves to an SSIF.
    for dur, clips, name in playlists:
        for c in clips:
            if c in ssif_set:
                info.update(method="mpls", duration_s=dur, playlist=name, clip=c, kind="ssif")
                return ssif_set[c], info
    # Phase 2 (2D): longest playlist whose clip resolves to an M2TS.
    for dur, clips, name in playlists:
        for c in clips:
            if c in m2ts_set:
                info.update(method="mpls", duration_s=dur, playlist=name, clip=c, kind="m2ts")
                return m2ts_set[c], info

    # Phase 3: no usable playlist -> largest SSIF, else largest M2TS.
    if ssif_set:
        p = max(ssif_set.values(), key=_safe_size)
        info.update(method="largest", kind="ssif",
                    clip=os.path.splitext(os.path.basename(p))[0])
        return p, info
    if m2ts_set:
        p = max(m2ts_set.values(), key=_safe_size)
        info.update(method="largest", kind="m2ts",
                    clip=os.path.splitext(os.path.basename(p))[0])
        return p, info
    return None, info


def find_feature_3d_ssif(path):
    """Back-compat: locate ONLY a 3D SSIF feature. Returns (ssif_path, info) or (None, info)."""
    feat, info = find_feature(path)
    if feat and info.get("kind") == "ssif":
        return feat, info
    return None, info


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


def _wait_volume_ready(drive, timeout_s=8.0):
    """Poll until the mounted volume's BDMV/PLAYLIST is listable.

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
        _wait_volume_ready(drv)
    return drv


def dismount_iso(iso_path):
    """Dismount a previously-mounted ISO. Best-effort; returns True if the command ran."""
    if not iso_path:
        return False
    _run_ps(f"Dismount-DiskImage -ImagePath {_ps_quote(iso_path)} | Out-Null")
    return True
