"""
BD3D offset metadata - the authored depth of Blu-ray 3D graphics planes.

On BD3D the PG (subtitle) plane is flat; its stereoscopic depth is carried as
MVC offset metadata: an SEI in the dependent view (payloadType 37 wrapping a
user_data_unregistered whose payload starts with the ASCII magic "OFMD") gives,
per GOP, per offset sequence, a per-frame pixel shift. The playlist's
STN_table_SS extension maps each PG stream (PID) to its offset_sequence_id.

Structure reverse-engineered and validated on a real disc (Avatar BD3D,
52 GOPs, cross-checked 3 ways against the MPLS):

  OFMD user_data:  [0:4]="OFMD"  [10]&0x7F=n_sequences  [11]=frames_in_gop
                   [14:14+n*frames] sequence-major bytes:
                   bit7 = direction (0=toward viewer, 1=behind), bits6:0 = px

  STN_table_SS (MPLS ExtensionData type=2 ver=1): after the dependent-video
  stream entry + attributes, per PG stream 2 bytes: [offset_sequence_id][flags].

Convention out of this module: DISPARITY normalized to eye width (1920), > 0 =
crossed = in front of the screen. The BD offset is a PER-EYE shift, so
disparity = 2 * signed_px / 1920.
"""

import logging
import struct

logger = logging.getLogger(__name__)

EYE_WIDTH_PX = 1920.0


def ofmd_scan(au_bytes, window=None):
    """Find and parse an OFMD offset-metadata SEI in a dependent-view AU.

    Args:
        au_bytes: Annex-B bytes (or anything bytes-like) of the access unit
            (base+dep combined is fine — the magic is searched, memchr-fast).
        window: optional scan bound; None scans the whole AU (the SEI sits in
            the dep portion of combined AUs, i.e. not necessarily at the head).

    Returns:
        (frames_in_gop, [per-sequence list of per-frame signed px offsets])
        or None when the AU carries no offset metadata (only GOP-start AUs do).
        Positive px = toward the viewer (crossed).
    """
    head = bytes(au_bytes if window is None else au_bytes[:window])
    i = head.find(b'OFMD')
    if i < 0:
        return None
    if not (head[i + 10] & 0x80 if i + 10 < len(head) else False):
        return None  # marker bit absent -> random 'OFMD' bytes in slice data
    # The SEI payload traverses the NAL emulation-prevention layer: any
    # 00 00 03 in the raw stream is 00 00 + EPB. Strip within a bounded
    # region so offset rows containing 0,0,x parse correctly.
    region = head[i:i + 4 + 10 + 2 + 32 * 64 + 64].replace(b'\x00\x00\x03', b'\x00\x00')
    if len(region) < 15:
        return None
    n_seq = region[10] & 0x7F
    frames = region[11]
    if not (0 < n_seq <= 32 and 0 < frames <= 64):
        return None
    body = region[14:14 + n_seq * frames]
    if len(body) < n_seq * frames:
        return None
    seqs = []
    for s in range(n_seq):
        row = body[s * frames:(s + 1) * frames]
        seqs.append([-(b & 0x7F) if (b & 0x80) else (b & 0x7F) for b in row])
    return frames, seqs


def offset_to_disparity(offset_px):
    """BD plane offset (per-eye px) -> normalized eye-width disparity (>0 = front)."""
    return 2.0 * float(offset_px) / EYE_WIDTH_PX


def parse_mpls_pg_offsets(mpls_path):
    """Map each PG stream PID -> offset_sequence_id from the MPLS STN_table_SS.

    Parses the first PlayItem's STN_table (PG PIDs, in order) and the first
    STN_table_SS block of ExtensionData (type=2, ver=1). Returns {} when the
    playlist has no SS extension (2D disc) or on any parse problem.
    """
    try:
        with open(mpls_path, 'rb') as f:
            d = f.read()
        pl_start = struct.unpack('>I', d[8:12])[0]
        ext_start = struct.unpack('>I', d[16:20])[0]
        if not ext_start:
            return {}

        # --- first PlayItem's primary STN_table: ordered PG PIDs ---
        q = pl_start + 10
        item_len = struct.unpack('>H', d[q:q + 2])[0]
        body = d[q + 2:q + 2 + item_len]
        o = 32 + 2 + 2                      # fixed PlayItem part + stn_len + reserved
        n_video, n_audio, n_pg, n_ig = body[o], body[o + 1], body[o + 2], body[o + 3]
        o += 7 + 5                          # counts + reserved

        def read_stream(o):
            se_len = body[o]; s = o + 1
            se_type = body[s]
            pid = None
            if se_type == 1:
                pid = struct.unpack('>H', body[s + 1:s + 3])[0]
            elif se_type == 2:
                pid = struct.unpack('>H', body[s + 3:s + 5])[0]
            elif se_type == 3:
                pid = struct.unpack('>H', body[s + 2:s + 4])[0]
            o = s + se_len
            sa_len = body[o]
            o += 1 + sa_len
            return pid, o

        pg_pids = []
        for _ in range(n_video + n_audio):
            _pid, o = read_stream(o)
        for _ in range(n_pg):
            pid, o = read_stream(o)
            pg_pids.append(pid)

        # --- ExtensionData -> STN_table_SS (type=2, ver=1) ---
        ext = d[ext_start:]
        n_entries = ext[11]
        blk = None
        eo = 12
        for _ in range(n_entries):
            etype = struct.unpack('>H', ext[eo:eo + 2])[0]
            ever = struct.unpack('>H', ext[eo + 2:eo + 4])[0]
            estart = struct.unpack('>I', ext[eo + 4:eo + 8])[0]
            el = struct.unpack('>I', ext[eo + 8:eo + 12])[0]
            if etype == 2 and ever == 1:
                blk = ext[estart:estart + el]
                break
            eo += 12
        if not blk:
            return {}

        b = blk
        o = 4                               # length(2) + popup flag/reserved(2)
        se_len = b[o]
        o += 1 + se_len                     # dependent-video stream_entry
        sa_len = b[o]
        o += 1 + sa_len                     # dependent-video stream_attributes
        o += 2                              # number_of_offset_sequences
        mapping = {}
        for i in range(n_pg):
            if o + 2 > len(b):
                break
            seq_id, _flags = b[o], b[o + 1]
            if i < len(pg_pids) and pg_pids[i] is not None:
                mapping[pg_pids[i]] = seq_id
            o += 2
        logger.info(f"[BD3D-DEPTH] MPLS PG offset sequences: "
                    f"{{{', '.join(f'0x{p:04X}: {s}' for p, s in mapping.items())}}}")
        return mapping
    except Exception as e:
        logger.warning(f"[BD3D-DEPTH] MPLS STN_table_SS parse failed: {e}")
        return {}
