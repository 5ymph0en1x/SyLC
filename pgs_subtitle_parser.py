# -*- coding: utf-8 -*-
"""
PGS/SUP Subtitle Parser - Parse Blu-ray PGS bitmap subtitles.

PGS (Presentation Graphic Stream) format consists of segments:
- PCS (0x14): Presentation Composition Segment
- WDS (0x15): Window Definition Segment
- PDS (0x16): Palette Definition Segment
- ODS (0x17): Object Definition Segment
- END (0x80): End of Display Set
"""

import struct
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, BinaryIO

logger = logging.getLogger(__name__)


@dataclass
class PGSPalette:
    """256-color RGBA palette."""
    colors: np.ndarray = field(default_factory=lambda: np.zeros((256, 4), dtype=np.uint8))


@dataclass
class PGSObject:
    """A single subtitle object (bitmap)."""
    id: int = 0
    version: int = 0
    width: int = 0
    height: int = 0
    rle_data: bytes = b''
    decoded: Optional[np.ndarray] = None  # Decoded indexed pixels


@dataclass
class PGSWindow:
    """Window definition for subtitle placement."""
    id: int = 0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class PGSCompositionObject:
    """Object placement within a composition."""
    object_id: int = 0
    window_id: int = 0
    x: int = 0
    y: int = 0
    crop_flag: bool = False
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0


@dataclass
class PGSDisplaySet:
    """A complete display set (subtitle frame)."""
    pts: float = 0.0  # Presentation timestamp in seconds (start)
    dts: float = 0.0  # Decode timestamp in seconds
    end_pts: float = -1.0  # End timestamp from END segment (-1 = use next DS)
    width: int = 1920
    height: int = 1080
    palette: Optional[PGSPalette] = None
    objects: Dict[int, PGSObject] = field(default_factory=dict)
    windows: Dict[int, PGSWindow] = field(default_factory=dict)
    compositions: List[PGSCompositionObject] = field(default_factory=list)
    rendered_image: Optional[np.ndarray] = None  # Final RGBA image
    render_x: int = 0
    render_y: int = 0


def decode_rle(rle_data: bytes, width: int, height: int) -> Optional[np.ndarray]:
    """Decode PGS RLE-compressed bitmap data."""
    # Validate dimensions
    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        logger.warning(f"[PGS-RLE] Invalid dimensions: {width}x{height}")
        return None

    if len(rle_data) == 0:
        logger.warning("[PGS-RLE] Empty RLE data")
        return None

    try:
        result = np.zeros((height, width), dtype=np.uint8)
        x, y = 0, 0
        i = 0

        while i < len(rle_data) and y < height:
            byte = rle_data[i]
            i += 1

            if byte == 0:
                # Control code
                if i >= len(rle_data):
                    break
                flag = rle_data[i]
                i += 1

                if flag == 0:
                    # End of line
                    x = 0
                    y += 1
                elif flag & 0xC0 == 0:
                    # Short run of zeros
                    run_length = flag & 0x3F
                    end_x = min(x + run_length, width)
                    if y < height:
                        result[y, x:end_x] = 0
                    x = end_x
                elif flag & 0xC0 == 0x40:
                    # Long run of zeros
                    if i >= len(rle_data):
                        break
                    run_length = ((flag & 0x3F) << 8) | rle_data[i]
                    i += 1
                    end_x = min(x + run_length, width)
                    if y < height:
                        result[y, x:end_x] = 0
                    x = end_x
                elif flag & 0xC0 == 0x80:
                    # Short run of color
                    run_length = flag & 0x3F
                    if i >= len(rle_data):
                        break
                    color = rle_data[i]
                    i += 1
                    end_x = min(x + run_length, width)
                    if y < height:
                        result[y, x:end_x] = color
                    x = end_x
                elif flag & 0xC0 == 0xC0:
                    # Long run of color
                    if i + 1 >= len(rle_data):
                        break
                    run_length = ((flag & 0x3F) << 8) | rle_data[i]
                    i += 1
                    color = rle_data[i]
                    i += 1
                    end_x = min(x + run_length, width)
                    if y < height:
                        result[y, x:end_x] = color
                    x = end_x
            else:
                # Single pixel
                if x < width and y < height:
                    result[y, x] = byte
                    x += 1

        return result

    except Exception as e:
        logger.error(f"[PGS-RLE] Decode error: {e}")
        return None


class PGSSubtitleParser:
    """
    Parser for PGS/SUP subtitle files.

    Supports two modes:
    - File mode: Load entire .sup file at once
    - Streaming mode: Parse segments incrementally as they arrive from M2TS demuxer
    """

    def __init__(self):
        self.display_sets: List[PGSDisplaySet] = []
        self._current_ds: Optional[PGSDisplaySet] = None
        self._pending_objects: Dict[int, bytes] = {}  # Object ID -> accumulated RLE data

        # Streaming mode state
        self._streaming_mode: bool = False
        self._pes_buffer: bytes = b''  # Accumulate PES packets
        self._last_rendered_idx: int = 0  # Track which DS have been rendered

    def load_from_file(self, filepath: str) -> bool:
        """Load and parse a .sup file."""
        try:
            with open(filepath, 'rb') as f:
                return self._parse_stream(f)
        except Exception as e:
            logger.error(f"[PGSParser] Failed to load {filepath}: {e}")
            return False

    def load_from_bytes(self, data: bytes) -> bool:
        """Load and parse from memory."""
        try:
            import io
            return self._parse_stream(io.BytesIO(data))
        except Exception as e:
            logger.error(f"[PGSParser] Failed to parse data: {e}")
            return False

    def _parse_stream(self, f: BinaryIO) -> bool:
        """Parse PGS stream."""
        self.display_sets = []
        self._current_ds = None
        self._pending_objects = {}
        # Buffers for segments that arrive before PCS
        self._pending_palette = None
        self._pending_ods_list = []  # List of PGSObject instances
        # 3D MVC format support
        self._pending_3d_ods = None  # Metadata from 3D ODS segments
        self._wds_3d_debug = 0
        # Reset debug counters
        self._seg_order_count = 0
        self._ods_debug_count = 0
        self._pcs_debug_count = 0

        segment_count = 0
        segment_types = {}  # Count each segment type
        while True:
            # Read segment header
            header = f.read(13)
            if len(header) < 13:
                break

            # Check magic
            if header[0:2] != b'PG':
                logger.warning(f"[PGSParser] Invalid magic at segment {segment_count}")
                break

            # Parse header
            pts_raw = struct.unpack('>I', header[2:6])[0]
            dts_raw = struct.unpack('>I', header[6:10])[0]
            segment_type = header[10]
            segment_size = struct.unpack('>H', header[11:13])[0]

            # Convert timestamps (90kHz -> seconds)
            pts = pts_raw / 90000.0
            dts = dts_raw / 90000.0

            # Read segment data
            segment_data = f.read(segment_size)
            if len(segment_data) < segment_size:
                break

            # Process segment
            self._process_segment(segment_type, pts, dts, segment_data)
            segment_count += 1
            segment_types[segment_type] = segment_types.get(segment_type, 0) + 1

        # Debug: Log first few display sets before rendering (debug level)
        if self.display_sets and logger.isEnabledFor(logging.DEBUG):
            for i, ds in enumerate(self.display_sets[:3]):
                obj_decoded = sum(1 for o in ds.objects.values() if o.decoded is not None)
                logger.debug(f"[PGS-DEBUG] DS[{i}] PTS={ds.pts:.2f}s: palette={ds.palette is not None}, "
                            f"compositions={len(ds.compositions)}, objects={len(ds.objects)}, decoded={obj_decoded}")

        # Render all display sets
        no_palette_count = 0
        no_objects_count = 0
        no_decoded_count = 0
        rendered_count = 0
        for ds in self.display_sets:
            had_palette = ds.palette is not None
            had_objects = len(ds.objects) > 0
            had_decoded = any(o.decoded is not None for o in ds.objects.values()) if had_objects else False
            self._render_display_set(ds)
            if ds.rendered_image is not None:
                rendered_count += 1
            elif not had_palette:
                no_palette_count += 1
            elif not had_objects:
                no_objects_count += 1
            elif not had_decoded:
                no_decoded_count += 1

        # Note: PGS timing is implicit - subtitles display from their PTS until
        # the next display set arrives. "Clear" display sets have no compositions.

        # Calculate end_pts for each subtitle based on next subtitle's PTS
        # This is necessary when there are no explicit "clear" display sets
        # Also apply a maximum duration to prevent "eternal" subtitles
        MAX_SUBTITLE_DURATION = 8.0  # seconds - typical subtitle max duration

        for i, ds in enumerate(self.display_sets):
            if ds.rendered_image is not None:
                # Find next display set (rendered or clear)
                if i + 1 < len(self.display_sets):
                    next_pts = self.display_sets[i + 1].pts
                    # Use the earlier of: next subtitle start OR max duration
                    ds.end_pts = min(next_pts, ds.pts + MAX_SUBTITLE_DURATION)
                else:
                    # Last subtitle - show for max duration
                    ds.end_pts = ds.pts + MAX_SUBTITLE_DURATION

        # Log summary
        logger.info(f"[PGSParser] Parsed {len(self.display_sets)} subtitles ({rendered_count} rendered) from {segment_count} segments")

        # Debug: Print PTS range and timing info
        # Debug PTS range (only at debug level)
        if self.display_sets and logger.isEnabledFor(logging.DEBUG):
            first_pts = self.display_sets[0].pts
            last_pts = self.display_sets[-1].pts
            rendered_ds = [ds for ds in self.display_sets if ds.rendered_image is not None]
            logger.debug(f"[PGSParser] PTS range: {first_pts:.2f}s - {last_pts:.2f}s, rendered={len(rendered_ds)}")

        return len(self.display_sets) > 0

    def _process_segment(self, seg_type: int, pts: float, dts: float, data: bytes):
        """Process a single PGS segment.

        Standard PGS segment types (per Blu-ray spec):
        - 0x14 = PDS (Palette Definition Segment)
        - 0x15 = ODS (Object Definition Segment)
        - 0x16 = PCS (Presentation Composition Segment)
        - 0x17 = WDS (Window Definition Segment)
        - 0x80 = END (End of Display Set)
        """
        # Track segment order for debugging (only at debug level)
        seg_order_count = getattr(self, '_seg_order_count', 0)
        if seg_order_count < 20:  # Log first 20 segments
            self._seg_order_count = seg_order_count + 1
            type_names = {0x14: 'PDS', 0x15: 'ODS', 0x16: 'PCS', 0x17: 'WDS', 0x80: 'END'}
            seg_name = type_names.get(seg_type, f'0x{seg_type:02X}')
            logger.debug(f"[PGS-ORDER] #{seg_order_count}: {seg_name} ({len(data)} bytes)")

        # Standard PGS segment type mapping (Blu-ray specification)
        if seg_type == 0x16:  # PCS - Presentation Composition Segment
            self._parse_pcs(pts, dts, data)
        elif seg_type == 0x17:  # WDS - Window Definition Segment
            self._parse_wds(data)
        elif seg_type == 0x14:  # PDS - Palette Definition Segment
            self._parse_pds(data)
        elif seg_type == 0x15:  # ODS - Object Definition Segment
            self._parse_ods(data)
        elif seg_type == 0x80:  # END - End of Display Set
            self._finalize_display_set(pts)

    def _parse_pcs(self, pts: float, dts: float, data: bytes):
        """Parse Presentation Composition Segment."""
        if len(data) < 11:
            return

        # Start new display set
        self._current_ds = PGSDisplaySet(pts=pts, dts=dts)

        # Parse video dimensions
        parsed_width = struct.unpack('>H', data[0:2])[0]
        parsed_height = struct.unpack('>H', data[2:4])[0]

        # Fallback to 1080p if dimensions are invalid (common in some Blu-ray PGS tracks)
        self._current_ds.width = parsed_width if parsed_width > 0 else 1920
        self._current_ds.height = parsed_height if parsed_height > 0 else 1080

        # 3D MVC Blu-ray format detection: Large PCS contains embedded palette
        # Normal PCS is ~11-25 bytes; 3D format has 1000+ bytes with palette at offset 7
        if len(data) > 100 and parsed_width == 0:
            self._parse_pcs_3d_palette(data)
            return  # 3D PCS has different structure, skip standard parsing

        # Parse composition state (byte 7)
        # 0x00 = Normal, 0x40 = Acquisition Point, 0x80 = Epoch Start
        composition_state = data[7]
        palette_update_flag = data[8]
        palette_id = data[9]
        num_objects = data[10]

        # Apply any pending palette/objects that arrived before this PCS
        pcs_count = getattr(self, '_pcs_debug_count', 0)
        if pcs_count < 5:
            self._pcs_debug_count = pcs_count + 1
            logger.debug(f"[PGS-PCS] #{pcs_count}: pts={pts:.2f}s, pending_palette={self._pending_palette is not None}, pending_ods={len(self._pending_ods_list)}")

        if self._pending_palette:
            self._current_ds.palette = self._pending_palette
            self._pending_palette = None
        if self._pending_ods_list:
            logger.debug(f"[PGS-PCS] Applying {len(self._pending_ods_list)} pending ODS objects")
            for obj in self._pending_ods_list:
                self._current_ds.objects[obj.id] = obj
            self._pending_ods_list = []

        # Inherit palette and objects from previous display set if not epoch start
        # (only if we didn't get pending data, which takes priority)
        # Epoch Start (0x80) means new palette/objects, others inherit
        if composition_state != 0x80 and self.display_sets:
            prev_ds = self.display_sets[-1]
            # Copy palette from previous (if we don't have one yet)
            if not self._current_ds.palette and prev_ds.palette:
                self._current_ds.palette = prev_ds.palette
            # Copy objects from previous (if we don't have any yet)
            if not self._current_ds.objects:
                for obj_id, obj in prev_ds.objects.items():
                    self._current_ds.objects[obj_id] = obj

        # Parse composition objects
        offset = 11
        for _ in range(num_objects):
            if offset + 8 > len(data):
                break

            comp = PGSCompositionObject()
            comp.object_id = struct.unpack('>H', data[offset:offset+2])[0]
            comp.window_id = data[offset + 2]
            comp.crop_flag = bool(data[offset + 3] & 0x80)
            comp.x = struct.unpack('>H', data[offset+4:offset+6])[0]
            comp.y = struct.unpack('>H', data[offset+6:offset+8])[0]
            offset += 8

            if comp.crop_flag and offset + 8 <= len(data):
                comp.crop_x = struct.unpack('>H', data[offset:offset+2])[0]
                comp.crop_y = struct.unpack('>H', data[offset+2:offset+4])[0]
                comp.crop_w = struct.unpack('>H', data[offset+4:offset+6])[0]
                comp.crop_h = struct.unpack('>H', data[offset+6:offset+8])[0]
                offset += 8

            self._current_ds.compositions.append(comp)

    def _parse_pcs_3d_palette(self, data: bytes):
        """Parse 3D MVC Blu-ray PCS with embedded palette.

        In 3D format, the PCS segment contains the full 256-color palette
        starting at offset 7 with 5-byte entries: [entry_id, Y, Cb, Cr, A]
        """
        # Set default video dimensions for 3D format
        self._current_ds.width = 1920
        self._current_ds.height = 1080

        # Create palette
        palette = PGSPalette()

        # Parse palette entries starting at offset 7
        # Format: [entry_id(1), Y(1), Cb(1), Cr(1), A(1)] repeated
        offset = 7
        entry_count = 0

        while offset + 5 <= len(data):
            entry_id = data[offset]
            y = data[offset + 1]
            cb = data[offset + 2]
            cr = data[offset + 3]
            alpha = data[offset + 4]
            offset += 5

            # Convert YCbCr to RGB (BT.601)
            r = int(max(0, min(255, y + 1.402 * (cr - 128))))
            g = int(max(0, min(255, y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128))))
            b = int(max(0, min(255, y + 1.772 * (cb - 128))))

            palette.colors[entry_id] = [r, g, b, alpha]
            entry_count += 1

        # Debug: Log first 3D PCS palette extraction only once
        if not getattr(self, '_pcs_3d_logged', False):
            self._pcs_3d_logged = True
            non_transparent = np.count_nonzero(palette.colors[:, 3])
            logger.debug(f"[PGS-3D] Embedded palette: {entry_count} entries, {non_transparent} with alpha")

        # Store palette for this display set
        self._current_ds.palette = palette

    def _parse_wds(self, data: bytes):
        """Parse Window Definition Segment."""
        if not self._current_ds or len(data) < 1:
            return

        num_windows = data[0]

        # 3D MVC Blu-ray format: num_windows=0 but large data = RLE bitmap container
        if num_windows == 0 and len(data) > 100:
            self._parse_wds_3d_bitmap(data)
            return

        offset = 1
        for _ in range(num_windows):
            if offset + 9 > len(data):
                break

            window = PGSWindow()
            window.id = data[offset]
            window.x = struct.unpack('>H', data[offset+1:offset+3])[0]
            window.y = struct.unpack('>H', data[offset+3:offset+5])[0]
            window.width = struct.unpack('>H', data[offset+5:offset+7])[0]
            window.height = struct.unpack('>H', data[offset+7:offset+9])[0]
            offset += 9

            self._current_ds.windows[window.id] = window

    def _parse_wds_3d_bitmap(self, data: bytes):
        """Parse 3D MVC Blu-ray WDS segment containing bitmap RLE data."""
        # 3D format: WDS contains RLE bitmap data when num_windows=0
        # Use pending 3D ODS metadata for dimensions
        if not hasattr(self, '_pending_3d_ods') or not self._pending_3d_ods:
            wds_count = getattr(self, '_wds_3d_debug', 0)
            if wds_count < 3:
                self._wds_3d_debug = wds_count + 1
                logger.warning(f"[PGS-WDS-3D] Got 3D WDS ({len(data)} bytes) but no pending 3D ODS metadata")
            return

        ods_meta = self._pending_3d_ods
        width = ods_meta.get('width', 0)
        height = ods_meta.get('height', 0)
        y_pos = ods_meta.get('y_pos', 0)  # Y position from ODS metadata
        object_id = ods_meta.get('object_id', 0)

        if width <= 0 or height <= 0:
            return

        # 3D WDS structure: byte 0 = num_windows (0), then header, then RLE
        # Skip to RLE data after header
        rle_start = 11
        rle_data = data[rle_start:]

        wds_count = getattr(self, '_wds_3d_debug', 0)
        if wds_count < 3:
            self._wds_3d_debug = wds_count + 1
            logger.debug(f"[PGS-WDS-3D] Creating object from WDS: id={object_id}, {width}x{height}, y_pos={y_pos}")

        # Create object with this data
        obj = PGSObject()
        obj.id = object_id
        obj.width = width
        obj.height = height
        obj.rle_data = rle_data

        # Decode RLE
        if width > 0 and height > 0 and len(rle_data) > 0:
            obj.decoded = decode_rle(rle_data, width, height)

        # Store in current display set
        if self._current_ds:
            self._current_ds.objects[object_id] = obj

            # For 3D format, fix video dimensions if they're invalid (0x0)
            if self._current_ds.width == 0:
                self._current_ds.width = 1920  # Standard 3D Blu-ray width
            if self._current_ds.height == 0:
                self._current_ds.height = 1080  # Standard 3D Blu-ray height

            # For 3D format, replace invalid compositions from malformed PCS
            self._current_ds.compositions.clear()
            comp = PGSCompositionObject()
            comp.object_id = object_id
            # Center subtitle horizontally within video frame
            comp.x = max(0, (self._current_ds.width - width) // 2)
            # Store the original y_pos from metadata for analysis
            # We'll use it in render to understand the format
            comp.y = y_pos  # Use metadata y_pos instead of 0
            # Store metadata in display set for debug
            if not hasattr(self._current_ds, 'meta_y_pos'):
                self._current_ds.meta_y_pos = y_pos
            self._current_ds.compositions.append(comp)

        # Clear pending metadata
        self._pending_3d_ods = None

    def _parse_pds(self, data: bytes):
        """Parse Palette Definition Segment."""
        if len(data) < 2:
            return

        palette_id = data[0]
        palette_version = data[1]

        palette = PGSPalette()
        offset = 2

        while offset + 5 <= len(data):
            entry_id = data[offset]
            y = data[offset + 1]
            cb = data[offset + 2]
            cr = data[offset + 3]
            alpha = data[offset + 4]
            offset += 5

            # Convert YCbCr to RGB (BT.601)
            r = int(max(0, min(255, y + 1.402 * (cr - 128))))
            g = int(max(0, min(255, y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128))))
            b = int(max(0, min(255, y + 1.772 * (cb - 128))))

            palette.colors[entry_id] = [r, g, b, alpha]

        # Store in current display set OR buffer for later
        if self._current_ds:
            self._current_ds.palette = palette
        else:
            self._pending_palette = palette

    def _parse_ods(self, data: bytes):
        """Parse Object Definition Segment."""
        if len(data) < 4:
            logger.warning(f"[PGS-ODS] Segment too short: {len(data)} bytes < 4")
            return

        object_id = struct.unpack('>H', data[0:2])[0]
        object_version = data[2]
        sequence_flag = data[3]

        # Debug: log first few ODS segments (debug level only)
        ods_count = getattr(self, '_ods_debug_count', 0)
        if ods_count < 5:
            self._ods_debug_count = ods_count + 1
            logger.debug(f"[PGS-ODS] Segment #{ods_count}: id={object_id}, flag=0x{sequence_flag:02X}, data_len={len(data)}")

        # Check for 3D MVC format: ODS with 10 bytes containing metadata only
        # Flag patterns like 0x2B, 0xE2, 0x81 with only 10 bytes = 3D metadata format
        if len(data) == 10 and len(data) < 11:
            # 3D format: bytes 4-9 contain dimensions and position
            # Structure: obj_id(2), version(1), flag(1), width(2), height(2), y_pos(2)
            width = struct.unpack('>H', data[4:6])[0]
            height = struct.unpack('>H', data[6:8])[0]
            y_pos = struct.unpack('>H', data[8:10])[0]

            if ods_count < 3:
                logger.debug(f"[PGS-ODS-3D] 3D metadata: id={object_id}, {width}x{height}, y_pos={y_pos}")

            # Store as pending 3D ODS metadata for WDS to use
            self._pending_3d_ods = {
                'object_id': object_id,
                'width': width,
                'height': height,
                'y_pos': y_pos
            }
            return

        # First/only segment in sequence (standard format)
        if sequence_flag & 0x80:
            # Need at least 11 bytes for first segment (header + dimensions)
            if len(data) < 11:
                return
            # Get object dimensions
            # data_length = struct.unpack('>I', b'\x00' + data[4:7])[0]  # 3 bytes
            width = struct.unpack('>H', data[7:9])[0]
            height = struct.unpack('>H', data[9:11])[0]
            rle_data = data[11:]

            obj = PGSObject()
            obj.id = object_id
            obj.version = object_version
            obj.width = width
            obj.height = height
            obj.rle_data = rle_data

            # Decode immediately if this is the last (or only) segment
            if sequence_flag & 0x40:
                if obj.width > 0 and obj.height > 0:
                    obj.decoded = decode_rle(obj.rle_data, obj.width, obj.height)

            # Store in current display set OR buffer for later
            if self._current_ds:
                self._current_ds.objects[object_id] = obj
            else:
                # Buffer as pending - store the complete object
                self._pending_ods_list.append(obj)

            # If not last segment, store for continuation
            if not (sequence_flag & 0x40):
                self._pending_objects[object_id] = obj  # Store object reference for continuation
        else:
            # Continuation segment - find the object to append to
            target_obj = None
            if self._current_ds and object_id in self._current_ds.objects:
                target_obj = self._current_ds.objects[object_id]
            elif object_id in self._pending_objects:
                target_obj = self._pending_objects[object_id]

            if target_obj:
                target_obj.rle_data += data[4:]
                # If last segment, decode now
                if sequence_flag & 0x40:
                    if target_obj.width > 0 and target_obj.height > 0:
                        target_obj.decoded = decode_rle(target_obj.rle_data, target_obj.width, target_obj.height)

    def _finalize_display_set(self, end_pts: float = -1.0):
        """Finalize current display set."""
        if self._current_ds:
            # Note: END segment's PTS is typically same as PCS PTS (not useful for duration)
            # Actual end time is determined by when next display set arrives
            self._current_ds.end_pts = -1  # Will use next DS's PTS
            self.display_sets.append(self._current_ds)
            self._current_ds = None
        # Clear all pending data (END without PCS means discard)
        self._pending_objects.clear()
        self._pending_palette = None
        self._pending_ods_list = []

    def _render_display_set(self, ds: PGSDisplaySet):
        """Render a display set to RGBA image."""
        if not ds.palette or not ds.compositions:
            # Debug: log why we skip
            if not ds.palette:
                logger.debug(f"[PGS-RENDER] Skipping DS at {ds.pts:.2f}s: no palette")
            return

        # Calculate bounding box of all compositions
        min_x, min_y = ds.width, ds.height
        max_x, max_y = 0, 0

        decoded_count = 0
        for comp in ds.compositions:
            obj = ds.objects.get(comp.object_id)
            if obj and obj.decoded is not None:
                decoded_count += 1
                x1, y1 = comp.x, comp.y
                x2, y2 = x1 + obj.width, y1 + obj.height
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)

        if min_x >= max_x or min_y >= max_y:
            # Debug: log why bounding box is invalid
            logger.debug(f"[PGS-RENDER] Skipping DS at {ds.pts:.2f}s: no valid objects (decoded={decoded_count}/{len(ds.compositions)}, objects={len(ds.objects)})")
            return

        # Create output image (just the bounding box)
        out_w = max_x - min_x
        out_h = max_y - min_y
        rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)

        # Render each object (vectorized for performance)
        for comp in ds.compositions:
            obj = ds.objects.get(comp.object_id)
            if obj and obj.decoded is not None:
                # Calculate position relative to bounding box
                rel_x = comp.x - min_x
                rel_y = comp.y - min_y

                # Calculate valid copy region
                src_y_start = max(0, -rel_y)
                src_x_start = max(0, -rel_x)
                src_y_end = min(obj.height, out_h - rel_y)
                src_x_end = min(obj.width, out_w - rel_x)

                dst_y_start = max(0, rel_y)
                dst_x_start = max(0, rel_x)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)

                if src_y_end > src_y_start and src_x_end > src_x_start:
                    # Get the indexed pixels for the region
                    indexed_region = obj.decoded[src_y_start:src_y_end, src_x_start:src_x_end]
                    # Apply palette using numpy fancy indexing (vectorized!)
                    rgba[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = ds.palette.colors[indexed_region]

        # Crop to actual content (non-transparent pixels) for proper positioning
        # Find rows and columns with any non-transparent pixels
        alpha_channel = rgba[:, :, 3]
        non_zero_rows = np.any(alpha_channel > 0, axis=1)
        non_zero_cols = np.any(alpha_channel > 0, axis=0)

        if not np.any(non_zero_rows) or not np.any(non_zero_cols):
            # No visible content
            logger.debug(f"[PGS-RENDER] Skipping DS at {ds.pts:.2f}s: no visible content in bitmap")
            return

        # Find bounding box of actual content
        row_indices = np.where(non_zero_rows)[0]
        col_indices = np.where(non_zero_cols)[0]
        content_top = row_indices[0]
        content_bottom = row_indices[-1] + 1
        content_left = col_indices[0]
        content_right = col_indices[-1] + 1

        # Crop to content with small padding
        padding = 2
        crop_top = max(0, content_top - padding)
        crop_bottom = min(out_h, content_bottom + padding)
        crop_left = max(0, content_left - padding)
        crop_right = min(out_w, content_right + padding)

        cropped_rgba = rgba[crop_top:crop_bottom, crop_left:crop_right]
        cropped_h, cropped_w = cropped_rgba.shape[:2]


        # ALWAYS use 1080p for positioning calculation (matches video output dimensions)
        # Some PGS tracks have different dimensions (e.g., 720x480 for French subtitles)
        # but the final render target is always 1920x1080 for MVC video
        video_w = 1920
        video_h = 1080

        # Center horizontally and position at bottom with margin
        bottom_margin = 50
        render_x = (video_w - cropped_w) // 2
        render_y = video_h - cropped_h - bottom_margin

        ds.rendered_image = cropped_rgba
        ds.render_x = max(0, render_x)
        ds.render_y = max(0, render_y)

    def get_subtitle_at_time(self, time_seconds: float) -> Optional[PGSDisplaySet]:
        """
        Get the subtitle that should be displayed at a given time.

        PGS timing: subtitle displays from its PTS until end_pts.
        end_pts is calculated from the next display set's PTS.

        Args:
            time_seconds: Current playback position

        Returns:
            PGSDisplaySet or None if no subtitle should be shown
        """
        result = None

        for ds in self.display_sets:
            if ds.pts <= time_seconds:
                # Check if subtitle has expired (past its end time)
                if ds.end_pts > 0 and time_seconds >= ds.end_pts:
                    result = None  # Subtitle has ended
                elif len(ds.compositions) == 0 or ds.rendered_image is None:
                    result = None  # "Clear" display set
                else:
                    result = ds  # Active subtitle
            else:
                break  # Past current time, stop iterating

        return result

    # =========================================================================
    # STREAMING MODE API - For real-time parsing from M2TS demuxer
    # =========================================================================

    def start_streaming(self):
        """
        Initialize streaming mode for real-time PGS parsing.
        Call this before feeding PES packets.
        """
        self._streaming_mode = True
        self._pes_buffer = b''
        self._last_rendered_idx = 0
        self.display_sets = []
        self._current_ds = None
        self._pending_objects = {}
        self._pending_palette = None
        self._pending_ods_list = []

    def feed_pes_packet(self, pes_data: bytes, pts: float = 0.0) -> List[PGSDisplaySet]:
        """
        Feed a PES packet containing PGS data for streaming parsing.
        Supports both M2TS format (with PG header) and MKV format (raw segments).

        Args:
            pes_data: Raw PGS data (PES or MKV format)
            pts: Presentation timestamp in seconds (from container)

        Returns:
            List of newly completed and rendered display sets (may be empty)
        """
        if not self._streaming_mode:
            self.start_streaming()

        new_display_sets = []

        # Accumulate data
        self._pes_buffer += pes_data

        # Detect format: M2TS (starts with 'PG') vs MKV (starts with segment type)
        while len(self._pes_buffer) >= 3:  # Minimum: type (1) + size (2)
            # Check if this is M2TS format (PG header)
            if self._pes_buffer[0:2] == b'PG':
                # M2TS format: PG + PTS(4) + DTS(4) + type(1) + size(2) = 13 bytes header
                if len(self._pes_buffer) < 13:
                    break  # Need more data

                pts_raw = struct.unpack('>I', self._pes_buffer[2:6])[0]
                dts_raw = struct.unpack('>I', self._pes_buffer[6:10])[0]
                segment_type = self._pes_buffer[10]
                segment_size = struct.unpack('>H', self._pes_buffer[11:13])[0]

                total_size = 13 + segment_size
                if len(self._pes_buffer) < total_size:
                    break  # Need more data

                segment_data = self._pes_buffer[13:total_size]
                self._pes_buffer = self._pes_buffer[total_size:]

                # Convert timestamps (90kHz -> seconds)
                seg_pts = pts_raw / 90000.0
                seg_dts = dts_raw / 90000.0

            else:
                # MKV format: Raw PGS segments without PG header
                # Format: type(1) + size(2) + data(size)
                segment_type = self._pes_buffer[0]

                # Validate segment type (0x14-0x18 or 0x80)
                valid_types = {0x14, 0x15, 0x16, 0x17, 0x18, 0x80}
                if segment_type not in valid_types:
                    # Unknown segment, skip 1 byte and try again
                    logger.debug(f"[PGS-MKV] Skipping unknown segment type 0x{segment_type:02X}")
                    self._pes_buffer = self._pes_buffer[1:]
                    continue

                segment_size = struct.unpack('>H', self._pes_buffer[1:3])[0]
                total_size = 3 + segment_size

                if len(self._pes_buffer) < total_size:
                    break  # Need more data

                segment_data = self._pes_buffer[3:total_size]
                self._pes_buffer = self._pes_buffer[total_size:]

                # Use container PTS for MKV
                seg_pts = pts
                seg_dts = pts

                logger.debug(f"[PGS-MKV] Segment type=0x{segment_type:02X}, size={segment_size}, pts={seg_pts:.3f}s")

            # Process segment
            prev_ds_count = len(self.display_sets)
            self._process_segment(segment_type, seg_pts, seg_dts, segment_data)

            # Check if new display set was completed
            if len(self.display_sets) > prev_ds_count:
                # Render newly added display sets
                for i in range(self._last_rendered_idx, len(self.display_sets)):
                    ds = self.display_sets[i]
                    self._render_display_set(ds)
                    # Calculate end_pts based on MAX_SUBTITLE_DURATION for now
                    # Will be updated when next DS arrives
                    ds.end_pts = ds.pts + 8.0  # 8 seconds max
                    if ds.rendered_image is not None:
                        new_display_sets.append(ds)
                self._last_rendered_idx = len(self.display_sets)

                # Update end_pts of previous subtitle
                if len(self.display_sets) >= 2:
                    prev_ds = self.display_sets[-2]
                    curr_ds = self.display_sets[-1]
                    prev_ds.end_pts = min(curr_ds.pts, prev_ds.pts + 8.0)

        return new_display_sets

    def feed_raw_segment(self, segment_type: int, pts: float, dts: float, data: bytes) -> Optional[PGSDisplaySet]:
        """
        Feed a single raw PGS segment (without PG header).

        Args:
            segment_type: Segment type (0x14=PDS, 0x15=ODS, 0x16=PCS, 0x17=WDS, 0x80=END)
            pts: Presentation timestamp in seconds
            dts: Decoding timestamp in seconds
            data: Segment payload data

        Returns:
            Newly completed and rendered display set, or None
        """
        if not self._streaming_mode:
            self.start_streaming()

        prev_ds_count = len(self.display_sets)
        self._process_segment(segment_type, pts, dts, data)

        # Check if new display set was completed
        if len(self.display_sets) > prev_ds_count:
            ds = self.display_sets[-1]
            self._render_display_set(ds)
            ds.end_pts = ds.pts + 8.0  # Default max duration

            # Update previous subtitle's end time
            if len(self.display_sets) >= 2:
                prev_ds = self.display_sets[-2]
                prev_ds.end_pts = min(ds.pts, prev_ds.pts + 8.0)

            if ds.rendered_image is not None:
                logger.debug(f"[PGS-STREAM] Subtitle ready at {ds.pts:.2f}s")
                return ds

        return None

    def stop_streaming(self):
        """
        Stop streaming mode and finalize any pending data.
        """
        if self._streaming_mode:
            # Finalize any pending display set
            if self._current_ds:
                self._finalize_display_set()
            self._streaming_mode = False
            self._pes_buffer = b''

    def clear_streaming_buffer(self):
        """
        Clear the streaming buffer and pending state (call on seek).
        Does NOT stop streaming mode - just resets the parse state.
        """
        self._pes_buffer = b''
        self._current_ds = None
        self._pending_objects = {}
        self._pending_palette = None
        self._pending_ods_list = []
