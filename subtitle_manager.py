"""
SubtitleManager - Synchronizes PGS subtitle display with video playback.

Manages timing synchronization between the PGS subtitle parser and the OpenGL
renderer, ensuring subtitles appear and disappear at the correct times.
"""

import logging
from typing import Optional, Callable, Tuple
from PySide6.QtCore import QObject, Signal, Slot, QTimer

from pgs_subtitle_parser import PGSSubtitleParser, PGSDisplaySet

logger = logging.getLogger(__name__)


class SubtitleManager(QObject):
    """
    Manages subtitle display synchronization with video playback.

    Connects to a video player's time updates and pushes subtitle data
    to OpenGL widgets at the correct times.
    """

    # Emitted when subtitle should be displayed (rgba_array, x, y, w, h, video_w, video_h)
    subtitle_changed = Signal(object, int, int, int, int, int, int)
    # Emitted when subtitle should be cleared
    subtitle_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._parser: Optional[PGSSubtitleParser] = None
        self._current_display_set: Optional[PGSDisplaySet] = None
        self._last_time_seconds: float = 0.0
        self._enabled: bool = False
        self._video_width: int = 1920
        self._video_height: int = 1080

        # Offset in seconds (can be adjusted for sync fine-tuning)
        self._time_offset: float = 0.0

        # Track if we need to update (avoid redundant updates)
        self._last_update_pts: float = -1.0

        # Streaming mode flag
        self._streaming_mode: bool = False

    def load_subtitle_file(self, filepath: str) -> bool:
        """
        Load a PGS/SUP subtitle file.

        Args:
            filepath: Path to .sup or .pgs file

        Returns:
            True if loaded successfully
        """
        self._parser = PGSSubtitleParser()
        success = self._parser.load_from_file(filepath)

        if success:
            logger.info(f"[SubtitleManager] Loaded {len(self._parser.display_sets)} display sets from {filepath}")
            self._current_display_set = None
            self._last_update_pts = -1.0
        else:
            logger.error(f"[SubtitleManager] Failed to load subtitle file: {filepath}")
            self._parser = None

        return success

    def load_subtitle_data(self, data: bytes) -> bool:
        """
        Load PGS subtitle from binary data (extracted from MKV).

        Args:
            data: Raw PGS/SUP binary data

        Returns:
            True if loaded successfully
        """
        self._parser = PGSSubtitleParser()
        success = self._parser.load_from_bytes(data)

        if success:
            logger.info(f"[SubtitleManager] Loaded {len(self._parser.display_sets)} display sets from data")
            self._current_display_set = None
            self._last_update_pts = -1.0
        else:
            logger.error("[SubtitleManager] Failed to load subtitle data")
            self._parser = None

        return success

    def set_enabled(self, enabled: bool):
        """Enable or disable subtitle display."""
        self._enabled = enabled
        logger.debug(f"[SubtitleManager] set_enabled({enabled})")
        if not enabled:
            self._current_display_set = None
            self._last_update_pts = -1.0
            self._last_img_key = None
            self.subtitle_cleared.emit()
            logger.debug("[SubtitleManager] Subtitles disabled")
        else:
            logger.debug("[SubtitleManager] Subtitles enabled")
            # Re-check current time to show subtitle if applicable
            self.update_time(self._last_time_seconds)

    def set_video_dimensions(self, width: int, height: int):
        """Set video dimensions for coordinate normalization."""
        self._video_width = width
        self._video_height = height

    def set_time_offset(self, offset_seconds: float):
        """Set time offset for subtitle synchronization fine-tuning."""
        self._time_offset = offset_seconds

    def clear(self):
        """Clear current subtitle and parser."""
        self._parser = None
        self._current_display_set = None
        self._last_update_pts = -1.0
        self._last_img_key = None
        self.subtitle_cleared.emit()

    @Slot(float)
    def update_time(self, time_seconds: float):
        """
        Update current playback time - checks if subtitle needs to change.

        This should be called from the video player's time update signal.

        Args:
            time_seconds: Current playback position in seconds
        """
        # Anti-jitter / multi-source guard: update_time() is driven by several clock sources
        # (clamped UI time, raw MPV time-pos, the periodic position poller) that disagree by
        # tens of ms and can step BACKWARDS. A backward blip at a subtitle transition snaps the
        # lookup to the previous display set -> "cross-display" flicker (next shows, previous
        # flashes, then next). Hold the subtitle clock monotonic for small backward steps; a
        # real seek (large jump, or on_seek()) resets it.
        if (time_seconds < self._last_time_seconds
                and (self._last_time_seconds - time_seconds) < 1.0):
            time_seconds = self._last_time_seconds
        self._last_time_seconds = time_seconds

        if not self._enabled or self._parser is None:
            return

        # Apply time offset
        adjusted_time = time_seconds + self._time_offset

        # Get subtitle for current time
        display_set = self._parser.get_subtitle_at_time(adjusted_time)

        # Debug: Log subtitle lookup periodically
        if int(adjusted_time * 2) % 20 == 0:  # Every 10 seconds
            logger.debug(f"[SubtitleManager] update_time({adjusted_time:.1f}s): display_set={display_set is not None}")

        # Check if subtitle changed
        if display_set is None:
            # No subtitle at this time
            if self._current_display_set is not None:
                self._current_display_set = None
                self._last_update_pts = -1.0
                self._last_img_key = None
                self.subtitle_cleared.emit()
        else:
            # Check if it's a different display set (by PTS)
            if self._last_update_pts != display_set.pts:
                self._current_display_set = display_set
                self._last_update_pts = display_set.pts

                # Render and emit
                if display_set.rendered_image is not None:
                    img = display_set.rendered_image
                    # Suppress re-render when the image is IDENTICAL to what's already on
                    # screen. Blu-ray PGS re-transmits the same subtitle as periodic epochs
                    # (different PTS, same pixels); re-emitting would re-upload the texture and
                    # flash near the end of the display. Compare a cheap content key.
                    img_key = (img.shape, hash(img.tobytes()))
                    if img_key == getattr(self, '_last_img_key', None):
                        return  # same subtitle re-sent -> already displayed, no re-render
                    self._last_img_key = img_key
                    x = display_set.render_x
                    y = display_set.render_y
                    h, w = img.shape[:2]
                    # Normalize against the PGS COMPOSITION resolution (the PCS
                    # video_descriptor the coordinates were authored against — almost
                    # always 1920x1080 for Blu-ray), NOT the decoded video size.
                    # They diverge for cropped content (e.g. a 2.39:1 film stored as
                    # 1920x816): a y=982 coordinate over a 816-tall reference normalizes
                    # to >1.0 and the subtitle lands off-screen. The decoded size is only
                    # a fallback when the parser didn't report a composition size.
                    ref_w = getattr(display_set, 'width', 0) or self._video_width
                    ref_h = getattr(display_set, 'height', 0) or self._video_height
                    self.subtitle_changed.emit(
                        img,
                        x, y, w, h,
                        ref_w, ref_h
                    )

    @Slot()
    def on_seek(self):
        """
        Called when a seek operation occurs.
        Forces subtitle re-evaluation at new position.
        """
        self._current_display_set = None
        self._last_update_pts = -1.0
        self._last_img_key = None
        self._last_time_seconds = -1.0  # reset monotonic clock so the post-seek time is honored

        # Clear streaming buffer if in streaming mode
        if self._streaming_mode and self._parser:
            self._parser.clear_streaming_buffer()

        # Clear immediately during seek
        self.subtitle_cleared.emit()

    @property
    def is_loaded(self) -> bool:
        """Check if a subtitle file is loaded."""
        return self._parser is not None and len(self._parser.display_sets) > 0

    @property
    def subtitle_count(self) -> int:
        """Get number of subtitle display sets."""
        if self._parser is None:
            return 0
        return len(self._parser.display_sets)

    # =========================================================================
    # STREAMING MODE API - For real-time PGS from MVC demuxer
    # =========================================================================

    def start_streaming(self):
        """
        Start streaming mode for real-time PGS parsing.
        Call this when using MVC decoder with PGS streaming enabled.
        """
        self._parser = PGSSubtitleParser()
        self._parser.start_streaming()
        self._streaming_mode = True
        self._current_display_set = None
        self._last_update_pts = -1.0
        self._last_img_key = None

    @Slot(bytes, float)
    def on_pgs_data(self, pgs_data: bytes, pts: float):
        """
        Handle incoming PGS data from MVC decoder (streaming mode).

        Args:
            pgs_data: Raw PGS segment data
            pts: Presentation timestamp in seconds
        """
        if not self._streaming_mode or not self._parser:
            logger.debug(f"[SubtitleManager] on_pgs_data IGNORED: streaming_mode={self._streaming_mode}, parser={self._parser is not None}")
            return

        try:
            # Check for zlib compression (MKV often uses zlib for PGS subtitles)
            # zlib signature: 0x78 (CMF) + 0x01/0x5E/0x9C/0xDA (FLG)
            if len(pgs_data) >= 2 and pgs_data[0] == 0x78:
                import zlib
                try:
                    decompressed = zlib.decompress(pgs_data)
                    pgs_data = decompressed
                except zlib.error:
                    pass  # Not zlib compressed, use as-is

            # Feed data to streaming parser - subtitles will be displayed
            # by update_time() when playback reaches their PTS
            self._parser.feed_pes_packet(pgs_data, pts)
        except Exception as e:
            logger.warning(f"[SubtitleManager] PGS streaming error: {e}")

    def stop_streaming(self):
        """Stop streaming mode and clean up."""
        if self._streaming_mode:
            if self._parser:
                self._parser.stop_streaming()
            self._streaming_mode = False

    @property
    def is_streaming(self) -> bool:
        """Check if streaming mode is active."""
        return self._streaming_mode


class SubtitleTrack:
    """Represents a subtitle track from an MKV file."""

    def __init__(self, track_id: int, language: str = "", name: str = "", codec: str = ""):
        self.track_id = track_id
        self.language = language
        self.name = name
        self.codec = codec  # "S_HDMV/PGS" for Blu-ray PGS

    def __repr__(self):
        label = self.name or self.language or f"Track {self.track_id}"
        return f"SubtitleTrack({self.track_id}, '{label}')"

    @property
    def display_name(self) -> str:
        """Get human-readable track name."""
        if self.name:
            return f"{self.name} ({self.language})" if self.language else self.name
        elif self.language:
            return self.language
        else:
            return f"Track {self.track_id}"
