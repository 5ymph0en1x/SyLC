"""
SubtitleExtractor - Extract PGS subtitle tracks from MKV/M2TS files.

Uses ffprobe to detect subtitle tracks.
For MKV files: Uses fast direct block extraction (5 seconds vs 80+ with ffmpeg)
For other files: Falls back to ffmpeg extraction.
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import fast MKV extractor
try:
    from fast_mkv_subtitle_extractor import FastMKVSubtitleExtractor
    FAST_MKV_AVAILABLE = True
except ImportError:
    FAST_MKV_AVAILABLE = False
    logger.warning("[SubtitleExtractor] Fast MKV extractor not available, using ffmpeg fallback")


@dataclass
class SubtitleTrackInfo:
    """Information about a detected subtitle track."""
    index: int  # Stream index in file
    track_id: int  # Track number (for display)
    codec: str  # Codec name (e.g., "hdmv_pgs_subtitle")
    language: str  # Language code (e.g., "eng", "fre")
    title: str  # Track title/name
    is_pgs: bool  # True if PGS format

    @property
    def display_name(self) -> str:
        """Get human-readable name for UI."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.language:
            lang_names = {
                'eng': 'English', 'fre': 'French', 'fra': 'French',
                'ger': 'German', 'deu': 'German', 'spa': 'Spanish',
                'ita': 'Italian', 'jpn': 'Japanese', 'chi': 'Chinese',
                'kor': 'Korean', 'por': 'Portuguese', 'rus': 'Russian'
            }
            lang_display = lang_names.get(self.language, self.language.upper())
            if lang_display not in parts:
                parts.append(lang_display)
        if not parts:
            parts.append(f"Track {self.track_id}")
        return " - ".join(parts)


class SubtitleExtractor:
    """
    Extracts PGS subtitle tracks from video files.
    """

    def __init__(self, ffprobe_path: str = "ffprobe", ffmpeg_path: str = "ffmpeg",
                 mkvextract_path: str = "mkvextract"):
        """
        Initialize extractor with paths to tools.

        Args:
            ffprobe_path: Path to ffprobe executable
            ffmpeg_path: Path to ffmpeg executable
            mkvextract_path: Path to mkvextract (mkvtoolnix) - FASTEST option
        """
        self._ffprobe = ffprobe_path
        self._ffmpeg = ffmpeg_path
        self._mkvextract = mkvextract_path
        self._temp_dir = tempfile.gettempdir()
        self._mkvextract_available = None  # Lazy check

    def detect_subtitle_tracks(self, filepath: str) -> List[SubtitleTrackInfo]:
        """
        Detect all subtitle tracks in a video file.

        Args:
            filepath: Path to MKV/M2TS file

        Returns:
            List of SubtitleTrackInfo objects
        """
        tracks = []

        try:
            # Run ffprobe to get stream info as JSON
            cmd = [
                self._ffprobe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "s",  # Only subtitle streams
                filepath
            ]

            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=creationflags
            )

            if result.returncode != 0:
                logger.error(f"[SubtitleExtractor] ffprobe failed: {result.stderr}")
                return tracks

            data = json.loads(result.stdout)
            streams = data.get("streams", [])

            track_num = 1
            for stream in streams:
                codec = stream.get("codec_name", "")
                codec_tag = stream.get("codec_tag_string", "")
                tags = stream.get("tags", {})

                # Check if PGS
                is_pgs = (
                    codec == "hdmv_pgs_subtitle" or
                    "pgs" in codec.lower() or
                    "S_HDMV/PGS" in stream.get("codec_long_name", "")
                )

                track_info = SubtitleTrackInfo(
                    index=stream.get("index", 0),
                    track_id=track_num,
                    codec=codec,
                    language=tags.get("language", ""),
                    title=tags.get("title", ""),
                    is_pgs=is_pgs
                )
                tracks.append(track_info)
                track_num += 1

            logger.info(f"[SubtitleExtractor] Found {len(tracks)} subtitle tracks, "
                       f"{sum(1 for t in tracks if t.is_pgs)} are PGS")

        except subprocess.TimeoutExpired:
            logger.error("[SubtitleExtractor] ffprobe timed out")
        except json.JSONDecodeError as e:
            logger.error(f"[SubtitleExtractor] Failed to parse ffprobe output: {e}")
        except FileNotFoundError:
            logger.error(f"[SubtitleExtractor] ffprobe not found at {self._ffprobe}")
        except Exception as e:
            logger.error(f"[SubtitleExtractor] Error detecting tracks: {e}")

        return tracks

    def extract_pgs_track(self, filepath: str, track_index: int,
                          output_path: Optional[str] = None,
                          progress_callback: Optional[callable] = None) -> Optional[str]:
        """
        Extract a PGS subtitle track to a .sup file.

        Args:
            filepath: Path to source video file
            track_index: Stream index of the subtitle track
            output_path: Optional output path (defaults to temp file)
            progress_callback: Optional callback(float) for progress updates (0.0-1.0)

        Returns:
            Path to extracted .sup file, or None on failure
        """
        if output_path is None:
            # Generate temp file path
            base = os.path.splitext(os.path.basename(filepath))[0]
            output_path = os.path.join(self._temp_dir, f"{base}_sub_{track_index}.sup")

        # For MKV files, use fast direct extraction (16x faster than ffmpeg!)
        # mk3d is the standard 3D MKV extension (same Matroska container format)
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ('.mkv', '.mk3d') and FAST_MKV_AVAILABLE:
            return self._extract_mkv_fast(filepath, track_index, output_path, progress_callback)

        # Fallback to ffmpeg for other formats
        return self._extract_ffmpeg(filepath, track_index, output_path)

    def _is_mkvextract_available(self) -> bool:
        """Check if mkvextract is available (cached)."""
        if self._mkvextract_available is None:
            # Try multiple paths: Nuitka bundle, local exe, system path
            import sys
            candidates = [
                # Nuitka onefile extracts data files to __compiled__.containing_dir
                os.path.join(getattr(sys, '_MEIPASS', ''), "mkvextract.exe"),  # PyInstaller compat
                os.path.join(os.path.dirname(sys.executable), "mkvextract.exe"),  # Same dir as .exe
                os.path.join(os.path.dirname(__file__), "mkvextract.exe"),  # Same dir as script
                "mkvextract.exe",  # Current working dir
                "mkvextract",  # System PATH
            ]
            for candidate in candidates:
                if not candidate:  # Skip empty paths
                    continue
                try:
                    result = subprocess.run(
                        [candidate, "--version"],
                        capture_output=True,
                        timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )
                    if result.returncode == 0:
                        self._mkvextract = candidate
                        self._mkvextract_available = True
                        logger.info(f"[SubtitleExtractor] mkvextract found at {candidate}")
                        return True
                except:
                    continue
            self._mkvextract_available = False
            logger.warning("[SubtitleExtractor] mkvextract not found - subtitle extraction will be slower")
        return self._mkvextract_available

    def _extract_mkvextract(self, filepath: str, track_index: int,
                            output_path: str,
                            progress_callback: Optional[callable] = None) -> Optional[str]:
        """Extract using mkvextract - FASTEST method (uses MKV index)."""
        try:
            import time
            import re
            start_time = time.time()

            # Normalize paths to avoid Windows short path issues
            filepath = os.path.normpath(filepath)
            output_path = os.path.normpath(output_path)

            # mkvextract uses track IDs starting from 0
            # track_index from ffprobe already matches this
            cmd = [
                self._mkvextract,
                "tracks",
                filepath,
                f"{track_index}:{output_path}"
            ]
            # Note: No --quiet - we need progress output

            logger.info(f"[SubtitleExtractor] mkvextract: track {track_index} -> {output_path}")

            # CREATE_NO_WINDOW prevents console flash on Windows
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

            if progress_callback:
                # Use Popen to read progress in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr to stdout
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    creationflags=creationflags
                )

                # Parse progress output: "Progress: 50%"
                progress_pattern = re.compile(r'Progress:\s*(\d+)%')
                last_progress = -1

                try:
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            match = progress_pattern.search(line)
                            if match:
                                progress_pct = int(match.group(1))
                                if progress_pct != last_progress:
                                    last_progress = progress_pct
                                    progress_callback(progress_pct / 100.0)

                    returncode = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning("[SubtitleExtractor] mkvextract timed out (>120s)")
                    return None
            else:
                # No progress callback - use simple run with output discarded
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    creationflags=creationflags
                )
                returncode = result.returncode

            if returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                elapsed = time.time() - start_time
                size = os.path.getsize(output_path)
                logger.info(f"[SubtitleExtractor] mkvextract: {elapsed:.1f}s, {size/1024/1024:.1f} MB")
                if progress_callback:
                    progress_callback(1.0)  # Ensure 100% is reported
                return output_path
            else:
                logger.warning(f"[SubtitleExtractor] mkvextract failed (code {returncode})")
                return None

        except subprocess.TimeoutExpired:
            logger.warning("[SubtitleExtractor] mkvextract timed out (>120s)")
        except Exception as e:
            logger.warning(f"[SubtitleExtractor] mkvextract error: {e}")
        return None

    def _extract_mkv_fast(self, filepath: str, track_index: int,
                          output_path: str,
                          progress_callback: Optional[callable] = None) -> Optional[str]:
        """Fast MKV extraction - tries mkvextract first, then ffmpeg, then Python scanner."""
        import time
        start_time = time.time()

        # PRIORITY 1: mkvextract (fastest - uses MKV internal index)
        if self._is_mkvextract_available():
            result = self._extract_mkvextract(filepath, track_index, output_path, progress_callback)
            if result:
                return result
            logger.info("[SubtitleExtractor] mkvextract failed, trying ffmpeg...")

        # PRIORITY 2: ffmpeg -c copy (fast, uses index)
        result = self._extract_ffmpeg(filepath, track_index, output_path)
        if result:
            return result

        # PRIORITY 3: Python scanner (slow but always works)
        logger.info("[SubtitleExtractor] ffmpeg failed, trying Python scanner...")
        try:
            # MKV track numbers are typically ffmpeg_index + 1
            mkv_track_num = track_index + 1

            logger.info(f"[SubtitleExtractor] Python scanner: stream {track_index} -> track {mkv_track_num}")

            extractor = FastMKVSubtitleExtractor()
            result = extractor.extract_subtitle_track(filepath, mkv_track_num, output_path)

            if result and os.path.exists(result) and os.path.getsize(result) > 0:
                elapsed = time.time() - start_time
                size = os.path.getsize(result)
                logger.info(f"[SubtitleExtractor] Python scanner: {elapsed:.1f}s, {size/1024/1024:.1f} MB")
                return result

        except Exception as e:
            logger.error(f"[SubtitleExtractor] Python scanner error: {e}")

        return None

    def _extract_ffmpeg(self, filepath: str, track_index: int,
                        output_path: str) -> Optional[str]:
        """Extract using ffmpeg (slower but more compatible)."""
        try:
            cmd = [
                self._ffmpeg,
                "-y",  # Overwrite output
                "-i", filepath,
                "-map", f"0:{track_index}",
                "-c", "copy",  # Copy without re-encoding
                output_path
            ]

            logger.info(f"[SubtitleExtractor] FFmpeg extracting track {track_index} to {output_path}")

            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout for large files
                creationflags=creationflags
            )

            if result.returncode != 0:
                logger.error(f"[SubtitleExtractor] ffmpeg extraction failed: {result.stderr}")
                return None

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"[SubtitleExtractor] Extracted to {output_path} "
                           f"({os.path.getsize(output_path)} bytes)")
                return output_path
            else:
                logger.error("[SubtitleExtractor] Output file empty or not created")
                return None

        except subprocess.TimeoutExpired:
            logger.error("[SubtitleExtractor] ffmpeg extraction timed out")
        except FileNotFoundError:
            logger.error(f"[SubtitleExtractor] ffmpeg not found at {self._ffmpeg}")
        except Exception as e:
            logger.error(f"[SubtitleExtractor] Extraction error: {e}")

        return None

    def extract_pgs_to_bytes(self, filepath: str, track_index: int) -> Optional[bytes]:
        """
        Extract a PGS subtitle track directly to memory.

        Args:
            filepath: Path to source video file
            track_index: Stream index of the subtitle track

        Returns:
            Raw PGS/SUP data as bytes, or None on failure
        """
        try:
            cmd = [
                self._ffmpeg,
                "-i", filepath,
                "-map", f"0:{track_index}",
                "-c", "copy",
                "-f", "sup",  # Force SUP output format
                "pipe:1"  # Output to stdout
            ]

            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
                creationflags=creationflags
            )

            if result.returncode != 0:
                logger.error(f"[SubtitleExtractor] ffmpeg pipe extraction failed")
                return None

            if len(result.stdout) > 0:
                logger.info(f"[SubtitleExtractor] Extracted {len(result.stdout)} bytes to memory")
                return result.stdout
            else:
                logger.error("[SubtitleExtractor] No data extracted")
                return None

        except subprocess.TimeoutExpired:
            logger.error("[SubtitleExtractor] ffmpeg extraction timed out")
        except Exception as e:
            logger.error(f"[SubtitleExtractor] Extraction error: {e}")

        return None


def get_pgs_tracks(filepath: str) -> List[SubtitleTrackInfo]:
    """
    Convenience function to get only PGS subtitle tracks.

    Args:
        filepath: Path to video file

    Returns:
        List of PGS subtitle tracks
    """
    extractor = SubtitleExtractor()
    all_tracks = extractor.detect_subtitle_tracks(filepath)
    return [t for t in all_tracks if t.is_pgs]


def extract_subtitle(filepath: str, track_index: int) -> Optional[str]:
    """
    Convenience function to extract a subtitle track.

    Args:
        filepath: Path to video file
        track_index: Stream index of subtitle track

    Returns:
        Path to extracted .sup file
    """
    extractor = SubtitleExtractor()
    return extractor.extract_pgs_track(filepath, track_index)
