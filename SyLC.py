# -*- coding: utf-8 -*-

"""
HDR/3D Video Player - Premium Edition
Description: A luxurious, high-quality HDR and 3D video player using PySide6 and libmpv.
             Optimized for 3D Framepacking output with Nvidia 3D Vision support.
             Compatible with Sony VPL-HW55ES projector.
Version: 1.0 - Premium Edition
"""

import sys
import os
import subprocess
import json
import tempfile
import time
import shutil
import glob
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Fix for mpv.dll loading issue
os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ["PATH"]
from PySide6.QtCore import (
    Qt, QTimer, Signal, QPoint, QRectF, QPointF
)
from PySide6.QtGui import (
    QPainter, QColor, QFont, QFontMetrics,
    QPen, QBrush, QLinearGradient, QRadialGradient, QPainterPath, QPolygonF, QRegion, QBitmap, QImage
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider,
    QPushButton, QHBoxLayout, QLabel, QFrame,
    QFileDialog, QComboBox, QMessageBox, QGraphicsOpacityEffect
)
from PySide6.QtGui import QPixmap
import mpv

# --- Style HDR Image Converter (Professionnel) ---
APP_STYLE = """
    QMainWindow, QWidget {
        background-color: #2b2b2b;
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }

    QLabel {
        font-size: 11px;
        color: #CCCCCC;
        font-weight: 400;
    }

    QGroupBox {
        font-size: 11px;
        font-weight: 500;
        color: #E0E0E0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 8px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0px 8px;
        color: #E0E0E0;
    }

    QPushButton {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 6px;
        padding: 8px;
        color: #FFFFFF;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    QPushButton:pressed {
        background-color: rgba(255, 255, 255, 0.2);
    }
    QPushButton:checked {
        background-color: #007ACC;
        border: 1px solid #0096FF;
    }
    QPushButton:checked:hover {
        background-color: #005A9E;
    }

    QSlider::groove:horizontal {
        border: none;
        height: 6px;
        background: #404040;
        border-radius: 3px;
        margin: 0px;
    }
    QSlider::handle:horizontal {
        background: #007ACC;
        border: 2px solid #FFFFFF;
        width: 18px;
        height: 18px;
        border-radius: 9px;
        margin: -6px 0;
    }
    QSlider::handle:horizontal:hover {
        background: #0096FF;
        width: 20px;
        height: 20px;
        border-radius: 10px;
        margin: -7px 0;
    }
    QSlider::add-page:horizontal {
        background: #353535;
        border-radius: 3px;
    }
    QSlider::sub-page:horizontal {
        background: #007ACC;
        border-radius: 3px;
    }

    QComboBox {
        background-color: #3C3C3C;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        padding: 6px 10px;
        color: #E0E0E0;
        font-size: 11px;
    }
    QComboBox:hover {
        background-color: #464646;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    QComboBox::drop-down {
        border: none;
        padding-right: 8px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid #CCCCCC;
        margin-right: 5px;
    }
    QComboBox QAbstractItemView {
        background-color: #3C3C3C;
        color: #E0E0E0;
        selection-background-color: #007ACC;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        padding: 2px;
        outline: none;
    }
    QComboBox QAbstractItemView::item {
        padding: 6px 10px;
        min-height: 22px;
    }
    QComboBox QAbstractItemView::item:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
"""


@lru_cache(maxsize=None)
def _resolve_external_tool(executable_name):
    """Return an absolute path to an external tool (ffmpeg/ffprobe) if available."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []
    if sys.platform == 'win32' and not executable_name.lower().endswith('.exe'):
        candidates.append(f"{executable_name}.exe")
    candidates.append(executable_name)

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return os.path.abspath(resolved)

        local_candidate = os.path.join(base_dir, candidate)
        if os.path.exists(local_candidate):
            return os.path.abspath(local_candidate)

    return None


def _describe_windows_returncode(returncode):
    """Return a human readable explanation for common Windows subprocess errors."""
    if returncode in (3221225781, -1073741515):  # 0xC0000135
        return (
            "Failed to start the executable (code 0xC0000135). "
            "This usually indicates that DLLs for ffmpeg/ffprobe are missing. "
            "Download a static build of ffmpeg from https://www.gyan.dev/ffmpeg/builds/ "
            "and place ffmpeg.exe/ffprobe.exe and their DLLs in the application's folder, "
            "or add the ffmpeg /bin folder to your PATH."
        )
    if returncode in (3221225501, -1073741795):  # 0xC0000025 or similar
        return (
            "The system prevented ffmpeg/ffprobe from running (code 0xC0000025). "
            "Check your antivirus or try running the application with sufficient privileges."
        )
    return None


def _check_ffmpeg_runtime(executable_path):
    """
    Checks if essential DLLs for ffmpeg/ffprobe are present (Windows).

    Returns:
        str | None: error message if a dependency is missing.
    """
    if sys.platform != 'win32' or not executable_path:
        return None

    folder = os.path.dirname(executable_path)
    required_bases = ['avcodec', 'avformat', 'avutil']
    missing = []
    for base in required_bases:
        pattern = os.path.join(folder, f"{base}-*.dll")
        if not glob.glob(pattern):
            missing.append(base)

    if missing:
        return (
            f"ffmpeg/ffprobe found but the following DLLs are missing in the same folder: "
            f"{', '.join(missing)}. Copy all DLLs provided with ffmpeg (from the /bin directory of the archive) "
            "next to the executables, or install a full static build."
        )

    return None


_STEREO_PRIORITY = {
    'none': 0,
    'tab': 1,
    'sbs': 2,
    'mvc': 3,
    'anaglyph': 1,
}


def _classify_stereo_mode(mode_str):
    """Normalise une valeur de stereo_mode vers sbs/tab/mvc/anaglyph."""
    if not mode_str:
        return None

    mode = mode_str.strip().lower()
    mode = mode.replace('-', '_').replace(' ', '_')

    if mode in ('mono', 'left', 'right', 'both', '2d'):
        return None

    if any(keyword in mode for keyword in ('anaglyph', 'cyan', 'magenta', 'red_cyan', 'cyan_red')):
        return 'anaglyph'

    if any(keyword in mode for keyword in (
        'frame_altern', 'framealternate', 'frame_packing', 'frame_sequential',
        'frame_packed', 'view_packed', 'mvc', 'framepacking', 'frameinterleaved',
        'block_lr', 'block_rl', 'packed'
    )):
        return 'mvc'

    if any(keyword in mode for keyword in (
        'top_bottom', 'bottom_top', 'tab', 'over_under', 'under_over',
        'block_tb', 'block_bt', 'topbottom', 'bt', 'tb'
    )):
        return 'tab'

    if any(keyword in mode for keyword in (
        'side_by_side', 'sbs', 'left_right', 'right_left',
        'row_interleaved', 'column_interleaved'
    )):
        return 'sbs'

    return None


def _promote_stereo_mode(result_dict, mode, mark_mvc=False):
    """Met à jour le résultat de détection 3D avec priorité."""
    if not mode:
        return

    priority = _STEREO_PRIORITY.get(mode, 0)
    current_priority = _STEREO_PRIORITY.get(result_dict.get('stereo_mode', 'none'), 0)

    if priority >= current_priority:
        result_dict['stereo_mode'] = mode

    result_dict['is_3d'] = True

    if mark_mvc or mode == 'mvc':
        result_dict['has_mvc_track'] = True


class Video3DAnalyzer:
    """
    Analyzes video files to detect 3D content.
    Uses ffprobe to extract metadata.
    """

    @staticmethod
    def analyze_file(file_path):
        """
        Analyzes a video file and returns its 3D properties.

        Returns:
            dict: {
                'is_3d': bool,
                'stereo_mode': str,  # 'mvc', 'sbs', 'tab', 'none'
                'has_mvc_track': bool,
                'width': int,
                'height': int
            }
        """
        result = {
            'is_3d': False,
            'stereo_mode': 'none',
            'has_mvc_track': False,
            'width': 0,
            'height': 0,
            'analysis_error': None
        }

        try:
            ffprobe_path = _resolve_external_tool('ffprobe')
            if not ffprobe_path:
                raise FileNotFoundError(
                    "ffprobe not found. Add ffprobe to the PATH or place ffprobe.exe "
                    "in the same folder as SyLC_3D_GUI.py."
                )

            runtime_issue = _check_ffmpeg_runtime(ffprobe_path)
            if runtime_issue:
                print(runtime_issue)
                result['analysis_error'] = runtime_issue
                raise FileNotFoundError(runtime_issue)

            cmd = [
                ffprobe_path,
                '-v', 'error',
                '-print_format', 'json',
                '-show_streams',
                '-show_format',
                file_path
            ]

            creationflags = 0
            if sys.platform == 'win32':
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                creationflags=creationflags
            )

            data = json.loads(completed.stdout or "{}")

            # Analyser les streams vidéo
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    result['width'] = stream.get('width', 0)
                    result['height'] = stream.get('height', 0)

                    # PRIORITÉ 1 : Détecter FramePacking par résolution (avant tout le reste)
                    # 1920×2205 = FramePacked 1080p (1920×1080 * 2 + 45 pour le blanking)
                    # 1920×2160 = FramePacked 1080p alternatif
                    # 3840×4320 = FramePacked 2160p (3840×2160 * 2)
                    width = result['width']
                    height = result['height']
                    is_framepacked = (width == 1920 and height in [2205, 2160]) or (width == 3840 and height == 4320)

                    if is_framepacked:
                        result['is_3d'] = True
                        result['has_mvc_track'] = True
                        result['stereo_mode'] = 'mvc'

                    codec_name = (stream.get('codec_name') or '').lower()
                    profile = (stream.get('profile') or '').lower()

                    # PRIORITÉ 2 : Codec et profil MVC
                    if codec_name in ('mvc', 'h264'):
                        if 'stereo' in profile or 'mvc' in profile:
                            _promote_stereo_mode(result, 'mvc', mark_mvc=True)

                    disposition = stream.get('disposition') or {}
                    if isinstance(disposition, dict) and disposition.get('dependent'):
                        _promote_stereo_mode(result, 'mvc', mark_mvc=True)

                    # PRIORITÉ 3 : Tags et side_data (seulement si pas déjà MVC par résolution)
                    if not is_framepacked:
                        # Inspecter les side_data pour les informations de stéréoscopie
                        for side_data in stream.get('side_data_list', []):
                            side_type = (
                                side_data.get('type')
                                or side_data.get('side_data_type')
                                or ''
                            ).lower()
                            if 'stereo3d' in side_type or 'stereo_3d' in side_type:
                                detected = (
                                    side_data.get('stereo_mode')
                                    or side_data.get('type')
                                    or side_data.get('layout')
                                    or side_data.get('view')
                                    or ''
                                )
                                classified = _classify_stereo_mode(detected)
                                if classified == 'mvc':
                                    _promote_stereo_mode(result, 'mvc', mark_mvc=True)
                                elif classified:
                                    _promote_stereo_mode(result, classified)

                        # Détecter les tags de stéréoscopie (Matroska, MakeMKV…)
                        tags = stream.get('tags') or {}
                        for key, value in tags.items():
                            if key.lower().startswith('stereo'):
                                classified = _classify_stereo_mode(value)
                                if classified:
                                    _promote_stereo_mode(result, classified)

            # Vérifier si le fichier contient un stream MVC séparé
            if not result['has_mvc_track']:
                for stream in data.get('streams', []):
                    if stream.get('codec_name') == 'mvc':
                        result['is_3d'] = True
                        result['has_mvc_track'] = True
                        result['stereo_mode'] = 'mvc'
                        break

        except subprocess.CalledProcessError as e:
            error_output = (e.stderr or e.stdout or '').strip()
            message = error_output if error_output else str(e)
            print(f"Error during 3D analysis (ffprobe): {message}")

            hint = _describe_windows_returncode(e.returncode)
            if hint:
                print(hint)
                result['analysis_error'] = hint
            else:
                result['analysis_error'] = message

            # If ffprobe is not available, try basic detection by name
            filename = os.path.basename(file_path).lower()
            if '3d' in filename or 'sbs' in filename or 'hsbs' in filename:
                result['is_3d'] = True
                result['stereo_mode'] = 'sbs'
            elif '3d' in filename and ('tab' in filename or 'htab' in filename):
                result['is_3d'] = True
                result['stereo_mode'] = 'tab'

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error during 3D analysis: {e}")
            result['analysis_error'] = str(e)
            # If ffprobe is not available, try basic detection by name
            filename = os.path.basename(file_path).lower()
            if '3d' in filename or 'sbs' in filename or 'hsbs' in filename:
                result['is_3d'] = True
                result['stereo_mode'] = 'sbs'
            elif '3d' in filename and ('tab' in filename or 'htab' in filename):
                result['is_3d'] = True
                result['stereo_mode'] = 'tab'

        return result


# ThreadPool GLOBAL pour extraction parallèle de thumbnails (max 2 workers)
_thumbnail_executor = ThreadPoolExecutor(max_workers=2)

def _extract_thumbnail_ffmpeg(video_file, time_pos):
    """Extract a thumbnail with ffmpeg (worker function for ThreadPoolExecutor)."""
    try:
        ffmpeg_path = _resolve_external_tool('ffmpeg')
        if not ffmpeg_path:
            return None

        temp_file = os.path.join(tempfile.gettempdir(), f"preview_{int(time.time()*1000000)}.jpg")

        # ultra-fast ffmpeg: -ss BEFORE -i, direct JPEG, 120x68, 2s timeout
        cmd = [
            ffmpeg_path,
            '-ss', str(time_pos),
            '-i', video_file,
            '-frames:v', '1',
            '-vf', 'scale=120:-1',
            '-q:v', '8',
            '-y',
            temp_file
        ]

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            creationflags=creationflags
        )

        if result.returncode == 0 and os.path.exists(temp_file):
            return temp_file
        return None
    except:
        return None




class PreviewTooltip(QLabel):
    """Widget to display the frame preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(120, 68)  # 16:9 aspect ratio
        self.setStyleSheet("""
            QLabel {
                background: #1a1a1a;
                border: 2px solid #007ACC;
                border-radius: 4px;
            }
        """)
        self.setScaledContents(True)
        self.hide()


class TimeSlider(QSlider):
    """Custom slider with time preview on hover."""

    preview_requested = Signal(float)  # Signal to request a frame preview
    extraction_done = Signal(float, str)  # time_pos, temp_file_path

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self._hover_time = 0
        self._is_hovering = False
        self._player = None
        self._preview_widget = PreviewTooltip(self)
        self._last_preview_time = -99
        self._preview_cache = {}  # LRU cache (100 frames)
        self._video_file = None
        self._extraction_timer = QTimer(self)  # Timer for debouncing
        self._extraction_timer.setSingleShot(True)
        self._extraction_timer.timeout.connect(self._do_extraction)
        self._pending_time = 0
        self._pending_mouse_x = 0
        self.extraction_done.connect(self._on_extraction_done)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._is_hovering = True
        self.update()

    def set_player(self, player):
        """Associate the MPV player for frame capture."""
        self._player = player

    def mouseMoveEvent(self, event):
        # Calculate the time corresponding to the mouse position
        if self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            value = int((pos / self.width()) * self.maximum())
            self._hover_time = max(0, min(value, self.maximum()))
            self._is_hovering = True

            # Show a tooltip with the time
            s = int(self._hover_time)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            time_str = f"{h:02}:{m:02}:{s:02}"
            self.setToolTip(time_str)

            # Request a preview (extraction from secondary MPV)
            # Threshold reduced to 0.5s for better responsiveness
            if self._video_file and abs(self._hover_time - self._last_preview_time) > 0.5:
                self._last_preview_time = self._hover_time
                self._request_on_demand_preview(self._hover_time, pos)

            # Force redraw to display the preview
            self.update()

        super().mouseMoveEvent(event)

    def set_video_file(self, video_path, duration):
        """Set the video file for on-demand extraction."""
        self._video_file = video_path
        self._preview_cache.clear()

    def _request_on_demand_preview(self, time_pos, mouse_x):
        """Request extraction (with 100ms debouncing)."""
        # Round for LRU cache (1-second tolerance)
        cache_key = round(time_pos)

        # Check the cache
        if cache_key in self._preview_cache:
            pixmap = self._preview_cache[cache_key]
            if not pixmap.isNull():
                print(f"[CACHE] Hit for {time_pos:.1f}s")
                self._preview_widget.setPixmap(pixmap)
                self._show_preview_at(mouse_x)
                return

        # Store for extraction after debouncing
        self._pending_time = time_pos
        self._pending_mouse_x = mouse_x
        self._extraction_timer.start(100)  # 100ms debouncing

    def _do_extraction(self):
        """Start ffmpeg extraction in the background (Thread Pool)."""
        time_pos = self._pending_time
        mouse_x = self._pending_mouse_x

        # Start asynchronous extraction in the ThreadPool
        future = _thumbnail_executor.submit(_extract_thumbnail_ffmpeg, self._video_file, time_pos)
        future.add_done_callback(lambda f: self._handle_extraction_result(f, time_pos, mouse_x))

    def _handle_extraction_result(self, future, time_pos, mouse_x):
        """Callback when extraction is finished."""
        try:
            temp_file = future.result()
            if temp_file:
                self.extraction_done.emit(time_pos, temp_file)
        except:
            pass

    def _on_extraction_done(self, time_pos, temp_file):
        """Process the extraction result (in the main Qt thread)."""
        try:
            cache_key = round(time_pos)

            # Load the image
            pixmap = QPixmap(temp_file)
            if not pixmap.isNull():
                # Cache it
                if len(self._preview_cache) > 100:
                    oldest = next(iter(self._preview_cache))
                    del self._preview_cache[oldest]
                self._preview_cache[cache_key] = pixmap

                # Show if still hovering
                if self._is_hovering and abs(time_pos - self._hover_time) < 3:
                    self._preview_widget.setPixmap(pixmap)
                    self._show_preview_at(self._pending_mouse_x)

            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
        except Exception as e:
            print(f"[ERROR] {e}")

    def _on_thumbnail_ready(self, time_pos, pixmap, mouse_x):
        """Thumbnail ready - display and cache."""
        cache_key = round(time_pos)

        # LRU Cache: limit to 50 frames (more cache = less extraction)
        if len(self._preview_cache) > 50:
            oldest_key = next(iter(self._preview_cache))
            del self._preview_cache[oldest_key]

        self._preview_cache[cache_key] = pixmap
        print(f"[CACHE] Added {time_pos:.1f}s to cache (total: {len(self._preview_cache)})")

        # ALWAYS show if hovering (wide tolerance of 5 seconds)
        if self._is_hovering and abs(time_pos - self._hover_time) < 5:
            print(f"[DISPLAY] Displaying thumbnail at {time_pos:.1f}s (hover: {self._hover_time:.1f}s)")
            self._preview_widget.setPixmap(pixmap)
            self._show_preview_at(mouse_x)
        else:
            print(f"[DISPLAY] NOT displaying: hovering={self._is_hovering}, diff={abs(time_pos - self._hover_time):.1f}s")

    def _show_preview_at(self, mouse_x):
        """Display the preview widget."""
        # Position the tooltip above the mouse
        global_pos = self.mapToGlobal(QPoint(int(mouse_x), 0))
        tooltip_x = global_pos.x() - self._preview_widget.width() // 2
        tooltip_y = global_pos.y() - self._preview_widget.height() - 10

        self._preview_widget.move(tooltip_x, tooltip_y)
        self._preview_widget.show()
        self._preview_widget.raise_()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._is_hovering = False
        self.setToolTip("")
        self._preview_widget.hide()
        self.update()

    def mousePressEvent(self, event):
        """Allows clicking directly on the timeline to change position."""
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            value = int((pos / self.width()) * self.maximum())
            self.setValue(max(0, min(value, self.maximum())))
            self.sliderMoved.emit(self.value())
        super().mousePressEvent(event)

    def paintEvent(self, event):
        # First, draw the normal slider
        super().paintEvent(event)

        # Then, draw the preview ON TOP
        if self._is_hovering and self.maximum() > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            # Position of the indicator
            preview_x = int((self._hover_time / self.maximum()) * self.width())

            # Vertical preview line - industrial style
            painter.setPen(QPen(QColor(0, 122, 204, 180), 2))
            painter.drawLine(preview_x, 0, preview_x, self.height())

            # Simple circle without effect
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 122, 204, 220)))
            painter.drawEllipse(QPointF(preview_x, self.height() // 2), 5, 5)


class IconButton(QPushButton):
    """Professional HDR Converter style button."""

    def __init__(self, icon_type, is_primary=False, parent=None):
        super().__init__(parent)
        self.icon_type = icon_type
        self.is_primary = is_primary

        if is_primary:
            self.setFixedSize(42, 42)
            self.setStyleSheet("""
                QPushButton {
                    background-color: #007ACC;
                    border: 1px solid #0096FF;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #005A9E;
                }
                QPushButton:pressed {
                    background-color: #004578;
                }
            """)
        else:
            self.setFixedSize(36, 36)
            self.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.08);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.15);
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 0.2);
                }
                QPushButton:checked {
                    background-color: #007ACC;
                    border: 1px solid #0096FF;
                }
            """)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Couleur selon état - SIMPLE
        color = QColor(255, 255, 255, 230)

        painter.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Dessiner l'icône selon le type
        if self.is_primary:
            center_x, center_y = 21, 21
        else:
            center_x, center_y = 18, 18

        if self.icon_type == 'play':
            path = QPainterPath()
            path.moveTo(center_x - 5, center_y - 7)
            path.lineTo(center_x + 8, center_y)
            path.lineTo(center_x - 5, center_y + 7)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color))

        elif self.icon_type == 'pause':
            painter.drawRoundedRect(QRectF(center_x - 6, center_y - 7, 4, 14), 1.5, 1.5)
            painter.drawRoundedRect(QRectF(center_x + 2, center_y - 7, 4, 14), 1.5, 1.5)

        elif self.icon_type == 'folder':
            path = QPainterPath()
            path.moveTo(center_x - 8, center_y - 5)
            path.lineTo(center_x - 3, center_y - 5)
            path.lineTo(center_x - 1, center_y - 7)
            path.lineTo(center_x + 8, center_y - 7)
            path.lineTo(center_x + 8, center_y + 7)
            path.lineTo(center_x - 8, center_y + 7)
            path.closeSubpath()
            painter.strokePath(path, QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        elif self.icon_type == 'fullscreen':
            # Coins expand - fins et discrets
            size = 5
            gap = 8
            painter.drawLine(QPointF(center_x - gap, center_y - gap), QPointF(center_x - gap, center_y - gap + size))
            painter.drawLine(QPointF(center_x - gap, center_y - gap), QPointF(center_x - gap + size, center_y - gap))

            painter.drawLine(QPointF(center_x + gap, center_y - gap), QPointF(center_x + gap, center_y - gap + size))
            painter.drawLine(QPointF(center_x + gap, center_y - gap), QPointF(center_x + gap - size, center_y - gap))

            painter.drawLine(QPointF(center_x - gap, center_y + gap), QPointF(center_x - gap, center_y + gap - size))
            painter.drawLine(QPointF(center_x - gap, center_y + gap), QPointF(center_x - gap + size, center_y + gap))

            painter.drawLine(QPointF(center_x + gap, center_y + gap), QPointF(center_x + gap, center_y + gap - size))
            painter.drawLine(QPointF(center_x + gap, center_y + gap), QPointF(center_x + gap - size, center_y + gap))

        elif self.icon_type == 'exit_fullscreen':
            # Coins contract - fins et discrets
            size = 5
            gap = 8
            painter.drawLine(QPointF(center_x - gap + size, center_y - gap + size), QPointF(center_x - gap + size, center_y - gap))
            painter.drawLine(QPointF(center_x - gap + size, center_y - gap + size), QPointF(center_x - gap, center_y - gap + size))

            painter.drawLine(QPointF(center_x + gap - size, center_y - gap + size), QPointF(center_x + gap - size, center_y - gap))
            painter.drawLine(QPointF(center_x + gap - size, center_y - gap + size), QPointF(center_x + gap, center_y - gap + size))

            painter.drawLine(QPointF(center_x - gap + size, center_y + gap - size), QPointF(center_x - gap + size, center_y + gap))
            painter.drawLine(QPointF(center_x - gap + size, center_y + gap - size), QPointF(center_x - gap, center_y + gap - size))

            painter.drawLine(QPointF(center_x + gap - size, center_y + gap - size), QPointF(center_x + gap - size, center_y + gap))
            painter.drawLine(QPointF(center_x + gap - size, center_y + gap - size), QPointF(center_x + gap, center_y + gap - size))

        elif self.icon_type == '3d':
            # Icône 3D stylisée - bien centrée
            font = QFont('Segoe UI', 11, QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.setPen(QPen(color, 1))
            # Utiliser la taille complète du bouton pour centrer
            painter.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, '3D')

        elif self.icon_type == 'volume':
            # Haut-parleur - proportionné
            path = QPainterPath()
            path.moveTo(center_x - 5, center_y - 3)
            path.lineTo(center_x - 2, center_y - 3)
            path.lineTo(center_x + 3, center_y - 7)
            path.lineTo(center_x + 3, center_y + 7)
            path.lineTo(center_x - 2, center_y + 3)
            path.lineTo(center_x - 5, center_y + 3)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color))

            # Ondes sonores - fines
            painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(QRectF(center_x + 5, center_y - 5, 7, 10), 16 * -60, 16 * 120)
            painter.drawArc(QRectF(center_x + 8, center_y - 8, 10, 16), 16 * -60, 16 * 120)


class ControlsOverlay(QWidget):
    """
    Premium widget for playback controls.
    Elegant design with sophisticated animations and visual effects.
    """
    # Signals emitted on user interaction
    play_toggled = Signal()
    fullscreen_toggled = Signal()
    seeked = Signal(float)
    volume_changed = Signal(int)
    file_opened = Signal()
    stereo_mode_changed = Signal(str)
    mode_3d_toggled = Signal(bool)
    audio_track_changed = Signal(int)  # Signal for audio track change

    def __init__(self, parent=None):
        super().__init__(parent)
        # Disable auto-fill to draw manually
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        # Use QGraphicsOpacityEffect for 50% transparency
        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(0.5)  # 50% transparent
        self.setGraphicsEffect(opacity_effect)

        # Transparent stylesheet - we draw in paintEvent
        self.setStyleSheet("""
            ControlsOverlay {
                background: transparent;
            }
            ControlsOverlay > QWidget {
                background: transparent;
            }
        """)

        # --- Button creation - Play/Pause as main button (blue) ---
        self.play_pause_button = IconButton('play', is_primary=True)
        self.play_pause_button.setCheckable(False)
        self.play_pause_button.setToolTip("Play / Pause (Space)")

        self.fullscreen_button = IconButton('fullscreen', is_primary=False)
        self.fullscreen_button.setToolTip("Fullscreen (F)")

        self.open_file_button = IconButton('folder', is_primary=False)
        self.open_file_button.setToolTip("Open a file")

        self.mode_3d_button = IconButton('3d', is_primary=False)
        self.mode_3d_button.setCheckable(True)
        self.mode_3d_button.setToolTip("3D Mode (3)")

        # Enhanced control widgets
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setMaximumWidth(120)
        self.volume_slider.setMinimumWidth(80)
        self.volume_slider.setToolTip("Volume")

        self.time_slider = TimeSlider(Qt.Orientation.Horizontal)

        # HDR Converter style labels
        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet("""
            font-size: 11px;
            font-weight: 400;
            color: #E0E0E0;
            padding: 0px 6px;
        """)

        self.duration_label = QLabel("00:00:00")
        self.duration_label.setStyleSheet("""
            font-size: 11px;
            font-weight: 400;
            color: #CCCCCC;
            padding: 0px 6px;
        """)

        # 3D Combobox with premium style
        self.stereo_mode_combo = QComboBox()
        self.stereo_mode_combo.addItems([
            "Auto",
            "MVC",
            "Side-by-Side",
            "Top-Bottom",
            "2D (Mono)"
        ])
        self.stereo_mode_combo.setMinimumWidth(160)
        self.stereo_mode_combo.setToolTip("Stereoscopic format")

        # Audio track combobox
        self.audio_track_combo = QComboBox()
        self.audio_track_combo.addItem("Audio: No track")
        self.audio_track_combo.setMinimumWidth(200)
        self.audio_track_combo.setToolTip("Select audio track")
        self.audio_track_combo.setEnabled(False)  # Disabled by default

        # Flat style 3D info label
        self.info_3d_label = QLabel("")
        self.info_3d_label.setStyleSheet("""
            color: #E0E0E0;
            font-weight: 400;
            font-size: 11px;
            padding: 6px 12px;
            background-color: rgba(0, 122, 204, 0.15);
            border-radius: 4px;
            border: 1px solid rgba(0, 122, 204, 0.3);
        """)
        self.info_3d_label.hide()  # Hidden by default

        # --- Ultra-compact layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 8, 16, 8)
        main_layout.setSpacing(6)
        main_layout.addStretch()

        # Time bar with optimal spacing
        time_layout = QHBoxLayout()
        time_layout.setSpacing(12)
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider, 1)
        time_layout.addWidget(self.duration_label)

        # Controls bar - balanced and compact layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        # Left group - file and 3D + info label + audio
        left_group = QHBoxLayout()
        left_group.setSpacing(8)
        left_group.addWidget(self.open_file_button)
        left_group.addWidget(self.mode_3d_button)
        left_group.addWidget(self.stereo_mode_combo)
        left_group.addWidget(self.audio_track_combo)
        left_group.addWidget(self.info_3d_label)

        # Center - playback
        center_group = QHBoxLayout()
        center_group.addWidget(self.play_pause_button)

        # Right group - volume and fullscreen
        right_group = QHBoxLayout()
        right_group.setSpacing(8)
        right_group.addWidget(self.volume_slider)
        right_group.addWidget(self.fullscreen_button)

        controls_layout.addLayout(left_group)
        controls_layout.addStretch(1)
        controls_layout.addLayout(center_group)
        controls_layout.addStretch(1)
        controls_layout.addLayout(right_group)

        main_layout.addLayout(time_layout)
        main_layout.addLayout(controls_layout)

        # --- Connections ---
        self.play_pause_button.clicked.connect(self.play_toggled)
        self.fullscreen_button.clicked.connect(self.fullscreen_toggled)
        self.open_file_button.clicked.connect(self.file_opened)
        self.volume_slider.valueChanged.connect(self.volume_changed)
        self.time_slider.sliderMoved.connect(lambda pos: self.seeked.emit(pos))
        self.mode_3d_button.toggled.connect(self.mode_3d_toggled)
        self.stereo_mode_combo.currentTextChanged.connect(self._on_stereo_mode_changed)
        self.audio_track_combo.currentIndexChanged.connect(self._on_audio_track_changed)

    def show_animated(self):
        """Shows the widget without animation to avoid flickering."""
        self.show()

    def hide_animated(self):
        """Hides the widget without animation to avoid flickering."""
        self.hide()

    def set_paused(self, is_paused):
        """Updates the play/pause icon."""
        # Change the icon type
        self.play_pause_button.icon_type = 'play' if is_paused else 'pause'
        self.play_pause_button.update()

    def set_fullscreen_icon(self, is_fullscreen):
        """Updates the fullscreen mode icon."""
        self.fullscreen_button.icon_type = 'exit_fullscreen' if is_fullscreen else 'fullscreen'
        self.fullscreen_button.update()

    def set_duration(self, seconds):
        if seconds is not None:
            self.time_slider.setRange(0, int(seconds))
        self.duration_label.setText(self.format_time(seconds))

    def set_time(self, seconds):
        if seconds is not None and not self.time_slider.isSliderDown():
            self.time_slider.setValue(int(seconds))
        self.time_label.setText(self.format_time(seconds))

    def format_time(self, seconds):
        if seconds is None:
            return "00:00:00"
        s = int(seconds)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02}:{m:02}:{s:02}"

    def set_3d_info(self, text):
        """Displays information about the 3D mode."""
        if text:
            self.info_3d_label.setText(text)
            self.info_3d_label.show()
        else:
            self.info_3d_label.hide()

    def enable_3d_controls(self, enabled):
        """Enables or disables the 3D controls."""
        self.mode_3d_button.setEnabled(enabled)
        self.stereo_mode_combo.setEnabled(enabled)

    def _on_stereo_mode_changed(self, text):
        """Emits the signal with the selected mode."""
        mode_map = {
            "Auto": "auto",
            "MVC": "mvc",
            "Side-by-Side": "sbs",
            "Top-Bottom": "tab",
            "2D (Mono)": "mono"
        }
        self.stereo_mode_changed.emit(mode_map.get(text, "auto"))

    def _on_audio_track_changed(self, index):
        """Emits the signal with the ID of the selected audio track."""
        if index > 0:  # Index 0 is "No track"
            # Retrieve the track ID stored in the user data
            track_id = self.audio_track_combo.itemData(index)
            if track_id is not None:
                self.audio_track_changed.emit(track_id)

    def update_audio_tracks(self, tracks):
        """
        Updates the list of available audio tracks.

        Args:
            tracks: List of tuples (id, title, language)
        """
        # Block signals during update
        self.audio_track_combo.blockSignals(True)

        # Clear the combobox
        self.audio_track_combo.clear()

        if not tracks or len(tracks) == 0:
            # No audio tracks
            self.audio_track_combo.addItem("Audio: No track")
            self.audio_track_combo.setEnabled(False)
        else:
            # Add the tracks
            self.audio_track_combo.addItem("Audio: Select a track")
            for track_id, title, lang in tracks:
                # Format the track name
                track_name = f"Audio: {title}"
                if lang:
                    track_name += f" ({lang})"

                self.audio_track_combo.addItem(track_name, track_id)

            self.audio_track_combo.setEnabled(True)
            # Select the first track by default (index 1)
            self.audio_track_combo.setCurrentIndex(1)

        # Unblock signals
        self.audio_track_combo.blockSignals(False)

    def paintEvent(self, event):
        """Draw the rounded background with true transparent corners."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create a rounded path
        rect = QRectF(0, 0, self.width(), self.height())
        path = QPainterPath()
        path.addRoundedRect(rect, 12, 12)

        # Opaque background (transparency is handled by QGraphicsOpacityEffect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(43, 43, 43, 255)))
        painter.drawPath(path)

        # Border
        painter.setPen(QPen(QColor(255, 255, 255, 38), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def resizeEvent(self, event):
        """Create a bitmap mask for truly transparent corners."""
        super().resizeEvent(event)

        # Create an image to draw the mask with antialiasing
        image = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)

        # Draw the rounded rectangle
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.GlobalColor.white)
        painter.setPen(Qt.PenStyle.NoPen)

        # Rounded rectangle = visible part
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 12, 12)
        painter.fillPath(path, Qt.GlobalColor.white)
        painter.end()

        # Convert to bitmap and apply the mask
        mask = QBitmap.fromImage(image.createAlphaMask())
        self.setMask(mask)



class InfoOverlay(QWidget):
    """Elegant welcome message in the center of the window - clickable to open a file."""

    file_clicked = Signal()

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        # REMOVE WA_TransparentForMouseEvents to make it clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Subtle pulsing animation
        self._opacity = 1.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._update_pulse)
        self._pulse_timer.start(30)
        self._pulse_direction = -1
        self._pulse_value = 0.0

    def _update_pulse(self):
        """Soft pulse animation."""
        self._pulse_value += self._pulse_direction * 0.02
        if self._pulse_value <= 0.0:
            self._pulse_value = 0.0
            self._pulse_direction = 1
        elif self._pulse_value >= 1.0:
            self._pulse_value = 1.0
            self._pulse_direction = -1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        center_x = self.width() // 2
        center_y = self.height() // 2 - 40

        # Simple folder icon - flat
        icon_color = QColor(0, 122, 204, 200)
        painter.setPen(QPen(icon_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        path.moveTo(center_x - 30, center_y - 12)
        path.lineTo(center_x - 10, center_y - 12)
        path.lineTo(center_x - 6, center_y - 20)
        path.lineTo(center_x + 30, center_y - 20)
        path.lineTo(center_x + 30, center_y + 20)
        path.lineTo(center_x - 30, center_y + 20)
        path.closeSubpath()
        painter.strokePath(path, QPen(icon_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        # Main text - flat
        text_y = center_y + 60
        font = QFont('Segoe UI', 14, QFont.Weight.Normal)
        painter.setFont(font)

        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(self.text)

        painter.setPen(QColor(224, 224, 224))
        painter.drawText(int(center_x - text_width/2), int(text_y), self.text)

        # Subtitle
        subtitle = "MP4, MKV, AVI, ISO (3D & HDR)"
        subtitle_font = QFont('Segoe UI', 10, QFont.Weight.Normal)
        painter.setFont(subtitle_font)
        fm2 = QFontMetrics(subtitle_font)
        subtitle_width = fm2.horizontalAdvance(subtitle)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(int(center_x - subtitle_width/2), int(text_y + 26), subtitle)

        # Application title - simple
        app_title = "SyLC Player"
        title_font = QFont('Segoe UI', 24, QFont.Weight.Normal)
        painter.setFont(title_font)
        fm3 = QFontMetrics(title_font)
        title_width = fm3.horizontalAdvance(app_title)

        painter.setPen(QColor(224, 224, 224))
        painter.drawText(int(center_x - title_width/2), 60, app_title)

        # Version - discreet
        edition = "3D Edition"
        edition_font = QFont('Segoe UI', 9, QFont.Weight.Normal)
        painter.setFont(edition_font)
        fm4 = QFontMetrics(edition_font)
        edition_width = fm4.horizontalAdvance(edition)
        painter.setPen(QColor(0, 122, 204, 180))
        painter.drawText(int(center_x - edition_width/2), 78, edition)

    def mousePressEvent(self, event):
        """Clicking on the overlay opens the file dialog."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.file_clicked.emit()
        super().mousePressEvent(event)


class PlayerWindow(QMainWindow):
    """
    The main application window.
    It contains the video widget and manages overlays (controls, messages).
    Full 3D support with framepacking.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SyLC Player - Premium 3D/HDR Edition")
        self.setStyleSheet(APP_STYLE)
        self.setMinimumSize(960, 540)

        # Player state
        self.is_playing = False
        self.has_media = False
        self.current_file_path = None

        # 3D state
        self.video_3d_info = None
        self.is_3d_enabled = False
        self.current_stereo_mode = 'auto'
        self.is_3d_capable = False

        # Window and layout configuration
        container = QWidget()
        self.setCentralWidget(container)
        self.layout = QVBoxLayout(container)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Video widget (the drawing surface for mpv)
        self.video_widget = QWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_widget)

        # Controls overlay (absolute positioning)
        self.controls_overlay = ControlsOverlay(container)
        self.controls_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Information overlay (absolute positioning) - CLICKABLE
        self.info_overlay = InfoOverlay("Click here or drop a file to start", container)

        # Accept drag and drop
        self.setAcceptDrops(True)

        # Timer to hide controls
        self.hide_controls_timer = QTimer(self)
        self.hide_controls_timer.setInterval(2500)
        self.hide_controls_timer.timeout.connect(self.hide_controls)
        self.setMouseTracking(True)
        container.setMouseTracking(True)
        self.video_widget.setMouseTracking(True)
        self.controls_overlay.setMouseTracking(True)

        self._initialize_player()
        self._connect_signals()

        self.update_ui_state()

        # Check if 3D Vision is available
        self._check_3d_vision_availability()

    def _check_3d_vision_availability(self):
        """Checks if Nvidia 3D Vision is active."""
        try:
            # Check if the nvstview.exe process is running (3D Vision emitter)
            result = subprocess.run(['tasklist'], capture_output=True, text=True)
            if 'nvstview.exe' in result.stdout or 'nvspcaps.exe' in result.stdout:
                self.is_3d_capable = True
                self.show_3d_notification("✓ Nvidia 3D Vision detected and active", success=True)
            else:
                self.is_3d_capable = False
                self.controls_overlay.mode_3d_button.setEnabled(False)
                self.show_3d_notification("⚠ 3D Vision not detected. Check 3DFixManager", success=False)
        except Exception as e:
            self.is_3d_capable = False
            self.controls_overlay.mode_3d_button.setEnabled(False)
            print(f"Could not check for 3D Vision: {e}")

    def show_3d_notification(self, message, success=True, permanent=False):
        """Displays a notification about 3D mode."""
        self.controls_overlay.set_3d_info(message)
        if success:
            self.controls_overlay.info_3d_label.setStyleSheet("""
                color: #E0E0E0;
                font-weight: 400;
                font-size: 11px;
                padding: 6px 12px;
                background-color: rgba(0, 122, 204, 0.15);
                border-radius: 4px;
                border: 1px solid rgba(0, 122, 204, 0.3);
            """)
        else:
            self.controls_overlay.info_3d_label.setStyleSheet("""
                color: #E0E0E0;
                font-weight: 400;
                font-size: 11px;
                padding: 6px 12px;
                background-color: rgba(255, 165, 0, 0.15);
                border-radius: 4px;
                border: 1px solid rgba(255, 165, 0, 0.3);
            """)

        # Only hide if not permanent
        if not permanent:
            QTimer.singleShot(5000, lambda: self.controls_overlay.set_3d_info(""))

    def _initialize_player(self):
        """Configures and initializes the mpv instance with optimal settings."""
        QTimer.singleShot(100, self._setup_mpv_player)

    def _setup_mpv_player(self):
        """Advanced MPV configuration with 3D support."""
        win_id = str(int(self.video_widget.winId()))

        # "High-end" base configuration + 3D
        mpv_config = {
            'wid': win_id,
            'vo': 'gpu-next',
            'gpu-api': 'd3d11',  # Critical for 3D Vision
            'hwdec': 'auto-copy',  # Best for 3D
            'target-trc': 'auto',
            'target-prim': 'auto',
            'target-peak': 'auto',
            'tone-mapping': 'auto',
            'hdr-compute-peak': 'yes',
            'input-default-bindings': True,
            # --- Cache optimizations to reduce lag (modern mpv options) ---
            'cache': 'yes',
            'demuxer-readahead-secs': 5, # Read 5 seconds ahead (for lag)
            'demuxer-max-bytes': '500M', # 500MB max for the demuxer
            'demuxer-max-back-bytes': '250M', # 250MB max for the demuxer backbuffer
            'osc': False,
            'volume': 100,
            'mute': 'no',

            # CRITICAL PARAMETERS FOR 3D
            'video-sync': 'display-resample',  # Perfect sync for 24fps
            'video-output-levels': 'full',  # Full range for HDMI
            'interpolation': 'no',  # Disable for 3D
            'blend-subtitles': 'video',  # Subtitles in the video stream

            # FRAMEPACKING 1920x2205@24fps for projector
            # 'override-display-fps': 24,  # Removed - Causes acceleration at 60fps
            # --- HDR/3D performance optimizations ---
            'dither-depth': 'auto',  # Optimizes dithering quality
            'interpolation': 'no',  # Maintained for 3D
            'tscale': 'oversample', # Better temporal scaling quality
            'interpolation-threshold': 0.001, # Very low tolerance for interpolation
            'video-sync': 'display-resample',  # Perfect sync for 24/60fps
            'vd-lavc-threads': 8, # Use more threads for decoding (if available)
            'gpu-shader-cache': 'yes', # Cache shaders for faster startup
        }

        try:
            self.player = mpv.MPV(**mpv_config)

            # Enable logging for debugging
            self.player['msg-level'] = 'all=info'

        except Exception as e:
            QMessageBox.critical(
                self,
                "MPV Error",
                f"Error initializing mpv: {e}\n\n"
                "Make sure mpv-2.dll is in the same folder."
            )
            sys.exit(1)

        # Observer les propriétés de mpv
        self.player.observe_property('time-pos', self.on_time_update)
        self.player.observe_property('duration', self.on_duration_change)
        self.player.observe_property('pause', self.on_pause_state_change)

        # Connecter le player au time_slider pour la prévisualisation
        self.controls_overlay.time_slider.set_player(self.player)

    def _connect_signals(self):
        """Connects UI signals to player commands."""
        self.controls_overlay.play_toggled.connect(self.toggle_play)
        self.controls_overlay.fullscreen_toggled.connect(self.toggle_fullscreen)
        self.controls_overlay.volume_changed.connect(lambda v: setattr(self.player, 'volume', v))
        self.controls_overlay.seeked.connect(lambda t: setattr(self.player, 'time-pos', t))
        self.controls_overlay.file_opened.connect(self.open_file_dialog)
        self.controls_overlay.mode_3d_toggled.connect(self.toggle_3d_mode)
        self.controls_overlay.stereo_mode_changed.connect(self.change_stereo_mode)
        self.controls_overlay.audio_track_changed.connect(self.change_audio_track)
        # Connect the click on the info overlay
        self.info_overlay.file_clicked.connect(self.open_file_dialog)

    def update_ui_state(self):
        """Updates the visibility of overlays based on the state."""
        if not self.has_media:
            self.info_overlay.show()
            self.controls_overlay.hide()
        else:
            self.info_overlay.hide()
            self.show_controls()

    def analyze_and_configure_3d(self, file_path):
        """Analyzes the file and automatically configures the 3D mode."""
        print(f"Analyzing 3D for file: {file_path}")

        self.video_3d_info = Video3DAnalyzer.analyze_file(file_path)

        if self.video_3d_info.get('analysis_error'):
            print(f"3D analysis warning: {self.video_3d_info['analysis_error']}")
            self.show_3d_notification(
                "3D analysis via ffprobe failed. Switching to Auto mode.",
                success=False,
                permanent=False
            )

        if self.video_3d_info['is_3d'] and self.video_3d_info['stereo_mode'] != 'none':
            stereo_mode = self.video_3d_info['stereo_mode']
            print(f"3D content detected: {stereo_mode}")

            # Enable 3D controls
            self.controls_overlay.enable_3d_controls(True)

            # Automatically select the correct mode
            mode_index = {
                'mvc': 1,  # MVC
                'sbs': 2,  # Side-by-Side
                'tab': 3,  # Top-Bottom
            }.get(stereo_mode, 0)  # Auto by default

            self.controls_overlay.stereo_mode_combo.setCurrentIndex(mode_index)

            # PERMANENT information message
            mode_names = {
                'mvc': 'MVC',
                'sbs': 'Side-by-Side',
                'tab': 'Top-Bottom'
            }
            self.show_3d_notification(
                f"3D File: {mode_names.get(stereo_mode, stereo_mode.upper())}",
                success=True,
                permanent=True
            )

            # Automatically enable 3D mode if it's MVC AND the monitor is 3D capable
            if stereo_mode == 'mvc' and self.is_3d_capable:
                self.controls_overlay.mode_3d_button.setChecked(True)

        else:
            print("2D content detected")
            self.controls_overlay.enable_3d_controls(False)
            self.show_3d_notification("2D content detected", success=True, permanent=True)

    def configure_3d_output(self, enable_3d=True, stereo_mode='auto'):
        """
        Configures the 3D output of mpv.

        Args:
            enable_3d: Enables or disables 3D mode
            stereo_mode: 'auto', 'mvc', 'sbs', 'tab', 'mono'
        """
        if not enable_3d:
            # Disable 3D - mono mode
            try:
                self.player['lavfi-complex'] = ''
                print("3D mode disabled")
            except:
                pass
            return

        print(f"3D configuration: mode={stereo_mode}")

        # Configuration based on content type
        if stereo_mode == 'mvc' or (stereo_mode == 'auto' and
                                    self.video_3d_info and
                                    self.video_3d_info.get('stereo_mode') == 'mvc'):
            # For MVC, let mpv handle it natively
            # Framepacking output will be handled by 3D Vision
            try:
                # Ensure that hardware decoding does not interfere
                self.player['hwdec'] = 'no'  # Software decoding for MVC

                # CRITICAL CONFIGURATION FOR PROJECTOR FRAMEPACKING
                # The projector requires 1920x2205@24fps to enable FramePacking
                self.player['override-display-fps'] = 24  # Force 24fps for FramePacking
                self.player['vf'] = 'scale=1920:2205'  # Force FramePacking resolution

                print("MVC mode enabled - FramePacking 1920x2205@24fps configured")

                self.show_3d_notification("3D MVC mode enabled (FramePacking 1920×2205@24fps)", success=True)
            except Exception as e:
                print(f"MVC configuration error: {e}")

        elif stereo_mode == 'sbs' or (stereo_mode == 'auto' and
                                      self.video_3d_info and
                                      self.video_3d_info.get('stereo_mode') == 'sbs'):
            # Convert SBS to appropriate output
            try:
                # Use the lavfi filter to process SBS
                self.player['lavfi-complex'] = '[vid1] stereo3d=sbsl:ml [vo]'
                print("SBS mode enabled")
                self.show_3d_notification("3D Side-by-Side mode enabled", success=True)
            except Exception as e:
                print(f"SBS configuration error: {e}")

        elif stereo_mode == 'tab' or (stereo_mode == 'auto' and
                                      self.video_3d_info and
                                      self.video_3d_info.get('stereo_mode') == 'tab'):
            # Convert TAB to appropriate output
            try:
                self.player['lavfi-complex'] = '[vid1] stereo3d=abl:ml [vo]'
                print("TAB mode enabled")
                self.show_3d_notification("3D Top-Bottom mode enabled", success=True)
            except Exception as e:
                print(f"TAB configuration error: {e}")

    def play_file(self, file_path):
        """Loads and starts playing a video file."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        print(f"Loading file: {file_path}")

        # Analyze the file for 3D
        self.analyze_and_configure_3d(file_path)

        self.current_file_path = file_path
        self.has_media = True
        self.player.play(file_path)
        self.player.pause = False
        self.update_ui_state()

        # Wait for the duration to be available then generate the cache
        QTimer.singleShot(500, lambda: self._init_thumbnail_cache(file_path))

        # Load available audio tracks
        self.load_audio_tracks()

        # Apply 3D configuration if enabled
        if self.is_3d_enabled:
            self.configure_3d_output(True, self.current_stereo_mode)

    def _init_thumbnail_cache(self, file_path):
        """Initializes the thumbnail cache once the duration is known."""
        try:
            duration = self.player.duration
            if duration and duration > 0:
                self.controls_overlay.time_slider.set_video_file(file_path, duration)
            else:
                # Retry in 500ms if the duration is not yet available
                QTimer.singleShot(500, lambda: self._init_thumbnail_cache(file_path))
        except:
            # Retry
            QTimer.singleShot(500, lambda: self._init_thumbnail_cache(file_path))

    # --- Gestionnaires d'événements mpv ---
    def on_time_update(self, _, value):
        self.controls_overlay.set_time(value)

    def on_duration_change(self, _, value):
        self.controls_overlay.set_duration(value or 0)

    def on_pause_state_change(self, _, is_paused):
        self.is_playing = not is_paused
        self.controls_overlay.set_paused(is_paused)

    # --- Commandes du lecteur ---
    def toggle_play(self):
        if self.has_media:
            self.player.pause = not self.player.pause

    def toggle_fullscreen(self):
        """Toggles between fullscreen and windowed mode."""
        if self.isFullScreen():
            self.showNormal()
            self.controls_overlay.set_fullscreen_icon(False)
        else:
            self.showFullScreen()
            self.controls_overlay.set_fullscreen_icon(True)

    def toggle_3d_mode(self, enabled):
        """Enables or disables 3D mode."""
        self.is_3d_enabled = enabled

        if self.has_media:
            self.configure_3d_output(enabled, self.current_stereo_mode)

            # Show status with info about the source file
            if self.video_3d_info and self.video_3d_info['is_3d']:
                mode_names = {
                    'mvc': 'MVC',
                    'sbs': 'Side-by-Side',
                    'tab': 'Top-Bottom'
                }
                stereo_mode = self.video_3d_info['stereo_mode']
                if enabled:
                    self.show_3d_notification(
                        f"3D File: {mode_names.get(stereo_mode, stereo_mode.upper())} - 3D Playback Active",
                        success=True,
                        permanent=True
                    )
                else:
                    self.show_3d_notification(
                        f"3D File: {mode_names.get(stereo_mode, stereo_mode.upper())} - Downgraded to 2D",
                        success=False,
                        permanent=True
                    )
            else:
                if enabled:
                    self.show_3d_notification("2D File - 3D Mode Enabled", success=True, permanent=True)
                else:
                    self.show_3d_notification("2D File", success=True, permanent=True)

    def change_stereo_mode(self, mode):
        """Changes the stereoscopic mode."""
        self.current_stereo_mode = mode
        print(f"Stereo mode changed: {mode}")

        if self.has_media and self.is_3d_enabled:
            self.configure_3d_output(True, mode)

    def change_audio_track(self, track_id):
        """Changes the active audio track."""
        if self.has_media:
            try:
                self.player.aid = track_id
                print(f"Audio track changed: ID {track_id}")
            except Exception as e:
                print(f"Error changing audio track: {e}")

    def load_audio_tracks(self):
        """Loads the list of audio tracks from MPV."""
        if not self.has_media:
            return

        try:
            # Wait a bit for MPV to load the metadata
            QTimer.singleShot(500, self._fetch_audio_tracks)
        except Exception as e:
            print(f"Error loading audio tracks: {e}")

    def _fetch_audio_tracks(self):
        """Fetches the audio tracks from MPV."""
        try:
            track_list = self.player.track_list
            audio_tracks = []

            for track in track_list:
                if track.get('type') == 'audio':
                    track_id = track.get('id')
                    title = track.get('title', f"Track {track_id}")
                    lang = track.get('lang', '')

                    audio_tracks.append((track_id, title, lang))

            print(f"Audio tracks found: {len(audio_tracks)}")

            # Update the UI
            self.controls_overlay.update_audio_tracks(audio_tracks)

        except Exception as e:
            print(f"Error fetching audio tracks: {e}")

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open a video",
            "",
            "Video Files (*.mkv *.mp4 *.avi *.iso);;All files (*.*)"
        )
        if file_path:
            self.play_file(file_path)

    # --- Gestion des overlays et de la souris ---
    def show_controls(self):
        """Shows the controls with animation."""
        self.controls_overlay.show_animated()
        self.controls_overlay.raise_()
        self.hide_controls_timer.start()
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def hide_controls(self):
        """Hides the controls with animation."""
        if not self.controls_overlay.underMouse():
            self.controls_overlay.hide_animated()
            if self.is_playing:
                self.setCursor(Qt.CursorShape.BlankCursor)

    def mouseMoveEvent(self, event):
        self.show_controls()
        super().mouseMoveEvent(event)

    # --- Gestion des événements de la fenêtre ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.play_file(file_path)

    def closeEvent(self, event):
        self.player.terminate()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'controls_overlay'):
            # Ultra-compact positioning of controls
            controls_height = 90
            controls_margin = 40
            self.controls_overlay.setGeometry(
                controls_margin,
                self.height() - controls_height - controls_margin,
                self.width() - 2 * controls_margin,
                controls_height
            )
        if hasattr(self, 'info_overlay'):
            self.info_overlay.setGeometry(0, 0, self.width(), self.height())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_3:
            # Shortcut to toggle 3D
            self.controls_overlay.mode_3d_button.setChecked(
                not self.controls_overlay.mode_3d_button.isChecked()
            )
        else:
            super().keyPressEvent(event)


class SplashScreen(QWidget):
    """Custom PySide6 splash screen."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Load the splash image
        splash_pixmap = QPixmap('splash.png')

        # If in a PyInstaller exe, look in _MEIPASS
        if not splash_pixmap.isNull() or True:
            if hasattr(sys, '_MEIPASS'):
                splash_path = os.path.join(sys._MEIPASS, 'splash.png')
                if os.path.exists(splash_path):
                    splash_pixmap = QPixmap(splash_path)

        if not splash_pixmap.isNull():
            self.setFixedSize(splash_pixmap.size())
            self.pixmap = splash_pixmap
        else:
            # Fallback: black splash screen with text
            self.setFixedSize(600, 400)
            self.pixmap = QPixmap(600, 400)
            self.pixmap.fill(QColor(43, 43, 43))

        # Center the screen
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)


if __name__ == '__main__':
    # Support multiprocessing on Windows
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)

    app = QApplication(sys.argv)

    # Show splash screen for 3 seconds
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Wait for 3 seconds
    QTimer.singleShot(3000, splash.close)

    # For the window ID to be correct under Wayland (Linux)
    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    player_window = PlayerWindow()
    player_window.show()

    sys.exit(app.exec())
