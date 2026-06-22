# -*- coding: utf-8 -*-
"""
Premium Controls Overlay - SyLC 3D Player
=========================================
Ultra-modern navigation bar reflecting the technical excellence of the 3D MVC player.

Features:
- Glassmorphism design with depth effects
- Real-time technical indicators (MVC, FPS, sync, buffer)
- Animated badges (3D Ready, HDR, MVC Active)
- Timeline with animated gradient and preview
- Sophisticated vector icons
- Elegant separators and fluid animations
"""

import os
import subprocess
import tempfile
import shutil
import logging
import math
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)
from PySide6.QtWidgets import (
    QWidget, QSlider, QPushButton, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QSizePolicy, QGraphicsDropShadowEffect, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF, QPropertyAnimation, QEasingCurve, Property, Slot
from PySide6.QtGui import (
    QPainter, QColor, QFont, QFontMetrics, QPen, QBrush,
    QPainterPath, QLinearGradient, QRadialGradient, QConicalGradient, QPixmap
)

# (license/subscription system removed - freeware build)


# =============================================================================
# HELPERS & THUMBNAIL EXTRACTION
# =============================================================================

def _resolve_external_tool(executable_name):
    """Return an absolute path to an external tool (ffmpeg/ffprobe) if available."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []
    if os.name == 'nt' and not executable_name.lower().endswith('.exe'):
        candidates.append(f"{executable_name}.exe")
    candidates.append(executable_name)

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

        local_candidate = os.path.join(base_dir, candidate)
        if os.path.exists(local_candidate):
            return local_candidate

    return None


_thumbnail_executor = ThreadPoolExecutor(max_workers=2)


def _extract_thumbnail_ffmpeg(video_file, time_pos):
    """Extract a thumbnail with ffmpeg (worker function)."""
    try:
        ffmpeg_path = _resolve_external_tool('ffmpeg')
        if not ffmpeg_path:
            return None

        temp_file = os.path.join(tempfile.gettempdir(), f"preview_{int(time.time() * 1000000)}.jpg")

        cmd = [
            ffmpeg_path,
            '-ss', str(time_pos),
            '-i', video_file,
            '-frames:v', '1',
            '-vf', 'scale=160:-1',  # Slightly larger than old version
            '-q:v', '5',
            '-y',
            temp_file
        ]

        creationflags = 0
        if os.name == 'nt':
            creationflags = 0x08000000  # CREATE_NO_WINDOW

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
        self.setFixedSize(160, 90)  # 16:9 aspect ratio
        self.setStyleSheet("""
            QLabel {
                background: #1a1a1a;
                border: 2px solid #00C8FF;
                border-radius: 6px;
            }
        """)
        self.setScaledContents(True)
        self.hide()


# =============================================================================
# PREMIUM COLOR PALETTE
# =============================================================================

class PremiumColors:
    """Premium color palette for the 3D player."""

    # Primary accent (Cyan 3D)
    ACCENT_PRIMARY = QColor(0, 200, 255)  # Bright cyan
    ACCENT_SECONDARY = QColor(0, 150, 220)  # Deep blue
    ACCENT_GLOW = QColor(0, 200, 255, 80)  # Cyan glow

    # Success / Active states
    SUCCESS = QColor(0, 230, 118)  # Active MVC green
    SUCCESS_GLOW = QColor(0, 230, 118, 60)

    # Warning / Info
    WARNING = QColor(255, 170, 0)  # Orange
    INFO = QColor(138, 180, 248)  # Info blue

    # Background layers (glassmorphism)
    BG_DARK = QColor(18, 18, 22)  # Main background
    BG_SURFACE = QColor(28, 28, 35)  # Surface
    BG_ELEVATED = QColor(38, 38, 48)  # Elevated
    BG_GLASS = QColor(45, 45, 55, 200)  # Glass effect

    # Borders
    BORDER_SUBTLE = QColor(255, 255, 255, 20)
    BORDER_GLOW = QColor(0, 200, 255, 40)

    # Text
    TEXT_PRIMARY = QColor(240, 240, 245)
    TEXT_SECONDARY = QColor(160, 165, 180)
    TEXT_MUTED = QColor(100, 105, 115)


# =============================================================================
# PREMIUM ICON BUTTON
# =============================================================================

class PremiumIconButton(QPushButton):
    """
    Button with premium vector icon and sophisticated hover effects.
    """

    def __init__(self, icon_type, size='medium', parent=None):
        super().__init__(parent)
        self.icon_type = icon_type
        self._hover_progress = 0.0
        self._press_progress = 0.0
        self._glow_intensity = 0.0
        # Size presets
        sizes = {
            'small': (32, 32),
            'medium': (40, 40),
            'large': (52, 52),
            'primary': (56, 56)
        }
        w, h = sizes.get(size, (40, 40))
        self.setFixedSize(w, h)
        self.is_primary = (size == 'primary')

        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._update_animation)
        self._anim_timer.setInterval(16)  # ~60 FPS
        self._animations_blocked = False  # V7b+++ STUTTER FIX

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("background: transparent; border: none;")

    def stop_animations(self):
        """Stop animation timer to reduce activity during cleanup."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        self._animations_blocked = True  # V7b+++ Block restart on hover

    def enable_animations(self):
        """Re-enable animations after MVC mode ends."""
        self._animations_blocked = False

    def enterEvent(self, event):
        # V7b+++ STUTTER FIX: Don't start animation if blocked (MVC mode)
        if not self._animations_blocked:
            self._anim_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Stop timer immediately on leave to prevent stuttering
        self._anim_timer.stop()
        super().leaveEvent(event)

    def _update_animation(self):
        changed = False

        # Hover animation - only check underMouse if we need to animate
        is_hovered = self.underMouse()
        target_hover = 1.0 if is_hovered else 0.0
        if abs(self._hover_progress - target_hover) > 0.01:
            self._hover_progress += (target_hover - self._hover_progress) * 0.15
            changed = True
        else:
            self._hover_progress = target_hover

        # Glow pulse for primary button - only when hovered or playing
        if self.is_primary and is_hovered:
            self._glow_intensity = 0.5 + 0.5 * abs((time.time() * 2) % 2 - 1)
            changed = True
        elif self.is_primary:
            self._glow_intensity = 0.5  # Static when not hovered

        if changed:
            self.update()
        elif not is_hovered:
            # Stop timer when animation is stable and not hovered
            self._anim_timer.stop()

    def paintEvent(self, event):
        # Guard against painting during destruction
        if not self.isVisible():
            return
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return  # Painter failed to initialize
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            cx, cy = self.width() / 2, self.height() / 2
            radius = min(self.width(), self.height()) / 2 - 2

            # === BACKGROUND ===
            if self.is_primary:
                # Primary button: gradient with glow
                gradient = QRadialGradient(cx, cy, radius * 1.2)
                gradient.setColorAt(0, QColor(0, 180, 240, 255))
                gradient.setColorAt(0.7, QColor(0, 130, 200, 255))
                gradient.setColorAt(1, QColor(0, 100, 180, 255))

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(QPointF(cx, cy), radius, radius)

                # Glow effect
                if self._glow_intensity > 0:
                    glow_gradient = QRadialGradient(cx, cy, radius * 1.5)
                    glow_alpha = int(60 * self._glow_intensity)
                    glow_gradient.setColorAt(0, QColor(0, 200, 255, glow_alpha))
                    glow_gradient.setColorAt(0.5, QColor(0, 200, 255, glow_alpha // 2))
                    glow_gradient.setColorAt(1, QColor(0, 200, 255, 0))
                    painter.setBrush(QBrush(glow_gradient))
                    painter.drawEllipse(QPointF(cx, cy), radius * 1.3, radius * 1.3)

                # Border glow
                painter.setPen(QPen(QColor(100, 220, 255, 150), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), radius, radius)
            else:
                # Secondary button: glass effect
                if self.isChecked():
                    # Active (Checked) State - Cyan Tint
                    bg_color = QColor(0, 140, 220, 160)
                    border_color = QColor(100, 220, 255, 200)
                    painter.setPen(QPen(border_color, 1.5))
                    painter.setBrush(QBrush(bg_color))
                    painter.drawRoundedRect(QRectF(2, 2, self.width() - 4, self.height() - 4), 10, 10)
                else:
                    # Normal / Hover State
                    bg_alpha = int(40 + 60 * self._hover_progress)
                    border_alpha = int(30 + 50 * self._hover_progress)

                    painter.setPen(QPen(QColor(255, 255, 255, border_alpha), 1))
                    painter.setBrush(QBrush(QColor(60, 65, 80, bg_alpha)))
                    painter.drawRoundedRect(QRectF(2, 2, self.width() - 4, self.height() - 4), 10, 10)

                    # Hover glow
                    if self._hover_progress > 0.1:
                        glow_color = QColor(0, 200, 255, int(30 * self._hover_progress))
                        painter.setPen(QPen(glow_color, 2))
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawRoundedRect(QRectF(2, 2, self.width() - 4, self.height() - 4), 10, 10)

            # === ICON ===
            icon_color = QColor(255, 255, 255, 240)
            if not self.isEnabled():
                icon_color = QColor(255, 255, 255, 80)

            self._draw_icon(painter, cx, cy, icon_color)
        except Exception:
            pass  # Ignore paint errors during seek/thread contention

    def _draw_icon(self, painter, cx, cy, color):
        """Draws the vector icon."""
        painter.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        scale = 0.4 if self.is_primary else 0.35
        s = min(self.width(), self.height()) * scale

        if self.icon_type == 'play':
            path = QPainterPath()
            path.moveTo(cx - s * 0.35, cy - s * 0.5)
            path.lineTo(cx + s * 0.5, cy)
            path.lineTo(cx - s * 0.35, cy + s * 0.5)
            path.closeSubpath()
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)

        elif self.icon_type == 'pause':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            bar_w = s * 0.25
            gap = s * 0.15
            painter.drawRoundedRect(QRectF(cx - gap - bar_w, cy - s * 0.45, bar_w, s * 0.9), 2, 2)
            painter.drawRoundedRect(QRectF(cx + gap, cy - s * 0.45, bar_w, s * 0.9), 2, 2)

        elif self.icon_type == 'stop':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(cx - s * 0.35, cy - s * 0.35, s * 0.7, s * 0.7), 3, 3)

        elif self.icon_type == 'folder':
            path = QPainterPath()
            # Folder shape
            path.moveTo(cx - s * 0.5, cy - s * 0.25)
            path.lineTo(cx - s * 0.2, cy - s * 0.25)
            path.lineTo(cx - s * 0.1, cy - s * 0.4)
            path.lineTo(cx + s * 0.5, cy - s * 0.4)
            path.lineTo(cx + s * 0.5, cy + s * 0.35)
            path.lineTo(cx - s * 0.5, cy + s * 0.35)
            path.closeSubpath()
            painter.strokePath(path, QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

        elif self.icon_type == 'fullscreen':
            pen = QPen(color, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            corners = [
                (cx - s * 0.4, cy - s * 0.4, 1, 1),  # TL
                (cx + s * 0.4, cy - s * 0.4, -1, 1),  # TR
                (cx - s * 0.4, cy + s * 0.4, 1, -1),  # BL
                (cx + s * 0.4, cy + s * 0.4, -1, -1),  # BR
            ]
            for x, y, dx, dy in corners:
                painter.drawLine(QPointF(x, y), QPointF(x + dx * s * 0.25, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y + dy * s * 0.25))

        elif self.icon_type == 'exit_fullscreen':
            pen = QPen(color, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            offset = s * 0.15
            corners = [
                (cx - offset, cy - offset, -1, -1),
                (cx + offset, cy - offset, 1, -1),
                (cx - offset, cy + offset, -1, 1),
                (cx + offset, cy + offset, 1, 1),
            ]
            for x, y, dx, dy in corners:
                painter.drawLine(QPointF(x, y), QPointF(x + dx * s * 0.25, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y + dy * s * 0.25))

        elif self.icon_type == 'volume':
            # Speaker cone
            path = QPainterPath()
            path.moveTo(cx - s * 0.25, cy - s * 0.15)
            path.lineTo(cx - s * 0.05, cy - s * 0.15)
            path.lineTo(cx + s * 0.15, cy - s * 0.4)
            path.lineTo(cx + s * 0.15, cy + s * 0.4)
            path.lineTo(cx - s * 0.05, cy + s * 0.15)
            path.lineTo(cx - s * 0.25, cy + s * 0.15)
            path.closeSubpath()
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)
            # Sound waves
            painter.setPen(QPen(color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(QRectF(cx + s * 0.1, cy - s * 0.2, s * 0.25, s * 0.4), -60 * 16, 120 * 16)
            painter.drawArc(QRectF(cx + s * 0.1, cy - s * 0.35, s * 0.4, s * 0.7), -55 * 16, 110 * 16)

        elif self.icon_type == 'volume_mute':
            # Muted speaker
            path = QPainterPath()
            path.moveTo(cx - s * 0.3, cy - s * 0.15)
            path.lineTo(cx - s * 0.1, cy - s * 0.15)
            path.lineTo(cx + s * 0.1, cy - s * 0.35)
            path.lineTo(cx + s * 0.1, cy + s * 0.35)
            path.lineTo(cx - s * 0.1, cy + s * 0.15)
            path.lineTo(cx - s * 0.3, cy + s * 0.15)
            path.closeSubpath()
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)
            # X mark
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(cx + s * 0.2, cy - s * 0.2), QPointF(cx + s * 0.45, cy + s * 0.2))
            painter.drawLine(QPointF(cx + s * 0.45, cy - s * 0.2), QPointF(cx + s * 0.2, cy + s * 0.2))

        elif self.icon_type == 'skip_back':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            # Bar
            painter.drawRect(QRectF(cx - s * 0.4, cy - s * 0.35, s * 0.12, s * 0.7))
            # Arrow
            path = QPainterPath()
            path.moveTo(cx + s * 0.35, cy - s * 0.35)
            path.lineTo(cx - s * 0.15, cy)
            path.lineTo(cx + s * 0.35, cy + s * 0.35)
            path.closeSubpath()
            painter.drawPath(path)

        elif self.icon_type == 'skip_forward':
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            # Bar
            painter.drawRect(QRectF(cx + s * 0.28, cy - s * 0.35, s * 0.12, s * 0.7))
            # Arrow
            path = QPainterPath()
            path.moveTo(cx - s * 0.35, cy - s * 0.35)
            path.lineTo(cx + s * 0.15, cy)
            path.lineTo(cx - s * 0.35, cy + s * 0.35)
            path.closeSubpath()
            painter.drawPath(path)

        elif self.icon_type == '3d':
            font = QFont('Segoe UI', int(s * 0.8), QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(QRectF(0, 0, self.width(), self.height()),
                             Qt.AlignmentFlag.AlignCenter, '3D')

        elif self.icon_type == 'disc':
            # Optical disc: outer ring + center hub (open a Blu-ray)
            painter.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            r = s * 0.5
            painter.drawEllipse(QPointF(cx, cy), r, r)
            painter.drawEllipse(QPointF(cx, cy), r * 0.35, r * 0.35)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), r * 0.12, r * 0.12)

        elif self.icon_type == 'archive':
            # Optical disc above a down-arrow: save/rip the disc to an .iso image
            painter.setPen(QPen(color, 1.7, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            dcy = cy - s * 0.16
            r = s * 0.34
            painter.drawEllipse(QPointF(cx, dcy), r, r)
            painter.drawEllipse(QPointF(cx, dcy), r * 0.34, r * 0.34)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, dcy), r * 0.11, r * 0.11)
            # down arrow (export to file)
            pen = QPen(color, 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ay0, ay1 = cy + s * 0.20, cy + s * 0.50
            painter.drawLine(QPointF(cx, ay0), QPointF(cx, ay1))
            painter.drawLine(QPointF(cx - s * 0.13, ay1 - s * 0.15), QPointF(cx, ay1))
            painter.drawLine(QPointF(cx + s * 0.13, ay1 - s * 0.15), QPointF(cx, ay1))


# =============================================================================
# PREMIUM STATUS BADGE
# =============================================================================

class PremiumStatusBadge(QWidget):
    """
    Animated status badge with glow effect.
    Displays player state (Ready, MVC Active, etc.)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = "Ready"
        self._status_type = "info"  # info, success, warning, error
        self._glow_phase = 0.0
        self._is_active = False

        self.setFixedHeight(28)
        self.setMinimumWidth(100)

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_glow)
        self._timer.setInterval(30)

    def set_status(self, text, status_type="info", active=False):
        """
        Sets the displayed status.

        Args:
            text: Text to display
            status_type: 'info', 'success', 'warning', 'error'
            active: If True, enables the glow animation
        """
        self._status = text
        self._status_type = status_type
        self._is_active = active

        if active and not self._timer.isActive():
            self._timer.start()
        elif not active:
            self._timer.stop()
            self._glow_phase = 0.0

        self.updateGeometry()
        self.update()

    def _update_glow(self):
        self._glow_phase = (self._glow_phase + 0.05) % (2 * 3.14159)
        self.update()

    def sizeHint(self):
        fm = QFontMetrics(QFont('Segoe UI', 10, QFont.Weight.Medium))
        text_width = fm.horizontalAdvance(self._status)
        return self.size().__class__(text_width + 40, 28)

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Colors by type
            colors = {
                'info': (PremiumColors.INFO, QColor(138, 180, 248, 30)),
                'success': (PremiumColors.SUCCESS, QColor(0, 230, 118, 40)),
                'warning': (PremiumColors.WARNING, QColor(255, 170, 0, 30)),
                'error': (QColor(255, 100, 100), QColor(255, 100, 100, 30)),
            }
            accent_color, bg_color = colors.get(self._status_type, colors['info'])

            # Background with glow if active
            rect = QRectF(0, 0, self.width(), self.height())

            if self._is_active:
                import math
                glow_intensity = 0.5 + 0.5 * math.sin(self._glow_phase)
                glow_alpha = int(40 + 40 * glow_intensity)
                glow_color = QColor(accent_color.red(), accent_color.green(), accent_color.blue(), glow_alpha)

                # Outer glow
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(glow_color))
                painter.drawRoundedRect(rect.adjusted(-2, -2, 2, 2), 16, 16)

            # Main background
            painter.setPen(QPen(accent_color.lighter(120), 1))
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 12, 12)

            # Indicator dot
            dot_x = 12
            dot_y = self.height() / 2

            if self._is_active:
                import math
                pulse = 0.7 + 0.3 * math.sin(self._glow_phase * 2)
                painter.setBrush(QBrush(accent_color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(dot_x, dot_y), 4 * pulse, 4 * pulse)
            else:
                painter.setBrush(QBrush(accent_color.darker(130)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(dot_x, dot_y), 3, 3)

            # Text
            font = QFont('Segoe UI', 10, QFont.Weight.Medium)
            painter.setFont(font)
            painter.setPen(PremiumColors.TEXT_PRIMARY)
            text_rect = rect.adjusted(24, 0, -8, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._status)
        except Exception:
            pass  # Ignore paint errors


# =============================================================================
# PREMIUM TECH INFO WIDGET
# =============================================================================

class PremiumTechInfo(QWidget):
    """
    Displays real-time technical information.
    Format: "1080p • 23.976 fps • MVC"
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._resolution = ""
        self._fps = ""
        self._codec = ""
        self._sync_status = "sync"  # sync, drift, error

        self.setFixedHeight(24)
        self.setMinimumWidth(150)

    def set_info(self, resolution="", fps="", codec=""):
        self._resolution = resolution
        self._fps = fps
        self._codec = codec
        self.update()

    def set_sync_status(self, status):
        """'sync', 'drift', 'error'"""
        self._sync_status = status
        self.update()

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

            font = QFont('Segoe UI', 9)
            painter.setFont(font)

            x = 4
            y = self.height() / 2 + 4

            # Resolution
            if self._resolution:
                painter.setPen(PremiumColors.TEXT_SECONDARY)
                painter.drawText(int(x), int(y), self._resolution)
                x += QFontMetrics(font).horizontalAdvance(self._resolution) + 8

                # Separator dot
                painter.setPen(PremiumColors.TEXT_MUTED)
                painter.setBrush(QBrush(PremiumColors.TEXT_MUTED))
                painter.drawEllipse(QPointF(x, self.height() / 2), 2, 2)
                x += 10

            # FPS
            if self._fps:
                painter.setPen(PremiumColors.TEXT_SECONDARY)
                painter.drawText(int(x), int(y), self._fps)
                x += QFontMetrics(font).horizontalAdvance(self._fps) + 8

                if self._codec:
                    painter.setPen(PremiumColors.TEXT_MUTED)
                    painter.setBrush(QBrush(PremiumColors.TEXT_MUTED))
                    painter.drawEllipse(QPointF(x, self.height() / 2), 2, 2)
                    x += 10

            # Codec with color
            if self._codec:
                codec_color = PremiumColors.SUCCESS if 'MVC' in self._codec.upper() else PremiumColors.INFO
                painter.setPen(codec_color)
                painter.drawText(int(x), int(y), self._codec)
        except Exception:
            pass  # Ignore paint errors


# =============================================================================
# PREMIUM TIMELINE SLIDER
# =============================================================================

class PremiumTimelineSlider(QSlider):
    """
    Premium timeline with animated gradient, buffer indicator and hover preview.
    Corrected version for consistent movement.
    """

    preview_requested = Signal(float)
    extraction_done = Signal(float, str)
    scrub_finished = Signal(float)

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)

        self._hover_pos = -1
        self._hover_time = 0
        self._buffer_progress = 0.0
        self._is_mvc_active = False
        self._gradient_offset = 0.0
        self._is_busy = False  # New flag to block interaction during seeks

        # Preview machinery
        self._player = None
        self._video_file = None
        self._preview_widget = PreviewTooltip(self)
        self._last_preview_time = -99
        self._preview_cache = {}
        self._extraction_timer = QTimer(self)
        self._extraction_timer.setSingleShot(True)
        self._extraction_timer.timeout.connect(self._do_extraction)
        self._pending_time = 0
        self._pending_mouse_x = 0

        self.extraction_done.connect(self._on_extraction_done)

        # Animation
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.setInterval(30)

        # === CORRECTED SCRUBBING SYSTEM ===
        self._is_scrubbing = False
        self._scrub_start_value = 0
        self._scrub_debounce_timer = QTimer(self)
        self._scrub_debounce_timer.setSingleShot(True)
        self._scrub_debounce_timer.setInterval(50)  # Reduced to 50ms for more responsiveness
        self._scrub_debounce_timer.timeout.connect(self._on_scrub_debounce_expired)
        self._pending_scrub_value = None

        self.setFixedHeight(30) # Increased height to preventing clipping
        self.setStyleSheet("background: transparent;")

    def set_busy(self, busy):
        """Block or unblock user interaction."""
        self._is_busy = busy
        self.update()

    def set_player(self, player):
        self._player = player

    def set_video_file(self, video_path, duration):
        """Configures the video file and duration for the slider."""
        import os
        import logging
        
        if not video_path: return
        
        norm_path = os.path.normpath(video_path)
        
        # Only reset cache if file actually changed
        if self._video_file != norm_path:
            self._video_file = norm_path
            self._preview_cache.clear()
            
            # Trigger initial thumbnail extraction immediately
            self._request_on_demand_preview(0.0, 0)

        if duration and duration > 0:
            # Use milliseconds for better precision
            new_max = int(duration * 1000)
            if self.maximum() != new_max:
                self.setRange(0, new_max)
            self.setEnabled(True)

    def set_mvc_active(self, active):
        self._is_mvc_active = active
        # V7b+++ STUTTER FIX: DON'T start animation timer in MVC mode!
        # The gradient animation was causing constant repaints (every 30ms)
        # which caused stuttering when the window had focus.
        # The animation is purely cosmetic - not needed during MVC playback.
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        self.update()

    def set_buffer_progress(self, progress):
        self._buffer_progress = max(0, min(1, progress))
        self.update()

    def _animate(self):
        self._gradient_offset = (self._gradient_offset + 0.02) % 1.0
        self.update()

    def enterEvent(self, event):
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover_pos = -1
        self._preview_widget.hide()
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        if self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            self._hover_pos = pos
            
            # Margin correction for accurate hover time
            margin = 16
            usable_width = self.width() - 2 * margin
            if usable_width > 0:
                normalized_pos = (pos - margin) / usable_width
                normalized_pos = max(0.0, min(1.0, normalized_pos))
                self._hover_time = normalized_pos * (self.maximum() / 1000.0)
            else:
                self._hover_time = 0

            # Real-time update during scrubbing
            if self._is_scrubbing:
                # Apply same margin logic for setting value
                value = int(self._hover_time * 1000)
                self.setValue(max(0, min(value, self.maximum())))
                # Immediate emission during drag
                self.sliderMoved.emit(self.value())

            if not self._is_mvc_active:
                self.preview_requested.emit(self._hover_time)

            # Preview Request
            if (not self._is_mvc_active and self._video_file
                    and abs(self._hover_time - self._last_preview_time) > 0.5):
                self._last_preview_time = self._hover_time
                self._request_on_demand_preview(self._hover_time, pos)

        self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self._is_busy: return  # Block interaction if busy

        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > 0:
            pos = event.position().x() if hasattr(event, 'position') else event.pos().x()
            
            # Margin correction
            margin = 16
            usable_width = self.width() - 2 * margin
            if usable_width > 0:
                normalized_pos = (pos - margin) / usable_width
                normalized_pos = max(0.0, min(1.0, normalized_pos))
                value = int(normalized_pos * self.maximum())
            else:
                value = 0
                
            self.setValue(max(0, min(value, self.maximum())))

            # Start scrubbing
            self._is_scrubbing = True
            self._scrub_start_value = value

            # Immediate emission for click
            self.sliderMoved.emit(self.value())
            self._pending_scrub_value = value
            self._scrub_debounce_timer.stop()

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """End of scrubbing - emits final seek with debounce."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_scrubbing:
            self._is_scrubbing = False
            final_value = self.value()

            # Emit final seek via debounce
            self._pending_scrub_value = final_value
            self._scrub_debounce_timer.start()

        super().mouseReleaseEvent(event)

    def _on_scrub_debounce_expired(self):
        """Called after debounce - emits scrub_finished signal."""
        if self._pending_scrub_value is not None:
            # Convert ms to seconds for the signal
            self.scrub_finished.emit(float(self._pending_scrub_value) / 1000.0)
            self._pending_scrub_value = None

    # --- Preview methods remain unchanged ---
    def _request_on_demand_preview(self, time_pos, mouse_x):
        # Thumbnail extraction is now allowed in MVC mode
        cache_key = round(time_pos)
        if cache_key in self._preview_cache:
            pixmap = self._preview_cache[cache_key]
            if not pixmap.isNull():
                self._preview_widget.setPixmap(pixmap)
                self._show_preview_at(mouse_x)
                return

        self._pending_time = time_pos
        self._pending_mouse_x = mouse_x
        self._extraction_timer.start(150)

    def _do_extraction(self):
        # V13 FIX: Disable thumbnail extraction during MVC playback
        # The ThreadPoolExecutor causes Windows 0xe24c4a02 exceptions during MVC decoding
        if self._is_mvc_active:
            return

        time_pos = self._pending_time
        mouse_x = self._pending_mouse_x
        future = _thumbnail_executor.submit(_extract_thumbnail_ffmpeg, self._video_file, time_pos)
        future.add_done_callback(lambda f: self._handle_extraction_result(f, time_pos))

    def _handle_extraction_result(self, future, time_pos):
        try:
            temp_file = future.result()
            if temp_file:
                self.extraction_done.emit(time_pos, temp_file)
        except:
            pass

    @Slot(float, str)
    def _on_extraction_done(self, time_pos, temp_file):
        try:
            cache_key = round(time_pos)
            pixmap = QPixmap(temp_file)
            if not pixmap.isNull():
                if len(self._preview_cache) > 120:
                    oldest = next(iter(self._preview_cache))
                    del self._preview_cache[oldest]
                self._preview_cache[cache_key] = pixmap

                if self._hover_pos >= 0 and abs(time_pos - self._hover_time) < 5:
                    self._preview_widget.setPixmap(pixmap)
                    self._show_preview_at(self._pending_mouse_x)
            try:
                os.remove(temp_file)
            except:
                pass
        except Exception:
            pass

    def _show_preview_at(self, mouse_x):
        global_pos = self.mapToGlobal(QPointF(int(mouse_x), 0))
        tooltip_x = global_pos.x() - self._preview_widget.width() // 2
        tooltip_y = global_pos.y() - self._preview_widget.height() - 15
        self._preview_widget.move(tooltip_x, tooltip_y)
        self._preview_widget.show()
        self._preview_widget.raise_()

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Dimensions
            margin = 16
            track_height = 6
            track_y = (self.height() - track_height) / 2
            track_rect = QRectF(margin, track_y, self.width() - 2 * margin, track_height)

            # === TRACK BACKGROUND ===
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(50, 55, 65)))
            painter.drawRoundedRect(track_rect, track_height / 2, track_height / 2)

            # === BUFFER INDICATOR ===
            if self._buffer_progress > 0:
                buffer_width = track_rect.width() * self._buffer_progress
                buffer_rect = QRectF(track_rect.x(), track_rect.y(), buffer_width, track_rect.height())
                painter.setBrush(QBrush(QColor(80, 85, 95)))
                painter.drawRoundedRect(buffer_rect, track_height / 2, track_height / 2)

            # === PROGRESS BAR ===
            if self.maximum() > 0:
                progress = self.value() / self.maximum()
                progress_width = track_rect.width() * progress
                progress_rect = QRectF(track_rect.x(), track_rect.y(), progress_width, track_rect.height())

                if self._is_mvc_active:
                    # Animated gradient for MVC mode
                    gradient = QLinearGradient(0, 0, track_rect.width(), 0)
                    offset = self._gradient_offset
                    gradient.setColorAt(0, PremiumColors.ACCENT_PRIMARY)
                    gradient.setColorAt((0.5 + offset) % 1.0, PremiumColors.SUCCESS)
                    gradient.setColorAt(1, PremiumColors.ACCENT_SECONDARY)
                    painter.setBrush(QBrush(gradient))
                else:
                    # Static gradient
                    gradient = QLinearGradient(0, 0, progress_width, 0)
                    gradient.setColorAt(0, PremiumColors.ACCENT_SECONDARY)
                    gradient.setColorAt(1, PremiumColors.ACCENT_PRIMARY)
                    painter.setBrush(QBrush(gradient))

                painter.drawRoundedRect(progress_rect, track_height / 2, track_height / 2)

            # === HANDLE ===
            if self.maximum() > 0:
                handle_x = track_rect.x() + (track_rect.width() * self.value() / self.maximum())
                handle_radius = 8 if self._hover_pos >= 0 else 6

                # Handle shadow
                painter.setBrush(QBrush(QColor(0, 0, 0, 40)))
                painter.drawEllipse(QPointF(handle_x, track_y + track_height / 2 + 1), handle_radius, handle_radius)

                # Handle
                gradient = QRadialGradient(handle_x, track_y + track_height / 2, handle_radius)
                gradient.setColorAt(0, QColor(255, 255, 255))
                gradient.setColorAt(1, QColor(220, 225, 235))
                painter.setBrush(QBrush(gradient))
                painter.setPen(QPen(PremiumColors.ACCENT_PRIMARY, 2))
                painter.drawEllipse(QPointF(handle_x, track_y + track_height / 2), handle_radius, handle_radius)

            # === HOVER INDICATOR ===
            if self._hover_pos >= 0 and self.maximum() > 0:
                painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
                painter.drawLine(QPointF(self._hover_pos, 0), QPointF(self._hover_pos, self.height()))

            # === PREVIEW MARKER ===
            if self._hover_pos >= 0 and self.maximum() > 0:
                # Draw a small dot on the track at the hover position
                painter.setBrush(QBrush(PremiumColors.ACCENT_GLOW))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(self._hover_pos, track_y + track_height / 2), 3, 3)
        except Exception:
            pass  # Ignore paint errors


# =============================================================================
# PREMIUM VOLUME SLIDER
# =============================================================================

class PremiumVolumeSlider(QWidget):
    """
    Compact volume slider with integrated icon.
    """

    volume_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._volume = 100
        self._is_muted = False
        self._hover = False
        self._dragging = False

        self.setFixedSize(110, 28)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_volume(self, value):
        self._volume = max(0, min(100, value))
        self._is_muted = (value == 0)
        self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_volume_from_pos(event.position().x())
            self._dragging = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_volume_from_pos(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        super().mouseReleaseEvent(event)

    def _update_volume_from_pos(self, x):
        slider_start = 28
        slider_width = self.width() - slider_start - 4
        if slider_width > 0:
            normalized = (x - slider_start) / slider_width
            volume = int(max(0, min(100, normalized * 100)))
            if volume != self._volume:
                self._volume = volume
                self._is_muted = (volume == 0)
                self.volume_changed.emit(volume)
                self.update()

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # === VOLUME ICON ===
            icon_color = PremiumColors.TEXT_SECONDARY if not self._is_muted else PremiumColors.TEXT_MUTED
            cx, cy = 14, self.height() / 2
            s = 10

            # Speaker cone
            path = QPainterPath()
            path.moveTo(cx - s * 0.35, cy - s * 0.2)
            path.lineTo(cx - s * 0.1, cy - s * 0.2)
            path.lineTo(cx + s * 0.15, cy - s * 0.45)
            path.lineTo(cx + s * 0.15, cy + s * 0.45)
            path.lineTo(cx - s * 0.1, cy + s * 0.2)
            path.lineTo(cx - s * 0.35, cy + s * 0.2)
            path.closeSubpath()
            painter.setBrush(QBrush(icon_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)

            if not self._is_muted and self._volume > 30:
                painter.setPen(QPen(icon_color, 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawArc(QRectF(cx + s * 0.1, cy - s * 0.25, s * 0.3, s * 0.5), -60 * 16, 120 * 16)
            if not self._is_muted and self._volume > 60:
                painter.drawArc(QRectF(cx + s * 0.1, cy - s * 0.4, s * 0.5, s * 0.8), -55 * 16, 110 * 16)

            # === SLIDER TRACK ===
            slider_x = 28
            slider_width = self.width() - slider_x - 4
            slider_y = self.height() / 2
            track_height = 4

            # Track background
            track_rect = QRectF(slider_x, slider_y - track_height / 2, slider_width, track_height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(60, 65, 75)))
            painter.drawRoundedRect(track_rect, 2, 2)

            # Progress
            progress_width = slider_width * (self._volume / 100)
            if progress_width > 0:
                progress_rect = QRectF(slider_x, slider_y - track_height / 2, progress_width, track_height)
                painter.setBrush(QBrush(PremiumColors.ACCENT_PRIMARY))
                painter.drawRoundedRect(progress_rect, 2, 2)

            # Handle
            if self._hover or self._dragging:
                handle_x = slider_x + progress_width
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(PremiumColors.ACCENT_PRIMARY, 1.5))
                painter.drawEllipse(QPointF(handle_x, slider_y), 5, 5)
        except Exception:
            pass  # Ignore paint errors


# =============================================================================
# VERTICAL SEPARATOR
# =============================================================================

class PremiumSeparator(QWidget):
    """Elegant vertical separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(1)
        self.setMinimumHeight(20)

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(255, 255, 255, 0))
            gradient.setColorAt(0.5, QColor(255, 255, 255, 40))
            gradient.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setPen(QPen(QBrush(gradient), 1))
            painter.drawLine(0, 4, 0, self.height() - 4)
        except Exception:
            pass  # Ignore paint errors


# =============================================================================
# PREMIUM CONTROLS OVERLAY - MAIN WIDGET
# =============================================================================

class PremiumSpectrumMeter(QWidget):
    """Audio-reactive spectrum-style visualiser fed by the player's real mpv level.

    A wide row of slim cyan bars with a pleasant spectral arch + lively per-band
    motion, so it dances with the music and fills the transport gap. Sober and
    on-theme. (mpv exposes overall level, not true FFT bands, so the per-band
    motion is a level-driven envelope rather than literal frequency content.)
    """
    BANDS = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 40)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setToolTip("Audio spectrum")
        n = self.BANDS
        self._h = [0.0] * n          # smoothed bar heights 0..1
        self._level = 0.0            # latest overall level 0..1
        self._peak = 0.0             # latest transient peak 0..1
        self._phase = 0.0
        self._env = []               # fixed spectral arch (fuller mids, taper highs)
        self._rate = []              # varied per-band wiggle speeds
        for i in range(n):
            x = i / (n - 1)
            env = 0.40 + 0.60 * max(0.0, math.sin(math.pi * (0.12 + 0.76 * x)))
            env *= (1.0 - 0.30 * x)
            self._env.append(env)
            self._rate.append(0.6 + 1.6 * ((i * 0.37 + 0.13) % 1.0))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(33)  # ~30 fps, only runs while there's signal

    def set_levels(self, level, peak=None):
        """Feed the overall normalized level (0..1); peak adds a transient kick."""
        self._level = max(0.0, min(1.0, float(level)))
        if peak is not None:
            self._peak = max(self._peak, max(0.0, min(1.0, float(peak))))
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self):
        self._phase += 0.16
        lvl = self._level
        n = self.BANDS
        active = lvl > 0.003
        for i in range(n):
            wig = 0.5 + 0.5 * math.sin(self._phase * self._rate[i] + i * 1.7)
            wig2 = 0.5 + 0.5 * math.sin(self._phase * self._rate[i] * 0.6 + i * 0.7 + 2.1)
            tg = lvl * self._env[i] * (0.5 + 0.7 * wig * wig2)
            tg = min(1.0, tg + self._peak * 0.18 * wig)
            cur = self._h[i]
            cur += (tg - cur) * (0.60 if tg > cur else 0.22)   # fast attack, slow release
            if cur < 0.001 and tg < 0.001:
                cur = 0.0
            self._h[i] = cur
            if cur > 0.003:
                active = True
        self._peak *= 0.90            # transient bleeds away between polls
        self.update()
        if not active:
            self._timer.stop()

    def paintEvent(self, event):
        p = QPainter(self)
        if not p.isActive():
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = self.BANDS
        top, bot = 6.0, h - 6.0
        span = bot - top
        gap = 2.5
        bar_w = max(2.0, (w - gap * (n + 1)) / n)
        rad = min(2.0, bar_w / 2.0)
        grad = QLinearGradient(0, top, 0, bot)
        grad.setColorAt(0.0, QColor(150, 233, 255))
        grad.setColorAt(0.4, QColor(40, 196, 255))
        grad.setColorAt(1.0, QColor(8, 90, 130))
        x = gap
        for i in range(n):
            # unlit track
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 12))
            p.drawRoundedRect(QRectF(x, top, bar_w, span), rad, rad)
            v = self._h[i]
            if v > 0.003:
                lit_top = bot - v * span
                p.save()
                clip = QPainterPath()
                clip.addRoundedRect(QRectF(x, top, bar_w, span), rad, rad)
                p.setClipPath(clip)
                p.setBrush(QBrush(grad))
                p.drawRect(QRectF(x, lit_top, bar_w, bot - lit_top))
                p.restore()
            x += bar_w + gap


class PremiumControlsOverlay(QWidget):
    """
    Ultra-modern premium controls bar for SyLC 3D player.

    Signals:
        play_toggled: Play/Pause
        stop_clicked: Stop
        fullscreen_toggled: Fullscreen
        seeked(float): Seek position
        volume_changed(int): Volume 0-100
        file_opened: Open file
        stereo_mode_changed(str): 3D mode changed
        mode_3d_toggled(bool): 3D on/off
        audio_track_changed(int): Audio track
        subtitle_track_changed(int): Subtitles
    """

    play_toggled = Signal()
    stop_clicked = Signal()
    fullscreen_toggled = Signal()
    seeked = Signal(float)
    volume_changed = Signal(int)
    file_opened = Signal()
    disc_opened = Signal()  # open a Blu-ray 3D disc/folder (auto-detect the feature)
    archive_requested = Signal()  # archive the current optical disc to an .iso image
    stereo_mode_changed = Signal(str)
    mode_3d_toggled = Signal(bool)
    audio_track_changed = Signal(int)
    subtitle_track_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Window flags for overlay
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Configures the user interface."""

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # === ROW 1: TIMELINE ===
        timeline_layout = QHBoxLayout()
        timeline_layout.setSpacing(12)

        # Time labels - clean, without border or box
        self.time_label = QLabel("00:00")
        self.time_label.setStyleSheet(
            f"color: {PremiumColors.TEXT_PRIMARY.name()}; font-size: 13px; font-weight: 500; font-family: 'Segoe UI'; background: transparent; border: none; padding: 0px;")
        self.time_label.setMinimumWidth(45)

        self.time_slider = PremiumTimelineSlider()

        self.duration_label = QLabel("00:00")
        self.duration_label.setStyleSheet(
            f"color: {PremiumColors.TEXT_SECONDARY.name()}; font-size: 13px; font-family: 'Segoe UI'; background: transparent; border: none; padding: 0px;")
        self.duration_label.setMinimumWidth(45)
        self.duration_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        timeline_layout.addWidget(self.time_label)
        timeline_layout.addWidget(self.time_slider, 1)
        timeline_layout.addWidget(self.duration_label)

        main_layout.addLayout(timeline_layout)

        # === ROW 2: CONTROLS ===
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        # --- LEFT GROUP: File, Audio, Subtitles ---
        left_group = QHBoxLayout()
        left_group.setSpacing(10)

        self.open_file_button = PremiumIconButton('folder', 'medium')
        self.open_file_button.setToolTip("Open file")
        self.open_disc_button = PremiumIconButton('disc', 'medium')
        self.open_disc_button.setToolTip("Open Blu-ray 3D (drive or BDMV folder)")
        self.archive_button = PremiumIconButton('archive', 'medium')
        self.archive_button.setToolTip("Archive this Blu-ray to an ISO image")
        self.archive_button.setEnabled(False)  # enabled only when a Blu-ray disc is the source

        # Audio track
        self.audio_track_combo = QComboBox()
        self.audio_track_combo.addItem("Audio")
        self.audio_track_combo.setMinimumWidth(160)
        self.audio_track_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._style_combo(self.audio_track_combo)

        # Subtitle track
        self.subtitle_track_combo = QComboBox()
        self.subtitle_track_combo.addItem("Subtitles")
        self.subtitle_track_combo.setMinimumWidth(160)
        self.subtitle_track_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._style_combo(self.subtitle_track_combo)

        left_group.addWidget(self.open_file_button)
        left_group.addWidget(self.open_disc_button)
        left_group.addWidget(self.archive_button)
        left_group.addWidget(self.audio_track_combo, 1)  # stretch factor 1
        left_group.addWidget(self.subtitle_track_combo, 1)  # stretch factor 1 (same width)

        # --- CENTER GROUP: Transport ---
        center_group = QHBoxLayout()
        center_group.setSpacing(8)
        center_group.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.skip_back_button = PremiumIconButton('skip_back', 'small')
        self.skip_back_button.setToolTip("Skip back 10s")

        self.stop_button = PremiumIconButton('stop', 'medium')
        self.stop_button.setToolTip("Stop")

        self.play_pause_button = PremiumIconButton('play', 'primary')
        self.play_pause_button.setToolTip("Play / Pause")

        self.skip_forward_button = PremiumIconButton('skip_forward', 'small')
        self.skip_forward_button.setToolTip("Skip forward 10s")

        center_group.addWidget(self.skip_back_button)
        center_group.addWidget(self.stop_button)
        center_group.addWidget(self.play_pause_button)
        center_group.addWidget(self.skip_forward_button)

        # --- RIGHT GROUP: 3D, Volume, Fullscreen ---
        right_group = QHBoxLayout()
        right_group.setSpacing(8)
        right_group.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Tech info
        self.tech_info = PremiumTechInfo()

        # Separator
        sep1 = PremiumSeparator()

        # Separator
        sep2 = PremiumSeparator()

        # 3D Mode button
        self.mode_3d_button = PremiumIconButton('3d', 'medium')
        self.mode_3d_button.setCheckable(True)
        self.mode_3d_button.setToolTip("Toggle 3D mode")

        # 3D format badge — a contextual cyan pill shown ONLY when a stereoscopic
        # stream is detected & adapted by edge264, so the user sees the player
        # recognised the stream. Hidden by default; the player sets the adaptive
        # label (Full-SBS 3D / SBS 3D / Full-TAB 3D / TAB 3D / MVC 3D).
        self.format_badge = QLabel()
        self.format_badge.setObjectName("formatBadge")
        self.format_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.format_badge.setStyleSheet(
            "#formatBadge {"
            " color: #7FEBFF;"
            " background-color: rgba(45, 212, 255, 0.13);"
            " border: 1px solid rgba(45, 212, 255, 0.55);"
            " border-radius: 9px; padding: 2px 10px;"
            " font-size: 11px; font-weight: 700; letter-spacing: 0.4px; }"
        )
        self.format_badge.hide()

        # Stereo mode combo
        self.stereo_mode_combo = QComboBox()
        self.stereo_mode_combo.addItems(["MVC", "Side-by-Side", "Top-Bottom"])
        self.stereo_mode_combo.setMinimumWidth(140)
        self.stereo_mode_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._style_combo(self.stereo_mode_combo)

        # Separator
        sep3 = PremiumSeparator()

        # Volume
        self.volume_slider = PremiumVolumeSlider()

        # Fullscreen
        self.fullscreen_button = PremiumIconButton('fullscreen', 'medium')
        self.fullscreen_button.setToolTip("Fullscreen (F)")

        # Layout
        right_group.addStretch()
        # right_group.addWidget(self.tech_info)
        # right_group.addWidget(sep1)


        # Audio spectrum visualiser — fills the gap between the transport and the 3D controls
        self.vu_meter = PremiumSpectrumMeter()
        right_group.addWidget(self.vu_meter)
        right_group.addSpacing(12)
        right_group.addWidget(self.format_badge)
        right_group.addSpacing(6)
        right_group.addWidget(self.mode_3d_button)
        right_group.addWidget(self.stereo_mode_combo)
        right_group.addWidget(sep2)
        right_group.addWidget(self.volume_slider)
        right_group.addWidget(self.fullscreen_button)

        # Assemble controls row
        # Balanced split (1:0:1) to perfectly center the transport controls
        controls_layout.addLayout(left_group, 1)
        controls_layout.addLayout(center_group, 0)
        controls_layout.addLayout(right_group, 1)

        main_layout.addLayout(controls_layout)

    def _style_combo(self, combo, icon=None):
        """Applies premium style to ComboBox."""
        combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(60, 65, 80, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                padding: 5px 10px;
                padding-right: 25px;
                color: #E0E0E0;
                font-size: 11px;
                font-family: 'Segoe UI';
                min-height: 22px;
            }
            QComboBox:hover {
                background-color: rgba(70, 75, 90, 0.9);
                border: 1px solid rgba(0, 200, 255, 0.4);
            }
            QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #AAA;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2d35;
                color: #E0E0E0;
                selection-background-color: rgba(0, 200, 255, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                padding: 4px;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 10px;
                min-height: 24px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: rgba(0, 200, 255, 0.15);
            }
        """)

    def _connect_signals(self):
        """Connects internal signals."""
        self.play_pause_button.clicked.connect(self.play_toggled)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.fullscreen_button.clicked.connect(self.fullscreen_toggled)
        self.open_file_button.clicked.connect(self.file_opened)
        self.open_disc_button.clicked.connect(self.disc_opened)
        self.archive_button.clicked.connect(self.archive_requested)
        self.volume_slider.volume_changed.connect(self.volume_changed)
        self.time_slider.sliderMoved.connect(lambda pos: self._on_slider_scrub(pos))
        # ANTI-SPAM: Use scrub_finished (debounced) instead of sliderReleased
        self.time_slider.scrub_finished.connect(self._emit_seek)
        # Keep sliderReleased as a backup for simple clicks
        self.time_slider.sliderReleased.connect(self._on_slider_released)
        self.mode_3d_button.toggled.connect(self.mode_3d_toggled)
        self.stereo_mode_combo.currentTextChanged.connect(self._on_stereo_mode_changed)
        self.audio_track_combo.currentIndexChanged.connect(self._on_audio_track_changed)
        self.subtitle_track_combo.currentIndexChanged.connect(self._on_subtitle_track_changed)

    def stop_all_animations(self):
        """Stop all button animations - call before destruction or MVC mode."""
        buttons = [
            self.open_file_button,
            self.open_disc_button,
            self.archive_button,
            self.skip_back_button,
            self.stop_button,
            self.play_pause_button,
            self.skip_forward_button,
            self.mode_3d_button,
            self.fullscreen_button
        ]
        for btn in buttons:
            if hasattr(btn, 'stop_animations'):
                btn.stop_animations()

    def enable_all_animations(self):
        """Re-enable button animations after MVC mode ends."""
        buttons = [
            self.open_file_button,
            self.open_disc_button,
            self.archive_button,
            self.skip_back_button,
            self.stop_button,
            self.play_pause_button,
            self.skip_forward_button,
            self.mode_3d_button,
            self.fullscreen_button
        ]
        for btn in buttons:
            if hasattr(btn, 'enable_animations'):
                btn.enable_animations()

    # === PUBLIC API ===

    def set_status_info(self, text, status_type="info", active=False):
        """Updates the status badge. (NO-OP: Badge removed)"""
        # self.status_badge.set_status(text, status_type, active)
        pass

    def set_format_badge(self, text, tooltip="Decoded by edge264"):
        """Show the contextual 3D-format badge (only for detected stereo streams)."""
        if not text:
            self.clear_format_badge()
            return
        self.format_badge.setText(text)
        self.format_badge.setToolTip(tooltip)
        self.format_badge.show()

    def clear_format_badge(self):
        """Hide the 3D-format badge (2D content / stopped / mpv fallback)."""
        self.format_badge.hide()

    def set_tech_info(self, resolution="", fps="", codec=""):
        """Updates the technical info."""
        self.tech_info.set_info(resolution, fps, codec)

    def set_mvc_active(self, active):
        """Activates/deactivates visual MVC mode."""
        self.time_slider.set_mvc_active(active)
        if active:
            # V7b+++ STUTTER FIX: Stop ALL button animations in MVC mode
            # These constant repaints cause stuttering when window has focus
            self.stop_all_animations()
        else:
            # V7b+++ Re-enable animations when leaving MVC mode
            self.enable_all_animations()

    def set_paused(self, is_paused):
        """Changes play/pause icon."""
        self.play_pause_button.icon_type = 'play' if is_paused else 'pause'
        self.play_pause_button.update()

    def set_fullscreen_icon(self, is_fullscreen):
        """Changes fullscreen icon."""
        self.fullscreen_button.icon_type = 'exit_fullscreen' if is_fullscreen else 'fullscreen'
        self.fullscreen_button.update()

    def set_duration(self, seconds):
        """Sets total duration."""
        if seconds is None:
            return
        new_max = int(seconds * 1000)
        if self.time_slider.maximum() != new_max:
            self.time_slider.setRange(0, new_max)
        self.time_slider.setEnabled(seconds > 0)
        self.duration_label.setText(self._format_time(seconds))

    def set_time(self, seconds):
        """Updates the current position."""
        if seconds is not None and not self.time_slider.isSliderDown():
            self.time_slider.setValue(int(seconds * 1000))
        self.time_label.setText(self._format_time(seconds))

    def set_buffer_progress(self, progress):
        """Updates the buffer indicator (0-1)."""
        self.time_slider.set_buffer_progress(progress)

    def enable_3d_controls(self, enabled):
        """Activates/deactivates 3D controls."""
        self.mode_3d_button.setEnabled(enabled)
        self.stereo_mode_combo.setEnabled(enabled)

    def update_audio_tracks(self, tracks):
        """Updates the audio track list."""
        self.audio_track_combo.blockSignals(True)
        self.audio_track_combo.clear()
        if not tracks:
            self.audio_track_combo.addItem("No tracks")
            self.audio_track_combo.setEnabled(False)
        else:
            self.audio_track_combo.addItem("Select...")
            for track_id, title, lang in tracks:
                # Compact but readable format
                if lang:
                    label = f"{title} [{lang.upper()}]"
                else:
                    label = title
                self.audio_track_combo.addItem(label, track_id)
            self.audio_track_combo.setEnabled(True)
            self.audio_track_combo.setCurrentIndex(1)
        self.audio_track_combo.blockSignals(False)

    def update_subtitle_tracks(self, tracks):
        """Updates the subtitle list."""
        logger.info(f"[UI] update_subtitle_tracks called with {len(tracks) if tracks else 0} tracks")
        self.subtitle_track_combo.blockSignals(True)
        self.subtitle_track_combo.clear()
        self.subtitle_track_combo.addItem("None", 0)
        if tracks:
            for track_id, title, lang in tracks:
                # Compact but readable format
                if lang:
                    label = f"{title} [{lang.upper()}]"
                else:
                    label = title
                logger.info(f"[UI]   Adding subtitle: track_id={track_id}")
                self.subtitle_track_combo.addItem(label, track_id)
        self.subtitle_track_combo.setEnabled(True)
        self.subtitle_track_combo.blockSignals(False)
        logger.info(f"[UI] subtitle_track_combo now has {self.subtitle_track_combo.count()} items")

    def update_subtitle_tracks_streaming(self, tracks):
        """Update subtitle list from streaming tracks (MVC demuxer format).

        Args:
            tracks: List of dicts with {trackNumber, codecId, language, name, isPGS}
        """
        logger.info(f"[UI] update_subtitle_tracks_streaming called with {len(tracks) if tracks else 0} tracks")
        self.subtitle_track_combo.blockSignals(True)
        self.subtitle_track_combo.clear()
        self.subtitle_track_combo.addItem("None", 0)
        if tracks:
            # Convert streaming format to tuple format and add to combo
            for i, track in enumerate(tracks):
                track_number = track.get('trackNumber', i + 1)
                name = (track.get('name') or '').strip()
                language = (track.get('language') or '').strip()
                is_pgs = track.get('isPGS', False)
                codec = track.get('codecId', '')

                # Format label with codec type indication
                if is_pgs:
                    codec_label = "PGS"
                elif 'UTF8' in codec:
                    codec_label = "SRT"
                else:
                    codec_label = (codec.split('/')[-1] if '/' in codec else codec) or ''

                # Drop meaningless placeholder names (e.g. "TRACK_1"); build a
                # readable label from language + codec instead.
                low = name.lower()
                is_placeholder = (not name) or (
                    low.startswith('track') and low[5:].strip(' _0123456789') == ''
                )
                parts = []
                if name and not is_placeholder:
                    parts.append(name)               # already conveys the language (BD PGS)
                elif language:
                    parts.append(f"[{language.upper()}]")
                if codec_label:
                    parts.append(f"({codec_label})")
                label = " ".join(parts) if parts else f"Subtitle {i + 1}"

                # Use 1-based index for UI (matches track selection logic in GUI)
                ui_index = i + 1
                logger.info(f"[UI]   Adding streaming subtitle: ui_index={ui_index}, trackNumber={track_number}, isPGS={is_pgs}")
                self.subtitle_track_combo.addItem(label, ui_index)

        self.subtitle_track_combo.setEnabled(True)
        self.subtitle_track_combo.blockSignals(False)
        logger.info(f"[UI] subtitle_track_combo now has {self.subtitle_track_combo.count()} items (streaming mode)")

    # === INTERNAL ===

    def _format_time(self, seconds):
        if seconds is None:
            return "00:00"
        s = int(seconds)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h > 0:
            return f"{h}:{m:02}:{s:02}"
        return f"{m:02}:{s:02}"

    def _emit_seek(self, value=None):
        """Emits seeked signal with value in seconds."""
        # The value received from scrub_finished is already in seconds
        if value is not None:
            self.seeked.emit(float(value))
        else:
            # Fallback: use the slider value (in ms) converted to seconds
            self.seeked.emit(self.time_slider.value() / 1000.0)

    def _on_slider_released(self):
        """
        Called when slider is released (backup for simple clicks).
        Debounced scrub_finished handles most cases, but this ensures
        simple clicks are processed even if scrub_finished didn't trigger.
        """
        # Do nothing - scrub_finished handles the seek via mouseReleaseEvent
        pass

    def _on_slider_scrub(self, pos):
        self.time_label.setText(self._format_time(pos / 1000.0))

    def _on_stereo_mode_changed(self, text):
        mode_map = {"MVC": "mvc", "Side-by-Side": "sbs", "Top-Bottom": "tab"}
        self.stereo_mode_changed.emit(mode_map.get(text, "auto"))

    def _on_audio_track_changed(self, index):
        if index > 0:
            track_id = self.audio_track_combo.itemData(index)
            if track_id is not None:
                self.audio_track_changed.emit(track_id)

    def _on_subtitle_track_changed(self, index):
        track_id = self.subtitle_track_combo.itemData(index)
        logger.info(f"[UI] _on_subtitle_track_changed: index={index}, track_id={track_id}")
        if track_id is not None:
            logger.info(f"[UI] Emitting subtitle_track_changed({track_id})")
            self.subtitle_track_changed.emit(track_id)
        else:
            logger.info(f"[UI] track_id is None, signal NOT emitted")

    # === PAINT ===

    def paintEvent(self, event):
        """Draws glassmorphism background."""
        try:
            painter = QPainter(self)
            if not painter.isActive():
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect = QRectF(0, 0, self.width(), self.height())

            # Background with glassmorphism effect
            path = QPainterPath()
            path.addRoundedRect(rect, 16, 16)

            # Main background
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(35, 38, 48, 245))
            gradient.setColorAt(1, QColor(25, 28, 35, 250))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawPath(path)

            # Top highlight (glass effect)
            highlight_path = QPainterPath()
            highlight_rect = QRectF(0, 0, self.width(), 40)
            highlight_path.addRoundedRect(highlight_rect, 16, 16)
            highlight_gradient = QLinearGradient(0, 0, 0, 40)
            highlight_gradient.setColorAt(0, QColor(255, 255, 255, 15))
            highlight_gradient.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setBrush(QBrush(highlight_gradient))
            painter.drawPath(highlight_path)

            # Border
            painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # Subtle cyan glow at top
            glow_gradient = QLinearGradient(0, 0, self.width(), 0)
            glow_gradient.setColorAt(0, QColor(0, 200, 255, 0))
            glow_gradient.setColorAt(0.5, QColor(0, 200, 255, 20))
            glow_gradient.setColorAt(1, QColor(0, 200, 255, 0))
            painter.setPen(QPen(QBrush(glow_gradient), 2))
            painter.drawLine(QPointF(20, 1), QPointF(self.width() - 20, 1))
        except Exception:
            pass  # Ignore paint errors


# =============================================================================
# COMPATIBILITY WRAPPER
# =============================================================================

class ControlsOverlay(PremiumControlsOverlay):
    """
    Compatibility wrapper to replace old ControlsOverlay.
    Inherits from PremiumControlsOverlay while maintaining existing API.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def show_animated(self):
        self.show()

    def hide_animated(self):
        self.hide()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    # Dark theme
    app.setStyleSheet("""
        QMainWindow { background-color: #1a1a1e; }
    """)

    window = QMainWindow()
    window.setWindowTitle("Premium Controls Overlay - Test")
    window.resize(1200, 200)

    overlay = PremiumControlsOverlay()
    overlay.setFixedHeight(120)
    window.setCentralWidget(overlay)

    # Test data
    overlay.set_duration(7200)  # 2 hours
    overlay.set_time(1234)
    overlay.set_tech_info("1920*1080", "23.976 fps", "MVC")
    overlay.set_mvc_active(True)
    overlay.update_audio_tracks([
        (1, "English", "eng"),
        (2, "French", "fra"),
        (3, "German", "deu"),
    ])
    overlay.update_subtitle_tracks([
        (1, "English", "eng"),
        (2, "French", "fra"),
    ])

    window.show()
    sys.exit(app.exec())
