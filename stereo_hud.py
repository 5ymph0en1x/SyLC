"""Stereo playback HUD for the native SyLC video surfaces.

The existing PremiumControlsOverlay remains the only source of actions/state.
This controller renders it into a transparent RGBA texture, sends that texture
to every live NativeFramepackWidget, and maps mouse/wheel input from either eye
back to the canonical Qt children. Combo boxes and QMenu-backed buttons use a
small texture-native popup so they never fall back to a single 2D Tool window.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QLinearGradient,
                           QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent)
from PySide6.QtWidgets import (QApplication, QAbstractButton, QComboBox, QLabel,
                               QSlider, QWidget)


logger = logging.getLogger("SyLC.StereoHUD")


_MOUSE_TYPES = {
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseButtonRelease,
    QEvent.Type.MouseButtonDblClick,
    QEvent.Type.MouseMove,
}


class StereoHudController(QObject):
    """
    Manages the 3D HUD (Heads-Up Display) overlay logic.
    Captures the off-screen PremiumControlsOverlay and generates a texture.
    """
    ACTIVE_MODES = {"mvc", "sbs", "tab", "glasses", "dual"}
    DEFAULT_DISPARITY = 0.003
    MAX_POPUP_ROWS = 8
    POPUP_ROW_H = 34

    def __init__(self, host, overlay):
        super().__init__(host)
        self._first_push_done = False
        self.host = host
        self.overlay = overlay
        self.active = False
        self.desired_visible = bool(overlay.isVisible())
        self.interacting = False
        # Horloge d'interaction propre au HUD : le poll de la barre s'en sert
        # pour faire expirer un drapeau `interacting` resté collé (il ne peut
        # pas utiliser _nav_last_activity, qu'il rafraîchit lui-même).
        self.last_interaction = 0.0

        self._targets = []
        self._target_rects = {}
        self._canvas_size = (1, 1)
        self._bar_offset = 0
        self._last_capture = 0.0
        self._last_active_signature = None
        self._pressed_widget = None
        self._pressed_output = None
        self._hover_widget = None

        # Popup is a dict with owner/kind/entries/rect/scroll/hover/pressed.
        # QComboBox entries store original item indices; menu entries store QAction.
        self._popup = None
        self._popup_stack = []

        self._timer = QTimer(self)
        self._timer.setInterval(50)  # smooth VU/timeline without video-rate CPU capture
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------ state
    def set_desired_visible(self, visible):
        self.desired_visible = bool(visible)
        if not self.desired_visible:
            self._close_popup()
        self.sync(force=True)

    @property
    def busy(self):
        """Keep auto-hide suspended during a drag or while a HUD menu is open."""
        return bool(self.interacting or self._popup is not None)

    def sync(self, force=False):
        mode = str(getattr(self.host, "current_stereo_mode", "")).lower()
        enabled = bool(getattr(self.host, "has_media", False)
                       and getattr(self.host, "is_3d_enabled", False)
                       and mode in self.ACTIVE_MODES)

        has_external = False
        fp = getattr(self.host, 'framepacking_window', None)
        if fp and getattr(fp, 'isVisible', lambda: False)():
            has_external = True
        eyes = getattr(self.host, 'eye_windows', None)
        if eyes and any(e and getattr(e, 'isVisible', lambda: False)() for e in eyes):
            has_external = True

        hide_native = not has_external

        targets = []
        if enabled:
            try:
                candidates = list(self.host._display_widgets())
            except Exception:
                candidates = []
            for widget in candidates:
                if not hide_native and widget is getattr(self.host, 'mvc_embedded_widget', None):
                    continue
                if not hasattr(widget, "set_hud") or not hasattr(widget, "clear_hud"):
                    continue
                if not getattr(widget, "_have_hud_api", True):
                    continue
                try:
                    if not widget.isVisible() or not widget.window().isVisible():
                        continue
                except RuntimeError:
                    continue
                targets.append(widget)

        signature = (enabled, mode, tuple(id(w) for w in targets))
        if not force and signature == self._last_active_signature:
            return
        self._last_active_signature = signature

        old_targets = list(self._targets)
        new_ids = {id(w) for w in targets}
        for widget in old_targets:
            if id(widget) not in new_ids:
                self._detach_target(widget, clear=True)
        old_ids = {id(w) for w in old_targets}
        for widget in targets:
            if id(widget) not in old_ids:
                self._attach_target(widget)
        self._targets = targets

        was_active = self.active
        self.active = bool(enabled and targets)

        time_slider = getattr(self.overlay, 'time_slider', None)
        if time_slider:
            # La preview native est une fenêtre Tool 2D positionnée en
            # coordonnées de la fenêtre principale : HUD actif (mono OU
            # pilote/FramePack), elle surgirait mal placée et en un seul
            # exemplaire par-dessus la sortie 3D. La vignette est composée
            # DANS la texture HUD à la place (chaque œil, alignée au survol).
            time_slider.setProperty('hud_mode', self.active)
            if self.active and hasattr(time_slider, '_preview_widget'):
                time_slider._preview_widget.hide()

        was_dont_show = self.overlay.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)

        if self.active:
            self.overlay.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, hide_native)
            
            if hide_native:
                # Mode Mono-écran (SBS dans la fenêtre principale). HUD et barre native liés.
                if self.desired_visible:
                    if was_dont_show != hide_native:
                        QTimer.singleShot(0, self._remap_native_overlay)
                    elif not self.overlay.isVisible():
                        self.overlay.setVisible(True)
                        
                    if not was_active or not self._first_push_done:
                        self._first_push_done = False
                        self._capture_and_push(force_refresh=True)
                else:
                    self.overlay.setVisible(False)
                    self._clear_targets(refresh=True)
            else:
                # Mode Pilote (FramePack). La barre native suit le cycle normal des 5 secondes
                if was_dont_show != hide_native:
                    QTimer.singleShot(0, self._remap_native_overlay)
                
                # Le HUD (sur le FramePack) s'efface ou se dessine indépendamment
                if self.desired_visible:
                    if not was_active or not self._first_push_done:
                        self._first_push_done = False
                        self._capture_and_push(force_refresh=True)
                else:
                    self._clear_targets(refresh=True)
        else:
            self._close_popup()
            self._clear_targets(refresh=True, targets=old_targets)
            self._first_push_done = False
            
            self.overlay.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)
            if was_dont_show:
                QTimer.singleShot(0, self._remap_native_overlay)
            elif self.desired_visible:
                self.overlay.setVisible(True)
                self.overlay.raise_()
            else:
                self.overlay.setVisible(False)

        if was_active != self.active:
            logger.info("[STEREO-HUD] %s mode=%s targets=%d (hide_native=%s)",
                        "enabled" if self.active else "disabled", mode, len(targets), hide_native)

    def _remap_native_overlay(self):
        """Force l'OS à détruire ou recréer physiquement la fenêtre native sans bloquer Qt."""
        if self.overlay and self.desired_visible:
            # Enlever les conditions ici garantit que hide() et show()
            # forcent le gestionnaire de l'OS à accepter le nouvel attribut.
            self.overlay.hide()
            self.overlay.show()
            self.overlay.raise_()

    def _capture_and_push(self, force_refresh=False):
        if not self.active or not self.desired_visible:
            return

        has_external = False
        fp = getattr(self.host, 'framepacking_window', None)
        if fp and getattr(fp, 'isVisible', lambda: False)():
            has_external = True
        eyes = getattr(self.host, 'eye_windows', None)
        if eyes and any(e and getattr(e, 'isVisible', lambda: False)() for e in eyes):
            has_external = True

        hide_native = not has_external

        was_visible = self.overlay.isVisible()
        # On ne triche avec la visibilité OS que si la barre native doit être cachée
        if hide_native:
            self.overlay.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            if not was_visible:
                self.overlay.setVisible(True)

        width = max(1, int(self.overlay.width()))
        height = max(1, int(self.overlay.height()))

        if width <= 1 or height <= 1:
            if hide_native and not was_visible:
                self.overlay.setVisible(False)
            return

        if not self._first_push_done:
            self.overlay.ensurePolished()
            self._first_push_done = True

        # Timeline hover vignette, against the REAL PremiumTimelineSlider API:
        # hover is `_hover_pos` (pixel x, -1 = none — NOT the old TimeSlider's
        # `_is_hovering`), the cache holds (pixmap, exact, idr_s) TUPLES keyed
        # by round(seconds), and `_nearest_cached` supplies the same approx
        # fallback the native tooltip shows while the exact thumb decodes.
        # The reserved band takes the ACTUAL pixmap height (a fixed 85 px band
        # clipped the XL tooltip entirely off-canvas).
        active_slider = getattr(self.overlay, 'time_slider', None)
        vignette_pix = None
        vignette_h = 0
        hover_px = getattr(active_slider, '_hover_pos', -1) if active_slider else -1
        if hover_px is not None and hover_px >= 0:
            hover_key = round(getattr(active_slider, '_hover_time', -1.0))
            cache = getattr(active_slider, '_preview_cache', None) or {}
            entry = cache.get(hover_key)
            candidate = entry[0] if isinstance(entry, tuple) else entry
            if candidate is None:
                nearest = getattr(active_slider, '_nearest_cached', None)
                if callable(nearest):
                    try:
                        candidate = nearest(hover_key)
                    except Exception:
                        candidate = None
            if candidate is not None and not candidate.isNull():
                vignette_pix = candidate
                vignette_h = candidate.height() + 34   # image + timestamp chip

        popup_h = max(self._popup_height(), vignette_h)
        self._bar_offset = popup_h
        canvas_h = height + popup_h

        image = QImage(width, canvas_h, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)

        if not painter.isActive():
            if hide_native and not was_visible:
                self.overlay.setVisible(False)
            return

        try:
            self.overlay.render(painter, QPoint(0, popup_h))

            if self._popup is not None:
                self._layout_popup(width, popup_h)
                self._paint_popup(painter)

            if vignette_pix is not None:
                # Centered on the hovered pixel (each eye shows it aligned
                # with its own hover tick), drawn entirely inside the reserved
                # band just above the bar — never clipped. `_hover_pos` is the
                # pixel x on the slider: no time/width math, no ms-vs-s trap
                # (maximum() is in MILLISECONDS on this slider).
                slider_pos = active_slider.mapTo(self.overlay, QPoint(0, 0))
                hover_x = slider_pos.x() + int(hover_px)

                pw, ph = vignette_pix.width(), vignette_pix.height()
                draw_x = max(8, min(hover_x - pw // 2, width - pw - 8))
                draw_y = max(4, popup_h - ph - 28)

                painter.setPen(QPen(QColor(69, 194, 245, 220), 2))
                painter.setBrush(QColor(20, 22, 30))
                painter.drawRoundedRect(
                    QRectF(draw_x, draw_y, pw, ph), 6, 6)
                painter.drawPixmap(draw_x, draw_y, vignette_pix)

                # Timestamp chip under the image — the same information the
                # native floating tooltip carries.
                t = int(max(0.0, active_slider._hover_time))
                label = f"{t // 3600:02d}:{(t // 60) % 60:02d}:{t % 60:02d}"
                painter.setFont(self._popup_font(9))
                metrics = painter.fontMetrics()
                chip_w = metrics.horizontalAdvance(label) + 16
                chip_h = metrics.height() + 6
                chip_x = max(8.0, min(float(hover_x) - chip_w / 2.0,
                                      float(width - chip_w - 8)))
                chip_y = float(draw_y + ph + 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(20, 22, 30, 235))
                painter.drawRoundedRect(
                    QRectF(chip_x, chip_y, chip_w, chip_h), 6, 6)
                painter.setPen(QColor(238, 241, 247))
                painter.drawText(
                    QRectF(chip_x, chip_y, chip_w, chip_h),
                    Qt.AlignmentFlag.AlignCenter, label)

        finally:
            painter.end()
            # Restauration de l'état UNIQUEMENT si on avait manipulé la visibilité
            if hide_native and not was_visible:
                self.overlay.setVisible(False)

        bpl = int(image.bytesPerLine())
        raw = np.frombuffer(image.bits(), dtype=np.uint8,
                            count=int(image.sizeInBytes())).reshape(canvas_h, bpl)

        rgba = np.array(raw[:, :width * 4].reshape(canvas_h, width, 4),
                        dtype=np.uint8, order="C", copy=True)
        self._canvas_size = (width, canvas_h)
        self._last_capture = time.monotonic()

        for target in list(self._targets):
            try:
                rect = self._hud_rect(target, width, canvas_h)
                self._target_rects[id(target)] = rect
                target.set_hud(rgba, *rect, disparity=self.DEFAULT_DISPARITY, opacity=1.0)
                if force_refresh and hasattr(target, "refresh_hud"):
                    target.refresh_hud()
            except RuntimeError:
                continue
            except Exception:
                pass

    def shutdown(self):
        self._timer.stop()
        self._clear_targets(refresh=False)
        for widget in list(self._targets):
            self._detach_target(widget, clear=False)
        self._targets = []
        self.active = False

    # --------------------------------------------------------------- rendering
    def _tick(self):
        self.sync()
        if not self.active or not self.desired_visible:
            return
        now = time.monotonic()
        playing = bool(getattr(self.host, "is_playing", False))
        min_period = 0.05 if playing or self.interacting else 0.20
        if now - self._last_capture >= min_period:
            self._capture_and_push(force_refresh=not playing)

    def _source_video_aspect(self, target):
        if hasattr(target, "source_aspect"):
            try:
                val = float(target.source_aspect)
                if val > 0.1:
                    return val
            except Exception:
                pass
        return self._eye_aspect(target)

    def _hud_rect(self, target, texture_w, texture_h):
        dims = getattr(target, "_last_eye_size", None)
        if dims and dims[0] > 0 and dims[1] > 0:
            target_w = float(dims[0])
            target_h = float(dims[1])
        else:
            target_w = float(max(1, target.width()))
            target_h = float(max(1, target.height()))
        
        video_aspect = self._source_video_aspect(target)
        screen_aspect = target_w / target_h
        
        safe_x = 10.0 / target_w
        rect_w = 1.0 - (2.0 * safe_x)
        
        if screen_aspect < video_aspect:
            # Letterbox: video height is smaller than screen height
            video_h = target_w / video_aspect
            black_bar_h = (target_h - video_h) / 2.0
            
            # Map coordinates relative to video frame into the bottom black bar
            v_bottom = 1.0 + ((black_bar_h - 20.0) / video_h)
            rect_h = float(texture_h) / video_h
            rect_y = v_bottom - rect_h
        else:
            # Pillarbox or exact fit: video fills height
            safe_bottom = 20.0 / target_h
            aspect = self._eye_aspect(target)
            rect_h = rect_w * aspect * (float(texture_h) / float(max(1, texture_w)))
            rect_y = 1.0 - safe_bottom - rect_h

        rect_h = min(0.90, max(0.04, rect_h))
        return (safe_x, rect_y, rect_w, rect_h)

    def _eye_aspect(self, target):
        mode = str(getattr(self.host, "current_stereo_mode", "")).lower()
        if mode == "glasses":
            try:
                value = float(self.host._glasses_eye_aspect())
                if value > 0.0:
                    return value
            except Exception:
                pass
        try:
            value = float(getattr(target, "source_aspect", 0.0))
            if value > 0.0:
                return value
        except Exception:
            pass
        dims = getattr(target, "_last_eye_size", None)
        if dims and dims[1]:
            return float(dims[0]) / float(dims[1])
        return 16.0 / 9.0

    def _clear_targets(self, refresh=False, targets=None):
        for target in list(self._targets if targets is None else targets):
            try:
                target.clear_hud()
                self._target_rects.pop(id(target), None)
                if refresh and hasattr(target, "refresh_hud"):
                    target.refresh_hud()
            except (RuntimeError, AttributeError):
                pass

    # --------------------------------------------------------------- input map
    def _attach_target(self, widget):
        widget.installEventFilter(self)
        widget.setMouseTracking(True)

    def _detach_target(self, widget, clear):
        try:
            widget.removeEventFilter(self)
            if clear:
                widget.clear_hud()
                widget.refresh_hud()
        except (RuntimeError, AttributeError):
            pass
        self._target_rects.pop(id(widget), None)

    def eventFilter(self, watched, event):
        if watched not in self._targets or not self.active:
            return super().eventFilter(watched, event)
        et = event.type()
        if et not in _MOUSE_TYPES and et != QEvent.Type.Wheel:
            return super().eventFilter(watched, event)
        self.last_interaction = time.monotonic()
        try:
            self.host._mark_activity()
        except Exception:
            pass
        if not self.desired_visible:
            # Barres masquées par l'auto-hide : cet événement sert UNIQUEMENT
            # à les réveiller (_mark_activity -> show_controls). Le poll
            # global est géométrique (frameGeometry/WindowFromPoint) et
            # fragile en multi-écrans/DPI — ce filtre est le chemin de réveil
            # fiable, il doit écouter même barre cachée. Ne rien router :
            # un clic donné barre invisible ne doit pas actionner un contrôle.
            return super().eventFilter(watched, event)
        if et == QEvent.Type.Wheel:
            handled = self._route_wheel(watched, event)
        else:
            handled = self._route_mouse(watched, event)
        if handled:
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _texture_point(self, output, position):
        rect = self._target_rects.get(id(output))
        if rect is None:
            return None
        uv = output.stereo_hud_video_uv(position.x(), position.y())
        if uv is None:
            return None
        rx, ry, rw, rh = rect
        # The visual disparity is only ~0.15% per eye: ignoring its sign here
        # keeps identical hit geometry in both copies and is far below the
        # deliberately enlarged touch targets.
        if rw <= 0.0 or rh <= 0.0:
            return None
        tx = (float(uv[0]) - rx) / rw * self._canvas_size[0]
        ty = (float(uv[1]) - ry) / rh * self._canvas_size[1]
        return QPointF(tx, ty)

    def _widget_at_texture_point(self, point):
        ox = int(round(point.x()))
        oy = int(round(point.y() - self._bar_offset))
        if ox < 0 or oy < 0 or ox >= self.overlay.width() or oy >= self.overlay.height():
            return None, None
        overlay_pos = QPoint(ox, oy)
        child = self.overlay.childAt(overlay_pos) or self.overlay
        local = child.mapFrom(self.overlay, overlay_pos)
        return child, QPointF(local)

    def _route_mouse(self, output, event):
        point = self._texture_point(output, event.position())
        et = event.type()

        popup_entry = self._popup_entry_at(point) if point is not None else None
        if self._popup is not None and et == QEvent.Type.MouseMove:
            hover = popup_entry[0] if popup_entry is not None else -1
            if self._popup.get("hover") != hover:
                self._popup["hover"] = hover
                self._last_capture = 0.0
            return point is not None
        if self._popup is not None and et == QEvent.Type.MouseButtonPress:
            if popup_entry is not None:
                self._popup["pressed"] = popup_entry[0]
                self.interacting = True
                return True
            self._close_popup()
            self._last_capture = 0.0
        elif self._popup is not None and et == QEvent.Type.MouseButtonRelease:
            if popup_entry is not None:
                row, entry = popup_entry
                if self._popup.get("pressed") == row:
                    self._activate_popup_entry(entry, point)
                self.interacting = False
                return True
            self._close_popup()

        if point is None:
            # Preserve a slider drag that leaves the HUD by clamping its local
            # position; otherwise release the press cleanly.
            if self._pressed_widget is None:
                return False
            point = QPointF(
                min(max(event.position().x(), 0.0), float(self.overlay.width() - 1)),
                float(self._bar_offset + self.overlay.height() - 1))

        child, local = self._widget_at_texture_point(point)
        if et == QEvent.Type.MouseMove and self._pressed_widget is not None:
            child = self._pressed_widget
            overlay_point = QPoint(int(round(point.x())),
                                   int(round(point.y() - self._bar_offset)))
            local = QPointF(child.mapFrom(self.overlay, overlay_point))
        if child is None:
            return False

        if et == QEvent.Type.MouseMove:
            self._update_hover(child)
        if et == QEvent.Type.MouseButtonPress:
            self._pressed_widget = child
            self._pressed_output = output
            self.interacting = True
            # QComboBox and menu buttons normally create a native top-level
            # popup on press. Consume that press: release opens our per-eye
            # texture popup instead.
            if isinstance(child, QComboBox) or self._button_menu(child) is not None:
                return True
        if et == QEvent.Type.MouseButtonRelease and self._pressed_widget is not None:
            child = self._pressed_widget
            overlay_point = QPoint(int(round(point.x())),
                                   int(round(point.y() - self._bar_offset)))
            local = QPointF(child.mapFrom(self.overlay, overlay_point))

        # Texture-native popups prevent Qt from creating one unsplit top-level
        # list over the movie. Open on release to retain normal button feedback.
        open_popup = (et == QEvent.Type.MouseButtonRelease
                      and child is self._pressed_widget)
        if open_popup and isinstance(child, QComboBox):
            self._open_combo_popup(child)
            self._pressed_widget = None
            self._pressed_output = None
            self.interacting = False
            return True
        menu = self._button_menu(child) if open_popup else None
        if menu is not None:
            self._open_menu_popup(child, menu)
            self._pressed_widget = None
            self._pressed_output = None
            self.interacting = False
            return True

        global_pos = QPointF(child.mapToGlobal(QPoint(int(local.x()), int(local.y()))))
        clone = QMouseEvent(et, local, global_pos, event.button(), event.buttons(),
                            event.modifiers(), event.pointingDevice())
        QApplication.sendEvent(child, clone)
        if et == QEvent.Type.MouseButtonRelease:
            self._pressed_widget = None
            self._pressed_output = None
            self.interacting = False
        self._last_capture = 0.0
        return True

    def _route_wheel(self, output, event):
        point = self._texture_point(output, event.position())
        popup_hit = self._popup_entry_at(point) if self._popup is not None else None
        if popup_hit is not None:
            _row, entry = popup_hit
            slider = entry.get("slider")
            if slider is not None:
                step = max(1, int(slider.singleStep()))
                slider.setValue(slider.value() + (step if event.angleDelta().y() > 0 else -step))
                slider.sliderReleased.emit()
                self._last_capture = 0.0
                return True
            total = len(self._popup["entries"])
            maximum = max(0, total - self.MAX_POPUP_ROWS)
            delta = -1 if event.angleDelta().y() > 0 else 1
            self._popup["scroll"] = min(max(0, self._popup["scroll"] + delta), maximum)
            self._last_capture = 0.0
            return True
        if point is None:
            return False
        child, local = self._widget_at_texture_point(point)
        if child is None:
            return False
        global_pos = QPointF(child.mapToGlobal(QPoint(int(local.x()), int(local.y()))))
        clone = QWheelEvent(local, global_pos, event.pixelDelta(), event.angleDelta(),
                            event.buttons(), event.modifiers(), event.phase(),
                            event.inverted(), event.source(), event.pointingDevice())
        QApplication.sendEvent(child, clone)
        self._last_capture = 0.0
        return True

    def _update_hover(self, child):
        if child is self._hover_widget:
            return
        if self._hover_widget is not None:
            QApplication.sendEvent(self._hover_widget, QEvent(QEvent.Type.Leave))
        self._hover_widget = child
        QApplication.sendEvent(child, QEvent(QEvent.Type.Enter))

    # ----------------------------------------------------------------- popups
    def _button_menu(self, child):
        if not isinstance(child, QAbstractButton):
            return None
        getter = getattr(child, "menu", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def _open_combo_popup(self, combo):
        entries = []
        model = combo.model()
        for index in range(combo.count()):
            enabled = True
            try:
                enabled = bool(model.flags(model.index(index, combo.modelColumn()))
                               & Qt.ItemFlag.ItemIsEnabled)
            except Exception:
                pass
            entries.append({"label": combo.itemText(index), "value": index,
                            "enabled": enabled, "checked": index == combo.currentIndex(),
                            "submenu": None})
        self._popup_stack = []
        self._set_popup(combo, "combo", entries)

    def _menu_entries(self, menu, include_back=False):
        # Preserve the canonical QMenu lifecycle: the player refreshes action
        # labels/enablement (current media, Cast availability, model status)
        # from aboutToShow. The HUD does not display the native window, so it
        # emits that preparation signal explicitly before mirroring actions.
        try:
            menu.aboutToShow.emit()
        except Exception:
            pass
        entries = []
        if include_back:
            entries.append({"label": "‹ Back", "value": "__back__", "enabled": True,
                            "checked": False, "submenu": None})
        for action in menu.actions():
            if not action.isVisible():
                continue
            if action.isSeparator():
                entries.append({"label": "", "value": None, "enabled": False,
                                "checked": False, "submenu": None, "separator": True})
                continue
            label = action.text().replace("&", "").split("\t", 1)[0].strip()
            if not label:
                # QWidgetAction sliders in the 2D->3D menu have no QAction
                # text. Mirror their QLabel/QSlider pair rather than silently
                # making fine stereo geometry unavailable in 3D mode.
                getter = getattr(action, "defaultWidget", None)
                holder = getter() if callable(getter) else None
                slider = holder.findChild(QSlider) if holder is not None else None
                label_widget = holder.findChild(QLabel) if holder is not None else None
                if slider is not None:
                    entries.append({"label": label_widget.text() if label_widget else "Value",
                                    "value": slider, "enabled": slider.isEnabled(),
                                    "checked": False, "submenu": None,
                                    "slider": slider})
                    continue
                # The native menus open with an eyebrow+subtitle header and
                # quiet section labels (_add_export_menu_header/_section).
                # Mirror them instead of dropping them, so a HUD menu reads
                # like the same menu, not a bare list of rows.
                if holder is not None:
                    eyebrow = subtitle = section = None
                    for lab in holder.findChildren(QLabel):
                        name = lab.objectName()
                        if name == "sylcMenuEyebrow":
                            eyebrow = lab.text()
                        elif name == "sylcMenuSubtitle":
                            subtitle = lab.text()
                        elif name == "sylcMenuSection":
                            section = lab.text()
                    if eyebrow or section:
                        entries.append({"label": eyebrow or section, "value": None,
                                        "enabled": False, "checked": False,
                                        "submenu": None,
                                        "header": "header" if eyebrow else "section",
                                        "sublabel": subtitle or ""})
                continue
            entries.append({"label": label, "value": action,
                            "enabled": action.isEnabled(),
                            "checked": action.isCheckable() and action.isChecked(),
                            "submenu": action.menu()})
        return entries

    def _open_menu_popup(self, owner, menu):
        self._popup_stack = []
        self._set_popup(owner, "menu", self._menu_entries(menu))

    def _set_popup(self, owner, kind, entries):
        self._popup = {"owner": owner, "kind": kind, "entries": entries,
                       "scroll": 0, "hover": -1, "pressed": -1,
                       "rect": QRectF()}
        self._last_capture = 0.0

    def _close_popup(self):
        self._popup = None
        self._popup_stack = []
        self.interacting = self._pressed_widget is not None
        self._last_capture = 0.0

    def _popup_height(self):
        if self._popup is None:
            return 0
        rows = min(self.MAX_POPUP_ROWS, max(1, len(self._popup["entries"])))
        return rows * self.POPUP_ROW_H + 16

    def _popup_font(self, point_size=10, weight=None, letter_spacing=0.0):
        """The popup's typography mirrors _EXPORT_MENU_STYLE (Segoe UI 10pt)."""
        font = QFont("Segoe UI")
        font.setPointSize(point_size)
        if weight is not None:
            font.setWeight(weight)
        if letter_spacing:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
        return font

    def _layout_popup(self, canvas_w, popup_h):
        if self._popup is None:
            return
        owner = self._popup["owner"]
        owner_pos = owner.mapTo(self.overlay, QPoint(0, 0))
        fm = QFontMetrics(self._popup_font())
        visible = self._popup["entries"][
            self._popup["scroll"]:self._popup["scroll"] + self.MAX_POPUP_ROWS]
        text_w = max((max(fm.horizontalAdvance(e.get("label", "")),
                          fm.horizontalAdvance(e.get("sublabel", "")))
                      for e in visible), default=160)
        width = min(canvas_w - 16, max(220, owner.width(), text_w + 64))
        x = min(max(8, owner_pos.x()), max(8, canvas_w - width - 8))
        self._popup["rect"] = QRectF(float(x), 8.0, float(width), float(popup_h - 12))

    def _paint_popup(self, painter):
        """Painted mirror of the native menus' _EXPORT_MENU_STYLE: same frame,
        same gradient hover, same header/section/disabled palette — so the HUD
        popup reads as the SAME menu the main bar opens, only texture-composed."""
        popup = self._popup
        if popup is None:
            return
        rect = popup["rect"]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._popup_font())
        path = QPainterPath()
        path.addRoundedRect(rect, 12.0, 12.0)
        painter.fillPath(path, QColor(20, 22, 30, 252))
        painter.setPen(QPen(QColor(69, 194, 245, 88), 1.0))
        painter.drawPath(path)

        start = popup["scroll"]
        entries = popup["entries"][start:start + self.MAX_POPUP_ROWS]
        for visible_row, entry in enumerate(entries):
            logical_row = start + visible_row
            row = QRectF(rect.x() + 6.0,
                         rect.y() + 4.0 + visible_row * self.POPUP_ROW_H,
                         rect.width() - 12.0, self.POPUP_ROW_H)
            if entry.get("separator"):
                painter.setPen(QPen(QColor(255, 255, 255, 24), 1.0))
                painter.drawLine(row.left() + 10.0, row.center().y(),
                                 row.right() - 10.0, row.center().y())
                continue
            header = entry.get("header")
            if header == "header":
                painter.setFont(self._popup_font(8, QFont.Weight.Bold, 1.0))
                painter.setPen(QColor(89, 216, 255))
                painter.drawText(row.adjusted(10.0, 3.0, -10.0, -row.height() * 0.5),
                                 Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                                 entry.get("label", ""))
                painter.setFont(self._popup_font(9))
                painter.setPen(QColor(146, 153, 170))
                painter.drawText(row.adjusted(10.0, row.height() * 0.5, -10.0, -1.0),
                                 Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                                 entry.get("sublabel", ""))
                painter.setFont(self._popup_font())
                continue
            if header == "section":
                painter.setFont(self._popup_font(8, QFont.Weight.DemiBold, 1.2))
                painter.setPen(QColor(123, 129, 148))
                painter.drawText(row.adjusted(12.0, 0.0, -10.0, 0.0),
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                                 entry.get("label", ""))
                painter.setFont(self._popup_font())
                continue
            hovered = logical_row == popup.get("hover")
            checked = bool(entry.get("checked"))
            if hovered:
                gradient = QLinearGradient(row.topLeft(), row.topRight())
                gradient.setColorAt(0.0, QColor(0, 200, 255, 58))
                gradient.setColorAt(1.0, QColor(77, 116, 255, 34))
                painter.setPen(QPen(QColor(74, 207, 255, 105), 1.0))
                painter.setBrush(gradient)
                painter.drawRoundedRect(row, 8.0, 8.0)
            elif checked:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 200, 255, 32))
                painter.drawRoundedRect(row, 8.0, 8.0)
            if not entry.get("enabled", True):
                color = QColor(151, 158, 177, 128)
            elif hovered or checked:
                color = QColor(255, 255, 255)
            else:
                color = QColor(240, 242, 247)
            painter.setPen(color)
            slider = entry.get("slider")
            if slider is not None:
                painter.drawText(row.adjusted(10.0, 0.0, -150.0, 0.0),
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                                 entry.get("label", "Value"))
                track_left = row.left() + max(120.0, row.width() * 0.48)
                track_right = row.right() - 32.0
                track_y = row.center().y()
                painter.setPen(QPen(QColor(255, 255, 255, 65), 3.0))
                painter.drawLine(track_left, track_y, track_right, track_y)
                span = max(1, slider.maximum() - slider.minimum())
                fraction = (slider.value() - slider.minimum()) / span
                knob_x = track_left + fraction * (track_right - track_left)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 200, 255))
                painter.drawEllipse(QPointF(knob_x, track_y), 5.0, 5.0)
                painter.setPen(color)
                painter.drawText(row.adjusted(0.0, 0.0, -8.0, 0.0),
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                                 str(slider.value()))
                continue
            prefix = "✓  " if checked else "    "
            painter.drawText(row.adjusted(10.0, 0.0, -26.0, 0.0),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             prefix + entry.get("label", ""))
            if entry.get("submenu") is not None:
                painter.drawText(row.adjusted(0.0, 0.0, -10.0, 0.0),
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, "›")
        painter.restore()

    def _popup_entry_at(self, point):
        if self._popup is None or point is None:
            return None
        rect = self._popup["rect"]
        if not rect.contains(point):
            return None
        row = int((point.y() - rect.y() - 4.0) // self.POPUP_ROW_H)
        index = self._popup["scroll"] + row
        if row < 0 or row >= self.MAX_POPUP_ROWS or index >= len(self._popup["entries"]):
            return None
        entry = self._popup["entries"][index]
        if entry.get("separator") or not entry.get("enabled", True):
            return None
        return index, entry

    def _activate_popup_entry(self, entry, point=None):
        popup = self._popup
        if popup is None:
            return
        value = entry.get("value")
        if value == "__back__":
            if self._popup_stack:
                owner, entries = self._popup_stack.pop()
                self._set_popup(owner, "menu", entries)
            return
        submenu = entry.get("submenu")
        if submenu is not None:
            self._popup_stack.append((popup["owner"], popup["entries"]))
            self._set_popup(popup["owner"], "menu",
                            self._menu_entries(submenu, include_back=True))
            return
        slider = entry.get("slider")
        if slider is not None:
            rect = popup["rect"]
            track_left = rect.left() + max(120.0, (rect.width() - 12.0) * 0.48) + 6.0
            track_right = rect.right() - 38.0
            px = point.x() if point is not None else track_left
            fraction = min(1.0, max(0.0, (px - track_left) /
                                    max(1.0, track_right - track_left)))
            value = round(slider.minimum() + fraction *
                          (slider.maximum() - slider.minimum()))
            slider.setValue(value)
            slider.sliderReleased.emit()
            popup["pressed"] = -1
            self._last_capture = 0.0
            return
        if popup["kind"] == "combo":
            popup["owner"].setCurrentIndex(int(value))
        elif value is not None:
            value.trigger()
        self._close_popup()
