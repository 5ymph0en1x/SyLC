"""Centralized keyboard action mapping for SyLC."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent


class KeyboardAction:
    PLAY_PAUSE = "play_pause"
    STOP = "stop"
    FULLSCREEN = "fullscreen"
    MUTE = "mute"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    SEEK_BACKWARD_5 = "seek_backward_5"
    SEEK_FORWARD_5 = "seek_forward_5"
    SEEK_BACKWARD_30 = "seek_backward_30"
    SEEK_FORWARD_30 = "seek_forward_30"
    AV_SYNC_DELAY = "av_sync_delay"
    AV_SYNC_ADVANCE = "av_sync_advance"
    TOGGLE_3D = "toggle_3d"
    ALWAYS_ON_TOP = "always_on_top"


def resolve_action(event: QKeyEvent):
    """Return KeyboardAction string or None."""
    key = event.key()
    mods = event.modifiers()
    ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

    if key == Qt.Key.Key_Space:
        return KeyboardAction.PLAY_PAUSE
    if key == Qt.Key.Key_Escape:
        return KeyboardAction.STOP
    if key == Qt.Key.Key_F:
        return KeyboardAction.FULLSCREEN
    if key == Qt.Key.Key_M:
        return KeyboardAction.MUTE
    if key == Qt.Key.Key_Left:
        if ctrl:
            return KeyboardAction.SEEK_BACKWARD_30
        return KeyboardAction.SEEK_BACKWARD_5
    if key == Qt.Key.Key_Right:
        if ctrl:
            return KeyboardAction.SEEK_FORWARD_30
        return KeyboardAction.SEEK_FORWARD_5
    if key == Qt.Key.Key_Up:
        return KeyboardAction.VOLUME_UP
    if key == Qt.Key.Key_Down:
        return KeyboardAction.VOLUME_DOWN
    if key == Qt.Key.Key_BracketLeft:
        return KeyboardAction.AV_SYNC_ADVANCE
    if key == Qt.Key.Key_BracketRight:
        return KeyboardAction.AV_SYNC_DELAY
    if key == Qt.Key.Key_D:
        return KeyboardAction.TOGGLE_3D
    if key == Qt.Key.Key_T and ctrl:
        return KeyboardAction.ALWAYS_ON_TOP
    return None


SHORTCUT_LABELS = {
    KeyboardAction.PLAY_PAUSE: "Espace",
    KeyboardAction.STOP: "Échap",
    KeyboardAction.FULLSCREEN: "F",
    KeyboardAction.MUTE: "M",
    KeyboardAction.SEEK_BACKWARD_5: "←",
    KeyboardAction.SEEK_FORWARD_5: "→",
    KeyboardAction.SEEK_BACKWARD_30: "Ctrl + ←",
    KeyboardAction.SEEK_FORWARD_30: "Ctrl + →",
    KeyboardAction.VOLUME_UP: "↑",
    KeyboardAction.VOLUME_DOWN: "↓",
    KeyboardAction.AV_SYNC_ADVANCE: "[",
    KeyboardAction.AV_SYNC_DELAY: "]",
    KeyboardAction.TOGGLE_3D: "D",
    KeyboardAction.ALWAYS_ON_TOP: "Ctrl + T",
}
