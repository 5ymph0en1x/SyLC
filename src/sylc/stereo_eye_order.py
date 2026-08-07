# -*- coding: utf-8 -*-
"""Small, dependency-free stereo eye-order helpers shared by playback/export.

The canonical contract used by the Apple MV-HEVC exporter is:

    left_first   first packed plane / MVC base view is the left eye
    right_first  first packed plane / MVC base view is the right eye
    unknown      the source carries no trustworthy association

Keeping ``unknown`` distinct is intentional.  Guessing left-first at detection
time made a right-base MVC or right-first packed source look valid while
silently producing a reversed Apple Spatial file.
"""

LEFT_FIRST = "left_first"
RIGHT_FIRST = "right_first"
UNKNOWN = "unknown"
VALID_EYE_ORDERS = frozenset((LEFT_FIRST, RIGHT_FIRST, UNKNOWN))


# RFC 9559, Matroska StereoMode values.  Only layouts SyLC can route are
# classified; checkerboard/interleaved values retain their eye order but are
# not advertised as exportable packed SBS/TAB.
_MATROSKA_MODES = {
    0: (None, UNKNOWN),
    1: ("sbs", LEFT_FIRST),
    2: ("tab", RIGHT_FIRST),
    3: ("tab", LEFT_FIRST),
    4: (None, RIGHT_FIRST),
    5: (None, LEFT_FIRST),
    6: (None, RIGHT_FIRST),
    7: (None, LEFT_FIRST),
    8: (None, RIGHT_FIRST),
    9: (None, LEFT_FIRST),
    10: ("anaglyph", UNKNOWN),
    11: ("sbs", RIGHT_FIRST),
    12: ("anaglyph", UNKNOWN),
    13: ("mvc", LEFT_FIRST),
    14: ("mvc", RIGHT_FIRST),
}


def normalise_eye_order(value, default=UNKNOWN):
    """Return a canonical eye order without converting ``unknown`` to a guess."""
    if isinstance(value, bool):
        return RIGHT_FIRST if value else LEFT_FIRST
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        LEFT_FIRST: LEFT_FIRST,
        "left": LEFT_FIRST,
        "left_eye_first": LEFT_FIRST,
        "lr": LEFT_FIRST,
        "l_r": LEFT_FIRST,
        RIGHT_FIRST: RIGHT_FIRST,
        "right": RIGHT_FIRST,
        "right_eye_first": RIGHT_FIRST,
        "rl": RIGHT_FIRST,
        "r_l": RIGHT_FIRST,
        UNKNOWN: UNKNOWN,
        "auto": UNKNOWN,
        "": UNKNOWN,
    }
    result = aliases.get(text)
    if result is not None:
        return result
    return default if default in VALID_EYE_ORDERS else UNKNOWN


def _optional_bool(value):
    """Parse ffprobe-style booleans; return None when the field is absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return None


def classify_stereo_value(value):
    """Return ``(layout, eye_order)`` from ffprobe/Matroska stereo metadata."""
    if value is None:
        return None, UNKNOWN
    text = str(value).strip().lower()
    try:
        numeric = int(text, 10)
    except (TypeError, ValueError):
        numeric = None
    if numeric in _MATROSKA_MODES:
        return _MATROSKA_MODES[numeric]

    mode = text.replace("-", "_").replace(" ", "_")
    if any(k in mode for k in (
            "anaglyph", "cyan", "magenta", "red_cyan", "cyan_red")):
        return "anaglyph", UNKNOWN

    if any(k in mode for k in (
            "frame_altern", "framealternate", "frame_packing",
            "frame_sequential", "frame_packed", "view_packed", "mvc",
            "framepacking", "frameinterleaved", "block_lr", "block_rl",
            "both_eyes_laced", "packed")):
        layout = "mvc"
    elif any(k in mode for k in (
            "top_bottom", "bottom_top", "tab", "over_under", "under_over",
            "block_tb", "block_bt", "topbottom", "tbr", "tbl")):
        layout = "tab"
    elif any(k in mode for k in (
            "side_by_side", "sbs", "left_right", "right_left",
            "sbsl", "sbsr")):
        layout = "sbs"
    else:
        return None, UNKNOWN

    right_markers = (
        "right_left", "right_eye_first", "right_first", "block_rl",
        "bottom_top", "under_over", "block_bt", "sbsr", "tbr",
    )
    left_markers = (
        "left_right", "left_eye_first", "left_first", "block_lr",
        "top_bottom", "over_under", "block_tb", "sbsl", "tbl",
    )
    if any(k in mode for k in right_markers):
        order = RIGHT_FIRST
    elif any(k in mode for k in left_markers):
        order = LEFT_FIRST
    else:
        order = UNKNOWN
    return layout, order


def eye_order_from_stereo_value(value, inverted=None):
    """Resolve eye order, letting an explicit AVStereo3D invert flag win."""
    layout, order = classify_stereo_value(value)
    inv = _optional_bool(inverted)
    if inv is not None and layout in ("sbs", "tab"):
        return RIGHT_FIRST if inv else LEFT_FIRST
    return order


def effective_view_swap(eye_order=UNKNOWN, *manual_swaps):
    """True when canonical output must swap the two decoded/packed input views."""
    swap = normalise_eye_order(eye_order) == RIGHT_FIRST
    for value in manual_swaps:
        swap ^= bool(value)
    return swap
