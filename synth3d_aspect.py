"""Aspect-bucket selection for Synth3D fixed-shape ONNX exports.

The native worker reports both encoded horizontal mattes and the uncropped
source dimensions.  This module turns either representation into a content
aspect ratio and selects the closest installed fixed rectangular graph.  It
deliberately never invents a path: absence, ambiguity or a poor shape match
keeps the square preset unchanged.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass


PATCH = 14
MAX_HEIGHT_ERROR = PATCH
MIN_PIXEL_SAVING = 0.10
# A crop-free 16:9 frame must remain on the square graph.  Sources at or above
# this ratio are considered genuinely pre-cropped widescreen masters.
MIN_NATIVE_WIDE_RATIO = 1.80


@dataclass(frozen=True)
class AspectSelection:
    model_path: str
    grid_width: int
    grid_height: int
    crop_top: int
    crop_bottom: int
    source_width: int
    source_height: int

    @property
    def crop_top_norm(self) -> float:
        return self.crop_top / float(self.source_height)

    @property
    def crop_bottom_norm(self) -> float:
        return self.crop_bottom / float(self.source_height)


def effective_content_ratio(source_width: int, source_height: int,
                            crop_top: int, crop_bottom: int) -> float | None:
    active_height = int(source_height) - int(crop_top) - int(crop_bottom)
    if source_width <= 0 or source_height <= 0 or active_height <= 0:
        return None
    if crop_top < 0 or crop_bottom < 0:
        return None
    return source_width / float(active_height)


def nearest_patch_height(width: int, ratio: float) -> int:
    if width <= 0 or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("width and ratio must be positive")
    # Round half upward rather than using Python's bankers' rounding.  When the
    # ideal height lies exactly between two patch grids, preserve more pixels.
    patches = max(1, int(math.floor((width / ratio) / PATCH + 0.5)))
    return patches * PATCH


def _model_root(filename: str, square_side: int) -> str:
    stem, _ = os.path.splitext(filename)
    suffix = f"_{int(square_side)}"
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def select_installed_aspect_model(square_model_path: str, square_side: int,
                                  source_width: int, source_height: int,
                                  crop_top: int, crop_bottom: int):
    """Return an :class:`AspectSelection` or ``None`` for the square fallback."""
    ratio = effective_content_ratio(
        source_width, source_height, crop_top, crop_bottom)
    if ratio is None or square_side <= 0 or not square_model_path:
        return None

    folder = os.path.dirname(square_model_path) or "."
    root = _model_root(os.path.basename(square_model_path), square_side)
    pattern = re.compile(
        rf"^{re.escape(root)}_(\d+)x(\d+)\.onnx$", re.IGNORECASE)
    try:
        names = os.listdir(folder)
    except OSError:
        return None

    ideal_height = square_side / ratio
    best = None
    best_key = None
    for name in names:
        match = pattern.match(name)
        if not match:
            continue
        width, height = map(int, match.groups())
        if width != square_side or width % PATCH or height % PATCH:
            continue
        if height <= 0 or height > width:
            continue
        saving = 1.0 - (width * height) / float(square_side * square_side)
        if saving < MIN_PIXEL_SAVING:
            continue
        height_error = abs(height - ideal_height)
        if height_error > MAX_HEIGHT_ERROR:
            continue
        candidate_ratio = width / float(height)
        ratio_error = abs(candidate_ratio / ratio - 1.0)
        # Prefer the closest tensor height.  Remaining ties favour lower ratio
        # distortion, then the taller graph (more detail), then filename for a
        # deterministic choice independent of os.listdir ordering.
        key = (height_error, ratio_error, -height, name.lower())
        path = os.path.join(folder, name)
        if os.path.isfile(path) and (best_key is None or key < best_key):
            best = AspectSelection(
                path, width, height, crop_top, crop_bottom,
                source_width, source_height)
            best_key = key
    return best
