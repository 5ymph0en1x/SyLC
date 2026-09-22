# -*- coding: utf-8 -*-
"""Unit tests for MacOSSynth3DEngine."""

import os
import numpy as np
import pytest

from sylc.macos_synth3d_engine import MacOSSynth3DEngine, IMAGENET_MEAN, IMAGENET_STD


def test_imagenet_constants():
    assert np.allclose(IMAGENET_MEAN, [0.485, 0.456, 0.406])
    assert np.allclose(IMAGENET_STD, [0.229, 0.224, 0.225])


def test_engine_initial_state():
    engine = MacOSSynth3DEngine()
    assert not engine.is_running()
    assert not engine.is_ready_for_frame()
    assert engine.status() == "Engine: off"
    assert engine.get_latest_depth() is None


def test_submit_frame_when_not_running():
    engine = MacOSSynth3DEngine()
    dummy_frame = np.zeros((518, 518, 3), dtype=np.uint8)
    # Should safely return False without error
    assert engine.submit_frame(dummy_frame) is False


def test_submit_qimage_null_handling():
    engine = MacOSSynth3DEngine()
    assert engine.submit_qimage(None) is False


def test_depth_inversion_and_quantization_math():
    """Verify that inverse depth mapping, percentile scaling, and RGBA packing are bitwise consistent."""
    # Synthetic raw depth map (distance: 1.0 to 10.0)
    raw_depth = np.linspace(1.0, 10.0, 518 * 518, dtype=np.float32).reshape((518, 518))
    
    # Invert to nearness (1.0 down to 0.1)
    inv_depth = 1.0 / raw_depth
    assert np.all(np.isfinite(inv_depth))
    assert inv_depth[0, 0] > inv_depth[-1, -1]  # Nearer objects have higher nearness

    # Percentile scaling
    p2, p98 = np.percentile(inv_depth, [2.0, 98.0])
    span = max(1e-6, float(p98 - p2))
    norm_depth = np.clip((inv_depth - p2) / span, 0.0, 1.0)
    assert 0.0 <= norm_depth.min() <= 0.05
    assert 0.95 <= norm_depth.max() <= 1.0

    # S-curve
    smooth = norm_depth * norm_depth * (3.0 - 2.0 * norm_depth)
    norm_depth = norm_depth + 0.25 * (smooth - norm_depth)
    norm_depth = np.clip(0.5 + 0.92 * (norm_depth - 0.5), 0.0, 1.0)

    # 16-bit uint16 quantization & RGBA byte packing
    depth_u16 = (norm_depth * 65535.0).astype(np.uint16)
    hi = (depth_u16 >> 8).astype(np.uint8)
    lo = (depth_u16 & 0xFF).astype(np.uint8)

    packed = np.zeros((518, 518, 4), dtype=np.uint8)
    packed[..., 0] = hi
    packed[..., 1] = lo
    packed[..., 3] = 255

    # Verify that decoding RGBA in shader matches original nearness
    decoded_nearness = (packed[..., 0].astype(np.float32) * (65280.0 / 65535.0) / 255.0 +
                        packed[..., 1].astype(np.float32) * (255.0 / 65535.0) / 255.0)
    # Error should be within 1 LSB of 16-bit quantization (1 / 65535 < 0.00002)
    max_err = np.max(np.abs(decoded_nearness - (depth_u16 / 65535.0)))
    assert max_err < 1.0 / 65535.0
