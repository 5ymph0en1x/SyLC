from pathlib import Path

import cv2
import numpy as np

from sylc.synth3d_matting_service import (
    MatAnyone2Runtime,
    MatAnyone2Service,
    MatteAdvector,
    MatteFrame,
)


def _runtime_stub():
    root = Path("unused")
    return MatAnyone2Runtime(root, root, root, root, root)


def test_matanyone_defaults_keep_full_yuv_and_cap_model_at_512(monkeypatch):
    monkeypatch.delenv("SYLC_MATANYONE2_AUTO_CAP", raising=False)
    monkeypatch.delenv("SYLC_MATANYONE2_TRANSPORT", raising=False)
    service = MatAnyone2Service(
        _runtime_stub(), target_fps=23.976, short_side=808)
    assert service.short_side == 512
    assert service.auto_cap == 512
    assert service.transport == "yuv420"


def test_direct_partition_percentile_matches_numpy():
    rng = np.random.default_rng(4207)
    values = rng.normal(size=(96, 160)).astype(np.float32)
    mask = rng.random(values.shape) > 0.37
    for q in (0.0, 50.0, 65.0, 75.0, 90.0, 100.0):
        sampled = values[::2, ::2][mask[::2, ::2]]
        if sampled.size > 4096:
            sampled = sampled[::int(np.ceil(sampled.size / 4096.0))]
        expected = float(np.percentile(sampled, q))
        actual = MatteAdvector._masked_percentile(values, mask, q, -1.0)
        assert actual == expected


def test_parallel_contour_flow_is_identical_to_sequential(monkeypatch):
    rng = np.random.default_rng(1949)
    base = rng.integers(16, 236, size=(96, 160), dtype=np.uint8)
    current = np.empty_like(base)
    current[:, :3] = base[:, :1]
    current[:, 3:] = base[:, :-3]
    alpha = np.zeros((96, 160), dtype=np.uint8)
    cv2.ellipse(alpha, (80, 48), (24, 35), 0, 0, 360, 255, -1)
    matte = MatteFrame(1, 1, 1000.0, alpha, 0.0, False, False)

    outputs = {}
    statuses = {}
    advectors = []
    try:
        for mode in ("bidirectional", "parallel"):
            monkeypatch.setenv("SYLC_SYNTH3D_CONTOUR_LOCK_MODE", mode)
            advector = MatteAdvector()
            advectors.append(advector)
            advector.note_frame(base, 1000.0)
            advector.note_frame(current, 1041.7)
            outputs[mode] = advector.advect(matte, 1041.7)
            statuses[mode] = advector.status()

        assert (outputs["parallel"] is None) == (
            outputs["bidirectional"] is None)
        if outputs["parallel"] is not None:
            assert np.array_equal(
                outputs["parallel"].alpha, outputs["bidirectional"].alpha)
            assert np.array_equal(
                outputs["parallel"].reliability,
                outputs["bidirectional"].reliability)
        for field in ("kind", "confidence", "local_reject_pct", "sparse_pct"):
            assert statuses["parallel"][field] == statuses["bidirectional"][field]
    finally:
        for advector in advectors:
            if advector._flow_pool is not None:
                advector._flow_pool.shutdown()
