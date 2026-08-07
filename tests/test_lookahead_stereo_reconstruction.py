import numpy as np

from sylc.lookahead_stereo_reconstruction import (
    FlowField,
    reconstruction_metrics,
    reconstruct_stereo,
)


def _moving_occluder_scene(height=48, width=128):
    yy, xx = np.mgrid[0:height, 0:width]
    background = np.empty((height, width, 3), dtype=np.float32)
    background[..., 0] = 0.16 + 0.68 * xx / (width - 1)
    background[..., 1] = 0.42 + 0.08 * np.sin(xx * 0.19)
    background[..., 2] = 0.58 + 0.07 * np.cos(yy * 0.23)

    current = background.copy()
    future = background.copy()
    current_depth = np.full((height, width), 0.20, np.float32)
    future_depth = current_depth.copy()
    rows = slice(6, height - 6)
    current[rows, 50:80] = (0.84, 0.31, 0.67)
    future[rows, 75:105] = (0.84, 0.31, 0.67)
    current_depth[rows, 50:80] = 0.90
    future_depth[rows, 75:105] = 0.90
    flow = FlowField(
        np.zeros((height, width), np.float32),
        np.zeros((height, width), np.float32),
        np.ones((height, width), np.float32),
    )
    return background, current, future, current_depth, future_depth, flow


def test_future_background_improves_only_stereo_holes():
    (background, current, future, current_depth,
     future_depth, flow) = _moving_occluder_scene()
    result = reconstruct_stereo(
        current, future, current_depth, flow,
        future_depth=future_depth,
        strength_pct=20.0,
        convergence=0.20,
        confidence_threshold=0.18,
    )

    accepted = np.concatenate((
        result.left.accepted.ravel(), result.right.accepted.ravel()))
    assert accepted.sum() > 100
    metrics = reconstruction_metrics(result)
    assert metrics["lookahead_coverage_pct"] > 15.0

    # Background sits exactly on the convergence plane, so its ground-truth
    # destination colour is the same x/y coordinate in either eye.
    baseline_error = []
    lookahead_error = []
    for eye in (result.left, result.right):
        mask = eye.accepted
        baseline_error.append(np.abs(eye.baseline[mask] - background[mask]))
        lookahead_error.append(np.abs(eye.lookahead[mask] - background[mask]))
        # Temporal evidence is forbidden from rewriting a valid source preimage.
        assert np.array_equal(
            eye.lookahead[eye.supported], eye.baseline[eye.supported])
    before = np.concatenate(baseline_error).mean()
    after = np.concatenate(lookahead_error).mean()
    assert after < 0.72 * before


def test_future_foreground_is_rejected_by_depth_evidence():
    (_, current, future, current_depth,
     future_depth, flow) = _moving_occluder_scene()
    result = reconstruct_stereo(
        current, future, current_depth, flow,
        future_depth=future_depth,
        strength_pct=20.0,
        convergence=0.20,
        confidence_threshold=0.18,
    )

    # In the future frame the foreground occupies x=75..104.  Any accepted
    # zero-flow candidate must therefore come from outside that interval.
    for eye in (result.left, result.right):
        _, xs = np.nonzero(eye.accepted)
        assert not np.any((xs >= 75) & (xs < 105))


def test_no_holes_means_exact_noop():
    height, width = 24, 40
    rng = np.random.default_rng(7)
    frame = rng.random((height, width, 3), dtype=np.float32)
    depth = np.full((height, width), 0.5, np.float32)
    flow = FlowField(
        np.zeros_like(depth), np.zeros_like(depth), np.ones_like(depth))
    result = reconstruct_stereo(
        frame, frame, depth, flow, strength_pct=1.5, convergence=0.5)
    for eye in (result.left, result.right):
        assert not eye.holes.any()
        assert np.array_equal(eye.lookahead, frame)


def test_flow_field_rejects_mismatched_shapes():
    with np.testing.assert_raises(ValueError):
        FlowField(
            np.zeros((3, 4), np.float32),
            np.zeros((3, 5), np.float32),
            np.ones((3, 4), np.float32),
        )

