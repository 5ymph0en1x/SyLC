"""Profile the complete MatAnyone2 + contour-lock path on a real video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sylc.synth3d_matting_service import (  # noqa: E402
    MatAnyone2Runtime,
    MatAnyone2Service,
    MatteAdvector,
)


def distribution(values):
    if not values:
        return {}
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "mean": float(np.mean(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def planar_i420(bgr):
    height, width = bgr.shape[:2]
    flat = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    luma = height * width
    chroma = luma // 4
    return (
        flat[:luma].reshape(height, width),
        flat[luma:luma + chroma].reshape(height // 2, width // 2),
        flat[luma + chroma:].reshape(height // 2, width // 2),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--seek", type=float, default=600.0,
                        help="start position in seconds")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    runtime = MatAnyone2Runtime.discover(ROOT)
    if runtime is None:
        raise SystemExit("complete offline MatAnyone2 runtime not found")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not fps > 0.0:
        raise SystemExit("video reports no usable frame rate")
    capture.set(cv2.CAP_PROP_POS_MSEC, args.seek * 1000.0)

    service = MatAnyone2Service(runtime, target_fps=fps, short_side=2160)
    if not service.start():
        raise SystemExit(f"worker did not start: {service.status()}")
    ready_deadline = time.monotonic() + 40.0
    while time.monotonic() < ready_deadline:
        status = service.status()
        if status["state"] in ("ready", "running"):
            break
        if status["state"] == "error":
            raise SystemExit(status["error"])
        time.sleep(0.05)
    else:
        raise SystemExit("worker startup timed out")

    advector = MatteAdvector()
    start = time.monotonic()
    frame_count = 0
    output_seen = 0
    lock_ms = []
    accepted = 0
    stage_samples = {name: [] for name in (
        "inference_ms", "model_ms", "upload_ms", "readback_ms", "prep_ms")}
    try:
        while time.monotonic() - start < max(1.0, args.seconds):
            ok, bgr = capture.read()
            if not ok:
                break
            planes = planar_i420(bgr)
            pts_ms = args.seek * 1000.0 + frame_count * 1000.0 / fps
            advector.note_frame(planes[0], pts_ms)
            service.configure_media(fps=fps, short_side=planes[0].shape[0])
            service.submit_yuv(planes, pts_ms)
            matte = service.latest_for_pts(pts_ms)
            if matte is not None:
                lock_started = time.perf_counter()
                projected = advector.advect(matte, pts_ms)
                lock_ms.append((time.perf_counter() - lock_started) * 1000.0)
                accepted += projected is not None

            status = service.status()
            if status["outputs"] != output_seen:
                output_seen = status["outputs"]
                if output_seen > 8:  # exclude seed/official warm-up frames
                    for name in stage_samples:
                        stage_samples[name].append(float(status[name]))
            frame_count += 1
            delay = start + frame_count / fps - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
    finally:
        capture.release()
        final_status = service.status()
        service.stop(1.0)
        if advector._flow_pool is not None:
            advector._flow_pool.shutdown()

    elapsed = time.monotonic() - start
    report = {
        "video": str(args.video.resolve()),
        "source_fps": fps,
        "presented_fps": frame_count / max(elapsed, 1.0e-9),
        "frames": frame_count,
        "outputs": output_seen,
        "dropped": int(final_status["dropped"]),
        "reported_output_fps": float(final_status["fps"]),
        "short_side": int(final_status["short_side"]),
        "transport": final_status["transport"],
        "stages": {name: distribution(values)
                   for name, values in stage_samples.items()},
        "contour_lock_ms": distribution(lock_ms),
        "contour_accepted": accepted,
        "contour_attempts": len(lock_ms),
        "contour_status": advector.status(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                             encoding="utf-8")


if __name__ == "__main__":
    main()
