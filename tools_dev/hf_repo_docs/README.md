---
license: apache-2.0
base_model:
  - depth-anything/DA3-BASE
  - depth-anything/DA3-SMALL
tags:
  - depth-estimation
  - onnx
  - stereoscopic
---

# SyLC 3D Player — depth models

ONNX exports of **Depth Anything V3** used by
[SyLC 3D Player](https://github.com/Symphoenix/SyLC) for real-time 2D→3D
stereoscopic conversion, plus a pre-built TensorRT engine cache.

## `onnx/` — the models (required)

Local re-exports at fixed inference grids, produced by `tools_dev/da3_to_onnx.py`
in the SyLC repository. Fixed shapes rather than dynamic axes: the dynamic
export's position-embedding `Resize` has no DirectML kernel and falls back to
the CPU, costing 342 ms per depth map against 55 ms here.

| Directory | Files | Size | Notes |
|---|---|---|---|
| `onnx/small/` | 10 | 960 MB | DA3-SMALL. Makes all three SyLC presets work. |
| `onnx/base/` | 10 | 3.67 GB | DA3-BASE. The quality upgrade. |

Each family holds the two square grids (756, 518) plus eight adaptive
rectangles (756×406/378/350/322, 518×280/266/238/210) selected automatically
for letterboxed or already-cropped wide sources.

The player downloads these itself — **2D→3D menu → Depth models…** — verifying each
file against a SHA-256 manifest. Manual download into `models/` works too.

## `trt/` — pre-built TensorRT engines (optional, RTX 40-series only)

See `trt/sm89-trt10.16.1.11/README.md`. Read it before downloading 3.1 GB you
may not be able to use.

## Licence and attribution

These are derivative exports of the Depth Anything V3 **Small** and **Base**
checkpoints, both **Apache-2.0**:

- https://huggingface.co/depth-anything/DA3-SMALL
- https://huggingface.co/depth-anything/DA3-BASE

Copyright © the Depth Anything V3 authors. Licence text:
https://www.apache.org/licenses/LICENSE-2.0

The **Large** and **Giant** DA3 variants are CC-BY-NC-4.0 and are **never**
exported, published or redistributed here.
