# Third-Party Notices

SyLC 3D Player's real-time 2D->3D AI conversion feature bundles or
redistributes the following third-party components. The Depth Anything V3
model weights below are no longer bundled in the player binary; they are
redistributed from the project's own HuggingFace repository
(https://huggingface.co/Symphoenix/sylc_TRT) and downloaded on demand. The
attribution obligation is unchanged by that: SyLC is still the redistributor.
This notice is provided to satisfy those attribution requirements; it does not
modify SyLC's own license.

## Depth Anything V3 Small (model weights)

- **What**: `models/da3_small_756.onnx`, the second candidate of the "Quality"
  depth preset's own model chain for real-time 2D->3D conversion (used when
  that preset's `da3_base_756.onnx` below is absent); and
  `models/da3_small_518.onnx`, the same weights re-exported at a 518x518
  inference grid, which is the "Performance" preset's model and the "Balanced"
  preset's second candidate; and the adaptive exports at 756x406, 756x378,
  756x350, 756x322, 518x280, 518x266, 518x238 and 518x210, selected for
  stable encoded mattes or an already-cropped wide source. Distributed in the
  `small` pack.
- **Source**: local re-exports at the fixed square and adaptive grids above,
  of the original Depth Anything V3 "Small" checkpoint at
  https://huggingface.co/depth-anything/DA3-SMALL
  (`tools_dev/da3_to_onnx.py --variant small --size 756`, the same command
  with `--size 518`, or paired `--width` / `--height`). Earlier development builds instead shipped
  `models/da3_small.onnx`, the community ONNX export at
  https://huggingface.co/onnx-community/depth-anything-v3-small (pinned
  revision `0b6a7f3bf5595f9950b91389e0da3a0de130324c`, see
  `tools_dev/fetch_da3_model.py`), which derives from the same DA3-SMALL
  checkpoint and carries the same license; it is no longer bundled.
- **License**: Apache License 2.0, verified against the model repository's
  card metadata (`license: apache-2.0`, model `depth-anything/DA3-SMALL`).
  Only the Small and Base checkpoints are used; the Large and Giant Depth
  Anything V3 variants are licensed CC-BY-NC-4.0 and are never fetched,
  bundled, or redistributed by SyLC.
- **Copyright**: (c) the Depth Anything V3 authors, per the model card at
  the source repository above.
- **License text**: https://www.apache.org/licenses/LICENSE-2.0

## Depth Anything V3 Base (model weights)

- **What**: `models/da3_base_756.onnx`, the depth-estimation model of the
  "Quality" depth preset (the default preset for real-time 2D->3D conversion)
  and the first candidate of its chain; `da3_small_756.onnx` above is the next
  candidate of that same chain when this file is absent. Also
  `models/da3_base_518.onnx`, the same weights re-exported at a 518x518
  inference grid, which is the "Balanced" preset's first candidate; and
  the adaptive Base exports at 756x406, 756x378, 756x350, 756x322, 518x280,
  518x266, 518x238 and 518x210. Distributed in the `base` pack.
- **Source**: local re-exports at the fixed square and adaptive grids above,
  of the original Depth Anything V3 "Base" checkpoint at
  https://huggingface.co/depth-anything/DA3-BASE
  (`tools_dev/da3_to_onnx.py --variant base --size 756`, the same command
  with `--size 518`, and paired `--width` / `--height` arguments for each
  adaptive graph).
- **License**: Apache License 2.0, verified verbatim against the model
  repository's `README.md` frontmatter (`license: apache-2.0`). Only the
  Small and Base checkpoints are used; the Large and Giant Depth Anything V3
  variants are licensed CC-BY-NC-4.0 (verified `license: cc-by-nc-4.0` on the
  DA3-GIANT model card) and are never fetched, bundled, or redistributed by
  SyLC.
- **Copyright**: (c) the Depth Anything V3 authors, per the model card at
  the source repository above.
- **License text**: https://www.apache.org/licenses/LICENSE-2.0

## ONNX Runtime

- **What**: `onnxruntime.dll`, the inference engine that runs the depth
  model (`mvc_realtime_demuxer/src/depth_engine.cpp` dynamically loads it; no
  static linking).
- **Version**: 1.24.4 (`Microsoft.ML.OnnxRuntime.DirectML` NuGet package).
- **Source**: https://github.com/Microsoft/onnxruntime
- **License**: MIT License.
- **Copyright**: (c) Microsoft Corporation.

## DirectML

- **What**: `DirectML.dll`, the DirectX 12 machine-learning execution
  provider ONNX Runtime uses for GPU-accelerated inference on Windows.
- **Version**: 1.15.4 (`Microsoft.AI.DirectML` NuGet package).
- **Source**: https://aka.ms/DirectML
- **License**: Microsoft Software License Terms (redistributable package;
  not open source). Distributed as-is per those terms alongside
  `onnxruntime.dll`.
- **Copyright**: (c) Microsoft Corporation.
