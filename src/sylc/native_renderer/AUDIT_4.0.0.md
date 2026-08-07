# Native Renderer — 4.0.0 code audit (no-hardware pass)

Verdict of the Phase-0 audit (code-level review of `native_renderer.cpp/.h`,
`python_bindings.cpp`, `native_framepack_widget.py`, `native_tap.py`): the
**S1+S2 implementation is correct and memory-safe as written.** No speculative
changes were made to working code in Phase 0 — retyping a decoder/shader path to
"prove" originality would only add risk. Each residual item below is tied to the
stage that can actually validate it on the 3D display.

## Confirmed safe (no action needed)
- `ComPtr` RAII throughout; `shutdown()` releases every D3D resource and is
  idempotent; the destructor calls `shutdown()` then frees `Impl`. No leaks.
- One internal `std::mutex` serializes present / upload / resize / pause /
  shutdown; no locked public method calls another locked one (no self-deadlock).
- cbuffer layout `static_assert(sizeof(FrameCB) == 48)` matches the HLSL
  `packoffset`s exactly.
- pybind: the GIL is released around the blocking calls (initialize / resize /
  present / pause / resume). `set_yuv_frame` validates ndim/dtype; `forcecast`
  guarantees contiguity before upload.
- Import graph + `NATIVE_RENDERER_AVAILABLE == True` verified under Python 3.14
  (rebuilt cp314 `.pyd`).

## Residual items — each addressed at the stage that can verify it
| # | Item | Stage |
|---|---|---|
| 1 | BT.601 matrix on BT.709 content — latent color error, faithfully reproduced for parity | **S2** (A/B vs Qt) |
| 2 | Subtitle rect normalized by video size, not the PGS composition reference (cropped 2.39:1) | **S4** |
| 3 | Seek/pause = flag under the shared mutex; explicit fence/flush before texture realloc not yet added | **S3** |
| 4 | `set_yuv_frame` holds the GIL across the memcpy (fine on the GUI thread; release it for the decode-thread push) | **S3 / S5b** |
| 5 | Framepack/SBS/TAB sample right-eye SRVs that may be null on the very first frame (renders black, not a crash) | **S2** |
| 6 | Single swapchain — simultaneous dual output (embedded 2D preview + detached 3D) not yet unified | **S4** |
| 7 | `is_paused()` / `is_hdr()` read status flags without the lock (benign status reads) | **S3** (atomics if needed) |

Cross-reference: `NATIVE_RENDERER_DESIGN.md` §9 (risk register) and §10 (latent issues).
Validation gate for every GPU-touching stage is **on-hardware 3D confirmation** —
which only the user's stereoscopic display can provide.
