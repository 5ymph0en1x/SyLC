# Native D3D11 Renderer — Design & Capture Spec

Status: **design complete, implementation staged (needs build toolchain + on-hardware 3D validation)**
Goal (Tokyo #3): replace the Qt RHI renderer with a native C++ D3D11 renderer so decoded YUV
planes go straight from edge264 into D3D11 textures, eliminating the per-frame Python copies.
Python keeps play/pause/seek + windowing.

This document is the **contract** the native renderer must satisfy. It is derived from a full
read of the current renderer (`framepacking_widget_d3d11.py`, `framepacking_window_d3d11.py`),
the decode/pacing path (`mvc_decoder.py`), the fan-out/control wiring (`SyLC_3D_Player.py`), the
build system (`mvc_realtime_demuxer/`), and a byte-exact decode of the baked shader.

---

## 0. The one architectural decision that de-risks everything

**Drive presentation from the existing decode-thread pacing — do NOT add a free-running
vblank-driven present loop.**

Two blocker-class hazards come *only* from the naive "C thread free-runs and pushes into D3D11"
framing; both vanish if the renderer presents on-arrival under the decode thread's clock:

1. **Double-pacing / A-V desync (blocker).** Frame scheduling is audio-locked in the *decode
   thread*: the velvet "liquid pacing", `_av_sync_offset_s = 0.75` (tuned to the *current* ~1-frame
   present latency), drift correction, and `_liquid_stretch` gate every emit
   (`mvc_decoder.py` ~608-625, 943-991, 3705-3715). A second clock in the renderer (vblank pull)
   double-paces and brings back the dense-scene judder that work removed. **Rule:** one present per
   pushed frame, present interval 1, single frame in flight, present-on-arrival.

2. **Seek/pause GPU race (blocker).** Today the `0xC0000005` seek crash (V8/V14) is prevented
   because `Qt.QueuedConnection` serializes the handoff onto the GUI/render thread and the **decode
   thread never touches GPU textures**; `pause_rendering()` is a cheap flag. If the decode thread
   both writes textures *and* reallocates them on seek, that serialization is gone. The native
   design keeps a **single GPU-owning thread** and an explicit **fence + pause gate** around seek
   reallocation (see §5).

Net: the renderer becomes native (owns device/swapchain/shader/textures, uploads from raw plane
pointers), but the **threading model is unchanged** — the decode thread, at the exact point it
currently calls `frameYUVReady.emit(...)`, instead calls `renderer.present(planes…)`. This is what
makes the rewrite tractable and keeps the hard-won timing valid.

---

## 1. Exact shader — already extracted, zero reconstruction risk

The baked `.qsb` already contained the compiled **HLSL SM5.0** variant Qt itself uses. Extracted
verbatim to:

- `shaders/yuv_framepack.vert.hlsl` — passthrough (position + texCoord → SV_Position, no transform).
- `shaders/yuv_framepack.frag.hlsl` — the full pixel shader.
- `shaders/yuv_framepack.frag.glsl.ref` — GLSL reference for human reading / diffing.

Resource bindings the C++ renderer MUST match (from the HLSL):

| Slot | Resource |
|---|---|
| `b0` | cbuffer `buf` = `{ int stereo_mode; int subtitle_enabled; /*pad8*/ float4 subtitle_rect; float sdr_white_level; }` |
| `t0` | `texSubtitle` (RGBA8, straight alpha, **not** sRGB-decoded) |
| `t1..t3` | `texY_L, texU_L, texV_L` (R8_UNORM) |
| `t4..t6` | `texY_R, texU_R, texV_R` (R8_UNORM) |
| `s0..s6` | one shared **linear / clamp-to-edge / no-mip** sampler bound to all slots |

Pixel-exact behaviors (do not "modernize"):

- **YUV→RGB = limited-range BT.601**: `y=(y-16/255)*1.16438353`, `u-=.5`, `v-=.5`,
  `r=y+1.402v`, `g=y-0.344136u-0.714136v`, `b=y+1.772u`, then `clamp(0,1)`.
  ⚠️ **Finding:** these are **BT.601** coefficients, applied to HD/Blu-ray content that is almost
  always **BT.709**. This is a *latent color bug in the current build*, not something the port
  introduces. **Reproduce it as-is** for a faithful port; fixing it is a separate, visible change
  (track as a follow-up, A/B on hardware).
- **Vertical flip in the fragment**: `y_flipped = 1.0 - texCoord.y` (not in the vertex UVs).
- **UV clamp** to `[0.001, 0.999]` on the bilinear path **and** both spline paths.
- **Stereo modes** (`stereo_mode`):
  - `0` = 2D: sample **L only**, bilinear.
  - `1` = framepack (RT is 1920×2205): top eye `y_flipped < 1080/2205 (0.4897958934307098)`,
    **45-px gap** band (`rows 1080..1125`) outputs `rgb=0` with `videoUV=(-1,-1)`, bottom eye
    `y_flipped > 1125/2205 (0.5102040767669678)`; bilinear. Use these constants **exactly**, not `0.5`.
  - `2` = SBS: `x<0.5`→L(`x*2`), else R(`(x-0.5)*2`); **spline36 horizontal** resample.
  - `3` = TAB: `y<0.5`→L(`y*2`), else R; **spline36 vertical** resample.
- **Subtitle**: rect-test `videoUV` in normalized `subtitle_rect`, sample `texSubtitle`,
  `rgb = lerp(rgb, sub.rgb, sub.a)`. (Spline36 only for SBS/TAB; bilinear for 2D/framepack.)
- **HDR**: final `rgb *= sdr_white_level`, output `float4(rgb, 1.0)`. There is **no PQ/HDR10/EOTF
  decode in-shader** — content is treated as SDR and scaled into the scRGB-linear FP16 buffer
  (scRGB 1.0 = 80 nits; `sdr_white_level = SDRWhiteLevel_nits / 80`).

---

## 2. Swapchain / HDR (what Qt did for free, now explicit)

- DXGI **flip-model** swapchain, `DXGI_FORMAT_R16G16B16A16_FLOAT`, **2 buffers, FLIP_DISCARD,
  present interval 1**. Match current ~1-frame latency — do **not** use a deep frame-latency-
  waitable queue (would add ~2 frames and invalidate the tuned `0.75 s` audio offset).
- Set scRGB color space explicitly: `IDXGISwapChain3::SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709)`.
- **Framepack RT is offscreen 1920×2205 RGBA16F**, then aspect-preserving-scaled (pillarbox/
  letterbox, §3) to the swapchain. Other modes render at window size.
- **HDR probe + SDR fallback (high):** probe `IDXGIOutput6::GetDesc1` for scRGB support. If absent,
  create an 8-bit swapchain — but then the scRGB-linear output is wrong; the SDR path needs its
  **own output encode** (the current code only "works" because Qt's SDR fallback + `sdr_white_level≈1`
  mask it). Specify the SDR encode explicitly.
- **Fullscreen = fake/borderless only** (Win32 `SetWindowLong`/`SetWindowPos` + `SWP_NOZORDER` +
  `DwmFlush()`), **never DXGI exclusive** — preserves the flip-model HDR swapchain. Re-query SDR
  white level on fullscreen enter.
- Dead code (do not port as live): `check_display_hdr_capability()`, `configure_window_for_hdr()`
  (DWM cloak/peek) — defined but never called.

## 3. Aspect / viewport

Pillarbox when wider than target, letterbox when taller; clear-to-black bars.
`target_aspect`: framepack `1920/2205`; every main-window layout (2D/SBS/TAB) uses
the **actual source display aspect**, so 2.39:1 remains 2.39:1 while switching
presentation.

## 4. Per-frame upload (native D3D11 reality differs from Qt)

- Inputs per frame: 6 planes (L/R × Y/U/V) as **separated** R8 planes + width/height/stride.
  edge264 emits **side-by-side UV** when `stride_c == 2*cw`; the Python `to_np` splits Cb/Cr
  (`mvc_decoder.py` ~4149-4160). The native path reads raw decoder chroma, so it **must port this
  split** or 4:2:0 chroma is garbage.
- **Copy ALL planes row-by-row honoring the real `D3D11_MAPPED_SUBRESOURCE.RowPitch`** (src pitch =
  plane width, dst pitch = mapped RowPitch). The Qt-era "Y tight / pad only U/V" note is a
  *Qt-API artifact*; in raw D3D11 even the 1920-wide Y can map to a padded pitch on some drivers.
  `ALIGN_BYTES=256` in the Python is **declared-but-unused** — read the real RowPitch, don't hardcode.
- Width-pad = **repeat last column** (not zero — avoids right-edge chroma artifacts); height-pad =
  zeros. Reproduce both.
- **Clear new textures to Y=16 / UV=128** (limited-range black + neutral chroma) on create **and on
  resize** — the `_needs_init_clear` path in Python is dead; don't rely on first-frame-overwrites.
- Textures: `R8_UNORM` SRVs, one shared linear/clamp sampler, no mips. Subtitle: `R8G8B8A8_UNORM`,
  straight alpha.
- **Resolution change**: detect plane-size != texture-size, **park the present, fence, reallocate
  all 7 textures + SRVs, resume**. (Today Qt's single render-thread serialization made this safe;
  native must make it explicit.)

## 5. Threading, seek/pause, teardown (the dangerous part)

- **GPU-owning thread = the decode thread** (it already calls the present point). No second present
  thread.
- **Seek/pause:** `pause()` must (a) stop the decode thread from issuing `Map`/`Present`, and (b)
  **fence/flush** so no in-flight GPU read remains, **before** any texture reallocation. A naive
  Python-flag port reproduces V8 with worse symptoms (GPU + CPU now race the same texture).
- **GIL discipline (high):** the present call from the C/decode thread must copy plane
  pointers+strides into C locals **while holding the GIL**, then **release the GIL** around the
  blocking `Map`/`memcpy`/`Present` (holding the GIL across `Present` freezes the UI and starves
  audio), then reacquire only to return. Mirror the existing `gil_scoped_release` + capsule/RAII
  idiom (`python_bindings.cpp` ~89-129). Note the V7b lesson: never touch a Python object un-GIL'd.
- **In-flight bookkeeping:** the present-complete fence must replicate all three current resets —
  clear on hide, re-arm on show only if a frame is pending **and not paused**, and never re-present
  while seek-paused (`framepacking_widget_d3d11.py:258-285`).
- **Teardown ordering (high):** anti-strobe = hide window + clear textures **before** decoder stop
  (`SyLC_3D_Player.py:3176-3184`); then stop present, **drain in-flight Present**, then release
  D3D11 textures/device, then interrupt the decode thread (`wait(5000)`, **never** `terminate()`
  inside edge264.dll). On a **fresh load with no seek**, `seekFinished` never fires, so MVC init must
  **explicitly resume** the present (V57 black-screen-on-reload, `SyLC_3D_Player.py:5008-5022`).

## 6. Dual output + control plane (what Python keeps)

- **Two outputs, normally simultaneous:** the embedded 2D left-eye preview stays **visible for
  timing sync** while the detached framepack window renders 3D (`SyLC_3D_Player.py:4849-4852`). The
  native renderer must support **two swapchains with independent stereo mode + per-swapchain HDR
  white level**, fed from one decoded frame.
- The current 2D-mode right-plane drop is a per-widget local optimization that only works in the
  two-independent-uploader model; a unified native uploader must upload right planes whenever **any**
  visible target is framepack/SBS/TAB.
- **Control API Python must still call** (these are the surviving surface): `create/destroy`,
  `set_stereo_mode(target, mode)`, `set_subtitle(target, rgba, rect, ref_w, ref_h)`, `present(planes…)`,
  `pause()/resume()`, `clear()`, window attach/move/fullscreen, plus `set_display_widget`-equivalent
  target selection.
- **Signals that must survive** even though `frameYUVReady` goes away: `seekFinished`, `seekIDRFound`,
  `frameTimestampReady` (**always emitted — drives the timeline/seek-bar independently of frame
  delivery**), `fps_update`/stats, `error`, `decoderCrashed`. Resume is gated on `seekFinished`
  (`SyLC_3D_Player.py:2563-2579`).
- **Subtitle normalization (medium):** PGS coords normalize against the **PGS composition reference**
  (`subtitle_manager.py:186-202`), NOT the decoded video size — collapsing the two breaks cropped
  2.39:1 content. Thread `ref_w/ref_h` through; place into a full-source overlay; force
  `subtitle_rect=(0,0,1,1)`; keep source+dest clipping.

## 7. Build integration

- Add a native renderer to the existing `mvc_demuxer_cpp` pybind11 module (`mvc_realtime_demuxer/`,
  built via CMake). Link `d3d11; dxgi; dxguid`. Expose a `NativeRenderer` pybind class mirroring the
  `FrameRingBuffer` exposure pattern.
- **Shaders compiled offline to DXBC `.cso`** with `fxc` (SM5.0) and embedded — avoids a runtime
  `d3dcompiler` dependency and keeps the MinGW/MSVC AV-safe build constraint intact (the native
  edge264 decoder is deliberately compiled out for AV reasons; do not disturb that gate).
- MinGW can link `-ld3d11 -ldxgi -ldxguid`. Keep the renderer in its own translation unit so the
  build option is isolated.

---

## 8. Staged implementation plan (each stage independently checkpointed)

Every stage that touches the GPU ends in **your** build (`fxc` + rebuild `.pyd`) + **on-hardware
visual confirm** — I can smoke-test imports/synthetic frames here but cannot validate 3D output.

- **S0 (done):** capture spec + exact HLSL + this design.
- **S1 — Static native window:** `NativeRenderer` creates the FP16 scRGB flip-model swapchain,
  clears to black, presents. Verify HDR swapchain comes up + fullscreen toggle preserves HDR. *No
  decode.*
- **S2 — One static YUV frame:** upload 6 R8 planes (from a synthetic/test frame) + cbuffer, compile
  `.cso`, draw. Verify color (against the Qt path side-by-side), all 4 stereo modes, aspect, the
  45-px framepack gap. *Pixel parity gate.*
- **S3 — Live single output:** decode thread calls `present(planes)` at the current emit point
  (replacing `frameYUVReady` for the embedded widget only); GIL discipline + seek/pause fence.
  Verify playback, **seek stress** (the V8 crash site), pause, smoothness vs. baseline (velvet probe).
- **S4 — Dual output + subtitles + teardown:** second swapchain for the detached framepack window,
  per-swapchain stereo/white-level, PGS overlay, full teardown/reload ordering (V57). Verify the
  normal 3D-playback state (embedded 2D + detached framepack simultaneously).
- **S5 — Cut over:** route all targets to native, remove the Qt RHI upload path (keep the widget shell
  only as an HWND host if needed). Re-validate the whole matrix; then delete dead Qt-RHI code.

## 9. Risk register (carry into every stage)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Double-pacing vs audio-locked liquid pacing | blocker | Decode-thread-driven present-on-arrival; no renderer clock; interval 1; 1 in flight |
| 2 | Seek/pause GPU race (C thread writes+reallocs textures) | blocker | Single GPU thread; fence+park around seek realloc; pause stops Map/Present first |
| 3 | Shader not byte-exact | high | Use the extracted HLSL verbatim; pixel-parity gate at S2 |
| 4 | HDR/SDR-fallback correctness (scRGB-linear into 8-bit) | high | Probe IDXGIOutput6; explicit SDR encode path; per-swapchain white level |
| 5 | GIL held across blocking Present (UI freeze / starve audio) | high | Copy ptrs under GIL → release → Map/memcpy/Present → reacquire |
| 6 | RowPitch: pad ALL planes, not just U/V | high | Always copy row-by-row at mapped RowPitch; ignore dead ALIGN_BYTES |
| 7 | edge264 side-by-side UV not split natively | high | Port the Cb/Cr split before upload |
| 8 | Qt-lost: WM_SIZE/ResizeBuffers, DPI, multi-monitor color-space, occlusion | high | Handle in the HWND message loop; ResizeBuffers on resize; re-probe color space on monitor move |
| 9 | Teardown/reload ordering (anti-strobe, no terminate, V57 resume) | high | Follow §5 ordering exactly |
| 10 | Latency tuning invalidates `0.75 s` audio offset | medium | 2-buffer FLIP_DISCARD, interval 1, match current latency |
| 11 | Subtitle uses PGS composition ref, not video size | medium | Thread ref_w/ref_h end-to-end; (0,0,1,1) rect; src+dst clip |
| 12 | Init-clear / resize-clear to 16/128 | medium | Explicit clear on create AND resize |

## 10. Known latent issues found during mapping (separate from the port)

- **BT.601 matrix on BT.709 content** (§1) — real color error in the current build; fix as a
  deliberate, hardware-verified change after parity is established.
- `sdr_white_level` is sampled once / on fullscreen-enter; no listener for Windows HDR-setting
  changes or monitor moves → stale white level on those events.
