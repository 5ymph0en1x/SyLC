# SyLC 3D Player — build manifest

Minimal, dependency-traced file set for running and building SyLC 3D Player.
Target: **Python 3.14**, Windows x64. Verified: the full import graph resolves
from this folder alone, and every bundled binary's non-system DLL dependencies
are present (closure check passed).

## Contents

### Python sources (entry + concretely-imported modules)
- `SyLC_3D_Player.py` — entry point
- `mvc_decoder.py` — MVC pipeline orchestration + presenter; the primary MKV/MVC
  demux/decode path is native C++
- `framepacking_window_d3d11.py` — the detached frame-packed 3D window (window management + borderless DWM handling). Its companion `framepacking_widget_d3d11.py` was the Qt RHI render surface and was **removed in 4.1.0** with the rest of the Qt/RHI path; rendering is the native C++ D3D11 engine below.
- `native_renderer/` — **native D3D11 renderer (Tokyo #3)**: `native_framepack_widget.py` (opt-in drop-in 3D widget), `native_tap.py` (A/B diagnostic), `gen_shader_header.py`, `shaders/`, and the design contract `NATIVE_RENDERER_DESIGN.md`
- `monitoring_overlay.py`, `premium_controls_overlay.py` — UI
- `subtitle_manager.py`, `subtitle_extractor.py`, `pgs_subtitle_parser.py`, `fast_mkv_subtitle_extractor.py` — PGS subtitles

> **Freeware build:** the license/subscription system has been removed entirely — no `license_system/`, `subscription_window.py`, license checks, or trial gates. All features are unlocked.

### Native extension (Python 3.14)
- `mvc_demuxer_cpp.cp314-win_amd64.pyd` — MKV/MVC demuxer **+ the native D3D11 renderer**. The renderer is exposed as `mvc_demuxer_cpp.NativeRenderer` when `mvc_demuxer_cpp.NATIVE_RENDERER_AVAILABLE` is `True`.

### Runtime binaries (traced dependency closure)
| Binary | Needed by | Its deps (bundled here) |
|---|---|---|
| `edge264.dll` | native `mvc_demuxer_cpp.MVCDecoder` (dynamic loading); Python/ctypes safety fallback | self-contained (system DLLs only) |
| `mvc_demuxer_cpp….pyd` | native MVC demux/decode + renderer | `ebml.dll`, `matroska.dll` |
| `mpv-2.dll` (libmpv) | `python-mpv` | self-contained (system DLLs only) |
| `ffprobe.exe` | PGS subtitle track detection | `avcodec-62`, `avdevice-62`, `avfilter-11`, `avformat-62`, `avutil-60`, `swresample-6`, `swscale-9` |

> `mpv-2.dll`, `avcodec-62.dll`, `avfilter-11.dll` are large (~100 MB each) runtime
> binaries: they are **git-ignored** (not committed) but must be present in this
> folder to run. `python-mpv` finds `mpv-2.dll` via `%PATH%`; the app adds its own
> directory to the DLL search path at startup. `ffprobe.exe` + the 7 `av*/sw*` DLLs
> are ONLY used for PGS subtitle track detection.

## Python dependencies
Install reproducibly with `uv sync --frozen` from the committed `uv.lock`.
`pip install -r requirements.txt` remains available for development only and
does not guarantee the exact dependency set used for release builds.
Runtime: PySide6, python-mpv, numpy, opencv-python. Build-only for the `.pyd`:
pybind11 (via vcpkg). The VC++ 2015–2022 redistributable (`MSVCP140`,
`VCRUNTIME140*`) must be present (the build tool bundles it).

## Native sources (rebuildable) + CPU-dispatched binaries
- `edge264/src/` — H.264/MVC decoder. `build_sylc_edge264.bat` builds a portable
  baseline plus x86-64-v2 and x86-64-v3 variants; edge264 selects the best supported
  implementation at runtime. GCC and winpthreads are linked statically, so the
  resulting DLL has no MinGW runtime DLL dependency.
- `mvc_realtime_demuxer/` — MKV/MVC demuxer **+ native D3D11 renderer** (pybind11 +
  libmatroska/libebml via vcpkg) **+ native edge264 wrapper**. Rebuilt with MSVC
  (CMake + VS2022, Python 3.14). The wrapper resolves `edge264.dll` with
  `LoadLibraryExW`/`GetProcAddress`; no MinGW archive is linked into the `.pyd`.

### Rebuilding `edge264.dll`
From an MSYS2/MinGW64-enabled checkout:

```bat
cd edge264
build_sylc_edge264.bat
copy /Y edge264_candidate.dll ..\edge264.dll
```

The response files deliberately avoid `-march=native`. This makes one DLL portable
across supported x64 machines while retaining AVX2/FMA/BMI acceleration through
runtime dispatch. Override the DLL used for diagnostics with
`SYLC_EDGE264_DLL=C:\full\path\edge264.dll`.

### Rebuilding the `.pyd` (Python 3.14, native renderer ON)
```
cmake -S mvc_realtime_demuxer -B build_py314 -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -Dpybind11_DIR=C:/vcpkg/installed/x64-windows/share/pybind11 ^
  -DPYBIND11_FINDPYTHON=ON ^
  -DPython_EXECUTABLE=<python-3.14>/python.exe ^
  -DPython_ROOT_DIR=<python-3.14> ^
  -DBUILD_NATIVE_RENDERER=ON
cmake --build build_py314 --config Release
```
Output: `build_py314/python/Release/mvc_demuxer_cpp.cp314-win_amd64.pyd` → copy to
the project root.

- `PYBIND11_FINDPYTHON=ON` + explicit `Python_EXECUTABLE`/`Python_ROOT_DIR` are
  required so pybind11 builds against the intended 3.14 interpreter rather than the
  vcpkg-bundled Python.
- `BUILD_NATIVE_RENDERER=ON` (default on Windows) compiles `src/native_renderer.cpp`,
  defines `SYLC_NATIVE_RENDERER`, and links `d3d11 dxgi dxguid d3dcompiler`. The HLSL
  shader is compiled at runtime via `d3dcompiler` (no offline `fxc` step). With it
  OFF, the module builds exactly as before and `NATIVE_RENDERER_AVAILABLE` is `False`.

The shipped x64 module requires AVX2. Ryzen Zen 2, including the Ryzen Z2 A, supports
the required instruction set. Worker count is selected adaptively; override it for
qualification with `SYLC_EDGE264_THREADS=2`, `3`, or `4`.

### 2D->3D (AI) prerequisites

Real-time 2D->3D conversion (Depth Anything V3) needs two kinds of runtime piece
that are not part of the vcpkg toolchain -- the inference runtime, and the model
files each depth preset can open:

- **ONNX Runtime + DirectML** (`onnxruntime.dll`, `DirectML.dll`): fetched via the
  NuGet packages `Microsoft.ML.OnnxRuntime.DirectML` (ORT built with the DirectML
  execution provider) and `Microsoft.AI.DirectML` (the DirectML redistributable
  itself). Both are dynamic-loaded at runtime (`LoadLibraryExW` + `GetProcAddress`
  in `mvc_realtime_demuxer/src/depth_engine.cpp`) -- `mvc_realtime_demuxer/CMakeLists.txt` only compiles
  against the vendored C API headers (`third_party/onnxruntime/include`) and links
  no `onnxruntime.lib`. The two DLLs just need to sit next to
  `mvc_demuxer_cpp….pyd` (project root during development; alongside
  `SyLC_3D_Player.exe` in a standalone build).
- **Models are per PRESET, not global.** Since round 4 there is no single
  "preferred model" and no single fallback: each of the three depth presets is
  one `(name, candidate models, inference grid)` entry in
  `SyLC_3D_Player.SYNTH3D_DEPTH_PRESETS`, and resolution walks that preset's
  own candidates, all exported for that preset's own grid. **Quality** (the
  default, 756) tries `da3_base_756.onnx`, then `da3_small_756.onnx`, then
  `da3_small.onnx`; **Balanced** (518) tries `da3_base_518.onnx` then
  `da3_small_518.onnx`; **Performance** (518) has `da3_small_518.onnx` alone.
  The bullets below describe the files, preset by preset.
- **Quality's first candidate, DA3-Base** (`models/da3_base_756.onnx`), exported
  at a fixed 756x756 inference grid with `tools_dev/da3_to_onnx.py`. Its two
  outputs are inverse-depth source data and DA3's native per-pixel confidence;
  the latter drives temporal fusion and prevents uncertain boundaries from
  setting the depth budget for the whole shot. This is the model a default
  install runs. Both DA3-Base and DA3-Small are Apache-2.0 models; DA3-Large and
  DA3-Giant remain CC-BY-NC-4.0 and are not selected or shipped by this path.
- **Quality's second candidate** (`models/da3_small_756.onnx`), the same
  DA3-SMALL weights re-exported by the same `tools_dev/da3_to_onnx.py` at the
  same fixed 756 grid. It replaced `models/da3_small.onnx` as the shipped 756
  Small export in round 3: that older community export declares height/width as
  DYNAMIC axes, so its position-embedding `Resize` has no DirectML kernel and
  onnxruntime silently routes that single node to the CPU at ~229 ms — 342 ms
  per depth map against 55 ms for the fixed-shape re-export, measured on the
  same GPU. Same weights, same Apache-2.0 license, 6.2x faster.
- **Quality's third and last candidate** (`models/da3_small.onnx`) is that older
  dynamic-axes export, kept at the bottom of the chain so an install directory
  holding only it still converts. It is a genuine, exercised candidate — the
  TensorRT engine probe builds and value-gates it at the 756 grid like every
  other candidate (see the engine-cost table below), which is why it is in the
  probe's list at all. It is simply the slowest of the three on DirectML, and
  the ordering above means it is only ever reached when neither fixed-shape
  export is installed. It is no longer bundled;
  `tools_dev/fetch_da3_model.py` remains its pinned, sha256-verified fetcher.
  The 518 presets never fall back to it: their candidates are same-grid only.
- **The two 518-grid exports** (`models/da3_base_518.onnx`,
  `models/da3_small_518.onnx`), added in round 4 to back **Balanced**
  (Base then Small at 518) and **Performance** (Small at 518). Same DA3-Base and
  DA3-Small weights as the 756 exports, same `tools_dev/da3_to_onnx.py`, same
  Apache-2.0 chain — only the fixed inference grid differs. A fixed-shape export
  is bound to its grid:
  `DepthEngine::init()` REJECTS a 518 export opened at 756 (and the reverse)
  with a named error rather than running it, which is why a model and its grid
  are declared together in one `SYNTH3D_DEPTH_PRESETS` entry and never
  re-derived anywhere else. See the preset table below for what they cost.
- **Adaptive aspect pool** (Base and Small at `756x406`, `756x378`,
  `756x350`, `756x322`, `518x280`, `518x266`, `518x238`, and `518x210`):
  the shared worker detects symmetric horizontal mattes encoded in the decoded
  image. After eight agreeing depth observations, the player derives the
  active-content ratio and selects the installed graph whose height is within
  one 14-pixel DA3 patch of the ideal tensor height. It crops source mattes
  before depth preparation and maps depth back only across the active picture;
  the matte itself stays at zero parallax. The worker also publishes coded
  dimensions when no matte exists, allowing already-cropped cinema masters
  such as 1920x800 to select the same pool immediately. Crop-free 16:9 sources
  are excluded by a 1.80:1 minimum native-wide ratio. Missing, asymmetric,
  transient or low-confidence mattes leave the square preset untouched.
  Generate a graph with, for example,
  `tools_dev/da3_to_onnx.py --variant both --width 756 --height 322`.
  Width and height must both be positive multiples of DA3's 14-pixel patch.
- **The experimental causal three-view graph**
  (`models/da3_base_756_t3.onnx`) exercises DA3's native multi-view transformer
  with a rolling three-frame window. The runtime accepts it transparently and
  reports `views=3`, but it is a qualification/Master artifact rather than the
  default: current DirectML throughput is substantially below the single-view
  graph.

- **How to get the models.** They are not in this repository and not in the
  binary: 4.61 GB across twenty graphs. Two routes:
  - **In the player** — 2D→3D menu → **Depth models…** → pick `Small` (960 MB, all
    three presets work) or `Base` (3.67 GB, the quality upgrade). Resumable,
    SHA-256 verified against `models/MANIFEST.json`, no account.
  - **By hand** — download from
    https://huggingface.co/Symphoenix/sylc_TRT (`onnx/small/`, `onnx/base/`)
    into `models/` next to the executable.

  Quality's **third** candidate, `models/da3_small.onnx`, is in neither pack —
  see `tools_dev/fetch_da3_model.py`, which is its acquisition path. It is the
  older dynamic-axes community export: same weights, but 2.8 maps/s against
  `da3_small_756.onnx`'s 13.6 on DirectML, because its position-embedding
  `Resize` has no DirectML kernel. Under TensorRT the two are equivalent.

- **Regenerating the manifest.** After re-exporting any graph:
  `tools_dev/gen_model_manifest.py` rehashes `models/` and rewrites
  `models/MANIFEST.json`. `tests/models/test_manifest_matches_player.py` pins
  the manifest equal to the player's own preset tables.

**Packaging whitelist:** `build_exe_v521.bat` bundles **no model weights at
all** — only `onnxruntime.dll`, `DirectML.dll` and `models/MANIFEST.json`: the
inference runtime and the download manifest. The twenty graphs are fetched at
runtime (see "How to get the models" above). It also never uses
`--include-data-dir=models=models`, which would sweep the whole tree in.

The exclusion list below still matters, for a different reason — these
artifacts must never be **published** either. The `t3` graph (a qualification
artifact, not a shipping model), `da3_base_756_fp16.onnx` and
`da3_small_518_clean.onnx` (round-3 measurement artifacts), the superseded
`da3_small.onnx`, the `da3_base_756.onnx.bak_*` backups, and the `DA3-SMALL/`,
`DA3-BASE/`, `DA3-GIANT/` checkpoint directories `tools_dev/da3_to_onnx.py`
uses to produce the ONNX exports. `DA3-GIANT/` in particular is CC-BY-NC-4.0
and is not redistributable — it must no more reach the HuggingFace repository
than it could reach the binary.

`tools_dev/gen_model_manifest.py` enforces this structurally rather than by
discipline: its `PACKS` table is a hardcoded allowlist of the twenty shipping
filenames, not a directory glob, so nothing else in `models/` can reach the
manifest even by accident.

`models_dev_nc/` is a separate directory for third-party model probes whose
licence chain is not verified Apache-2.0 end to end — evaluated locally, never
bundled, never referenced by the player. See its own `README.md`. The rule:
only an artifact with a clean Apache-2.0 chain may live in `models/`;
everything else goes there or is not downloaded at all.

### Adaptive aspect benchmark

The 756x322 Scope bucket contains 243,432 depth pixels instead of 571,536 for
756x756: **57.4% fewer pixels**. On the same RTX 4090, DirectML runtime and
moving-checkerboard service bench:

| graph | infer_ms | flow_ms | stab_ms | delivered depth fps |
|---|---:|---:|---:|---:|
| `da3_base_756.onnx` | 110.5 | 32.1 | 10.8 | 6.2 |
| `da3_base_756x322.onnx` | 52.1 | 18.5 | 5.2 | 12.6 |
| `da3_small_756.onnx` | 64.1 | 33.5 | 10.2 | 8.6 |
| `da3_small_756x322.onnx` | 33.7 | 18.8 | 5.4 | 16.3 |

That is a **52.9% inference-latency reduction / 2.03x delivered depth rate**
for the default Base graph, and **47.4% / 1.90x** for Small. Detection
deliberately costs the first eight square-grid observations for encoded
mattes; a pre-cropped source can switch from its coded dimensions immediately.
After the switch, inference, flow and temporal stabilization all run on the rectangular grid.
Bars introduced only by the display/window compositor are not part of the
decoded pixels and therefore need neither detection nor masking.

The selected crop is scoped to the current medium and is cleared on file,
depth-preset and seek changes. A title that changes aspect ratio continuously
without a seek is a remaining prototype limitation: the rectangular prep no
longer sees pixels outside its current ROI, so automatic mid-shot
re-certification needs a separate low-resolution full-frame probe.

TensorRT caches and verifies engines per ONNX graph. An existing marker that
attests only square graphs therefore keeps the running square TensorRT service
instead of downgrading an adaptive switch to DirectML. Re-run
`tools_dev/setup_tensorrt.py --engine-probe` after installing these profiles;
the probe now builds and records both 756x322 graphs with their rectangular
optimization shapes. Until that succeeds, adaptive Scope selection remains
inactive for a live TensorRT service.

### Depth presets: the consolidated bench (single source of truth)

The user picks one of three depth presets (Quality / Balanced / Performance) in
the 2D→3D menu; each is one `(name, candidate models, inference grid)` entry in
`SyLC_3D_Player.SYNTH3D_DEPTH_PRESETS`. **This is the only throughput table in
this file** — the TensorRT section below points back here rather than carrying
its own copy. (User-facing summaries of these same numbers live in
`RELEASE_NOTES.md` and the validation checklist; if you change a figure here,
change it there too.)

**How to read it, and the rule that produced it:** `infer_ms` is the model's own
inference latency. **maps/s is the MEASURED service rate** — the `fps` field of
`NativeRenderer.synth3d_status()`, i.e. the whole worker cycle (optical flow +
inference + stabilizer + readback), which is what the eye actually sees.
**Never divide 1000 by `infer_ms` to get maps/s**: that mistake shipped once
already, and on today's numbers it would claim 1000/13.4 = **74.6** maps/s for
Quality under TensorRT against a **measured 29.0** (14.9 once the scene moves).
Flow and the stabilizer do not care how fast the model is — they scale with the
GRID — so they dominate as soon as inference gets cheap.

RTX 4090, `tools_dev/bench_depth_models.py`, 640x360 checkerboard source, ~12 s
per model, **one model per subprocess** (see that script's docstring: a single
process benching the whole list measures the tail under GPU memory pressure the
head never saw, off by up to 10x). TensorRT columns are measured with
`SYLC_BENCH_ORT_DIR=G:\SyLC-main\ort_tensorrt` and pre-warmed engines.

**Two regimes, both real, and the difference is large.** A depth cycle only
pays for optical flow and the stabilizer's reprojection when the source is
actually MOVING: below the service's motion threshold the scene is classified
static and both stages are skipped outright (`shared_depth_service.cpp`,
`direct_motion_mean / time_scale >= 0.008f`; the status line then reads
`flow=0.00`). So:

- **static** = a locked-off shot. Also, historically, everything this bench ever
  printed, because it pushed one identical frame per iteration.
- **moving** = the pan the bench now runs by default (1 px per pushed frame),
  where flow and reprojection are in the cycle. This is the regime most real
  content is in.

| preset | model | grid | DML maps/s (static / **moving**) | TRT maps/s (static / **moving**) |
|---|---|---|---|---|
| **Quality** (default) | `da3_base_756.onnx` | 756 | 8.6 / **7.4** | 29.0 / **14.9** |
| **Balanced** | `da3_base_518.onnx` | 518 | 16.4 / **13.3** | 37.9 / **23.3** |
| **Performance** | `da3_small_518.onnx` | 518 | 22.8 / **15.0** | 47.6 / **25.0** |

Where each cycle goes on **moving** content (ms):

| preset | provider | infer_ms | flow_ms | stab_ms |
|---|---|---|---|---|
| Quality | DirectML | 92.8 | 25.7 | 9.9 |
| Quality | TensorRT | 20.5 | **28.4** | 10.3 |
| Balanced | DirectML | 49.8 | 16.6 | 5.2 |
| Balanced | TensorRT | 15.4 | 18.2 | 5.5 |
| Performance | DirectML | 41.3 | 16.2 | 5.7 |
| Performance | TensorRT | 12.2 | 18.2 | 5.9 |

Read the Quality/TensorRT row twice: **optical flow costs more than the model
does** (28.4 vs 20.5 ms). That is the whole reason the static numbers flatter
TensorRT so badly — they omit the single most expensive stage.

Graphs no preset selects, DirectML maps/s (static / **moving**), from the same
script's full sweep: `da3_small_756.onnx` 13.6 / **10.1** ·
`da3_small.onnx` 2.8 / **2.4** · `da3_base_756_fp16.onnx` 12.7 / **10.0** ·
`da3_base_756_t3.onnx` 2.9 / **2.0**. Their TensorRT figures are Task 4's and
static only: `da3_small_756.onnx` 22.8, `da3_small.onnx` 23.9. The other two
have **no TensorRT engine** — not a zero: they are in no preset, so
`--engine-probe` never compiled one and TensorRT was still compiling when the
12 s window expired. Exactly what an unprobed graph would do to a playback
session, which is why the player refuses TensorRT for any model the marker does
not name.

Reproducibility of the static Quality figures across three independent round-4
runs: 93.8 / 94.4 / 95.2 ms DirectML and 13.6 / 13.6 / 13.4 ms TensorRT (~1.5%).
Note that inference itself reads slower under motion (TensorRT Quality 13.4 ->
20.5 ms): the flow stage is CPU-thread-parallel and contends with the inference
submission path, so "inference latency" is not independent of what else the
cycle is doing.

**What the numbers say.** The grid is a real lever on both providers, and on
moving content it is a bigger one than the static figures suggested:

- **DirectML**: Balanced delivers 1.80x Quality on moving content (13.3 vs 7.4).
- **TensorRT**: Balanced delivers **1.56x** Quality on moving content (23.3 vs
  14.9). The static numbers put that gap at only 1.31x, and an earlier draft of
  this section concluded from it that "a TensorRT user pays very little for
  Quality and should stay on it". **That conclusion was an artifact of the
  static bench** and is withdrawn: the stages the static run skipped are exactly
  the ones that grow with the grid, so 1.31x was a floor, not the answer.

Inference alone still behaves the way TensorRT's fusion predicts (Base costs
about the same at 756 and 518), but inference is no longer the thing that
decides the delivered rate once flow is in the cycle. **The honest guidance is
that all three presets are a genuine trade on both providers, and the author's
own A/B is what should settle it** — which is what the validation checklist
asks for.

Absolute rates still depend on content and on the source cadence (see the
"maps/s is capped by the source" note below), so treat these as a comparison
between presets on one machine rather than as a promise.

**Round-3 figures are superseded** (that round's TensorRT table read 18.3 ms /
18.0 maps/s for `da3_base_756.onnx`; today the same graph reads 13.4 ms /
29.0 maps/s static). Two of the three components of that gap now have an
explanation, and one does not. The 18.0 -> 29.0 move is a cycle going from
55.6 ms to 34.5 ms, i.e. 21.1 ms saved:

| component | round 3 | round 4 | delta | status |
|---|---|---|---|---|
| flow + stabilizer | 4.5 + 17.0 | 2.6 + 10.4 | **-8.5 ms** | **explained** |
| inference | 18.3 | 13.4 | -4.9 ms | unexplained |
| everything else (readback, prep, queueing) | 15.8 | 7.9 | -7.9 ms | unexplained |

The flow/stabilizer line is **provider-independent work that round 3
parallelized between the two measurements** (its two waves: the flow estimator,
then the stabilizer's reprojection and the luma/histogram pass — see round-3
`progress.md`, which measured reproject 50.1 -> 10.2 ms and the step loop
22.8 -> 14.0 ms at 8 threads). The DirectML side corroborates it: the DirectML
*rate* also moved, 8.1 -> 8.6-8.7 maps/s, while its inference did not
(93.9 -> 94.4-95.2 ms). An earlier draft argued "the DirectML column is
unchanged, so the change is TensorRT-specific" — that argument compared
`infer_ms` only and is withdrawn. Applying just the 8.5 ms saving to round 3's
cycle predicts 21.3 maps/s, so this accounts for roughly 40% of the gap.

What remains genuinely unexplained is the inference drop and the residual term.
Two things weaken 18.3 as a stable reference: it was itself the **mean of three
runs reading 19.5 / 19.1 / 16.2 ms** (round-3 `task-4-report.md`), a 3.3 ms
spread whose low end is already heading toward today's value; and the
process-isolation hypothesis — the obvious suspect, since round 3 benched the
whole list in one process — was tested and **FALSIFIED**: the same graph benched
third in a single process, behind the two smaller ones exactly as round 3 did,
still reads 13.6 ms / 28.7 maps/s.

**Preset switching keeps every visited service warm, on purpose.** The
process-wide service registry is keyed by (model, runtime, grid) and **never
evicts**: cycling Quality -> Balanced -> Performance leaves three ORT sessions
and three worker threads resident (roughly 400 + 400 + 100 MB of model weights,
plus each provider's own workspace). That is the deliberate trade behind
"switch back is instant" — the price of an LRU eviction would be paid on every
A/B comparison, which is the main thing this feature exists for. An eviction
policy (idle timeout, or capping the registry at two entries) is future work,
tracked as a known limitation below.

**TensorRT is gated PER PRESET, so re-run the probe after adding a model.**
`ort_tensorrt\.trt_verified` names every graph its engine probe actually built,
and `_synth3d_ort_dir(model_path)` prefers TensorRT only for a graph the marker
NAMES — an unattested model gets DirectML instead of a multi-minute engine
compile in the middle of playback. Consequence: **dropping a new `.onnx` into
`models\` (or adding a preset) silently leaves that graph on DirectML until you
re-run `setup_tensorrt.py --engine-probe`**, which walks every candidate of
every preset present on disk and rewrites the marker. The status line is the
tell — `provider=DirectML` on one preset and `provider=TensorRT` on another,
same machine, is this gate doing its job, not a bug. See "TensorRT opt-in"
below for the probe, its cost, and the freshness rule.

At runtime, renderers using the same (model, runtime, grid) triple attach to one
process-wide inference service. Only a renewable leader surface performs the
depth readback at the preset's grid; every surface consumes the same immutable
stabilized depth map. The service remains warm across an interactive disable, so neither
disabling nor switching presentation surfaces waits for ONNX Runtime teardown.
The established inverse-depth field is transported with conservative
quarter-resolution forward/backward luma flow before confidence-aware fusion.
Surface boundaries and low-confidence regions additionally use a five-sample,
motion-compensated temporal surface memory. A local stability score expands
the useful history on quiet shots, while moving silhouettes immediately
collapse back to the current observation. The robust selector returns an
actually observed foreground or background layer instead of averaging both
into a synthetic “rubber” contour. Motion activation also considers local
32-pixel tiles, so a moving face or neck cannot disappear inside a low
full-frame motion average.
Scene-persistent percentile grading prevents shot-wide depth breathing; the GPU
warp uses edge/depth-guided sampling, Jacobian fold detection, bilateral
background candidates, and a soft disocclusion blend.
`NativeRenderer.synth3d_status()` exposes the provider, the active inference
grid (`side=`), inference FPS, map age, attached-client count, cut count, compensated mean motion, mean flow in pixels,
adaptive EMA alpha, mean DA3 confidence, and source scene-change score for
qualification. The shared-service status also reports mean local stability and
effective temporal-history support.

**Known limitations (depth presets, round 4).** All three are deliberate, all
three are cheap to live with, and none of them can bite silently:

- **The service registry never evicts** (see the preset table above). Visiting
  all three presets in one session keeps ~900 MB of model weights and three
  worker threads resident until the process exits. An idle-timeout or
  two-entry-LRU policy is the obvious future work; it was not done here because
  every eviction policy makes the A/B comparison this feature exists for slower.
- **A preset change that cannot re-arm leaves the 3D button checked.** Changing
  the preset while synthesis is running re-arms it with
  `toggle_synth3d(False)` then `(True)`, and `_synth3d_rearming` suppresses the
  framepack window's "user closed the 3D output" teardown for the microseconds
  between the two legs. If the ON leg early-returns because synthesis stopped
  being supported/eligible in between (the model file deleted mid-session,
  playback stopped between opening the menu and clicking), that suppressed
  teardown never runs: `_synth3d_active` is correctly False and the decoder is
  correctly back on the embedded 2D widget (`configure_3d_output(False)` does
  that itself on both the MVC and HEVC paths), but the 3D button stays checked
  and `is_3d_enabled` stays True until the next toggle. **The apparently
  obvious repair does not work**: re-running `configure_3d_output(False)` with
  the flag cleared cannot resurrect the teardown. That teardown runs off
  `visibilityChanged`, which on this path is emitted from the window's
  `hideEvent` — and Qt sends no `hideEvent` to an already-hidden window
  (verified: two consecutive `hide()` calls deliver exactly one event). The
  signal does have a second emitter, the explicit
  `enter/exit_fake_fullscreen()` pair, but neither is on the re-arm path.
  A real fix has to perform the button/`is_3d_enabled`
  teardown directly, which also means touching cast lifecycle — more risk than
  a stale checkbox in a corner case is worth, so it is documented rather than
  attempted.
- **The depth preset persists in `QSettings`** (`SyLC` / `SyLC3DPlayer`, key
  `synth3d/depth_preset`) while the rest of the player's preferences live in
  `~/.sylc3d_player.json`. Two stores, chosen so the preset survives before the
  JSON store is loaded; worth unifying if a second Qt-side setting ever appears.

### TensorRT opt-in (local, never redistributed)

An optional, NVIDIA-only execution provider for the same DA3 ONNX models
above. **Measured on moving content with the default Quality preset (RTX 4090,
round 4): 4.5x on inference latency, 2.0x on the delivered depth-map rate**
(92.8 -> 20.5 ms, 7.4 -> 14.9 maps/s). On a static shot the same pair reads
7.1x and 3.4x — the gap between the two ratios, and between the two regimes, is
the whole lesson of the depth-preset table above; see it before quoting any of
these. TensorRT compiles
the graph into a GPU-specific engine
on first use of a given model+shape -- this takes a few minutes, then the
compiled engine is cached on disk and subsequent launches are fast. It lives
entirely in
`ort_tensorrt/`, a flat directory of DLLs next to the project root during
development (alongside `SyLC_3D_Player.exe` in a standalone build), and is
selected instead of the root `onnxruntime.dll`/`DirectML.dll` pair by the
player at process start -- never both in the same process (see the round-3
Global Constraints). If `ort_tensorrt/` is absent, behavior is byte-for-byte
identical to today's DirectML-only path.

**Status: ACQUIRED, COMPLETE, AND ENGINE-BUILD VERIFIED** (completed
2026-07-29). `ort_tensorrt\` exists (1.977 GB, 20 files), its TensorRT
execution provider registers cleanly against a real ONNX Runtime session
(see "How the probe works" below), AND -- as of the same-day Task 4 update
below -- a REAL TensorRT engine build + inference has been proven to
complete successfully (cold compile 202.3s for `da3_base_756.onnx` / 232.7s
for `da3_small_756.onnx`; cached warm start 3.3-4.3s) with
`ort_tensorrt\.trt_verified` as the proof. Registration alone (the original
acquisition gate) turned out NOT to be sufficient evidence a real engine
build would succeed; see "Round 3 Task 4 update" below for the two-part fix
(a completed cuDNN9 modular DLL set + a missing `SetDllDirectoryW` call) and
why `--engine-probe`'s marker is now the player's actual gate, not DLL
presence.

**Measured TensorRT throughput: see the depth-preset table above** — it carries
both providers side by side and is the only throughput table in this file.
Headline for the default preset (Quality, `da3_base_756.onnx`): inference
92.8 -> 20.5 ms and the delivered depth rate 7.4 -> 14.9 maps/s on moving
content (8.6 -> 29.0 on a static shot).

**Read the maps/s columns, not the inference ratio.** `infer_ms` is the model's
own latency; the rate the user perceives is the whole worker cycle, which also
carries optical flow, the stabilizer and readback — and on moving content at
756 the flow stage alone outweighs a TensorRT inference. That is why a ~4.5x
inference win delivers ~2x. (An early round-3 draft claimed ~55/s by dividing
1000 by `infer_ms`; on today's numbers that arithmetic would claim 1000/13.4 =
74.6 maps/s against a measured 29.0. It is banned here for good.)

**maps/s is capped by the source cadence — a bench figure above it is
headroom, not frames.** In playback the depth service is fed one frame per
present, so on a 24 fps film the depth map cannot refresh more than 24 times a
second however fast the model runs; the bench feeds a service with no such
source and therefore measures its ceiling, not a playback rate. The bench's own
feed is capped too — it sleeps 20 ms between pushes, i.e. **up to ~48 pushes per
second**, and the fastest static cell (Performance/TensorRT, 47.6) sits at that
limit rather than above it, so read it as "at least 48" and not as a measured
maximum. It costs nothing for the conclusions here, which all rest on the moving
column. Two
consequences worth knowing before reading a status line in anger: a preset
measured at 25-47 maps/s here will simply track the source in playback (the
headroom absorbs busy scenes instead of raising the rate), while a preset
measured below the source cadence — Quality at 7.4 (DirectML) or 14.9
(TensorRT) on moving content — is the case where depth genuinely refreshes
slower than the picture, and that is what the eye can catch.

**The moving-content column is corroborated by real playback.** Round 3's
analysis of the author's own session logs (round-3 `progress.md`, the flow
parallelization entries) measured in-situ `flow_ms` at 68-108 while the scene
moved against 12.8 on a calm shot — i.e. ~8 maps/s moving versus 17.5 calm —
and after its two parallelization waves predicted **12-18 maps/s in situ**
(expected `flow_ms` ~25-35, `stab_ms` ~14). This bench's moving Quality/TensorRT
row now reads `flow_ms` 28.4, `stab_ms` 10.3 and **14.9 maps/s**: inside that
predicted band, from an independent measurement. The static column never
reproduced it, which is precisely the blind spot the panning frame closes.

Base is faster than Small under TensorRT on inference latency -- the reverse of
their DirectML ordering -- because TensorRT's fusion/kernel-selection has more
to work with on the larger graph. Base is what the default Quality preset
already selects, so a TRT user gets it with no further configuration.

**The whole flow, end to end**, on a machine with an NVIDIA GPU:

1. `.venv\Scripts\python.exe tools_dev\setup_tensorrt.py` -- acquires and
   assembles `ort_tensorrt\`. Add `--sdk-dir <path>` if you have an official
   NVIDIA TensorRT SDK extracted locally; it takes priority over the download
   routes but is not required. Nothing is installed into the project venv.
2. `.venv\Scripts\python.exe tools_dev\setup_tensorrt.py --engine-probe` --
   builds and value-gates a real engine for EVERY depth-preset graph present
   in `models\`, then stamps `.trt_verified` naming each one. **This is the
   one-time multi-minute compile, once per graph** (measured on the RTX 4090:
   ~16 minutes for the current five graphs from a genuinely cold cache,
   ~2.5-4 min per graph; 15s once the engines are cached).
3. Start the player. `_synth3d_ort_dir()` sees the complete directory + the
   fresh marker and selects TensorRT; the 2D->3D status line reads
   `provider=TensorRT`.

Steps 1 and 2 are separate on purpose: step 1 can succeed while step 2 fails
(registration is not proof of a working build -- that is exactly what round 3
found), and only step 2's marker unlocks the player's preference.

**Get it:** `.venv\Scripts\python.exe tools_dev\setup_tensorrt.py`. It never
installs into the project venv -- it fetches ONNX Runtime's GPU build from
NuGet (`Microsoft.ML.OnnxRuntime.Gpu` -> its pinned
`Microsoft.ML.OnnxRuntime.Gpu.Windows` dependency; the versionless
meta-package URL alone does NOT contain the native DLLs, only a thin
`.props`/`.targets` wrapper -- the script resolves and fetches the real
platform package at the exact pinned version: 1.28.0, confirmed to be
NuGet's true latest), then acquires TensorRT + cuDNN/cuBLAS via `uv` in a
disposable scratch venv (`%TEMP%`-side, never the project `.venv`):

```
uv venv <scratch> --python 3.12
uv pip install --python <scratch> tensorrt-cu13==10.16.1.11
uv pip install --python <scratch> nvidia-cudnn-cu13
```

**The route that actually works, and why round 1 missed it entirely:**
`tensorrt-cu13` (and `tensorrt-cu13-libs`) are published on PyPI as tiny
(~15-18KB) `wheel-stub`-backed sdists -- not real content, just a PEP 517
build backend whose `build_wheel()` step reaches out to
`https://pypi.nvidia.com/` **at build time** and downloads the real,
2+ GB, platform-matched wheel from NVIDIA's own index. Any static
wheel-listing check (what round 1 did) sees nothing there; only actually
letting the sdist build run reveals it. `pip install --only-binary=:all:`
(round 1's flag on every invocation) makes pip refuse to build ANY sdist,
which silently defeats this exact mechanism -- that flag is now dropped for
this specific package.

**Version pin matters -- TensorRT's DLL naming encodes its major version,
and the loader matches by EXACT filename.** `tensorrt-cu13`'s "latest"
(11.1.0.106) produces `nvinfer_11.dll`; the fetched ORT 1.28.0 GPU build's
`onnxruntime_providers_tensorrt.dll` statically imports `nvinfer_10.dll` by
literal name (confirmed via a real PE-import-table read) -- Windows does
NOT treat these as interchangeable, no matter how "complete" the 11.x set
is. `tensorrt-cu13` also publishes a 10.x-line (CUDA 13 support was
backported onto TensorRT's 10.x series before 11.x shipped: 10.13.2.6
through 10.16.1.11 exist). **`10.16.1.11`** (the latest 10.x release) is
pinned in the script, empirically verified to produce the exact
`nvinfer_10.dll` name needed and to pass the real session-API gate below.

**cuBLAS naming churn:** `nvidia-cublas-cu13` is explicitly deprecated
upstream (its own build step prints "use `nvidia-cublas` instead" and
fails); `nvidia-cudnn-cu13` transitively installs the correctly-named
`nvidia-cublas` package, which is why the script installs cuDNN and gets
cuBLAS "for free" rather than requesting it directly.

**Harvest is minimal, not the whole `tensorrt_libs` tree:** 6 files --
`nvinfer_10.dll`, `nvinfer_plugin_10.dll`, `nvonnxparser_10.dll`,
`cudnn64_9.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`. Empirically verified
sufficient (a 9-file test directory: these 6 + the 3 ORT DLLs) -- the
~2 GB of `nvinfer_builder_resource_*_10.dll` (per-GPU-architecture
precompiled kernels, needed only when actually *building* a TensorRT engine
for a specific SM target, not for provider registration) and `nvrtc*`/
`nvblas*.dll` are NOT included. **Note for whoever wires up real inference
(T4):** if engine building on the RTX 4090 (compute capability 8.9, "sm89")
fails needing a precompiled-kernel resource, `nvinfer_builder_resource_sm89_10.dll`
from this exact same `tensorrt-cu13==10.16.1.11` install (already cached
under `%TEMP%\sylc_tensorrt_setup\uv_scratch_venv`) is the fix -- not
pre-included here to keep the footprint modest, and because TensorRT can
often JIT-compile PTX as a fallback so it wasn't clear it's actually needed.

**How the probe works, and why a bare DLL load is the WRONG test for this
DLL:** a plain `LoadLibraryExW` of `onnxruntime_providers_tensorrt.dll` in
isolation fails with WinError 1114 (DLL init routine failed) -- reproduced
consistently across four different TensorRT 10.x point releases, all with
matching cuBLAS/cuDNN present. That is NOT a missing-dependency signature
(which would be error 126); something in this DLL's own static
initialization behaves differently when force-loaded standalone versus
loaded the way ONNX Runtime actually loads it internally. Proof: calling the
REAL ORT C API in the exact sequence ORT itself uses --
`OrtGetApiBase()->GetApi(24)` -> `CreateEnv` -> `CreateSessionOptions` ->
`OrtSessionOptionsAppendExecutionProvider_Tensorrt(options, 0)` -- with the
IDENTICAL files in the IDENTICAL directory, **succeeds cleanly**. This live
session-API call (not a bare DLL load) is `setup_tensorrt.py`'s actual
completeness gate; the old bare-load + PE-import dependency walk is kept
only as a diagnostic for when the real test fails.

**Fallbacks, in order, if the primary `uv` route fails:** (1) plain `pip
install tensorrt-cu13==10.16.1.11` in the disposable embeddable-Python-3.12
venv (same recipe as `tools_dev/convert_fp16.py`), WITHOUT
`--only-binary=:all:` so the sdist bootstrap can run; (2) a **read-only**
harvest from `G:\SyLC-main\.venv\Lib\site-packages\tensorrt_libs\` etc., if
those packages are already installed there by some other means -- this path
only ever copies files out, never writes to `.venv`.

**Verify without re-running acquisition:** `setup_tensorrt.py --verify-manual`
probes `ort_tensorrt\` exactly as it sits -- zero downloads, no staging
directory, and it never deletes or modifies anything there, pass or fail.
Manually-populated files (staging or final directory) also survive reruns
of the full script: `_assemble_stage()` merges rather than wipes, so a
human dropping files into `.ort_tensorrt_staging\` and re-running
`setup_tensorrt.py` (no flags) gets them auto-promoted the moment the probe
passes.

**Remove it:** delete the `ort_tensorrt\` directory. Nothing else references
it when it's absent.

**Round 3 Task 4 update (2026-07-29): registration was never the whole
story.** Real engine-BUILD testing (`CreateSession`, never exercised by the
registration-only probe above) found the runtime was still incomplete in a
way that does not fail gracefully:

- **cuDNN 9's modular backend was missing.** `cudnn64_9.dll` above is only
  cuDNN 9's thin front-end *dispatcher* (267 KB); it loads several backend
  engine DLLs (`cudnn_ops64_9.dll`, `cudnn_cnn64_9.dll`, `cudnn_adv64_9.dll`,
  `cudnn_graph64_9.dll`, `cudnn_heuristic64_9.dll`,
  `cudnn_engines_precompiled64_9.dll`, `cudnn_engines_runtime_compiled64_9.dll`,
  `cudnn_engines_tensor_ir64_9.dll`, `cudnn_ext64_9.dll`) ON DEMAND at first
  real `cudnnCreate()` -- i.e. at TensorRT engine build time, never at
  provider registration. Their absence does not return an `OrtStatus` error:
  it **hard-crashes the process** (`Invalid handle. Cannot load symbol
  cudnnCreate`, NTSTATUS `0xC0000409`, a native fail-fast abort). All 9 files
  are now in `HARVEST_PATTERNS` -- confirmed already present in the SAME
  `nvidia-cudnn-cu13` install this script performs, no new download needed.
- **`nvinfer_builder_resource_sm89_10.dll`** (the RTX 4090's architecture,
  compute capability 8.9) -- the watch-item flagged when this runtime was
  first acquired -- is now also in `HARVEST_PATTERNS`.

**`--sdk-dir <path>`: a user-provided official NVIDIA TensorRT SDK as a
first-class, durable acquisition route.** Point this at an extracted
official SDK zip (its DLLs live under `<sdk>\bin\`, NOT `\lib\` -- `\lib\`
only holds `.lib` *import* libraries, unused by this dynamic-loading
project). This route is deliberately NOT hardcoded to any one TensorRT major
version: it reads the REAL PE import table of the `onnxruntime_providers_
tensorrt.dll` this run just fetched from NuGet to determine the EXACT
filenames that build actually requires, then searches `<sdk>\bin\`
(recursively) for exact-name matches -- winning over the `uv`/pip/venv routes
on any conflict. A full miss (nothing matches) means the SDK is a different
TensorRT major version than this ORT build imports by literal filename
(Windows does not treat e.g. `nvinfer_11.dll` as satisfying a `nvinfer_10.dll`
import), reported explicitly rather than silently substituting an
incompatible version. Verified against the author's official TensorRT
11.1.0.106 SDK (`C:\TensorRT-11.1.0.106`): the currently-published
ONNX Runtime GPU build (1.28.0, NuGet's true latest as of this writing) still
imports `nvinfer_10.dll` by exact name, so the 11.x SDK's files do not
satisfy it -- the wheel route (`tensorrt-cu13==10.16.1.11`) remains this
machine's working pairing. See task-4-report.md for the full version-sweep
evidence and the TensorRT-11 pairing decision.

**`--engine-probe`: the "playback never dies" completeness gate, and the
one-time engine compile.**
Registration succeeding (the probe above) is NOT proof a real engine BUILD
will succeed -- see the cuDNN finding above. `--engine-probe` runs a REAL
engine build + inference (reusing `tests/synth3d/_trt_engine_probe.py`'s
"engine" mode) against `ort_tensorrt\` **in an isolated child process**, so a
native crash there can never take this script down. **A clean build is also
not proof of a numerically correct one** (fix round 1, F1):
`trt_fp16_enable=1` could silently run an fp32 artifact on fp16 kernels and
produce garbage, so the probe also asserts the same near-vs-far polarity check
`test_depth_engine.py` uses on its test scene before calling anything a pass.
On success, it writes `ort_tensorrt\.trt_verified`: ORT version, pinned
TensorRT version, UTC timestamp, total elapsed time, **one `probe_model=` line
per attested graph** and a `probe_detail_<n>=` line carrying that graph's own
grid, build time and value-gate numbers.

**It probes every graph the PLAYER can open** (round 4) --
`_engine_probe_graphs()` walks `ENGINE_PROBE_MODEL_CANDIDATES`, which mirrors
`SyLC_3D_Player.SYNTH3D_DEPTH_PRESETS` entry for entry, models **and**
inference grid, and builds an engine for every candidate present in `models\`.
`tests/synth3d/test_trt_optin.py::test_engine_probe_candidates_match_player`
pins the two equal, so the lists cannot drift apart silently. It probes every
candidate rather than each preset's first choice because the player's gate is
checked against the model a preset RESOLVES to, and that resolution reads the
disk at enable time while the marker only has to stay newer than the `*.dll`
files: remove `da3_base_756.onnx` from an install and Quality resolves to
`da3_small_756.onnx` without invalidating anything.

This all matters because TensorRT's engine cache is keyed per graph: compiling
one model's engine does not warm another's. Round 3 Task 4 probed
`da3_small_756.onnx` while the player loads `da3_base_756.onnx`, so a freshly
set-up machine would have passed verification and then still paid a full cold
compile at the user's FIRST in-playback enable -- the exact wait this step
exists to absorb. Task 5 corrected it, and round 4 generalized it: the player
now prefers `ort_tensorrt\` only for a graph the marker actually NAMES
(`synth3d_marker_attests`), so an unprobed model gets DirectML instead of a
multi-minute compile in the middle of playback. (Round 3's "smallest model
compiles fastest" rationale was also empirically wrong on this graph family:
Base compiled in 202.3s against Small's 232.7s. TensorRT's optimizer, not
parameter count, dominates compile time here.)

**So: the multi-minute compile happens HERE, during setup, not during
playback.** Budget ~2.5-4 minutes PER COLD GRAPH. Round 4 measured, RTX 4090,
the whole five-graph preset table (538.8s wall for the run, against a cache
that already held round 3's two 756 engines). This table is the authority for
ENGINE COST and the value gate; throughput lives in the depth-preset table
above and nowhere else:

| graph | grid | cold build | warm start | value gate near/far |
|---|---|---|---|---|
| `da3_base_756.onnx` | 756 | 202.3s (round 3) | 4.0s | 1.1177 / 1.0478 |
| `da3_small_756.onnx` | 756 | 232.7s (round 3) | 2.6s | 1.0740 / 1.0047 |
| `da3_small.onnx` | 756 | 228.2s | 2.5s | 1.0732 / 1.0052 |
| `da3_base_518.onnx` | 518 | 156.6s | 3.4s | 1.3852 / 1.0211 |
| `da3_small_518.onnx` | 518 | 146.8s | 2.3s | 1.1380 / 0.9579 |

Summing the cold column: **~966s (16.1 min) for all five from an empty cache**
-- that, not the 538.8s wall above, is what a fresh machine pays. All five warm
in 14.7s together, producing byte-identical value-gate numbers to their cold
builds -- the disk cache, not a warm in-process session, is what makes the
reruns fast. The engines live in `ort_tensorrt\trt_cache\` (~831 MB
for these five; 92-354 MB each, and NOT ordered by model size -- the 518 Base
engine is the largest of them). **Never reclaim that space by deleting
`trt_cache\` alone**: the freshness check looks for `*.dll` newer than the
marker, so an emptied cache leaves `.trt_verified` present, fresh and still
attesting -- the next in-playback enable then rebuilds the engine it needs,
which is the multi-minute compile the marker exists to keep out of playback.
Delete the whole `ort_tensorrt\` directory, or re-run `--engine-probe`
afterwards to refill the cache offline. Until the probe passes and stamps the
marker, the player simply keeps using DirectML -- nothing waits, nothing
degrades.

**Re-running the FULL setup wipes the engine cache.** `setup_tensorrt.py`
with no flags promotes its staging directory over `ort_tensorrt\` with an
`rmtree` + rename, which takes `ort_tensorrt\trt_cache\` and `.trt_verified`
with it. That is deliberate (a re-acquired runtime's old engines are not
guaranteed valid), but it means **every full re-run costs another cold engine
compile for EVERY graph** -- ~16 minutes for the current five (~2.5-4 min
each; the 538.8s figure in the table above is NOT the number to budget here,
it had two engines already cached). Use
`--verify-manual` (read-only) or `--engine-probe`
(no downloads, no staging) when you only want to re-check an existing
`ort_tensorrt\`; reserve the flagless full run for an actual re-acquisition.
**`SyLC_3D_Player.py`'s `_synth3d_ort_dir()` requires this marker's presence
AND freshness** (no `*.dll` in the directory newer than the marker -- see
fix round 1, F2) **in addition to the DLL checks** before ever preferring
`ort_tensorrt\` over the DirectML root -- DLL presence alone is deliberately
not enough. A failed probe removes any stale marker from a previous run, so
the player correctly falls back to DirectML, closing the realistic crash
path at acquisition time, until a fresh `--engine-probe` passes. Run it once
after any acquisition or SDK change:
```
.venv\Scripts\python.exe tools_dev\setup_tensorrt.py --engine-probe
```

**The completed cuDNN set above was necessary but NOT sufficient -- the real
second half of the fix was a missing `SetDllDirectoryW`.** Even with all 9
modular cuDNN backend DLLs physically present, `--engine-probe` still
crashed identically. Root cause (found by comparing against this script's
OWN already-working registration probe, which already called
`SetDllDirectoryW(runtime_dir)` -- `DepthEngine::init()` in
`depth_engine.cpp` had no equivalent): `LOAD_WITH_ALTERED_SEARCH_PATH` on the
`LoadLibraryExW` call for `onnxruntime.dll` only helps resolve THAT DLL's
own direct dependencies; it does not extend to nested, runtime-time
`LoadLibrary` calls made later by a transitively-loaded DLL --
`cudnn64_9.dll` loads its own modular backend files ON DEMAND, at first real
`cudnnCreate()`, and that nested call uses the plain OS default search order.
Fixed with a single added line,
`SetDllDirectoryW(cfg.ort_dir)`, mirroring this script's own working
pattern (process-global, so it also covers nested loads). Confirmed: FIRST
engine compile (`da3_small_756.onnx`) 232.7s, cached warm start 3.3s -- see
task-4-report.md for the full diagnosis and the bench numbers.

**Never packaged (verified):** `build_exe_v521.bat` bundles files by exact,
hardcoded name only (`--include-data-files=onnxruntime.dll=...`,
`--include-data-files=DirectML.dll=...`, etc.) -- there is no
`--include-data-dir` sweeping `ort_tensorrt/`, `models/`, or the project
root, and no pattern in the file matches `ort_tensorrt` at all (confirmed by
`grep -i ort_tensorrt build_exe_v521.bat` -> no output). Nuitka's own
dependency-following auto-bundling only picks up DLLs that are actual
`ctypes`/import dependencies of already-included Python modules; nothing in
the shipped source imports from `ort_tensorrt/` (the player selects its
directory by path, dynamically, at runtime, before opening any ORT
session), so Nuitka's scanner has no path into it either. Extending the
whitelist to include it is a decision for a future gate, not automatic.

## Standalone build (no-console exe)
Built with **Nuitka --standalone** via `build_exe.bat` (bundles edge264 /
mpv-2 / ffprobe + av*/sw* as data files; ebml, matroska and the
`.pyd` are auto-bundled by Nuitka's dependency scan). The `build_nuitka/` output is
git-ignored — rebuild via `build_exe.bat`.

**`build_exe_onefile.bat` is not part of the v5.2.1 release.** It still carries
v5.0.0 version stamps throughout and predates the 2D→3D packaging whitelist, so
it bundles neither `onnxruntime.dll`, `DirectML.dll`, `models/MANIFEST.json` nor
`model_fetcher` / `model_download_dialog`: a binary built from it plays video but
answers `models/MANIFEST.json is missing from this install` to every AI action.
v5.2.1 ships a single asset, the portable folder from `build_exe_v521.bat`.
