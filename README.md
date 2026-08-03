<div align="center">

# SyLC 3D Player

<p align="center"><img src="splash.png" alt="SyLC 3D Player Logo" width="250" /></p>
<p align="center"><img src="GUI.jpg" alt="SyLC 3D Player Interface" width="1000" /></p>

### A free, open-source player for the 3D format the industry left behind.

*Stereoscopic 3D Blu-ray (MVC) playback, decoded from scratch, rendered in native HDR — given to the community, no strings attached.*

![Version](https://img.shields.io/badge/version-5.2.1-1f6feb?style=for-the-badge)
![Platform](https://img.shields.io/badge/Windows-x64-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-free%20%26%20open--source-2ea44f?style=for-the-badge)

![3D](https://img.shields.io/badge/3D-MVC%20stereoscopic-e10098?style=for-the-badge)
![HDR](https://img.shields.io/badge/HDR-Direct3D%2011-5c2d91?style=for-the-badge)
![Decoder](https://img.shields.io/badge/decoder-edge264%20BSD-fe7a16?style=for-the-badge)
![Audio](https://img.shields.io/badge/audio-libmpv-eb5d2a?style=for-the-badge)

</div>

---

## Why this exists

In 2017 the industry quietly killed 3D. Blu-ray players stopped shipping it, TVs dropped it, and the software that could play **3D Blu-rays** — encoded in a format called **MVC** — was discontinued one app at a time. The discs didn't disappear. The collections didn't disappear. The *players* did.

And here's the cruel part: **MVC can't be played by the tools everyone already has.** When you rip a 3D Blu-ray to an MKV, you get an H.264 stream carrying **two interleaved camera views** — left and right eye, the second view encoded as differences against the first. FFmpeg — the engine inside VLC, MPC-HC, and nearly every "it plays everything" player — **decodes only the base view and silently throws the 3D away.** You get a flat 2D picture and no warning. The depth is *in the file*. Nothing on your machine will show it to you.

**SyLC 3D Player is the answer to that problem.** It is a complete, from-scratch stereoscopic pipeline — its own MVC decoder, its own demuxer, its own HDR renderer — built over months specifically so that your 3D library plays again, in full quality, on modern hardware. It is **free, open-source, and unencumbered**. No license, no activation, no trial, no telemetry.

As far as we know, it is **the only actively-developed, open-source player that truly decodes MVC** — both eyes — and renders it in real HDR.

---

## What makes it unique

- 🧬 **It doesn't lean on FFmpeg for the hard part.** The 3D is decoded by a custom in-house H.264/**MVC** decoder that reconstructs *both* views — the thing mainstream players can't do.
- 🎞️ **Every H.264 runs on the in-house decoder.** Not just MVC — plain 2D, **Full-SBS (FSBS)**, and H.264 in **MP4 / AVI / MOV / raw** all decode through edge264 and its pipeline; mpv is only a fallback for other codecs.
- 🌈 **True HDR, not a tone-mapped fake.** Frames land in a 16-bit-float **scRGB** Direct3D 11 swapchain; a GPU shader does YUV→RGB and the stereo frame-packing in one pass. HDR10/PQ is preserved end to end.
- 🥽 **Real 3D output.** Frame-packed stereo to a detached window for 3D TVs, projectors and HMDs — plus an embedded 2D preview.
- 🎯 **Pixel-exact.** The decoder's luma output has been verified byte-for-byte against FFmpeg's base view. It's not "close enough" — it's correct.
- 🪶 **Self-contained.** One executable, or one portable folder. Nothing to install, no codec packs, no system pollution.
- 💿 **Archive your discs.** Image the 3D Blu-ray you're watching to a **byte-perfect `.iso`** from inside the player — one click, no admin, no external tool — so a failing optical drive can't take your collection with it.
- 🔍 **A timeline that shows you where you'll land.** Hover the seek bar and a large preview thumbnail follows your cursor — decoded **in-process by the same engine as playback**, live even during MVC playback and on mounted Blu-ray ISOs. Click, and you land **exactly** on the frame the preview promised.

---

## Under the hood

For the curious, here is what is actually happening between the file and your eyes — and why each step was hard enough to be interesting.

### 1. The decoder — `edge264`, taught to see in stereo
The heart of the player is **[edge264](https://github.com/tvlabs/edge264)**, a remarkable single-translation-unit H.264 decoder with hand-written SIMD kernels — **SSE2→AVX2** on x86, **NEON** on ARM. It is fast, lean, and BSD-licensed. But like everything else, it spoke only 2D.

This project extends it into a real **MVC (Annex H)** decoder: a second *dependent* view that predicts itself from the *base* view across the inter-view boundary, a per-view **decoded-picture-buffer** that has to honour `max_dec_frame_buffering` *separately* for each eye, SPS↔Subset-SPS fallback, PPS inheritance, frame-pairing, and graceful buffer-overflow handling so the two eyes never drift apart. Getting two interdependent H.264 bitstreams to march in lockstep, frame for frame, is most of the engineering.

### 2. The demuxer — pulling two eyes out of one container
A dedicated **C++ demuxer** (pybind11, on top of **libmatroska/libebml**) opens the MKV, finds the MVC track, and de-interleaves the base and dependent NAL units into the exact order the decoder expects — feeding a zero-copy ring buffer so decode never waits on I/O.

The same native module now owns the complete hot path into edge264: it locates the
decoder DLL, resolves and validates its C ABI, keeps compressed NAL storage alive
until edge264's completion callback, copies returned planes into owned frame storage,
and releases the Python GIL while decoding. Normal MVC playback therefore has no
per-NAL Python/ctypes boundary. The older ctypes decoder remains only as an automatic
safety fallback if native initialization fails repeatedly.

### 3. The renderer — HDR all the way to the panel, in native C++ D3D11
Decoded YUV planes are uploaded straight to the GPU. A **Direct3D 11** shader converts colour and assembles the stereo frame inside an **RGBA16F (scRGB)** HDR surface — the format Windows uses for native HDR — so there is no SDR round-trip and no OpenGL→DXGI copy tax. As of **4.1.0 the renderer is a ground-up _native C++ D3D11 engine_** (code-named *Tokyo #3*) that takes decoded planes **straight into D3D11 textures** with no per-frame Python/Qt copy — lower latency and less memory churn. It was pixel-validated byte-for-byte against the previous Qt renderer on real 3D hardware, and now **fully replaces it** — the Qt/RHI path is gone. For **Full-SBS** content each eye is **letterboxed into its frame-pack slot, never stretched**, so per-eye resolution is preserved.

### 4. The real-time problem — and moving the hot path out of Python
Audio rides on **libmpv**; video is slaved to mpv's clock so the two stay locked. For a long time MVC decode was **single-threaded** (the multiview decoder wasn't thread-safe), which made timing brutal: decoding a single key frame can take ~100 ms, and on a naïve loop that froze the picture once per GOP — a visible hitch every second. The fix was to **decouple presentation from decoding** (a dedicated presenter thread with back-pressure so the buffer absorbs the spikes) and then to wrestle the **CPython GIL** itself — `sys.setswitchinterval(0.0005)` was the decisive change that stopped the decode thread from starving the presenter. Result on a dense scene: **16 fps with 33 % dropped frames → a steady 24 fps with zero drops.**

As of **4.5.0 the decoder itself is multithreaded** — a task-engine race and a scene-cut
picture-ordering bug were hunted down and fixed. The current native pipeline chooses its
worker count from the available logical/physical CPU budget (three workers on a typical
4-core/8-thread Zen 2 handheld) and releases the GIL across native demux/decode calls.
`SYLC_EDGE264_THREADS` remains available for controlled A/B qualification.

### 5. 2D→3D — one depth engine, many surfaces, and a hard deadline

Turning a flat film into stereo in real time is a scheduling problem before it
is a machine-learning one. The model is the easy part: **Depth Anything V3**
infers a depth map, the renderer warps the frame into a stereo pair. The hard
part is that the answer is due **every 41.7 ms**, forever, while the same GPU is
also decoding video and drawing two output surfaces.

The pipeline that came out of it looks like this. A **single process-wide
inference service** owns the model, so attaching more surfaces — embedded
preview, frame-packed window, two projector eyes, SyLC Cast — never duplicates
model work; one surface holds a short renewable *input leader* lease and does
the GPU readback, and everyone consumes the same published map on their own
device. The service worker overlaps stages deliberately: the GPU infers map *N*
while the CPU runs boundary refinement, temporal fusion and geometry for map
*N-1*, with optical flow farmed to a persistent thread pool. Depth is
**temporally stabilized in video time, not compute time** — the two clocks are
kept separate on purpose, so a faster or slower GPU changes how *often* the
depth refreshes, never how the depth *behaves*.

Cuts get special treatment, because a depth map is always a frame or two behind
the picture it describes. A **look-ahead scout** runs a miniature analysis of
each decoded frame far enough ahead of presentation to see a cut coming, and
the renderer flattens disparity through the transition rather than warping the
first frame of a new shot with the previous shot's geometry. The temporal
history, the accumulated background plate and the depth EMA all re-prime on the
confirmed cut.

Everything above is instrumented and readable at runtime — the 2D→3D menu
footer reports the provider, the active preset and grid, the delivered depth
rate and the age of the current map, and the log carries the full per-stage
breakdown. That instrumentation is not decoration: all three fixes in 5.2.1
were found by reading it, and two of them turned out to be the opposite of what
the symptom suggested.

---

## War stories

Months of work hide inside a few one-line fixes. A taste:

- **The "Frankenstein" banding.** *Gravity* and other demanding discs came out sliced with horizontal bands of wrong colour. The cause was buried deep in dequantization: when a picture declared a scaling matrix but supplied no lists and the sequence had none either, the decoder fell back to a **flat-16** matrix instead of the **H.264 default** matrices the spec mandates. One wrong fallback, an entire film corrupted. Fixed in the PPS parser.
- **The decoder that worked everywhere but Windows.** Every slice failed with `EBADMSG`. The culprit: Windows' `<windows.h>` defines `min`/`max` as **macros**, which silently replaced edge264's own inline `min`/`max` and made the **CABAC** arithmetic diverge bit-for-bit. The fix is three characters — `NOMINMAX` — and finding it took considerably longer than typing it.
- **The deadlock between two eyes.** Under load the per-view buffers could wedge against each other; it took an entry-guard bypass, a graceful frame-bump path, and a force-complete with chroma concealment to guarantee the stereo pair always advances.
- **The corruption that wasn't ours.** Some 3D discs played with maddening, repeatable artifacts — a strobing band, a "stair-step" stutter, transient blocks of garbage, always in the same spots. After a long decoder hunt, the truth was humbler and stranger: the **optical drive itself was returning corrupt reads.** Imaging the very same disc to an `.iso` and playing *that* is flawless — the decoder had been right all along. That finding is exactly why 4.0.0 ships a built-in **disc→ISO archiver**: a clean image routes around a dying drive.

- **The 4K film that converted faster than the 1080p one.** A 4K HEVC master hit 24 depth maps per second; the *same film* in 1080p H.264 managed 19.5. Same inference grid, same engine — and the 1080p file's GPU inference was measured **2.3× faster** (9 ms vs 21 ms). A component that is faster inside a system that is slower is never a throughput problem; it is a *phase* problem. The depth engine turned out to be idle-waiting 45 % of every cycle: the readback copy it needs was being issued but **not submitted** — a 1080p frame carries so little GPU work that the driver never accumulated enough to flush its command buffer on its own, so the copy was still unreadable a whole frame later, no work was handed to the engine at all, and it lost 41.7 ms. On the 4K path the heavy per-frame work flushed it implicitly, which is precisely why the *harder* file never missed. One `Flush()` after the copy: stalled hand-offs went from 436 in 1991 to **1**, and the cycle locked to 41.7 ms.
- **The cut that tore one frame, every time.** Every shot change produced a single wavy, "gravitational-lensing" frame. The cut *detection* was never at fault — it is reliable — and every piece of temporal state (flow, plate, EMA, history) was correctly re-primed. The culprit was the **advisory** that carries the news: it was refreshed by a **10 Hz** timer while frames present at **24 fps**, so it stayed frozen for up to 100 ms. On the first frame of a new shot it still read "a cut is coming in +δ ms" — so the renderer re-opened up to 55 % of the disparity budget and kept filling disocclusions from a background plate made entirely of *the previous shot's* pixels. Dead-reckoning the advisory against its own age puts the deadline back where it belongs.
- **The seek that cost a second of depth.** Every seek deliberately bounced the adaptive selection back through the square inference grid so the new position could re-earn its verdict. Sound in principle — a film really can switch between Scope and IMAX reels — but on a natively wide master the selection comes from the **coded frame dimensions**, and those belong to the file, not to the position: no seek can invalidate them. The detour was protecting nothing, and charging ~1-3 s of flat-then-square-then-flat depth and two full temporal re-primes for it. Now only a selection derived from an encoded matte still takes the safe route.

This is the kind of work that doesn't show up in a feature list — but it's the difference between "plays MVC" and *plays MVC correctly, every frame, on every disc.*

---

## Features

- **3D MVC playback** — H.264 Stereo High (profile 128), both views decoded in-house.
- **All H.264 through edge264** — MVC, plain 2D, and **Full-SBS (FSBS)** all decode in-house; **mpv is only a fallback** for non-H.264 codecs or if edge264 can't handle a stream.
- **Full-SBS (FSBS) 3D** — one H.264 stream carrying both eyes is detected, split, and frame-packed (each eye **letterboxed, not stretched**); a contextual on-screen badge confirms the detected format.
- **Native C++ Direct3D 11 rendering** with **HDR (PQ)** preservation and high-quality scaling — the sole render path (the legacy Qt/RHI engine was removed in 4.1.0).
- **2D→3D AI synthesis** — turn a flat film into stereo 3D in real time. A
  bundled **Depth Anything V3** model estimates depth from the frames being
  played and the renderer synthesizes the second eye from it, with temporal
  stabilization and cut handling so the depth doesn't shimmer or jump at a cut.
  Three **depth presets** in the 2D→3D menu — **Quality** (Base model, 756×756
  inference grid), **Balanced** (same model, 518×518) and **Performance**
  (Small model, 518×518) — trade depth detail against how often the depth map
  refreshes; switchable during playback and remembered between sessions. Runs
  on **DirectML** (any Direct3D 11 GPU, nothing to install); an **optional,
  local, opt-in NVIDIA TensorRT** path lifts that refresh rate by ~1.7-2x on
  supported cards. It is installed from inside the player — the **Depth
  models…** dialog detects the GPU, downloads the runtime and either fetches
  prebuilt engines for it or compiles them locally, then proves a real engine
  builds and infers correctly on your machine before the player will use it.
  No Python toolchain, and never required.
  **5.2.1 makes the conversion keep up with the film.** The converter now
  reaches the source cadence on ordinary 1080p H.264 films instead of
  hovering around 19-20 depth maps per second (see the war story below), it
  no longer tears the first frame of every new shot, and a click in the
  timeline no longer costs a second of flat, half-finished depth. On
  letterboxed or natively wide **Scope** masters it also detects the black
  bars and infers on a **rectangular grid** — 756×322 instead of 756×756,
  57 % fewer depth pixels, roughly twice the depth rate for the same picture,
  because black bars carry no depth worth computing.
- **Frame-packed 3D output** (detached window) + embedded 2D view.
- **Broad container support** — **MKV / MP4 / AVI / MOV / FLV / WebM / raw `.h264`** (the native C++ demuxer + a libavformat-backed demuxer), all decoded by edge264.
- **Raw Blu-ray streams** — plays **SSIF** (3D) and **M2TS** (2D) directly, *no remux*, with frame-accurate seeking.
- **Open a whole Blu-ray** — point SyLC at a **disc/drive, a BDMV folder, or an `.iso`**; the feature film is auto-detected by **duration-based main-title detection** (3D SSIF preferred, 2D otherwise). ISO images are **auto-mounted without admin rights** and released on exit.
- **Archive a Blu-ray to ISO** — image the disc you're playing to a **byte-perfect `.iso`** from inside the player (no admin, no external tool); resilient to a flaky drive, with optional **SHA-256** verification.
- **Non-H.264 compatibility** — VC-1 / MPEG-2 / HEVC… (incl. **2D Blu-rays**) play through libmpv at the correct aspect.
- **PGS (Blu-ray) subtitles** — streamed in real time, **labelled by language** (from the disc's CLPI), and shown on **both the 3D and the 2D** views.
- **Timeline preview thumbnails** — hover the seek bar for an instant **320×180 preview** with a time pill, decoded **in-process by edge264** (no external processes). Live during MVC playback, on mounted **Blu-ray ISOs**, and for packed-stereo sources (single eye shown); on physical discs the timeline fills itself from frames already decoded, at zero disc I/O. **Clicking lands exactly on the previewed frame** — Blu-ray open-GOP recovery points included.
- **Multithreaded in-house decoding** — edge264 uses an adaptive worker count
  (three on a typical 4C/8T handheld), with `SYLC_EDGE264_THREADS` available for
  qualification; scene-cut ordering is fixed in the decoder.
- **Live A/V sync trim** to cancel your system's audio-output latency — nudge it by ear with `[` and `]` (persisted across sessions). True container-PTS timestamps with micro-pacing keep lip-sync honest.
- **Instant, smooth seeking** — no post-seek lag, audio back immediately, and the preview shows the exact landing frame before you click.
- **Completely free** — every feature unlocked, forever.

### Keyboard shortcuts
| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `Esc` | Exit fullscreen |
| `]` / `[` | Delay / advance the video for A/V sync (±50 ms) |

---

## Native x64 build — no emulation

| Flavor | Asset | Notes |
|---|---|---|
| **Portable folder** | `SyLC_3D_Player_v5.2.0_win-x64.zip` | Unzip anywhere and run `SyLC_3D_Player.exe` — no extraction step, no installer. |

Everything needed to play video — decoder, demuxer, audio, codecs, Python
runtime — is bundled.

The **2D→3D depth models** are not. Depth Anything V3 weighs 4.61 GB across the
twenty exported graphs, so they live in a
[HuggingFace repository](https://huggingface.co/Symphoenix/sylc_TRT) and are
fetched from inside the player, via **Depth models…** in the 2D→3D menu. Two packs:
**Small** (960 MB) makes all three depth presets work, **Base** (3.67 GB) is the
quality upgrade. Downloads resume after an interruption, every file is SHA-256
verified, and **no account is needed**.

The Windows module requires **AVX2**. edge264 contains baseline, x86-64-v2 and
x86-64-v3 implementations and dispatches to the best supported one at runtime. Its
SIMD hot loop therefore runs natively on Zen 2/Z2 A and other supported AVX2 CPUs:
real silicon, no translation layer.
*(The ARM64/NEON port lives in the codebase, but **5.2.0 ships Windows x64 only**.)*

---

## System requirements

Ryzen Z2 A / low-power Zen 2 qualification and tuning:
[docs/Z2A_VALIDATION.md](docs/Z2A_VALIDATION.md).

- **Windows 10/11 (x64)**.
- A **Direct3D 11**-capable GPU (an HDR display to enjoy HDR).
- A CPU with **AVX2** (standard since ~2013).
- Input: a **3D MKV** (MVC track), a **raw Blu-ray stream** (`.ssif` / `.m2ts`), a **BDMV disc/folder**, or a **Blu-ray `.iso`**. Rip with **MakeMKV**, or just point SyLC at the disc. (2D files of any codec play through libmpv.)

> **No remux required for Blu-rays.** Open the **disc/drive**, the **BDMV folder**, or the **`.iso` directly** — SyLC mounts the image (no admin), finds the 3D feature by duration, and streams the **SSIF** straight off it. `.iso` opens via *Open file* or drag-and-drop; a disc/folder via the **disc** button or drag-and-drop.

---

## Get started

1. Download the asset for your platform from **Releases**.
2. Unzip `SyLC_3D_Player_v5.2.0_win-x64.zip` anywhere and run `SyLC_3D_Player.exe`.
3. For **2D→3D AI conversion**: open the **2D→3D menu → Depth models…** and pick
   a pack. Start with **Small** (960 MB) — every preset works on it. Skip this
   if you only play 3D content; nothing else needs it.
4. Open your 3D content — a **MKV**, a raw **`.ssif` / `.m2ts`**, a **BDMV folder**, or a Blu-ray **`.iso`** (drag-and-drop, the **Open file** button, or the **disc** button). Send the frame-packed window to your 3D display and enjoy.

Nothing to install. Everything — decoder, demuxer, audio, codecs, Python runtime — is bundled.

---

## Build from source

Everything needed lives in this repository: the Python application, the **decoder sources** (`edge264/`), the **demuxer sources** (`mvc_realtime_demuxer/`), the binaries, and the build scripts. Full details in **[`BUILD.md`](BUILD.md)**.

The short version (x64):

```bat
:: edge264 decoder (MSYS2 / MinGW64) — portable runtime-dispatched build
cd edge264
build_sylc_edge264.bat
copy /Y edge264_candidate.dll ..\edge264.dll
cd ..

:: standalone no-console build (Nuitka + MSVC 2022)
build_exe_v520.bat
```

`build_exe_v520.bat` is the current release script — and the only one used to
build v5.2.0: a one-folder standalone build that bundles `onnxruntime.dll`,
`DirectML.dll` and `models/MANIFEST.json` — the inference runtime and the
download manifest, but no model weights.

The older `build_exe_onefile.bat` is **not part of the v5.2.0 release**. It is
still stamped 5.0.0 and bundles none of the three files above, so a binary built
from it reports `models/MANIFEST.json is missing from this install` and cannot
do AI conversion at all. That is why the release ships one asset, the portable
folder, and no single-file `.exe`.

Prerequisites: **Python 3.14**, `pip install -r requirements.txt` + `nuitka` +
`pybind11`, **MSVC 2022**, and **MSYS2/MinGW64 GCC** for edge264. The checked-in
response files build baseline/v2/v3 objects explicitly; do not replace them with
`-march=native` for a redistributable release. `BUILD.md` documents the 2D→3D
prerequisites (ONNX Runtime + DirectML, the per-preset model files) and the
optional TensorRT flow.

---

## Architecture at a glance

```
   MKV (MVC)
      │
      ▼
 ┌──────────────┐   base + dependent NAL units (zero-copy ring buffer)
 │  C++ demuxer │ ───────────────────────────────────────────────►
 │ libmatroska  │
 └──────────────┘
      │
      ▼
 ┌──────────────┐   two interdependent H.264 views, decoded in lockstep
 │   edge264    │   (AVX2 on x64 · NEON on ARM64 · GIL released)
 │  MVC decoder │ ───────────────────────────────────────────────►
 └──────────────┘
      │ YUV planes
      ▼
 ┌──────────────┐   YUV→RGB + stereo frame-packing in one GPU pass
 │ Native D3D11 │   RGBA16F (scRGB) HDR swapchain
 │  HDR shader  │ ──────────────►  3D display / projector / HMD
 └──────────────┘
                    audio ── libmpv ──► clock that video is slaved to
```

*As of 4.1.0 the **native C++ D3D11** renderer above is the sole render path (the Qt/RHI engine was removed); decoded planes go straight into D3D11 textures with no per-frame Python/Qt copy. Full-SBS eyes are letterboxed into the frame-pack slot, not stretched.*

---

## License & credits

**Free & open-source.** The **edge264** decoder is **BSD**-licensed (see `edge264/LICENSE_BSD.txt`). SyLC also stands on the shoulders of great GPL/LGPL projects — please honour their licenses when redistributing.

- **[edge264](https://github.com/tvlabs/edge264)** — the fast H.264/AVC decoder this project extends to MVC
- **[libmpv / mpv](https://mpv.io/)** — audio engine
- **[libmatroska / libebml](https://www.matroska.org/)** — Matroska demuxing
- **[FFmpeg](https://ffmpeg.org/)** — `ffprobe` for stream & subtitle analysis
- **[Qt / PySide6](https://www.qt.io/)** — UI and Direct3D 11 rendering
- **[Nuitka](https://nuitka.net/)** — standalone compilation

---

<div align="center">

**Built over months, for the love of the format — and given freely to everyone who refused to let 3D die.**

*If SyLC brought one of your discs back to life, that's the whole reward. Long live open source. 🥂*

</div>
