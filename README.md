<div align="center">

# SyLC 3D Player

<img src="splash.png" alt="SyLC 3D Player Logo" width="250" />

### A free, open-source player for the 3D format the industry left behind.

*Stereoscopic 3D Blu-ray (MVC) playback, decoded from scratch, rendered in native HDR — given to the community, no strings attached.*

![Version](https://img.shields.io/badge/version-3.1.0-1f6feb?style=for-the-badge)
![Platform](https://img.shields.io/badge/Windows-x64%20%7C%20ARM64-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
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
- 🌈 **True HDR, not a tone-mapped fake.** Frames land in a 16-bit-float **scRGB** Direct3D 11 swapchain; a GPU shader does YUV→RGB and the stereo frame-packing in one pass. HDR10/PQ is preserved end to end.
- 🥽 **Real 3D output.** Frame-packed stereo to a detached window for 3D TVs, projectors and HMDs — plus an embedded 2D preview.
- 🎯 **Pixel-exact.** The decoder's luma output has been verified byte-for-byte against FFmpeg's base view. It's not "close enough" — it's correct.
- 🪶 **Self-contained.** One executable (x64) or one portable folder (ARM64). Nothing to install, no codec packs, no system pollution.

---

## Under the hood

For the curious, here is what is actually happening between the file and your eyes — and why each step was hard enough to be interesting.

### 1. The decoder — `edge264`, taught to see in stereo
The heart of the player is **[edge264](https://github.com/tvlabs/edge264)**, a remarkable single-translation-unit H.264 decoder with hand-written SIMD kernels — **SSE2→AVX2** on x86, **NEON** on ARM. It is fast, lean, and BSD-licensed. But like everything else, it spoke only 2D.

This project extends it into a real **MVC (Annex H)** decoder: a second *dependent* view that predicts itself from the *base* view across the inter-view boundary, a per-view **decoded-picture-buffer** that has to honour `max_dec_frame_buffering` *separately* for each eye, SPS↔Subset-SPS fallback, PPS inheritance, frame-pairing, and graceful buffer-overflow handling so the two eyes never drift apart. Getting two interdependent H.264 bitstreams to march in lockstep, frame for frame, is most of the engineering.

### 2. The demuxer — pulling two eyes out of one container
A dedicated **C++ demuxer** (pybind11, on top of **libmatroska/libebml**) opens the MKV, finds the MVC track, and de-interleaves the base and dependent NAL units into the exact order the decoder expects — feeding a zero-copy ring buffer so decode never waits on I/O.

### 3. The renderer — HDR all the way to the panel
Decoded YUV planes are uploaded straight to the GPU. A Qt **RHI / Direct3D 11** shader converts colour and assembles the stereo frame inside an **RGBA16F (scRGB)** HDR surface — the format Windows uses for native HDR — so there is no SDR round-trip and no OpenGL→DXGI copy tax.

### 4. The real-time problem — and the Python GIL
Audio rides on **libmpv**; video is slaved to mpv's clock so the two stay locked. But MVC decode is **single-threaded** (the multiview decoder isn't thread-safe), which makes timing brutal: decoding a single key frame can take ~100 ms, and on a naïve loop that froze the picture once per GOP — a visible hitch every second. The fix was to **decouple presentation from decoding** (a dedicated presenter thread with back-pressure so the buffer absorbs the spikes) and then to wrestle the **CPython GIL** itself — `sys.setswitchinterval(0.0005)` was the decisive change that stopped the decode thread from starving the presenter. Result on a dense scene: **16 fps with 33 % dropped frames → a steady 24 fps with zero drops.**

---

## War stories

Months of work hide inside a few one-line fixes. A taste:

- **The "Frankenstein" banding.** *Gravity* and other demanding discs came out sliced with horizontal bands of wrong colour. The cause was buried deep in dequantization: when a picture declared a scaling matrix but supplied no lists and the sequence had none either, the decoder fell back to a **flat-16** matrix instead of the **H.264 default** matrices the spec mandates. One wrong fallback, an entire film corrupted. Fixed in the PPS parser.
- **The decoder that worked everywhere but Windows.** Every slice failed with `EBADMSG`. The culprit: Windows' `<windows.h>` defines `min`/`max` as **macros**, which silently replaced edge264's own inline `min`/`max` and made the **CABAC** arithmetic diverge bit-for-bit. The fix is three characters — `NOMINMAX` — and finding it took considerably longer than typing it.
- **The deadlock between two eyes.** Under load the per-view buffers could wedge against each other; it took an entry-guard bypass, a graceful frame-bump path, and a force-complete with chroma concealment to guarantee the stereo pair always advances.

This is the kind of work that doesn't show up in a feature list — but it's the difference between "plays MVC" and *plays MVC correctly, every frame, on every disc.*

---

## Features

- **3D MVC playback** — H.264 Stereo High (profile 128), both views decoded in-house.
- **Direct3D 11 / Qt RHI rendering** with **HDR (PQ)** preservation and high-quality scaling.
- **Frame-packed 3D output** (detached window) + embedded 2D view.
- **Matroska (MKV)** input with an MVC track, via the native demuxer.
- **Raw Blu-ray streams** — plays **SSIF** (3D) and **M2TS** (2D) directly, *no remux*, with frame-accurate seeking.
- **Open a whole Blu-ray** — point SyLC at a **disc/drive, a BDMV folder, or an `.iso`**; the feature film is auto-detected by **duration-based main-title detection** (3D SSIF preferred, 2D otherwise). ISO images are **auto-mounted without admin rights** and released on exit.
- **Broad 2D compatibility** — any 2D video plays through libmpv (H.264 / VC-1 / MPEG-2…), including **2D Blu-rays**, at the correct aspect.
- **PGS (Blu-ray) subtitles**, streamed in real time.
- **Live A/V sync trim** to cancel your system's audio-output latency — nudge it by ear with `[` and `]`.
- **Instant, smooth seeking** — no post-seek lag.
- **Completely free** — every feature unlocked, forever.

### Keyboard shortcuts
| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `Esc` | Exit fullscreen |
| `]` / `[` | Delay / advance the video for A/V sync (±50 ms) |

---

## Two native builds — no emulation

| Platform | Asset | Notes |
|---|---|---|
| **Windows x64** | `SyLC_3D_Player_v3.1.0_win-x64.exe` | Single self-contained file. Built for the **x86-64-v3 (AVX2)** baseline — runs natively on any AVX2 CPU (Haswell 2013+ / Zen 1+). |
| **Windows on ARM** | `SyLC_3D_Player_v3.1.0_win-arm64.zip` | Portable folder, **100 % native ARM64** (Snapdragon / Adreno) — every binary cross-compiled to aarch64, zero x64 emulation. |

The decoder's SIMD hot loop is compiled for each architecture's vector unit (AVX2 / NEON), so you get the real silicon, not a translation layer.

---

## System requirements

- **Windows 10/11 (x64)** or **Windows 11 on ARM (ARM64)**.
- A **Direct3D 11**-capable GPU (an HDR display to enjoy HDR).
- x64: a CPU with **AVX2** (standard since ~2013).
- Input: a **3D MKV** (MVC track), a **raw Blu-ray stream** (`.ssif` / `.m2ts`), a **BDMV disc/folder**, or a **Blu-ray `.iso`**. Rip with **MakeMKV**, or just point SyLC at the disc. (2D files of any codec play through libmpv.)

> **No remux required for Blu-rays.** Open the **disc/drive**, the **BDMV folder**, or the **`.iso` directly** — SyLC mounts the image (no admin), finds the 3D feature by duration, and streams the **SSIF** straight off it. `.iso` opens via *Open file* or drag-and-drop; a disc/folder via the **disc** button or drag-and-drop.

---

## Get started

1. Download the asset for your platform from **Releases**.
2. **x64:** run `SyLC_3D_Player_v3.1.0_win-x64.exe`. **ARM64:** unzip and run `SyLC_3D_Player.exe`.
3. Open your 3D content — a **MKV**, a raw **`.ssif` / `.m2ts`**, a **BDMV folder**, or a Blu-ray **`.iso`** (drag-and-drop, the **Open file** button, or the **disc** button). Send the frame-packed window to your 3D display and enjoy.

Nothing to install. Everything — decoder, demuxer, audio, codecs, Python runtime — is bundled.

---

## Build from source

Everything needed lives in this repository: the Python application, the **decoder sources** (`edge264/`), the **demuxer sources** (`mvc_realtime_demuxer/`), the binaries, and the build scripts. Full details in **[`BUILD.md`](BUILD.md)** (x64) and **`BUILD_ARM.md`** (ARM64).

The short version (x64):

```bat
:: edge264 decoder (MSYS2 / UCRT64) — portable AVX2 build
gcc -shared -o edge264.dll -O3 -march=x86-64-v3 -flax-vector-conversions edge264/src/edge264.c -lpthread

:: one-file no-console executable (Nuitka + MSVC 2022)
build_exe_onefile.bat
```

Prerequisites: **Python 3.13**, `pip install -r requirements.txt` + `nuitka` + `pybind11`, **MSVC 2022**, and **MSYS2/GCC** for edge264. Swap `-march=x86-64-v3` for `-march=znver3` (or `native`) if you're building only for your own machine and want every last drop of Zen 3.

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
 │  D3D11 / RHI │   RGBA16F (scRGB) HDR swapchain
 │  HDR shader  │ ──────────────►  3D display / projector / HMD
 └──────────────┘
                    audio ── libmpv ──► clock that video is slaved to
```

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


