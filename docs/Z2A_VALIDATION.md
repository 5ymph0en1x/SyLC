# Ryzen Z2 A validation

Zen 2 supports the AVX2/BMI2/FMA baseline used by SyLC. The Z2 A performance
profile is nevertheless constrained by four physical cores, shared LPDDR memory
and a 6–20 W configurable power envelope.

## Baseline

Test while plugged in, at 20 W, with overlays/casting/thumbnails disabled and
the external display initially set to 60 Hz. Use the same local media file for
every run; do not benchmark from an optical drive.

SyLC automatically selects three edge264 workers on an 8-logical-CPU system.
For an A/B comparison, launch from `cmd.exe` with one of:

```bat
set SYLC_EDGE264_THREADS=2
set SYLC_EDGE264_THREADS=3
set SYLC_EDGE264_THREADS=4
```

Record for at least five minutes:

- displayed FPS and dropped frames;
- frame-time p50, p95 and p99;
- CPU/GPU clocks, utilization, temperature and package power;
- whether detached frame-packing is active;
- media codec, resolution, bit depth and stereo layout.

The best setting is the lowest stable p99, not necessarily the highest average
decode FPS. Two or three workers can outperform four when all four physical
cores otherwise contend with audio, GUI and presentation work.

## Renderer and HEVC checks

When the detached frame-packing window is visible, only that output should be
fed frames. A large regression versus embedded 2D should therefore be reported
with `SYLC_HEVC_DIAG=1` logs from:

```text
%LOCALAPPDATA%\SyLC3DPlayer\logs\sylc-player.log
```

For packed HEVC only, compare hardware copy-back with software decoding:

```bat
set SYLC_HEVC_HW=1
set SYLC_HEVC_HW=0
```

MV-HEVC remains a software-decoded path. Neither result should be generalized
to MVC/H.264 without testing the corresponding source format.

## Stability qualification

Before calling a Z2 A build qualified, run:

- 8 hours of continuous MVC playback;
- 500 random seeks;
- repeated stop/load cycles;
- sleep/resume and display hot-plug;
- corrupted/truncated input;
- 15 W and 20 W power modes.

Attach `sylc-player.log` and `crash_log.txt` from the log directory to any
failure report.
