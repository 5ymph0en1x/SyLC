# Pre-built TensorRT engines — sm89 / TensorRT 10.16.1.11

**Read this before downloading 3.1 GB.**

TensorRT engines are not portable. These were built for:

- **Compute capability sm89** — Ada Lovelace, i.e. GeForce RTX 40-series.
  A 30-series (sm86), 50-series (sm120), AMD or Intel GPU cannot load them.
- **TensorRT 10.16.1.11**, against the ONNX Runtime build that
  `tools_dev/setup_tensorrt.py` fetches. A different TensorRT version rejects
  them.

If any of that does not match your machine, ignore this directory: TensorRT will
build its own engines, which is the normal path and works everywhere.

## What they save

21 engines covering every graph a SyLC depth preset can open. Compiling them
locally costs roughly 200 seconds each — about **70 minutes** from a cold cache.

## They do not skip verification

SyLC still runs its own local engine probe and only writes the `.trt_verified`
marker after a real engine build plus an inference that passes a near/far
polarity gate **on your machine**. These engines make that probe fast; they are
never taken on trust. Playback never gambles on an unverified runtime.

## Use

```
python tools_dev/setup_tensorrt.py --fetch-engines --engine-probe
```
