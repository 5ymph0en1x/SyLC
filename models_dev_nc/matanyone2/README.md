# MatAnyone 2 local runtime assets

These files support SyLC's optional, asynchronous human-contour guidance.
They are intentionally outside `models/` and are not release assets.

## Pinned assets

- `matanyone2.pth`
  - source: https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth
  - size: 141,429,115 bytes
  - SHA-256: `5E9821E4087231427376B437C85BB6E072B41E582314F06FD524F75BC4AF5914`
  - licence: NTU S-Lab License 1.0, non-commercial use/redistribution only.
- `lraspp_mobilenet_v3_large-d234d4ea.pth`
  - source: https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth
  - size: 13,097,061 bytes
  - SHA-256: `D234D4EAE9D55D5F76DE18B77CF0DC62C66FE5C5482758209D00F950C92BB280`
  - purpose: Pascal VOC person-class seed; MatAnyone 2 then reconstructs the alpha.

The worker runs in `../matanyone2_runtime/.venv`, created with `uv`, and is
discovered only when both checkpoints and the isolated Python executable are
present. Playback never downloads a model and always falls back to depth-only
synthesis if this optional runtime is absent or unhealthy.

Recreate the pinned local runtime with:

```powershell
powershell -ExecutionPolicy Bypass -File tools_dev/setup_matanyone2_runtime.ps1
```

The upstream `uv.lock` targets PyTorch `cu128`. This is deliberate even on a
machine whose NVIDIA driver reports CUDA 13.1: the driver is backward
compatible with the CUDA 12.8 runtime carried inside the PyTorch wheels, and
the system CUDA toolkit is neither loaded nor required.

## Runtime controls

- `SYLC_MATANYONE2=0`: disable the optional worker.
- `SYLC_MATANYONE2_FPS=5`: sampling target (1–15 fps).
- `SYLC_MATANYONE2_SHORT_SIDE=720`: maximum short side (256–1080 px).
- `SYLC_MATANYONE2_PYTHON`, `SYLC_MATANYONE2_MODELS`,
  `SYLC_MATANYONE2_WORKER`: explicit deployment overrides.

The default `auto` mode starts MatAnyone 2 only when the complete offline
runtime is found. The native depth renderer remains authoritative for geometry;
the alpha is used exclusively for ownership, hole-fill veto, local stereo
safety and fractional contour reconstruction.
