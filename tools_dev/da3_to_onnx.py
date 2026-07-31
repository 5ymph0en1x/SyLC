"""Convert Depth Anything 3 (Small / Base) to a depth-only ONNX graph.

DA3 ships as PyTorch + safetensors, while SyLC's native 2D->3D path runs its
depth estimator through ONNX Runtime. This exporter produces the single
depth-only ``.onnx`` graph consumed by that path.

What gets exported
------------------
Only the backbone and depth head are called. Camera, sky, ray and
Gaussian-Splat branches are deliberately excluded because they are irrelevant
to SyLC and contain dynamic control-flow operators unsuitable for this static
runtime graph. The remaining graph is DINOv2 (ViT-S for Small, ViT-B for Base)
plus the DualDPT depth head.

Why a fixed input
-----------------
The transformer uses 14-pixel patches, 2D RoPE and a learned positional
embedding. The default 756x756 input is a 54x54 patch grid; rectangular
exports use the same rule independently on both axes. Positional
interpolation is evaluated and frozen during export, yielding a predictable
quality graph with no dynamic spatial axes. The single-view (S=1) dimension
also keeps reference-view selection constant. Only the batch axis is dynamic.

Input  : ``image``  float32 (B, 3, HEIGHT, WIDTH), ImageNet-normalised RGB.
Output : ``depth``  float32 (B, h_out, w_out) -- raw DA3 depth (distance-like,
         smaller = nearer). SyLC resizes it to the frame and inverts it at
         runtime.

Usage
-----
    tools_dev/.da3_export_venv/Scripts/python.exe tools_dev/da3_to_onnx.py

This defaults to DA3-Base at 756x756 and validates PyTorch against ONNX Runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# --- Locations -------------------------------------------------------------
# SyLC keeps the official source isolated under tools_dev/_vendor and the
# original Hugging Face checkpoints under models/DA3-<VARIANT>.  Exported
# runtime graphs live alongside them but never overwrite a source checkpoint.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENDOR = DEFAULT_ROOT / "tools_dev" / "_vendor" / "depth-anything-3" / "src"
DEFAULT_SRC_ROOT = DEFAULT_ROOT / "models"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "models"

VARIANTS = {
    "small": {"src": "DA3-SMALL", "onnx": "da3_small.onnx"},
    "base": {"src": "DA3-BASE", "onnx": "da3_base.onnx"},
}

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class DepthOnlyWrapper(nn.Module):
    """Wraps DepthAnything3 as ``-> (depth, confidence) (B,H,W)``.

    Calls the inner ``DepthAnything3Net`` directly (bypassing the api-level
    bf16 autocast) so the whole graph stays fp32, matching the head, which
    already runs autocast-disabled.

    ``fold_preprocess`` makes the graph take a uint8 ``(B,H,W,3)`` letterboxed
    canvas and do the ImageNet normalize + channel transpose on-device, so the
    runtime only pays the cv2 letterbox on the CPU (the numpy normalize was
    ~6 ms/frame at 1080p).
    """

    def __init__(self, da3: nn.Module, fold_preprocess: bool = False,
                 frames: int = 1):
        super().__init__()
        self.net = da3.model  # DepthAnything3Net
        self.fold_preprocess = fold_preprocess
        self.frames = frames
        if fold_preprocess:
            self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.fold_preprocess:
            # uint8 (B,H,W,3) -> normalized float (B,3,H,W)
            x = image.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            image = (x - self._mean) / self._std
        # Single-frame exports keep the convenient 4D input contract; temporal
        # exports receive an explicit (B,S,3,H,W) clip.
        x = image.unsqueeze(1) if image.dim() == 4 else image
        # Call the two depth-producing stages directly. Current upstream DA3
        # no longer exposes the older skip_camera/skip_sky keyword arguments;
        # bypassing those post-head branches is both clearer and keeps their
        # quantile/control-flow operators out of the ONNX graph.
        feats, _ = self.net.backbone(
            x, cam_token=None, export_feat_layers=[], ref_view_strategy="middle")
        height, width = x.shape[-2:]
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.net._process_depth_head(feats, height, width)
        depth = out["depth"]
        confidence = out.get("depth_conf")
        if confidence is None:
            # Compatibility with a custom checkpoint whose head has no
            # confidence channel. Keeping the second output stable lets the
            # native runtime use one contract for all current DA3 variants.
            confidence = torch.ones_like(depth)
        # Net returns (B,S,H,W). Publish the newest view: causal playback can
        # feed [t-S+1..t] without adding future-frame latency.
        if depth.dim() == 4:
            depth = depth[:, -1]
            confidence = confidence[:, -1]
        elif depth.dim() == 5:
            depth = depth[:, -1, 0]
            confidence = confidence[:, -1, 0]
        return depth.float(), confidence.float()


def _patch_position_getter() -> None:
    """Replace RoPE's ``cartesian_prod`` grid with an ONNX-exportable equivalent.

    ``torch.cartesian_prod`` has no ONNX symbolic. The grid is a constant (it
    only depends on patch dims, not input data), so meshgrid + stack is an exact
    drop-in that the exporter can fold to a constant.
    """
    from depth_anything_3.model.dinov2.layers.rope import PositionGetter

    def _call(self, batch_size, height, width, device):
        key = (height, width)
        if key not in self.position_cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            self.position_cache[key] = torch.stack(
                (yy.reshape(-1), xx.reshape(-1)), dim=-1
            )
        cached = self.position_cache[key]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    PositionGetter.__call__ = _call


def load_da3(model_dir: Path, vendor_root: Path):
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object

    _patch_position_getter()
    with (model_dir / "config.json").open("r", encoding="utf-8") as fh:
        model_config = json.load(fh)["config"]
    net = create_object(OmegaConf.create(model_config))

    # Official checkpoints store the network below the API wrapper's "model."
    # prefix. Loading the network directly avoids importing DA3's unrelated
    # video, point-cloud and Gaussian-export dependencies.
    checkpoint_path = model_dir / "model.safetensors"
    checkpoint = load_file(str(checkpoint_path), device="cpu")
    state = {
        (name[6:] if name.startswith("model.") else name): tensor
        for name, tensor in checkpoint.items()
    }
    # Safetensors de-duplicates tied tensors and records the aliases in file
    # metadata. Recreate those names before asking PyTorch for a strict match.
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as fh:
        aliases = fh.metadata() or {}
    for alias, source in aliases.items():
        alias = alias[6:] if alias.startswith("model.") else alias
        source = source[6:] if source.startswith("model.") else source
        if source in state:
            state[alias] = state[source]
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/network mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}")
    net.eval()
    return SimpleNamespace(model=net)


def export_variant(
    variant: str,
    src_root: Path,
    vendor_root: Path,
    out_dir: Path,
    size: int,
    width: int,
    height: int,
    opset: int,
    device: str,
    validate: bool,
    fold_preprocess: bool = False,
    frames: int = 1,
) -> Path:
    info = VARIANTS[variant]
    model_dir = src_root / info["src"]
    # Every export carries its grid in the name (da3_base_518.onnx), because
    # the grid is now what a caller must match: DepthEngine rejects a
    # fixed-shape graph whose H/W disagrees with the requested side. 518 used
    # to be treated as "the canonical size" and got no suffix -- which for the
    # Small variant resolved to models/da3_small.onnx and would silently
    # overwrite the historical dynamic-axes community export still kept there
    # as a measurement point.
    width = width or size
    height = height or size
    suffix = f"_{width}" if width == height else f"_{width}x{height}"
    if frames > 1:
        suffix += f"_t{frames}"
    onnx_name = info["onnx"][:-5] + suffix + ".onnx"
    out_path = out_dir / onnx_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DA3 {variant} ===")
    print(f"  weights : {model_dir}")
    print(f"  output  : {out_path}  (grid={width}x{height}, frames={frames}, "
          f"fold_preprocess={fold_preprocess})")
    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"DA3 weights not found under {model_dir}")

    t0 = time.time()
    da3 = load_da3(model_dir, vendor_root)
    wrapper = DepthOnlyWrapper(
        da3, fold_preprocess=fold_preprocess, frames=frames).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    if fold_preprocess and frames > 1:
        raise ValueError("--fold-preprocess is currently single-frame only")
    if fold_preprocess:
        dummy = torch.randint(
            0, 255, (1, height, width, 3), dtype=torch.uint8, device=device)
    elif frames > 1:
        dummy = torch.randn(1, frames, 3, height, width,
                            dtype=torch.float32, device=device)
    else:
        dummy = torch.randn(
            1, 3, height, width, dtype=torch.float32, device=device)

    # Static H/W (RoPE + pos_embed are size-locked); only batch is dynamic.
    t0 = time.time()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["image"],
            output_names=["depth", "confidence"],
            dynamic_axes={
                "image": {0: "batch"},
                "depth": {0: "batch"},
                "confidence": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  exported in {time.time() - t0:.1f}s "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")

    if validate:
        validate_variant(
            wrapper, out_path, width, height, device, fold_preprocess, frames)
    return out_path


def validate_variant(wrapper: nn.Module, out_path: Path, width: int, height: int,
                     device: str,
                     fold_preprocess: bool = False, frames: int = 1) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    batch = 1 if frames > 1 else 2
    if fold_preprocess:
        sample = rng.integers(
            0, 255, (batch, height, width, 3), dtype=np.uint8)
    elif frames > 1:
        sample = rng.standard_normal(
            (batch, frames, 3, height, width)).astype(np.float32)
    else:
        sample = rng.standard_normal(
            (batch, 3, height, width)).astype(np.float32)

    with torch.inference_mode():
        torch_depth, torch_conf = wrapper(torch.from_numpy(sample).to(device))
        torch_depth = torch_depth.cpu().numpy()
        torch_conf = torch_conf.cpu().numpy()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(out_path), providers=providers)
    onnx_depth, onnx_conf = sess.run(["depth", "confidence"], {"image": sample})

    if torch_depth.shape != onnx_depth.shape:
        raise SystemExit(
            f"  [FAIL] shape mismatch torch={torch_depth.shape} onnx={onnx_depth.shape}"
        )
    diff = np.abs(torch_depth - onnx_depth)
    conf_diff = np.abs(torch_conf - onnx_conf)
    denom = np.abs(torch_depth).mean() + 1e-6
    print(
        f"  [validate] out shape {onnx_depth.shape} | "
        f"max abs {diff.max():.4e} | mean abs {diff.mean():.4e} | "
        f"rel {diff.mean() / denom:.4e} | "
        f"conf mean abs {conf_diff.mean():.4e} | "
        f"conf range {onnx_conf.min():.3f}..{onnx_conf.max():.3f} "
        f"(mean {onnx_conf.mean():.3f}) | providers={sess.get_providers()}"
    )
    if diff.mean() / denom > 1e-2:
        print("  [WARN] relative error > 1e-2; inspect before trusting depth output")
    else:
        print("  [OK] torch vs onnxruntime match within tolerance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["small", "base", "both"], default="base",
                   help="Checkpoint tier to export (default: Base quality tier)")
    p.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT,
                   help="Folder holding Small/ and Base/ DA3 weight dirs")
    p.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR,
                   help="Vendored DA3 source root (contains depth_anything_3/)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Where to write da3_*.onnx (this project's models/ folder)")
    p.add_argument("--size", type=int, default=756,
                   help="Square input side, must be a multiple of 14 (default: 756 = 54x54 quality grid)")
    p.add_argument("--width", type=int, default=0,
                   help="Fixed input width; use together with --height to override --size")
    p.add_argument("--height", type=int, default=0,
                   help="Fixed input height; use together with --width to override --size")
    p.add_argument("--frames", type=int, default=1,
                   help="Causal temporal views per inference; 1 or 3 recommended")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Trace device; cpu keeps the graph fp32 and deterministic")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument("--fold-preprocess", dest="fold_preprocess", action="store_true",
                   help="Bake ImageNet normalize into the graph; input becomes uint8 (B,size,size,3)")
    p.set_defaults(validate=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.width > 0) != (args.height > 0):
        raise SystemExit("--width and --height must be provided together")
    width = args.width or args.size
    height = args.height or args.size
    if width <= 0 or height <= 0 or width % 14 != 0 or height % 14 != 0:
        raise SystemExit(
            f"inference grid must be positive multiples of 14, got {width}x{height}")
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    variants = ["small", "base"] if args.variant == "both" else [args.variant]
    written = []
    for v in variants:
        written.append(
            export_variant(v, args.src_root, args.vendor, args.out_dir,
                           args.size, width, height,
                           args.opset, args.device, args.validate,
                           args.fold_preprocess, args.frames)
        )
    print("\nDone. Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
