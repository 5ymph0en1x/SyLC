"""Isolated PyTorch worker used by :mod:`synth3d_matting_service`.

Stdout is a binary protocol channel.  All model/import diagnostics are routed
to stderr so an upstream library print can never corrupt a matte payload.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


_PROTOCOL = sys.stdout.buffer
sys.stdout = sys.stderr


def _emit(message: dict, payload: bytes = b"") -> None:
    _PROTOCOL.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
    if payload:
        _PROTOCOL.write(payload)
    _PROTOCOL.flush()


def _read_exact(size: int) -> bytes:
    chunks = []
    remaining = size
    stream = sys.stdin.buffer
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise EOFError("host closed in frame payload")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-checkpoint", required=True)
    parser.add_argument("--short-side", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--scene-threshold", type=float, default=0.26)
    return parser.parse_args()


class Runtime:
    def __init__(self, args):
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        from matanyone2.inference.inference_core import InferenceCore
        from matanyone2.utils.get_default_model import get_matanyone2_model

        self.cv2, self.np, self.torch, self.F = cv2, np, torch, F
        self.InferenceCore = InferenceCore
        if not torch.cuda.is_available():
            raise RuntimeError("MatAnyone 2 requires a CUDA GPU in the live worker")
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        checkpoint = Path(args.checkpoint)
        seed_checkpoint = Path(args.seed_checkpoint)
        if not checkpoint.is_file() or not seed_checkpoint.is_file():
            raise FileNotFoundError("offline model assets are incomplete")
        self.model = get_matanyone2_model(str(checkpoint), self.device)
        with torch.no_grad():
            self.model.cfg.max_internal_size = int(args.short_side)

        self.seed_model = lraspp_mobilenet_v3_large(weights=None, num_classes=21)
        seed_state = torch.load(seed_checkpoint, map_location="cpu", weights_only=True)
        self.seed_model.load_state_dict(seed_state)
        self.seed_model.to(self.device).eval()

        self.processor = None
        self.previous_probe = None
        self.previous_pts = None
        self.warmup = max(0, min(10, int(args.warmup)))
        self.scene_threshold = max(0.05, min(0.8, float(args.scene_threshold)))

    def reset(self):
        self.processor = None
        self.previous_probe = None
        self.previous_pts = None

    def _scene_cut(self, rgb, pts_ms):
        cv2, np = self.cv2, self.np
        gray = cv2.cvtColor(cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        cut = False
        if self.previous_probe is not None:
            mean_delta = float(np.mean(np.abs(gray - self.previous_probe)))
            cut = mean_delta >= self.scene_threshold
        if self.previous_pts is not None and pts_ms >= 0:
            delta = pts_ms - self.previous_pts
            cut = cut or delta < -40.0 or delta > 1800.0
        self.previous_probe = gray
        self.previous_pts = pts_ms
        return cut

    def _human_seed(self, image):
        torch, F, np, cv2 = self.torch, self.F, self.np, self.cv2
        height, width = image.shape[-2:]
        seed_h = max(256, int(round(520 * height / min(height, width))))
        seed_w = max(256, int(round(520 * width / min(height, width))))
        sample = F.interpolate(image.unsqueeze(0), (seed_h, seed_w), mode="bilinear",
                               align_corners=False)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = self.seed_model((sample - mean) / std)["out"]
            probability = torch.softmax(logits.float(), dim=1)[0, 15:16]
            labels = logits.argmax(dim=1, keepdim=True)[0]
            mask = ((probability > 0.30) & ((labels == 15) | (probability > 0.58))).float()
            mask = F.interpolate(mask.unsqueeze(0), (height, width), mode="nearest")[0, 0]
        binary = (mask.cpu().numpy() * 255.0).astype(np.uint8)
        count, labels_np, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        cleaned = np.zeros_like(binary)
        minimum = max(96, int(width * height * 0.00045))
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum:
                cleaned[labels_np == label] = 255
        if not np.any(cleaned):
            return None
        radius = max(3, int(round(min(height, width) / 240.0))) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        return torch.from_numpy(cleaned).float().to(self.device)

    def process(self, rgb, pts_ms):
        torch, np = self.torch, self.np
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device, non_blocking=False).permute(2, 0, 1).float().div_(255.0)
        scene_cut = self._scene_cut(rgb, pts_ms)
        if scene_cut:
            self.processor = None
            torch.cuda.empty_cache()

        seeded = False
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            if self.processor is None:
                seed = self._human_seed(image)
                if seed is None:
                    return np.zeros(rgb.shape[:2], dtype=np.uint8), False, scene_cut
                self.processor = self.InferenceCore(self.model, cfg=self.model.cfg,
                                                     device=self.device)
                output = self.processor.step(image, seed, objects=[1])
                output = self.processor.step(image, first_frame_pred=True)
                for _ in range(self.warmup):
                    output = self.processor.step(image, first_frame_pred=True)
                seeded = True
            else:
                output = self.processor.step(image)
            alpha = self.processor.output_prob_to_mask(output)
        matte = torch.clamp(alpha.float() * 255.0, 0, 255).byte().cpu().numpy()
        return np.ascontiguousarray(matte), seeded, scene_cut


def main():
    args = _parse_args()
    _emit({"kind": "status", "state": "loading"})
    try:
        runtime = Runtime(args)
    except Exception as exc:
        _emit({"kind": "status", "state": "error", "error": repr(exc)})
        raise
    _emit({"kind": "status", "state": "ready"})

    generation = 1
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return
        message = json.loads(line.decode("utf-8"))
        kind = message.get("kind")
        if kind == "shutdown":
            return
        if kind == "reset":
            generation = int(message.get("generation", generation + 1))
            runtime.reset()
            _emit({"kind": "status", "state": "ready"})
            continue
        if kind != "frame":
            continue
        size = int(message.get("size", 0))
        payload = _read_exact(size)
        frame_generation = int(message["generation"])
        if frame_generation != generation:
            continue
        started = time.perf_counter()
        try:
            import cv2
            import numpy as np
            encoded = np.frombuffer(payload, dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("invalid JPEG frame")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            alpha, seeded, cut = runtime.process(rgb, float(message["pts_ms"]))
            elapsed = (time.perf_counter() - started) * 1000.0
            raw = alpha.tobytes(order="C")
            _emit({
                "kind": "matte", "generation": frame_generation,
                "pts_ms": float(message["pts_ms"]), "width": int(alpha.shape[1]),
                "height": int(alpha.shape[0]), "size": len(raw),
                "inference_ms": elapsed, "seeded": seeded, "scene_cut": cut,
            }, raw)
        except Exception as exc:
            runtime.reset()
            _emit({"kind": "error", "generation": frame_generation,
                   "error": repr(exc)})


if __name__ == "__main__":
    main()
