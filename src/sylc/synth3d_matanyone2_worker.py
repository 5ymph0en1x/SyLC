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
        torch.set_float32_matmul_precision("high")

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
        self.warmup_remaining = 0
        self.scene_threshold = max(0.05, min(0.8, float(args.scene_threshold)))

    def reset(self):
        self.processor = None
        self.previous_probe = None
        self.previous_pts = None
        self.warmup_remaining = 0

    def _scene_cut_probe(self, gray, pts_ms):
        np = self.np
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

    def _scene_cut_rgb(self, rgb, pts_ms):
        cv2, np = self.cv2, self.np
        gray = cv2.cvtColor(
            cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return self._scene_cut_probe(gray, pts_ms)

    def _scene_cut_luma(self, luma, pts_ms, sample_peak=None):
        cv2, np = self.cv2, self.np
        peak = float(sample_peak if sample_peak is not None else
                     (1023 if luma.dtype == np.uint16 else 255))
        gray = cv2.resize(
            luma, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        return self._scene_cut_probe(gray / peak, pts_ms)

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

    def _run_model(self, image, scene_cut, upload_begin, upload_end):
        torch, np = self.torch, self.np
        if scene_cut:
            self.processor = None
            self.warmup_remaining = 0
        model_begin = torch.cuda.Event(enable_timing=True)
        model_end = torch.cuda.Event(enable_timing=True)

        seeded = False
        model_begin.record()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            if self.processor is None:
                seed = self._human_seed(image)
                if seed is None:
                    model_end.record()
                    model_end.synchronize()
                    return (np.zeros(tuple(image.shape[-2:]), dtype=np.uint8),
                            False, scene_cut,
                            upload_begin.elapsed_time(upload_end),
                            model_begin.elapsed_time(model_end), 0.0)
                self.processor = self.InferenceCore(self.model, cfg=self.model.cfg,
                                                     device=self.device)
                output = self.processor.step(image, seed, objects=[1])
                output = self.processor.step(image, first_frame_pred=True)
                # MatAnyone's official warmup consumes the following distinct
                # video frames. Repeating this same T0 image here wastes GPU
                # work and teaches temporal memory no new observation.
                self.warmup_remaining = self.warmup
                seeded = True
            elif self.warmup_remaining > 0:
                output = self.processor.step(image, first_frame_pred=True)
                self.warmup_remaining -= 1
            else:
                output = self.processor.step(image)
            alpha = self.processor.output_prob_to_mask(output)
        model_end.record()
        # The following CPU readback would synchronize the same stream anyway;
        # making the boundary explicit gives honest upload/model/readback
        # timings without introducing an extra frame of buffering.
        model_end.synchronize()
        upload_ms = upload_begin.elapsed_time(upload_end)
        model_ms = model_begin.elapsed_time(model_end)
        readback_started = time.perf_counter()
        matte = torch.clamp(alpha.float() * 255.0, 0, 255).byte().cpu().numpy()
        readback_ms = (time.perf_counter() - readback_started) * 1000.0
        return (np.ascontiguousarray(matte), seeded, scene_cut,
                upload_ms, model_ms, readback_ms)

    def process_rgb(self, rgb, pts_ms):
        torch, np = self.torch, self.np
        scene_cut = self._scene_cut_rgb(rgb, pts_ms)
        upload_begin = torch.cuda.Event(enable_timing=True)
        upload_end = torch.cuda.Event(enable_timing=True)
        upload_begin.record()
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device, non_blocking=False).permute(2, 0, 1).float().div_(255.0)
        upload_end.record()
        return self._run_model(image, scene_cut, upload_begin, upload_end)

    def process_yuv420(self, y, u, v, out_height, out_width,
                       sample_peak, pts_ms):
        """Upload planar decoded video and perform BT.709 conversion on CUDA."""
        torch, F, np = self.torch, self.F, self.np
        peak = float(sample_peak)
        if peak not in (255.0, 1023.0, 65472.0):
            raise ValueError(f"invalid YUV sample peak: {peak}")
        scene_cut = self._scene_cut_luma(y, pts_ms, peak)
        upload_begin = torch.cuda.Event(enable_timing=True)
        upload_end = torch.cuda.Event(enable_timing=True)
        upload_begin.record()
        def plane_tensor(plane):
            return torch.from_numpy(np.ascontiguousarray(plane)).to(
                self.device, non_blocking=False).float().div_(peak)[None, None]

        target = (int(out_height), int(out_width))
        yy = F.interpolate(plane_tensor(y), target, mode="bilinear",
                           align_corners=False, antialias=True)
        uu = F.interpolate(plane_tensor(u), target, mode="bilinear",
                           align_corners=False, antialias=True)
        vv = F.interpolate(plane_tensor(v), target, mode="bilinear",
                           align_corners=False, antialias=True)
        c = torch.relu(yy - 16.0 / 255.0) * 1.164383
        d = uu - 128.0 / 255.0
        e = vv - 128.0 / 255.0
        image = torch.cat((c + 1.792741 * e,
                           c - 0.213249 * d - 0.532909 * e,
                           c + 2.112402 * d), dim=1).clamp_(0.0, 1.0)[0]
        upload_end.record()
        return self._run_model(image, scene_cut, upload_begin, upload_end)


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
            decode_started = time.perf_counter()
            encoding = str(message.get("encoding", "jpeg")).lower()
            yuv_planes = None
            if encoding in ("yuv420p8", "yuv420p16"):
                source_width = int(message.get("source_width", 0))
                source_height = int(message.get("source_height", 0))
                width = int(message["width"])
                height = int(message["height"])
                dtype = np.uint16 if encoding == "yuv420p16" else np.uint8
                samples = source_width * source_height * 3 // 2
                if (source_width <= 0 or source_height <= 0 or
                        source_width % 2 or source_height % 2 or
                        width <= 0 or height <= 0 or
                        len(payload) != samples * np.dtype(dtype).itemsize):
                    raise ValueError("invalid planar YUV420 frame dimensions")
                packed = np.frombuffer(payload, dtype=dtype).copy()
                luma_count = source_width * source_height
                chroma_count = luma_count // 4
                y = packed[:luma_count].reshape(source_height, source_width)
                u = packed[luma_count:luma_count + chroma_count].reshape(
                    source_height // 2, source_width // 2)
                v = packed[luma_count + chroma_count:].reshape(
                    source_height // 2, source_width // 2)
                sample_peak = int(message.get(
                    "sample_peak", 1023 if dtype == np.uint16 else 255))
                yuv_planes = (y, u, v, height, width, sample_peak)
                rgb = None
            elif encoding == "rgb8":
                width = int(message["width"])
                height = int(message["height"])
                if width <= 0 or height <= 0 or len(payload) != width * height * 3:
                    raise ValueError("invalid raw RGB frame dimensions")
                # Own the bytes before returning to stdin.  This remains a
                # cheap memcpy and avoids JPEG's contour loss and codec work.
                rgb = np.frombuffer(payload, dtype=np.uint8).reshape(
                    height, width, 3).copy()
            elif encoding == "jpeg":
                encoded = np.frombuffer(payload, dtype=np.uint8)
                bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("invalid JPEG frame")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"unsupported frame encoding: {encoding}")
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if yuv_planes is not None:
                (alpha, seeded, cut, upload_ms, model_ms,
                 readback_ms) = runtime.process_yuv420(
                    *yuv_planes, float(message["pts_ms"]))
            else:
                (alpha, seeded, cut, upload_ms, model_ms,
                 readback_ms) = runtime.process_rgb(
                    rgb, float(message["pts_ms"]))
            elapsed = (time.perf_counter() - started) * 1000.0
            raw = alpha.tobytes(order="C")
            _emit({
                "kind": "matte", "generation": frame_generation,
                "pts_ms": float(message["pts_ms"]), "width": int(alpha.shape[1]),
                "height": int(alpha.shape[0]), "size": len(raw),
                "inference_ms": elapsed, "seeded": seeded, "scene_cut": cut,
                "decode_ms": decode_ms, "upload_ms": upload_ms,
                "model_ms": model_ms, "readback_ms": readback_ms,
            }, raw)
        except Exception as exc:
            runtime.reset()
            _emit({"kind": "error", "generation": frame_generation,
                   "error": repr(exc)})


if __name__ == "__main__":
    main()
