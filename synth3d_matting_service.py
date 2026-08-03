"""Asynchronous MatAnyone 2 bridge for the native 2D -> 3D renderer.

The player stays on Python 3.14 while the PyTorch runtime lives in an isolated
``uv`` environment.  Frames cross the process boundary through a tiny binary
stdio protocol.  Submission is latest-only: decoder/presentation pacing is
never allowed to wait for matting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np


log = logging.getLogger("SyLC.MatAnyone2")


@dataclass(frozen=True)
class MatAnyone2Runtime:
    python: Path
    worker: Path
    checkpoint: Path
    seed_checkpoint: Path
    model_root: Path

    @classmethod
    def discover(cls, source_root: str | os.PathLike[str]) -> Optional["MatAnyone2Runtime"]:
        """Find a complete offline runtime; never downloads during playback."""
        source = Path(source_root).resolve()
        exe_root = Path(os.path.abspath(os.path.dirname(os.sys.argv[0])))
        roots = []
        for root in (exe_root, source):
            if root not in roots:
                roots.append(root)

        worker_env = os.environ.get("SYLC_MATANYONE2_WORKER", "").strip()
        worker_candidates = ([Path(worker_env)] if worker_env else []) + [
            root / "synth3d_matanyone2_worker.py" for root in roots
        ]
        python_env = os.environ.get("SYLC_MATANYONE2_PYTHON", "").strip()
        python_candidates = ([Path(python_env)] if python_env else []) + [
            root / "models_dev_nc" / "matanyone2_runtime" / ".venv" / "Scripts" / "python.exe"
            for root in roots
        ]
        model_env = os.environ.get("SYLC_MATANYONE2_MODELS", "").strip()
        model_roots = ([Path(model_env)] if model_env else []) + [
            root / "models_dev_nc" / "matanyone2" for root in roots
        ]

        worker = next((p.resolve() for p in worker_candidates if p.is_file()), None)
        python = next((p.resolve() for p in python_candidates if p.is_file()), None)
        for model_root in model_roots:
            checkpoint = model_root / "matanyone2.pth"
            seed = model_root / "lraspp_mobilenet_v3_large-d234d4ea.pth"
            if python and worker and checkpoint.is_file() and seed.is_file():
                return cls(python, worker, checkpoint.resolve(), seed.resolve(),
                           model_root.resolve())
        return None


@dataclass(frozen=True)
class MatteFrame:
    sequence: int
    generation: int
    pts_ms: float
    alpha: np.ndarray
    inference_ms: float
    seeded: bool
    scene_cut: bool

    def matches(self, pts_ms: float, max_pts_delta_ms: float) -> bool:
        if not math.isfinite(pts_ms) or not math.isfinite(self.pts_ms):
            return True
        # Never project a mask from the future onto an older frame after a seek.
        delta = pts_ms - self.pts_ms
        return -40.0 <= delta <= max_pts_delta_ms


def _to_u8_plane(plane: np.ndarray) -> np.ndarray:
    src = np.asarray(plane)
    if src.ndim != 2:
        raise ValueError("YUV planes must be two-dimensional")
    if src.dtype == np.uint8:
        return src
    if src.dtype == np.uint16:
        # The native R16 path stores 10-bit samples in the low bits.
        return np.clip((src.astype(np.float32) * (255.0 / 1023.0)) + 0.5,
                       0, 255).astype(np.uint8)
    raise TypeError(f"unsupported YUV plane dtype: {src.dtype}")


def yuv420_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray,
                  max_short_side: int = 720) -> np.ndarray:
    """Convert planar limited-range BT.709 YUV420 to a capped RGB frame."""
    y8, u8, v8 = (_to_u8_plane(p) for p in (y, u, v))
    height, width = y8.shape
    if height < 2 or width < 2:
        raise ValueError("invalid luma dimensions")

    scale = min(1.0, float(max_short_side) / float(min(height, width)))
    out_w = max(16, int(round(width * scale / 2.0)) * 2)
    out_h = max(16, int(round(height * scale / 2.0)) * 2)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    yy = cv2.resize(y8, (out_w, out_h), interpolation=interp).astype(np.float32)
    uu = cv2.resize(u8, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    vv = cv2.resize(v8, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    c = np.maximum(0.0, yy - 16.0) * 1.164383
    d = uu - 128.0
    e = vv - 128.0
    rgb = np.empty((out_h, out_w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(c + 1.792741 * e, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(c - 0.213249 * d - 0.532909 * e, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(c + 2.112402 * d, 0, 255).astype(np.uint8)
    return rgb


class MatAnyone2Service:
    """Latest-only asynchronous client for ``synth3d_matanyone2_worker.py``."""

    def __init__(self, runtime: MatAnyone2Runtime, *, target_fps: float = 5.0,
                 short_side: int = 720, max_pts_delta_ms: float = 650.0,
                 warmup: int = 3):
        self.runtime = runtime
        self.target_fps = max(1.0, min(15.0, float(target_fps)))
        self.short_side = max(256, min(1080, int(short_side)))
        self.max_pts_delta_ms = max(100.0, float(max_pts_delta_ms))
        self.warmup = max(0, min(10, int(warmup)))

        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._pending = None
        self._controls = []
        self._latest: Optional[MatteFrame] = None
        self._generation = 1
        self._sequence = 0
        self._last_submit_pts = -math.inf
        self._stopping = False
        self._state = "off"
        self._error = ""
        self._submitted = 0
        self._dropped = 0
        self._outputs = 0
        self._started_at = 0.0
        self._last_output_at = 0.0
        self._output_fps = 0.0
        self._last_inference_ms = 0.0

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            self._stopping = False
            self._state = "loading"
            self._error = ""
        command = [
            str(self.runtime.python), "-u", str(self.runtime.worker),
            "--checkpoint", str(self.runtime.checkpoint),
            "--seed-checkpoint", str(self.runtime.seed_checkpoint),
            "--short-side", str(self.short_side),
            "--warmup", str(self.warmup),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TORCH_HOME"] = str(self.runtime.model_root / "torch-cache")
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, env=env,
                startupinfo=startupinfo, creationflags=creationflags)
        except Exception as exc:
            with self._lock:
                self._state, self._error = "error", str(exc)
            log.exception("[MATANYONE2] worker start failed")
            return False
        with self._lock:
            self._proc = proc
            self._started_at = time.monotonic()
        threading.Thread(target=self._pump, name="MatAnyone2-submit", daemon=True).start()
        threading.Thread(target=self._read_stdout, name="MatAnyone2-result", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="MatAnyone2-log", daemon=True).start()
        threading.Thread(target=self._watch_process, name="MatAnyone2-watch", daemon=True).start()
        return True

    def submit_yuv(self, planes, pts_ms: float) -> bool:
        if not self.running or not planes or len(planes) != 3:
            return False
        pts = float(pts_ms)
        schedule_pts = pts if math.isfinite(pts) and pts >= 0 else time.monotonic() * 1000.0
        with self._wake:
            interval = 1000.0 / self.target_fps
            if schedule_pts >= self._last_submit_pts and schedule_pts - self._last_submit_pts < interval:
                return False
            if schedule_pts < self._last_submit_pts - 100.0:
                # Defensive local reset if a caller forgot the explicit seek hook.
                self._reset_locked("backward PTS")
            try:
                copied = tuple(np.ascontiguousarray(np.asarray(p)).copy() for p in planes)
            except Exception as exc:
                self._error = f"plane copy: {exc}"
                return False
            if self._pending is not None:
                self._dropped += 1
            self._pending = (self._generation, pts, copied)
            self._last_submit_pts = schedule_pts
            self._submitted += 1
            self._wake.notify()
        return True

    def latest_for_pts(self, pts_ms: float) -> Optional[MatteFrame]:
        with self._lock:
            item = self._latest
            generation = self._generation
        if item is None or item.generation != generation:
            return None
        return item if item.matches(float(pts_ms), self.max_pts_delta_ms) else None

    def reset(self, reason: str = "host reset") -> None:
        with self._wake:
            self._reset_locked(reason)
            self._wake.notify()

    def _reset_locked(self, reason: str) -> None:
        self._generation += 1
        self._latest = None
        self._pending = None
        self._last_submit_pts = -math.inf
        generation = self._generation
        log.info("[MATANYONE2] reset generation=%d reason=%s", generation, reason)
        # Control writes are performed by the pump thread too. A CUDA worker
        # that is temporarily not reading stdin must never block the GUI/seek
        # thread on a full pipe.
        self._controls.append(
            {"kind": "reset", "generation": generation, "reason": reason})

    def stop(self, timeout: float = 0.0) -> None:
        with self._wake:
            self._stopping = True
            proc = self._proc
            self._pending = None
            self._latest = None
            self._wake.notify_all()
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=max(0.05, timeout))
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        with self._lock:
            self._proc = None
            self._state = "off"

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "error": self._error,
                "submitted": self._submitted,
                "dropped": self._dropped,
                "outputs": self._outputs,
                "fps": self._output_fps,
                "inference_ms": self._last_inference_ms,
                "generation": self._generation,
            }

    def _pump(self) -> None:
        while True:
            with self._wake:
                while not self._controls and self._pending is None and not self._stopping:
                    self._wake.wait(timeout=0.5)
                    if self._proc is None or self._proc.poll() is not None:
                        return
                if self._stopping:
                    return
                if self._controls:
                    control = self._controls.pop(0)
                    generation = pts = planes = None
                elif self._state not in ("ready", "running", "degraded"):
                    self._wake.wait(timeout=0.1)
                    continue
                else:
                    control = None
                    generation, pts, planes = self._pending
                    self._pending = None
            try:
                if control is not None:
                    self._send_header(control)
                    continue
                rgb = yuv420_to_rgb(*planes, max_short_side=self.short_side)
                ok, encoded = cv2.imencode(
                    ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError("JPEG encoder refused frame")
                payload = encoded.tobytes()
                self._send_header({
                    "kind": "frame", "generation": generation, "pts_ms": pts,
                    "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                    "encoding": "jpeg", "size": len(payload),
                }, payload)
            except Exception as exc:
                with self._lock:
                    self._error = f"submit: {exc}"
                log.warning("[MATANYONE2] frame submission failed: %s", exc)

    def _send_header(self, header: dict, payload: bytes = b"", *,
                     tolerate_failure: bool = False) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            if tolerate_failure:
                return
            raise BrokenPipeError("MatAnyone 2 worker is not running")
        data = (json.dumps(header, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                proc.stdin.write(data)
                if payload:
                    proc.stdin.write(payload)
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            if not tolerate_failure:
                raise

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            block = stream.read(remaining)
            if not block:
                raise EOFError("worker closed in payload")
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while not self._stopping:
                line = proc.stdout.readline()
                if not line:
                    return
                message = json.loads(line.decode("utf-8"))
                kind = message.get("kind")
                if kind == "status":
                    with self._wake:
                        self._state = str(message.get("state", self._state))
                        self._error = str(message.get("error", ""))
                        self._wake.notify()
                    continue
                if kind == "error":
                    with self._lock:
                        self._state = "degraded"
                        self._error = str(message.get("error", "worker error"))
                    log.warning("[MATANYONE2] %s", self._error)
                    continue
                if kind != "matte":
                    continue
                width, height = int(message["width"]), int(message["height"])
                size = int(message["size"])
                if width <= 0 or height <= 0 or size != width * height:
                    raise ValueError("invalid matte payload dimensions")
                payload = self._read_exact(proc.stdout, size)
                alpha = np.frombuffer(payload, dtype=np.uint8).reshape(height, width).copy()
                generation = int(message["generation"])
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._sequence += 1
                    self._last_inference_ms = float(message.get("inference_ms", 0.0))
                    self._latest = MatteFrame(
                        self._sequence, generation, float(message["pts_ms"]), alpha,
                        self._last_inference_ms, bool(message.get("seeded", False)),
                        bool(message.get("scene_cut", False)))
                    self._outputs += 1
                    now = time.monotonic()
                    if self._last_output_at > 0.0:
                        instantaneous = 1.0 / max(1e-3, now - self._last_output_at)
                        self._output_fps = (instantaneous if self._output_fps <= 0.0
                                            else 0.20 * instantaneous
                                            + 0.80 * self._output_fps)
                    self._last_output_at = now
                    self._state = "running"
                    self._error = ""
        except Exception as exc:
            if not self._stopping:
                with self._lock:
                    self._state, self._error = "error", f"protocol: {exc}"
                log.exception("[MATANYONE2] result protocol failed")

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while not self._stopping:
            line = proc.stderr.readline()
            if not line:
                return
            log.info("[worker] %s", line.decode("utf-8", "replace").rstrip())

    def _watch_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        if not self._stopping and code != 0:
            with self._lock:
                self._state = "error"
                if not self._error:
                    self._error = f"worker exited with code {code}"
            log.error("[MATANYONE2] worker exited with code %d", code)
