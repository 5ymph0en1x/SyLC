# -*- coding: utf-8 -*-
"""Asynchronous AI depth inference engine for SyLC on macOS (Apple Silicon / arm64).

Executes Depth Anything V3 ONNX models using ONNX Runtime with CoreMLExecutionProvider
(Apple Neural Engine / Metal GPU) and CPUExecutionProvider fallback.
Runs on a background worker thread so the Qt GUI and video playback are never blocked.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization constants (matching Depth Anything V3 & DepthEngine contract)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MacOSSynth3DEngine:
    """Asynchronous depth inference worker for macOS."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = False
        self._generation = 0
        self._worker_thread: Optional[threading.Thread] = None

        # Model & Session state
        self._model_path: Optional[str] = None
        self._side: int = 518
        self._session = None
        self._input_name: str = "image"
        self._output_name: str = "depth"
        self._input_rank: int = 4
        self._active_provider: str = "none"

        # Frame queue (single-slot buffer: newest frame always wins, drops oldest)
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_pts: float = 0.0

        # Output depth map (thread-safe exchange)
        self._latest_depth_bytes: Optional[bytes] = None
        self._latest_width: int = 0
        self._latest_height: int = 0
        self._latest_pts: float = 0.0
        self._has_new_depth: bool = False

        # Stabilization state (EMA + Tone range)
        self._ema_depth: Optional[np.ndarray] = None
        self._tone_lo: Optional[float] = None
        self._tone_hi: Optional[float] = None
        self._previous_input_time: Optional[float] = None

        # Diagnostics & Metrics
        self._infer_count: int = 0
        self._last_infer_ms: float = 0.0
        self._fps: float = 0.0
        self._last_fps_time: float = time.time()
        self._frames_since_fps_calc: int = 0
        self._error_message: Optional[str] = None
        self._depth_min: float = 0.0
        self._depth_max: float = 0.0

    def start(self, model_path: str, side: int = 518) -> None:
        """Start the background inference worker for the given model path and grid side."""
        # Never call stop() while holding _lock: stop() takes the same
        # non-reentrant lock. The previous code deadlocked the Qt thread on
        # every first activation, leaving mpv audio alive and video frozen.
        with self._lock:
            if self._running and self._model_path == model_path and self._side == side:
                return

        if not self.stop():
            with self._lock:
                self._error_message = (
                    "previous CoreML worker is still retiring; try again later")
            logger.warning("[2D3D macOS] Start refused while previous worker retires")
            return

        with self._lock:
            self._generation += 1
            generation = self._generation
            self._model_path = model_path
            self._side = side
            self._running = True
            self._has_new_depth = False
            self._ema_depth = None
            self._tone_lo = None
            self._tone_hi = None
            self._previous_input_time = None
            self._error_message = None

            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(generation,),
                name="MacOSSynth3DEngine",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("[2D3D macOS] Engine starting for %s (grid=%d)",
                        os.path.basename(model_path), side)

    def stop(self) -> bool:
        """Stop worker thread and release ONNX runtime session."""
        thread = None
        with self._lock:
            if not self._running and self._worker_thread is None:
                return True
            self._running = False
            self._generation += 1
            self._cond.notify_all()
            thread = self._worker_thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        retired = thread is None or not thread.is_alive()

        with self._lock:
            if retired and self._worker_thread is thread:
                self._worker_thread = None
            if retired:
                self._session = None
            self._pending_frame = None
            self._latest_depth_bytes = None
            self._has_new_depth = False

        if retired:
            logger.info("[2D3D macOS] Engine stopped")
        else:
            logger.warning("[2D3D macOS] Worker still retiring after stop timeout")
        return retired

    def is_running(self) -> bool:
        with self._lock:
            return self._running and self._session is not None

    def is_ready_for_frame(self) -> bool:
        """Returns True if the worker is running, has an initialized session, and the queue is empty."""
        with self._lock:
            return (self._running and self._session is not None
                    and self._pending_frame is None)

    def submit_frame(self, frame_rgb: np.ndarray, pts: float = 0.0) -> bool:
        """Submit an RGB frame (H, W, 3) uint8 to the worker without blocking.
        
        If a previous unconsumed frame is already waiting, it is replaced immediately.
        """
        with self._lock:
            if not self._running or self._session is None:
                return False
            self._pending_frame = frame_rgb
            self._pending_pts = pts
            self._cond.notify()
            return True

    def submit_qimage(self, qimage, pts: float = 0.0) -> bool:
        """Helper to extract an RGB numpy array from a QImage and submit it."""
        from PySide6.QtGui import QImage
        if qimage is None or qimage.isNull():
            return False
        w = qimage.width()
        h = qimage.height()
        if w <= 0 or h <= 0:
            return False
        # Convert to RGBA8888 for guaranteed contiguous 4-byte pixels without stride padding
        if qimage.format() != QImage.Format.Format_RGBA8888:
            qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        ptr = qimage.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        # Keep RGB channels
        rgb = np.ascontiguousarray(arr[:, :, :3])
        return self.submit_frame(rgb, time.monotonic())

    def get_latest_depth(self) -> Optional[Tuple[bytes, int, int]]:
        """Retrieve the latest stabilized depth map as packed RGBA bytes (width, height).
        
        Returns None if no new depth map has arrived since last query.
        """
        with self._lock:
            if not self._has_new_depth or self._latest_depth_bytes is None:
                return None
            self._has_new_depth = False
            return self._latest_depth_bytes, self._latest_width, self._latest_height

    def status(self) -> str:
        """Return diagnostic status string matching SyLC format."""
        with self._lock:
            if not self._running:
                return "Engine: off"
            if self._error_message:
                return f"state=error msg={self._error_message}"
            if self._session is None:
                return "state=init provider=AppleSilicon"
            model_name = os.path.basename(self._model_path or "unknown")
            provider = "CoreML" if "CoreML" in self._active_provider else ("CPU" if "CPU" in self._active_provider else self._active_provider)
            return (f"state=running provider={provider} "
                    f"model={model_name} grid={self._side}x{self._side} "
                    f"side={self._side} "
                    f"infer={self._last_infer_ms:.1f}ms age_ms={self._last_infer_ms:.0f} "
                    f"fps={self._fps:.1f} depth={self._depth_min:.3f}..{self._depth_max:.3f} "
                    f"clients=1")

    # =========================================================================
    # Worker Thread Loop
    # =========================================================================

    def _init_session(self, generation: int) -> bool:
        """Initialize the ONNX Runtime session in the background worker thread."""
        try:
            import onnxruntime as ort
        except ImportError:
            self._error_message = "onnxruntime is not installed"
            logger.error("[2D3D macOS] %s", self._error_message)
            return False

        model_path = self._model_path
        if not model_path or not os.path.isfile(model_path):
            self._error_message = f"Model not found: {model_path}"
            logger.error("[2D3D macOS] %s", self._error_message)
            return False

        t0 = time.time()
        logger.info("[2D3D macOS] Loading model: %s", model_path)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True

        available = ort.get_available_providers()
        logger.info("[2D3D macOS] Available providers: %s", available)

        session = None
        active_provider = "none"

        # Try CoreML first if available
        if 'CoreMLExecutionProvider' in available:
            try:
                session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'],
                )
                active_provider = session.get_providers()[0]
                logger.info("[2D3D macOS] Session created with provider: %s", active_provider)
            except Exception as e:
                logger.warning("[2D3D macOS] CoreML session failed (%s); falling back to CPU", e)
                session = None

        if session is None:
            try:
                session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=['CPUExecutionProvider'],
                )
                active_provider = session.get_providers()[0]
                logger.info("[2D3D macOS] Session created with CPU provider: %s", active_provider)
            except Exception as e:
                self._error_message = f"Failed to load session: {e}"
                logger.exception("[2D3D macOS] Session creation failed")
                return False

        # Probe model inputs & outputs
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            self._error_message = "Invalid model: missing inputs or outputs"
            return False

        input_name = inputs[0].name
        # Exporters may include auxiliary tensors before the actual prediction.
        # Prefer an explicitly named depth output instead of silently feeding an
        # unrelated first tensor to the stereo shader.
        depth_outputs = [out for out in outputs if "depth" in out.name.lower()]
        output_name = (depth_outputs[0] if depth_outputs else outputs[0]).name
        input_rank = len(inputs[0].shape)

        elapsed = (time.time() - t0) * 1000.0
        logger.info("[2D3D macOS] Model initialized in %.1f ms "
                    "(input=%s rank=%d output=%s available_outputs=%s)",
                    elapsed, input_name, input_rank, output_name,
                    [out.name for out in outputs])

        with self._lock:
            if not self._running or generation != self._generation:
                return False
            self._session = session
            self._input_name = input_name
            self._output_name = output_name
            self._input_rank = input_rank
            self._active_provider = active_provider

        return True

    def _worker_loop(self, generation: int) -> None:
        """Main execution loop for the background depth inference thread."""
        if not self._init_session(generation):
            with self._lock:
                if generation == self._generation:
                    self._running = False
            return

        while True:
            frame_rgb = None
            pts = 0.0

            with self._lock:
                while (self._running and generation == self._generation
                       and self._pending_frame is None):
                    self._cond.wait(timeout=0.1)

                if not self._running or generation != self._generation:
                    break

                frame_rgb = self._pending_frame
                pts = self._pending_pts
                self._pending_frame = None

            if frame_rgb is None:
                continue

            try:
                self._process_frame(frame_rgb, pts, generation)
            except Exception as e:
                logger.exception("[2D3D macOS] Inference failed on frame: %s", e)

    def _process_frame(self, frame_rgb: np.ndarray, pts: float,
                       generation: int) -> None:
        """Preprocess, infer, stabilize, and package a single video frame."""
        t0 = time.perf_counter()
        target_side = self._side

        # 1. Resize if frame resolution differs from requested model grid
        h, w = frame_rgb.shape[:2]
        if h != target_side or w != target_side:
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (target_side, target_side),
                                   interpolation=cv2.INTER_AREA)

        # 2. ImageNet normalization: (x / 255.0 - mean) / std
        f32 = frame_rgb.astype(np.float32) / 255.0
        norm = (f32 - IMAGENET_MEAN) / IMAGENET_STD

        # 3. Transpose from HWC to CHW and add batch dimension
        chw = np.transpose(norm, (2, 0, 1))
        if self._input_rank == 5:
            tensor = np.expand_dims(chw, axis=(0, 1))  # (1, 1, 3, H, W)
        else:
            tensor = np.expand_dims(chw, axis=0)        # (1, 3, H, W)

        # 4. Inference
        with self._lock:
            session = self._session
            input_name = self._input_name
            output_name = self._output_name

        if session is None:
            return

        raw_out = session.run([output_name], {input_name: tensor})[0]
        raw_depth = np.squeeze(raw_out)  # (H, W)

        # 5. Invert output: Depth Anything V3 metric/affine distance -> inverse depth
        inv_depth = np.where(
            (raw_depth > 1e-6) & np.isfinite(raw_depth),
            1.0 / raw_depth,
            0.0
        ).astype(np.float32)

        # 6. Robust percentile tone mapping (2nd & 98th percentile)
        p2, p98 = np.percentile(inv_depth, [2.0, 98.0])

        # Pixel-wise EMA assumes consecutive, aligned frames. At the observed
        # ~0.8 depth fps, the previous map can belong to another shot or object
        # position. Do not mix it into a new map or retain its tone scale.
        consecutive = (self._previous_input_time is not None
                       and 0.0 < pts - self._previous_input_time <= 0.25)
        self._previous_input_time = pts
        if not consecutive:
            self._ema_depth = None

        # Temporal tone smoothing only for closely spaced input frames.
        if not consecutive or self._tone_lo is None or self._tone_hi is None:
            self._tone_lo = float(p2)
            self._tone_hi = float(p98)
        else:
            self._tone_lo = 0.90 * self._tone_lo + 0.10 * float(p2)
            self._tone_hi = 0.90 * self._tone_hi + 0.10 * float(p98)

        tone_span = max(1e-6, self._tone_hi - self._tone_lo)
        norm_depth = np.clip((inv_depth - self._tone_lo) / tone_span, 0.0, 1.0)

        # Smooth Hermite S-curve for contrast depth separation
        smooth = norm_depth * norm_depth * (3.0 - 2.0 * norm_depth)
        norm_depth = norm_depth + 0.25 * (smooth - norm_depth)
        norm_depth = np.clip(0.5 + 0.92 * (norm_depth - 0.5), 0.0, 1.0)
        depth_min = float(np.min(norm_depth))
        depth_max = float(np.max(norm_depth))

        # 7. Temporal Exponential Moving Average (EMA) smoothing
        if self._ema_depth is None or self._ema_depth.shape != norm_depth.shape:
            self._ema_depth = norm_depth
        else:
            self._ema_depth = 0.65 * norm_depth + 0.35 * self._ema_depth

        # 8. Quantize to 16-bit uint16 and pack into 4-channel RGBA
        # R = high byte (bits 8-15), G = low byte (bits 0-7), B = 0, A = 255
        depth_u16 = (self._ema_depth * 65535.0).astype(np.uint16)
        dh, dw = depth_u16.shape
        hi = (depth_u16 >> 8).astype(np.uint8)
        lo = (depth_u16 & 0xFF).astype(np.uint8)

        packed = np.zeros((dh, dw, 4), dtype=np.uint8)
        packed[..., 0] = hi
        packed[..., 1] = lo
        packed[..., 3] = 255

        packed_bytes = packed.tobytes()
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        # Update metrics
        self._infer_count += 1
        self._last_infer_ms = infer_ms
        self._frames_since_fps_calc += 1
        now = time.time()
        dt = now - self._last_fps_time
        if dt >= 1.0:
            self._fps = self._frames_since_fps_calc / dt
            self._frames_since_fps_calc = 0
            self._last_fps_time = now

        # Publish depth map
        with self._lock:
            if not self._running or generation != self._generation:
                return
            self._latest_depth_bytes = packed_bytes
            self._depth_min = depth_min
            self._depth_max = depth_max
            self._latest_width = dw
            self._latest_height = dh
            self._latest_pts = pts
            self._has_new_depth = True
