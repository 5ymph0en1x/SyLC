# -*- coding: utf-8 -*-
"""
FramePacking Display Widget using QRhiWidget for native D3D11 HDR support.

This replaces QOpenGLWidget with QRhiWidget which:
- Uses D3D11 natively on Windows (no OpenGL→DXGI copy)
- Supports HDR swapchain (scRGB/HDR10)
- Preserves HDR in fullscreen without stuttering in windowed mode

Requirements:
- Qt 6.6+ (QRhiWidget was added in Qt 6.6)
- PySide6 6.6+
- Shaders compiled with qsb tool (yuv_framepack.vert.qsb, yuv_framepack.frag.qsb)
"""

import os
import sys
import struct
import logging
import numpy as np
try:
    import velvet_probe  # read-only timing probe; no-op unless SYLC_VELVET_PROBE=1
except Exception:  # pragma: no cover - keep player runnable if probe is absent
    class _VelvetNoop:
        ENABLED = False
        @staticmethod
        def _noop(*a, **k):
            return None
        on_emit = on_present = on_drop = on_hold = on_bulkdrop = record = tick = incr = _noop
        now = __import__('time').perf_counter
    velvet_probe = _VelvetNoop()
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QByteArray, QSize, QFile, QIODevice
from PySide6.QtGui import QMatrix4x4, QColor
from PySide6.QtWidgets import QWidget

# Check Qt version for QRhiWidget support
from PySide6 import __version__ as PYSIDE_VERSION

logger = logging.getLogger(__name__)

# QRhiWidget requires Qt 6.6+
QT_VERSION_TUPLE = tuple(int(x) for x in PYSIDE_VERSION.split('.')[:2])
HAS_RHI_WIDGET = QT_VERSION_TUPLE >= (6, 6)

if HAS_RHI_WIDGET:
    try:
        from PySide6.QtWidgets import QRhiWidget
        from PySide6.QtGui import (
            QRhi, QRhiBuffer, QRhiTexture, QRhiSampler,
            QRhiShaderResourceBindings, QRhiGraphicsPipeline,
            QRhiShaderStage, QRhiVertexInputLayout, QRhiVertexInputBinding,
            QRhiVertexInputAttribute, QRhiCommandBuffer, QRhiResourceUpdateBatch,
            QRhiSwapChain, QShader, QRhiDepthStencilClearValue,
            QRhiTextureSubresourceUploadDescription, QRhiTextureUploadDescription,
            QRhiTextureUploadEntry, QRhiShaderResourceBinding
        )
        logger.info(f"[D3D11-HDR] QRhiWidget available (PySide6 {PYSIDE_VERSION})")
    except ImportError as e:
        HAS_RHI_WIDGET = False
        logger.warning(f"[D3D11-HDR] QRhiWidget import failed: {e}")
else:
    logger.warning(f"[D3D11-HDR] QRhiWidget requires PySide6 6.6+, got {PYSIDE_VERSION}")


# =============================================================================
# EMBEDDED SHADERS - For true onefile deployment (no external files needed)
# =============================================================================
import base64
import tempfile
import atexit

# Base64-encoded compiled shaders (generated from .qsb files)
_VERTEX_SHADER_B64 = """
AAALTXic1VZbaxNBFJ5cau3GtunFVK2tI31JsMRNaUWMrUgVLRRaWylCCCFNtnUg2Q27syFSAv4K
n/0Nvvrmb/DH+CJ6zs7Z7mST9EFBcMpkd75z+c5thzLGpphaSdhZJpnFemyXOfDnsibhiV+02IiV
ZR1Q9pgAYwFvNhmxq4wSsBdZl9XGMF5p/DcrQc80Pa/RsxmpJL+nkmuTqJph19k9QucgLjTOgAk+
X+0f7xc92Sxubpkon2GpS4JZqiqStOsCKxIgGdgLsHNoD1LEVsk/njN0DvyDvzmmqvGaTQTvjHww
DUsQltCwJGEp8pvTeBZAivL7IE2Rbg7esBAc9hLoT1JhkgGWhiooX4iX6TxFGMoN4g/lhhYT+psm
f1iPm/A7Q1y4HtB5lvTvwjlL+rNB3Cqn0EeWOOeJf44456m2KeJYJI4EcSwGuFphjDlNjr5vaTHh
+XZw+vgM+VbIN8bxCGJKsmiIEP8ByAQ8t6l+N6gGOyCdonMoWyLuN2CBPHcIYxq2TBjGdwhW00Ft
lO4yxYrxPQf9FervAuWInKuk/xM8PFZjn6ZZxEOTRd/Cl7Wu5XrCsXnJNA2jW3c/CPucd63GBu/W
pNXbdRy3WTbqUrri1JeWEo0VdBxPSHBXBl+OaHKkzReMC4PDihzybc0Fis5btUMyBRm42syHrta5
WTTXealoFspG3zBi2fRYtLRsNv7PbD5p2XyOstmCbBxfDmci7HgKIfJvY8dPYiMKPfnVk3UpGvys
5dTlpu60bOiiwWwGJeNwLTNPun5D8uPDvaOT2q7reF5tz+74ktKMGfAn/O3Ld7sHB0cvzLKucFmM
SKEEGY4mOPDlEINWz9EcAyUAneMTrSD9sEfQb1n7s0YplsFWnQW9OlPNGk5BjcRQ7ThU+9yqCXwP
w9CC0KTFwYA6UTS6UtSuIKEoRQWMiEtZO8FBKelIcTDxgdEa0h2onz5qqOla0nftGF1fH+s03d60
Et/WhN1o+U2LP21bst6qwX8BLXG6Y0S4J9rNh/hTfL9jGL6HF5Bdb1tep96weGAVzRXSmEg9dpwq
Fd+z3HzLadhmoVodO1KVSlhnVOrHKYQ95pOoVC7vvHyMQPsudK1SIWTAXlq9KAn1lg8ZuUD34ShU
q+EwRfq4t/lFX5E68X4JOzZhzlD3aexBc+QlpXXZUc3F7k4EEdCtFbu5ghVre9T/cbLfMIZuMA==
"""

_FRAGMENT_SHADER_B64 = """
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
"""

def _get_shader_dir() -> Path:
    """Get shader directory - always uses embedded shaders extracted to temp."""
    # Always extract embedded shaders to temp (works for both dev and compiled)
    shader_temp_dir = Path(tempfile.gettempdir()) / "sylc_shaders_v2"
    shader_temp_dir.mkdir(exist_ok=True)

    vert_path = shader_temp_dir / "yuv_framepack.vert.qsb"
    frag_path = shader_temp_dir / "yuv_framepack.frag.qsb"

    # Decode shaders
    vert_data = base64.b64decode(_VERTEX_SHADER_B64)
    frag_data = base64.b64decode(_FRAGMENT_SHADER_B64)

    # Always write (ensures latest version)
    vert_path.write_bytes(vert_data)
    frag_path.write_bytes(frag_data)

    logger.info(f"[D3D11-HDR] Shaders ready: {shader_temp_dir}")
    logger.info(f"[D3D11-HDR] Vertex: {len(vert_data)} bytes, Fragment: {len(frag_data)} bytes")

    return shader_temp_dir

SHADER_DIR = _get_shader_dir()
VERTEX_SHADER_PATH = SHADER_DIR / "yuv_framepack.vert.qsb"
FRAGMENT_SHADER_PATH = SHADER_DIR / "yuv_framepack.frag.qsb"


def load_shader(path: Path) -> 'QShader':
    """Load a compiled .qsb shader file."""
    if not path.exists():
        logger.error(f"[D3D11-HDR] Shader not found: {path}")
        return QShader()

    f = QFile(str(path))
    if not f.open(QIODevice.ReadOnly):
        logger.error(f"[D3D11-HDR] Cannot open shader: {path}")
        return QShader()

    data = f.readAll()
    f.close()

    shader = QShader.fromSerialized(data)
    if not shader.isValid():
        logger.error(f"[D3D11-HDR] Invalid shader: {path}")
        return QShader()

    logger.info(f"[D3D11-HDR] Loaded shader: {path.name}")
    return shader


class FramepackingDisplayWidgetD3D11(QRhiWidget if HAS_RHI_WIDGET else QWidget):
    """
    D3D11-native framepacking display widget with HDR support.

    This widget uses Qt's RHI (Rendering Hardware Interface) which:
    - Uses D3D11 directly on Windows
    - Supports HDR swapchain formats (scRGB, HDR10)
    - No OpenGL→DXGI copy overhead
    """

    frame_displayed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        if not HAS_RHI_WIDGET:
            logger.error("[D3D11-HDR] QRhiWidget not available!")
            return

        # Request D3D11 backend explicitly
        self.setApi(QRhiWidget.Api.Direct3D11)

        # CRITICAL: Request HDR color buffer format (scRGB - Windows native HDR)
        # This is what enables true HDR output! Without this, D3D11 uses SDR format.
        try:
            # RGBA16F = 16-bit float per channel = scRGB (Windows HDR native format)
            self.setColorBufferFormat(QRhiWidget.TextureFormat.RGBA16F)
            logger.info("[D3D11-HDR] HDR color buffer format set: RGBA16F (scRGB)")
        except Exception as e:
            logger.warning(f"[D3D11-HDR] Could not set HDR format: {e}")

        # NOTE: Fixed color buffer size will be set dynamically based on stereo mode
        # Framepack mode uses 1920x2205, other modes use widget's natural size
        # See set_stereo_mode() for the logic

        # Optimization: Disable background painting (we draw full screen)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        # V7b++ STUTTER FIX: Prevent widget from receiving focus/input events
        # This reduces DWM overhead when the parent window is focused
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # Pass mouse to parent

        # Connect frameSubmitted signal for proper frame pacing
        self.frameSubmitted.connect(self._on_frame_submitted)
        self._frame_in_flight = False  # Track if a frame is being rendered

        # Enable HDR format if supported
        self._hdr_enabled = False

        # Graphics resources
        self._rhi = None
        self._pipeline = None
        self._vertex_buffer = None
        self._index_buffer = None
        self._uniform_buffer = None
        self._sampler = None
        self._shader_bindings = None
        self._vertex_shader = None
        self._fragment_shader = None

        # YUV textures (6 total: Y, U, V for left and right)
        # + 1 subtitle texture
        self._textures = [None] * 7
        self._texture_size = (1920, 1080)
        # Actual per-eye SOURCE frame size of the YUV textures. Starts at 1080p (Blu-ray 3D)
        # but is re-sized to the real decoded frame on the fly so edge264 can display 2D
        # (or any-resolution) content without green padding / right-edge ghosting.
        self._yuv_src_size = (1920, 1080)

        # Stereo mode
        self._stereo_mode = 1  # 0=2D, 1=FramePack, 2=SBS, 3=TAB
        self.current_stereo_mode = 1  # Public attribute for compatibility
        self._subtitle_enabled = 0
        self._subtitle_rect = (0.0, 0.8, 1.0, 0.2)  # Default subtitle position

        # Pending frame data
        self._pending_frame = None
        self._has_video = False
        self.has_video = False  # Public accessor for compatibility
        self._initialized = False
        self._rendering_paused = False  # Pause during seek

        # Initial vertex upload needed
        self._needs_vertex_upload = True

        # Concurrency control
        self._is_updating = False  # Prevent re-entrant render calls

        # Subtitle support (placeholders for compatibility)
        self._subtitle_texture_data = None

        # SDR white level for HDR displays (1.0 = SDR, ~2.5 = HDR typical)
        self._sdr_white_level = self._query_sdr_white_level()

        # === PERFORMANCE OPTIMIZATIONS ===
        # Pre-allocated uniform buffer data (avoids struct.pack allocations per frame)
        self._uniform_data = bytearray(48)
        self._uniform_view = memoryview(self._uniform_data)

        # Pre-allocated subtitle texture buffer (avoids numpy allocation per subtitle)
        self._subtitle_buffer = None  # Lazy init on first subtitle

        # Frame drop counter for monitoring
        self._frames_dropped = 0
        self._frames_rendered = 0

        logger.info(f"[D3D11-HDR] FramepackingDisplayWidgetD3D11 created (fixed buffer: 1920x2205, SDR white level: {self._sdr_white_level:.2f})")

    def hideEvent(self, event):
        """Reset the in-flight flag when widget is hidden.

        Without this, if the widget is hidden between set_frame_yuv_views()
        (which sets _frame_in_flight=True and queues update()) and the
        actual render(), Qt never calls render() on a hidden widget so
        _on_frame_submitted() never fires, and _frame_in_flight stays True
        forever — every subsequent set_frame_yuv_views() then silently drops
        its frame even after the widget becomes visible again.
        """
        self._frame_in_flight = False
        super().hideEvent(event)

    def showEvent(self, event):
        """Ensure pending frame gets rendered when widget becomes visible."""
        self._frame_in_flight = False
        super().showEvent(event)
        if self._pending_frame and not self._rendering_paused:
            self.update()

    def _on_frame_submitted(self):
        """Called when GPU has finished processing a frame - enables next frame."""
        if velvet_probe.ENABLED:
            velvet_probe.on_present()
        self._frame_in_flight = False
        # If we have a pending frame, schedule next update immediately
        if self._pending_frame and not self._rendering_paused:
            self.update()

    def set_stereo_mode(self, mode_str: str):
        """Set stereo mode: '2d', 'framepack', 'sbs', 'tab'"""
        mode_map = {'2d': 0, 'framepack': 1, 'sbs': 2, 'tab': 3}
        self._stereo_mode = mode_map.get(mode_str.lower(), 1)
        self.current_stereo_mode = self._stereo_mode  # Sync public attribute

        # Set fixed color buffer size ONLY for framepack mode
        # Other modes should use the widget's natural size for proper scaling
        if self._stereo_mode == 1:  # Framepack
            self.setFixedColorBufferSize(QSize(1920, 2205))
        else:
            self.setFixedColorBufferSize(QSize())  # Use widget's natural size

        self.update()

    def _query_sdr_white_level(self) -> float:
        """
        Query Windows SDR white level setting for proper HDR brightness.
        Returns multiplier: 1.0 for SDR displays, ~2.0-3.5 for HDR displays.
        """
        import ctypes
        from ctypes import Structure, c_uint32, c_int32, byref, sizeof, POINTER

        try:
            # DisplayConfig API structures
            class DISPLAYCONFIG_DEVICE_INFO_HEADER(Structure):
                _fields_ = [
                    ("type", c_uint32),
                    ("size", c_uint32),
                    ("adapterId_LowPart", c_uint32),
                    ("adapterId_HighPart", c_int32),
                    ("id", c_uint32),
                ]

            class DISPLAYCONFIG_SDR_WHITE_LEVEL(Structure):
                _fields_ = [
                    ("header", DISPLAYCONFIG_DEVICE_INFO_HEADER),
                    ("SDRWhiteLevel", c_uint32),  # in 1000ths of a nit (e.g., 1000 = 1 nit)
                ]

            # Constants
            QDC_ONLY_ACTIVE_PATHS = 0x00000002
            DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL = 0x0B

            user32 = ctypes.windll.user32

            # Get buffer sizes
            num_paths = c_uint32(0)
            num_modes = c_uint32(0)
            result = user32.GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, byref(num_paths), byref(num_modes))
            if result != 0 or num_paths.value == 0:
                logger.debug("[D3D11-HDR] GetDisplayConfigBufferSizes failed, using default SDR white level")
                return 1.0

            # Allocate arrays
            class DISPLAYCONFIG_PATH_INFO(Structure):
                _fields_ = [("data", c_uint32 * 18)]  # Simplified, 72 bytes

            class DISPLAYCONFIG_MODE_INFO(Structure):
                _fields_ = [("data", c_uint32 * 16)]  # Simplified, 64 bytes

            paths = (DISPLAYCONFIG_PATH_INFO * num_paths.value)()
            modes = (DISPLAYCONFIG_MODE_INFO * num_modes.value)()

            # Query display config
            result = user32.QueryDisplayConfig(
                QDC_ONLY_ACTIVE_PATHS,
                byref(num_paths),
                paths,
                byref(num_modes),
                modes,
                None
            )
            if result != 0:
                logger.debug("[D3D11-HDR] QueryDisplayConfig failed, using default SDR white level")
                return 1.0

            # Query SDR white level for first active path
            if num_paths.value > 0:
                # Extract target adapterId and id from path
                path_data = paths[0].data
                # targetInfo starts at offset 32 bytes (8 uint32s) in PATH_INFO
                # adapterId is at offset 0 of targetInfo (8 bytes = 2 uint32s)
                # id is at offset 8 bytes of targetInfo (1 uint32)
                adapter_low = path_data[8]
                adapter_high = path_data[9]
                target_id = path_data[10]

                # Prepare SDR white level query
                sdr_info = DISPLAYCONFIG_SDR_WHITE_LEVEL()
                sdr_info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL
                sdr_info.header.size = sizeof(DISPLAYCONFIG_SDR_WHITE_LEVEL)
                sdr_info.header.adapterId_LowPart = adapter_low
                sdr_info.header.adapterId_HighPart = adapter_high
                sdr_info.header.id = target_id

                result = user32.DisplayConfigGetDeviceInfo(byref(sdr_info))
                if result == 0:
                    # SDRWhiteLevel is in 1000ths of a nit
                    # Reference white for SDR is 80 nits (SDRWhiteLevel = 80000)
                    # Windows default for HDR is usually 200-280 nits
                    sdr_nits = sdr_info.SDRWhiteLevel / 1000.0
                    # scRGB 1.0 = 80 nits, so multiplier = sdr_nits / 80
                    multiplier = sdr_nits / 80.0
                    logger.info(f"[D3D11-HDR] SDR white level: {sdr_nits:.0f} nits, multiplier: {multiplier:.2f}")
                    return multiplier

            logger.debug("[D3D11-HDR] Could not query SDR white level, using default")
            return 1.0

        except Exception as e:
            logger.debug(f"[D3D11-HDR] SDR white level query failed: {e}, using default")
            return 1.0

    def refresh_sdr_white_level(self):
        """Refresh SDR white level (call after HDR settings change)."""
        self._sdr_white_level = self._query_sdr_white_level()
        logger.info(f"[D3D11-HDR] SDR white level refreshed: {self._sdr_white_level:.2f}")

    def initialize(self, cb: 'QRhiCommandBuffer'):
        """
        Called when widget initializes or configuration changes.
        Set up HDR swapchain and graphics resources.
        """
        if self._initialized:
            return

        self._rhi = self.rhi()
        if not self._rhi:
            logger.error("[D3D11-HDR] Failed to get QRhi instance")
            return

        backend_name = self._rhi.backendName()
        logger.info(f"[D3D11-HDR] RHI Backend: {backend_name}")
        logger.info(f"[D3D11-HDR] Driver Info: {self._rhi.driverInfo()}")

        # Check if D3D11 backend for HDR
        if backend_name == "D3D11":
            logger.info("[D3D11-HDR] D3D11 backend detected - HDR should be supported")
            self._hdr_enabled = True
        else:
            logger.warning(f"[D3D11-HDR] Non-D3D11 backend: {backend_name}")

        # Create graphics pipeline and resources
        success = self._create_resources()
        if success:
            self._initialized = True
            logger.info("[D3D11-HDR] Initialization complete")
        else:
            logger.error("[D3D11-HDR] Initialization failed")

    def _create_resources(self) -> bool:
        """Create GPU resources: buffers, textures, pipeline."""
        try:
            # Load shaders
            self._vertex_shader = load_shader(VERTEX_SHADER_PATH)
            self._fragment_shader = load_shader(FRAGMENT_SHADER_PATH)

            if not self._vertex_shader.isValid() or not self._fragment_shader.isValid():
                logger.error("[D3D11-HDR] Failed to load shaders")
                return False

            # Vertex buffer for fullscreen quad (triangle strip)
            # Format: position (x, y), texcoord (u, v)
            # Note: texcoord Y is NOT inverted here because shader already does y_flipped
            vertices = np.array([
                # position    texcoord
                -1.0, -1.0,   0.0, 0.0,  # bottom-left
                 1.0, -1.0,   1.0, 0.0,  # bottom-right
                -1.0,  1.0,   0.0, 1.0,  # top-left
                 1.0,  1.0,   1.0, 1.0,  # top-right
            ], dtype=np.float32)

            self._vertex_buffer = self._rhi.newBuffer(
                QRhiBuffer.Type.Immutable,
                QRhiBuffer.UsageFlag.VertexBuffer,
                vertices.nbytes
            )
            if not self._vertex_buffer.create():
                logger.error("[D3D11-HDR] Failed to create vertex buffer")
                return False

            # Store vertices for upload
            self._vertex_data = vertices.tobytes()

            # Uniform buffer
            # Layout (matching shader with std140):
            # - int stereo_mode (4 bytes)
            # - int subtitle_enabled (4 bytes)
            # - padding (8 bytes to align vec4 to 16)
            # - vec4 subtitle_rect (16 bytes)
            # - float sdr_white_level (4 bytes)
            # Total: 36 bytes, padded to 48 for alignment
            self._uniform_buffer = self._rhi.newBuffer(
                QRhiBuffer.Type.Dynamic,
                QRhiBuffer.UsageFlag.UniformBuffer,
                48
            )
            if not self._uniform_buffer.create():
                logger.error("[D3D11-HDR] Failed to create uniform buffer")
                return False

            # Create YUV textures
            if not self._create_yuv_textures():
                return False

            # Sampler for texture filtering
            self._sampler = self._rhi.newSampler(
                QRhiSampler.Filter.Linear,
                QRhiSampler.Filter.Linear,
                QRhiSampler.Filter.None_,
                QRhiSampler.AddressMode.ClampToEdge,
                QRhiSampler.AddressMode.ClampToEdge
            )
            if not self._sampler.create():
                logger.error("[D3D11-HDR] Failed to create sampler")
                return False

            # Create shader resource bindings
            if not self._create_shader_bindings():
                return False

            # Create graphics pipeline
            if not self._create_pipeline():
                return False

            return True

        except Exception as e:
            logger.error(f"[D3D11-HDR] Resource creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_yuv_textures(self) -> bool:
        """Create YUV textures for left and right eyes, sized to the current SOURCE frame.

        Blu-ray 3D is always 1080p/eye, but 2D-via-edge264 (and any non-1080p stream) needs
        textures matching the real frame; _resize_yuv_textures() updates self._yuv_src_size and
        re-runs this. NOTE: this is the SOURCE size, not self._texture_size (the 1920x2205
        framepack render target).
        """
        w, h = self._yuv_src_size

        # Pre-initialize buffer for first-frame clear (kept ready so the very first
        # render doesn't show uninitialized GPU memory as magenta noise).
        import numpy as _np
        self._init_y_data = _np.full((h, w), 16, dtype=_np.uint8)        # limited-range black
        self._init_uv_data = _np.full((h // 2, w // 2), 128, dtype=_np.uint8)  # neutral chroma
        self._needs_init_clear = True

        # Y textures (full resolution)
        for i in [0, 3]:  # texY_L, texY_R
            tex = self._rhi.newTexture(QRhiTexture.Format.R8, QSize(w, h))
            if not tex.create():
                logger.error(f"[D3D11-HDR] Failed to create Y texture {i}")
                return False
            self._textures[i] = tex

        # U and V textures (half resolution for 4:2:0)
        for i in [1, 2, 4, 5]:  # texU_L, texV_L, texU_R, texV_R
            tex = self._rhi.newTexture(QRhiTexture.Format.R8, QSize(w // 2, h // 2))
            if not tex.create():
                logger.error(f"[D3D11-HDR] Failed to create UV texture {i}")
                return False
            self._textures[i] = tex

        # Subtitle texture (RGBA, same resolution as video)
        self._textures[6] = self._rhi.newTexture(QRhiTexture.Format.RGBA8, QSize(w, h))
        if not self._textures[6].create():
            logger.error("[D3D11-HDR] Failed to create subtitle texture")
            return False

        logger.info(f"[D3D11-HDR] Created YUV textures: {w}x{h}")
        return True

    def _resize_yuv_textures(self, w, h) -> bool:
        """Recreate the YUV (+subtitle) textures at a new SOURCE size, then rebind.

        Called from the render thread (inside _upload_frame_textures, before beginPass — so
        outside any render pass, which is safe for QRhi resource (de)allocation) when the
        decoded frame size differs from the current textures (e.g. a non-1080p 2D H.264 stream).
        Without this the frame is padded into 1080p textures -> green band + right-edge ghost.
        """
        w = max(2, int(w))
        h = max(2, int(h))
        if (w, h) == self._yuv_src_size and all(t is not None for t in self._textures):
            return True
        try:
            for i in range(len(self._textures)):
                t = self._textures[i]
                if t is not None:
                    try:
                        t.destroy()
                    except Exception:
                        pass
                    self._textures[i] = None
            self._yuv_src_size = (w, h)
            if not self._create_yuv_textures():
                return False
            if not self._create_shader_bindings():
                return False
            logger.info(f"[D3D11-HDR] YUV textures resized to source {w}x{h}")
            return True
        except Exception as e:
            logger.error(f"[D3D11-HDR] YUV resize to {w}x{h} failed: {e}")
            return False

    def _create_shader_bindings(self) -> bool:
        """Create shader resource bindings."""
        bindings = [
            # Uniform buffer at binding 0
            QRhiShaderResourceBinding.uniformBuffer(
                0,
                QRhiShaderResourceBinding.StageFlag.VertexStage |
                QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._uniform_buffer
            ),
            # YUV textures at bindings 1-6
            QRhiShaderResourceBinding.sampledTexture(
                1, QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._textures[0], self._sampler
            ),
            QRhiShaderResourceBinding.sampledTexture(
                2, QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._textures[1], self._sampler
            ),
            QRhiShaderResourceBinding.sampledTexture(
                3, QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._textures[2], self._sampler
            ),
            QRhiShaderResourceBinding.sampledTexture(
                4, QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._textures[3], self._sampler
            ),
            QRhiShaderResourceBinding.sampledTexture(
                5, QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._textures[4], self._sampler
            ),
            QRhiShaderResourceBinding.sampledTexture(
                6, QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._textures[5], self._sampler
            ),
            # Subtitle texture at binding 7
            QRhiShaderResourceBinding.sampledTexture(
                7, QRhiShaderResourceBinding.StageFlag.FragmentStage,
                self._textures[6], self._sampler
            ),
        ]

        self._shader_bindings = self._rhi.newShaderResourceBindings()
        self._shader_bindings.setBindings(bindings)

        if not self._shader_bindings.create():
            logger.error("[D3D11-HDR] Failed to create shader bindings")
            return False

        return True

    def _create_pipeline(self) -> bool:
        """Create graphics pipeline."""
        self._pipeline = self._rhi.newGraphicsPipeline()

        # Shader stages
        self._pipeline.setShaderStages([
            QRhiShaderStage(QRhiShaderStage.Type.Vertex, self._vertex_shader),
            QRhiShaderStage(QRhiShaderStage.Type.Fragment, self._fragment_shader),
        ])

        # Vertex input layout
        input_layout = QRhiVertexInputLayout()
        input_layout.setBindings([
            QRhiVertexInputBinding(4 * 4)  # stride: 4 floats * 4 bytes
        ])
        input_layout.setAttributes([
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute.Format.Float2, 0),     # position
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute.Format.Float2, 8),    # texCoord
        ])
        self._pipeline.setVertexInputLayout(input_layout)

        # Triangle strip topology
        self._pipeline.setTopology(QRhiGraphicsPipeline.Topology.TriangleStrip)

        # Shader resource bindings
        self._pipeline.setShaderResourceBindings(self._shader_bindings)

        # Render pass descriptor from the render target
        rt = self.renderTarget()
        if rt:
            self._pipeline.setRenderPassDescriptor(rt.renderPassDescriptor())

        if not self._pipeline.create():
            logger.error("[D3D11-HDR] Failed to create graphics pipeline")
            return False

        logger.info("[D3D11-HDR] Graphics pipeline created")
        return True

    def set_frame_yuv_views(self, y_l_or_tuple, u_l_or_right=None, v_l=None, y_r=None, u_r=None, v_r=None):
        """
        Upload YUV frame data for both eyes.

        Supports multiple calling conventions:
        1. set_frame_yuv_views((y_l, u_l, v_l), (y_r, u_r, v_r)) - tuple pairs
        2. set_frame_yuv_views(y_l, u_l, v_l, y_r, u_r, v_r) - 6 separate arrays

        Args:
            y_l_or_tuple: Y plane for left eye, OR tuple (Y, U, V) for left
            u_l_or_right: U plane for left eye, OR tuple (Y, U, V) for right
            v_l, y_r, u_r, v_r: Remaining planes if using 6-argument form
        """
        # Don't accept frames during seek
        if self._rendering_paused:
            return

        # Detect calling convention
        if isinstance(y_l_or_tuple, tuple):
            # Called as (left_tuple, right_tuple)
            y_l, u_l, v_l_local = y_l_or_tuple
            if isinstance(u_l_or_right, tuple):
                y_r_local, u_r_local, v_r_local = u_l_or_right
            else:
                y_r_local, u_r_local, v_r_local = None, None, None
        else:
            # Called as 6 separate arrays
            y_l = y_l_or_tuple
            u_l = u_l_or_right
            v_l_local = v_l
            y_r_local = y_r
            u_r_local = u_r
            v_r_local = v_r

        # Track frame drops: if a previous _pending_frame is being overwritten
        # before render() consumed it, that frame is lost. Happens when decoder
        # is faster than GPU + display sync — visible as a "smoother but with
        # holes" playback. Use this counter in the monitoring overlay if needed.
        if self._pending_frame is not None:
            self._frames_dropped += 1
            if velvet_probe.ENABLED:
                velvet_probe.on_drop()

        # PERF OPTIM (modified A): in 2D mode the shader only samples *_L,
        # so passing right planes uploads ~3 MB/frame to the GPU for nothing.
        # Drop them at the source — the decoder reuses the same numpy arrays
        # for the other render target, so no decode work is wasted.
        if self._stereo_mode == 0:
            y_r_local = u_r_local = v_r_local = None

        self._pending_frame = (y_l, u_l, v_l_local, y_r_local, u_r_local, v_r_local)
        self._has_video = True
        self.has_video = True  # Sync public attribute

        # Frame pacing using frameSubmitted signal:
        # Only request new update if no frame is currently in flight
        # This prevents frame queue buildup which causes stuttering
        if not self._frame_in_flight and not self._is_updating:
            self._frame_in_flight = True
            self.update()

    def render(self, cb: 'QRhiCommandBuffer'):
        """
        Render the current frame.
        Called by Qt when widget needs to be redrawn.
        """
        self._is_updating = True  # Lock to prevent concurrent frame submits

        if not self._rhi:
            self._is_updating = False
            return

        rt = self.renderTarget()
        if not rt:
            self._is_updating = False
            return

        # Get/create resource update batch
        batch = self._rhi.nextResourceUpdateBatch()

        # Upload vertex data on first render
        if self._needs_vertex_upload and self._vertex_buffer:
            batch.uploadStaticBuffer(self._vertex_buffer, self._vertex_data)
            self._needs_vertex_upload = False

        # Upload uniform data using pre-allocated buffer (PERF: avoids allocations)
        if self._uniform_buffer:
            # std140 layout: int(4) + int(4) + padding(8) + vec4(16) + float(4) = 36 bytes
            # Write directly into pre-allocated buffer using struct.pack_into
            struct.pack_into('<ii', self._uniform_data, 0,
                            self._stereo_mode, self._subtitle_enabled)
            # Bytes 8-15 are padding (already zero from initialization)
            struct.pack_into('<4f', self._uniform_data, 16, *self._subtitle_rect)
            struct.pack_into('<f', self._uniform_data, 32, self._sdr_white_level)
            # Bytes 36-47 are padding (already zero)
            # PERF: pass the bytearray directly via buffer protocol — no per-frame bytes() copy.
            batch.updateDynamicBuffer(self._uniform_buffer, 0, 48, self._uniform_data)

        # Upload pending frame data
        if self._pending_frame:
            self._upload_frame_textures(batch)
            self._pending_frame = None

        # Upload pending subtitle texture
        if self._subtitle_texture_data:
            self._upload_subtitle_texture(batch)

        # Clear color (black)
        clear_color = QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)

        # Begin render pass
        cb.beginPass(rt, clear_color, QRhiDepthStencilClearValue(1.0, 0), batch)

        # Draw if pipeline and resources are ready
        if self._pipeline and self._shader_bindings and self._vertex_buffer and self._has_video:
            cb.setGraphicsPipeline(self._pipeline)
            cb.setShaderResources(self._shader_bindings)

            # Set viewport WITH aspect-ratio preservation.
            #   stereo_mode 0 (2D), 2 (SBS), 3 (TAB) → display aspect = 1920:1080 (16:9)
            #   stereo_mode 1 (FramePack)            → display aspect = 1920:2205
            # When the render target is wider than target, we pillarbox (black
            # bars on sides); when taller, we letterbox (black bars top/bottom).
            # The render target was cleared to black just above, so the bars are
            # naturally black without extra work.
            output_size = rt.pixelSize()
            ow = output_size.width()
            oh = output_size.height()
            if self._stereo_mode == 1:
                target_aspect = 1920.0 / 2205.0
            elif self._stereo_mode == 0:
                # 2D: use the ACTUAL source frame aspect (any resolution), so non-16:9 2D
                # content (e.g. 2.39:1) is letterboxed correctly instead of stretched.
                sw, sh = self._yuv_src_size
                target_aspect = (sw / sh) if sh else (1920.0 / 1080.0)
            else:
                target_aspect = 1920.0 / 1080.0
            if oh > 0 and ow > 0:
                out_aspect = ow / oh
                if out_aspect > target_aspect:
                    # Wider than target → pillarbox
                    vh = oh
                    vw = int(round(oh * target_aspect))
                    vx = (ow - vw) // 2
                    vy = 0
                else:
                    # Taller than target → letterbox
                    vw = ow
                    vh = int(round(ow / target_aspect))
                    vx = 0
                    vy = (oh - vh) // 2
                cb.setViewport(QRhiViewport(vx, vy, vw, vh))
            else:
                cb.setViewport(QRhiViewport(0, 0, ow, oh))

            # Bind vertex buffer and draw
            vb_binding = [(self._vertex_buffer, 0)]
            cb.setVertexInput(0, vb_binding)
            cb.draw(4)  # 4 vertices for triangle strip quad

        cb.endPass()

        self._is_updating = False  # Unlock for next frame
        self.frame_displayed.emit()

    def _upload_frame_textures(self, batch: 'QRhiResourceUpdateBatch'):
        """Upload YUV frame data to GPU textures.

        Differential strategy after exhaustive testing:
        - Y plane (1920 wide): tight packing works fine. No padding.
        - U/V planes (960 wide): D3D11 internal row pitch is aligned (likely
          1024 bytes), so tight 960-byte rows mismatch the destination's
          memory layout. We pad each U/V row to 1024 bytes AND declare the
          padded stride via setDataStride(1024).

        Why differential? Tests showed Y tight = correct, U/V tight = broken
        colored noise inside the visible image silhouette (err5.jpg). Conversely
        when both planes were padded, both showed banding (err4.jpg). Only the
        narrow U/V planes need padding to satisfy D3D11's pitch alignment.
        """
        if not self._pending_frame:
            return

        y_l, u_l, v_l, y_r, u_r, v_r = self._pending_frame
        planes = [y_l, u_l, v_l, y_r, u_r, v_r]

        import numpy as _np

        # Dimension-aware: match the YUV textures to the ACTUAL decoded frame size, so any
        # resolution (e.g. a non-1080p 2D H.264 stream via edge264) displays without green
        # padding or right-edge ghosting. The Y plane is delivered at the true frame size
        # (the decoder crops the stride), so its shape is the source size. Safe here: we are
        # on the render thread, before beginPass.
        if y_l is not None and getattr(y_l, 'ndim', 0) >= 2:
            src_h, src_w = int(y_l.shape[0]), int(y_l.shape[1])
            if (src_w, src_h) != self._yuv_src_size:
                self._resize_yuv_textures(src_w, src_h)

        # D3D11 typical row pitch alignment. R8 textures of narrow width need
        # explicit padding to this multiple. Wider textures (>= ALIGN) already
        # satisfy alignment naturally and should NOT be padded.
        ALIGN_BYTES = 256

        for i, plane in enumerate(planes):
            if plane is not None and self._textures[i]:
                tex = self._textures[i]
                tex_size = tex.pixelSize()
                tex_w = tex_size.width()
                tex_h = tex_size.height()

                arr = plane if (hasattr(plane, 'flags') and getattr(plane.flags, 'c_contiguous', False)) else _np.ascontiguousarray(plane)

                # Make array shape match texture exactly.
                src_w = int(arr.shape[1]) if arr.ndim >= 2 else int(arr.size)
                src_h = int(arr.shape[0]) if arr.ndim >= 2 else 1

                if src_w != tex_w:
                    if src_w > tex_w:
                        arr = _np.ascontiguousarray(arr[:, :tex_w])
                    else:
                        # Pad by REPEATING the last column. Zero-padding would
                        # produce strong chroma artifacts at the right edge when
                        # the GPU samples slightly past the valid region.
                        padded = _np.empty((src_h, tex_w), dtype=arr.dtype)
                        padded[:, :src_w] = arr
                        # Broadcast last column across the padding region
                        padded[:, src_w:] = arr[:, src_w - 1:src_w]
                        arr = padded
                if src_h != tex_h:
                    if src_h > tex_h:
                        arr = _np.ascontiguousarray(arr[:tex_h, :])
                    else:
                        padded = _np.zeros((tex_h, arr.shape[1]), dtype=arr.dtype)
                        padded[:src_h, :] = arr
                        arr = padded

                # Tight upload + explicit dataStride = tex_w (row size in bytes).
                # Qt RHI then knows source stride and copies row-by-row to
                # D3D11's pitch-aligned staging buffer. Without explicit stride,
                # Qt may misinterpret narrow R8 textures (U/V at 960) and shift
                # rows progressively, producing horizontal band artifacts.
                data = arr.tobytes()
                sub_desc = QRhiTextureSubresourceUploadDescription(data)
                sub_desc.setDataStride(tex_w)

                upload_desc = QRhiTextureUploadDescription(
                    QRhiTextureUploadEntry(0, 0, sub_desc)
                )

                batch.uploadTexture(tex, upload_desc)

    def _upload_subtitle_texture(self, batch: 'QRhiResourceUpdateBatch'):
        """Upload subtitle RGBA texture to GPU.

        Strategy: Place the subtitle at its correct pixel position in a full-resolution
        transparent texture. Then set subtitle_rect to (0, 0, 1, 1) so the shader samples
        the entire texture directly. The alpha channel handles transparency.

        PERF: Uses pre-allocated buffer to avoid numpy allocation per subtitle.
        """
        if not self._subtitle_texture_data:
            return

        rgba_array, norm_x, norm_y, norm_w, norm_h = self._subtitle_texture_data

        # Subtitle texture is at index 6
        if self._textures[6] is None:
            return

        tex = self._textures[6]
        tex_size = tex.pixelSize()
        tex_w, tex_h = tex_size.width(), tex_size.height()

        sub_h, sub_w = rgba_array.shape[:2]

        # PERF: Reuse pre-allocated buffer instead of creating new array each time
        if self._subtitle_buffer is None or self._subtitle_buffer.shape != (tex_h, tex_w, 4):
            self._subtitle_buffer = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
        else:
            # Clear only - faster than reallocating
            self._subtitle_buffer.fill(0)

        # Calculate pixel position from normalized video coordinates
        px = int(norm_x * tex_w)
        py = int(norm_y * tex_h)

        # Clamp to valid range
        end_x = min(px + sub_w, tex_w)
        end_y = min(py + sub_h, tex_h)
        start_x = max(px, 0)
        start_y = max(py, 0)

        # Calculate source region
        src_start_x = start_x - px
        src_start_y = start_y - py
        src_end_x = src_start_x + (end_x - start_x)
        src_end_y = src_start_y + (end_y - start_y)

        if end_x > start_x and end_y > start_y:
            # Copy subtitle to the correct position in the buffer
            self._subtitle_buffer[start_y:end_y, start_x:end_x] = rgba_array[src_start_y:src_end_y, src_start_x:src_end_x]

        # Upload the texture (buffer is already contiguous)
        data = self._subtitle_buffer.tobytes()

        sub_desc = QRhiTextureSubresourceUploadDescription(data)
        sub_desc.setDataStride(tex_w * 4)  # 4 bytes per pixel (RGBA)

        upload_desc = QRhiTextureUploadDescription(
            QRhiTextureUploadEntry(0, 0, sub_desc)
        )

        batch.uploadTexture(tex, upload_desc)

        # Update subtitle_rect to cover entire texture - the subtitle is already positioned
        self._subtitle_rect = (0.0, 0.0, 1.0, 1.0)

        # Clear pending subtitle data after upload
        self._subtitle_texture_data = None

    def releaseResources(self):
        """Clean up GPU resources."""
        self._pipeline = None
        self._vertex_buffer = None
        self._index_buffer = None
        self._uniform_buffer = None
        self._sampler = None
        self._shader_bindings = None
        self._textures = [None] * 7
        self._initialized = False
        logger.info("[D3D11-HDR] Resources released")

    def set_subtitle(self, rgba_array, x, y, w, h, video_width=1920, video_height=1080):
        """Set PGS subtitle for overlay.

        Args:
            rgba_array: numpy array (H, W, 4) RGBA uint8
            x, y: position in video coordinates (pixels)
            w, h: subtitle dimensions (pixels)
            video_width, video_height: video frame dimensions for normalization
        """
        if rgba_array is None or not isinstance(rgba_array, np.ndarray):
            self.clear_subtitle()
            return

        if rgba_array.dtype != np.uint8 or len(rgba_array.shape) != 3 or rgba_array.shape[2] != 4:
            logger.warning(f"[D3D11-SUBTITLE] Invalid array: dtype={rgba_array.dtype}, shape={rgba_array.shape}")
            self.clear_subtitle()
            return

        # Normalize coordinates to 0-1 range
        norm_x = x / video_width
        norm_y = y / video_height
        norm_w = w / video_width
        norm_h = h / video_height

        # Store pending subtitle data for upload
        # The upload function will place the subtitle at the correct position in a full-size texture
        self._subtitle_texture_data = (np.ascontiguousarray(rgba_array), norm_x, norm_y, norm_w, norm_h)
        self._subtitle_enabled = 1
        # Use full texture rect since subtitle is positioned in the texture itself
        self._subtitle_rect = (0.0, 0.0, 1.0, 1.0)
        self.update()

    def clear_subtitle(self):
        """Hide subtitle overlay."""
        # Only update if subtitle was actually enabled (avoid unnecessary repaints)
        if self._subtitle_enabled == 1:
            self._subtitle_enabled = 0
            self._subtitle_texture_data = None
            self.update()

    def clear_textures(self):
        """Clear all textures to black (used after seek)."""
        # Just mark that we need new frames
        self._has_video = False
        self.has_video = False  # Sync public attribute
        self._pending_frame = None
        self.update()

    def pause_rendering(self):
        """Pause rendering during seek to prevent access violations."""
        self._rendering_paused = True
        self._pending_frame = None
        self._has_video = False

    def resume_rendering(self):
        """Resume rendering after seek completes."""
        self._rendering_paused = False


# Import QRhiViewport if available
if HAS_RHI_WIDGET:
    try:
        from PySide6.QtGui import QRhiViewport
    except ImportError:
        # Fallback: create a simple class
        class QRhiViewport:
            def __init__(self, x, y, w, h, minDepth=0.0, maxDepth=1.0):
                self.x = x
                self.y = y
                self.w = w
                self.h = h
                self.minDepth = minDepth
                self.maxDepth = maxDepth


# Fallback message if QRhiWidget not available
if not HAS_RHI_WIDGET:
    logger.error(f"""
    ============================================================
    QRhiWidget NOT AVAILABLE

    To use D3D11 native HDR rendering, you need:
    - PySide6 6.6 or later
    - Qt 6.6 or later

    Current PySide6 version: {PYSIDE_VERSION}

    Install with: pip install PySide6>=6.6
    ============================================================
    """)


def check_hdr_support():
    """Check if system supports HDR output via D3D11."""
    if not HAS_RHI_WIDGET:
        return False, "QRhiWidget not available (need PySide6 6.6+)"

    # Check if shaders are compiled
    if not VERTEX_SHADER_PATH.exists() or not FRAGMENT_SHADER_PATH.exists():
        return False, f"Compiled shaders not found in {SHADER_DIR}"

    return True, "D3D11 HDR support available"


def check_display_hdr_capability():
    """
    Check if the primary display supports HDR using DXGI.
    Returns (hdr_supported, max_luminance, color_space_info)
    """
    import ctypes
    from ctypes import wintypes, POINTER, byref, c_void_p, c_uint, c_float

    try:
        # Load DXGI
        dxgi = ctypes.windll.dxgi

        # DXGI interfaces
        IID_IDXGIFactory1 = (ctypes.c_byte * 16)(
            0x70, 0x06, 0x7d, 0x77, 0x3e, 0x14, 0x00, 0x45,
            0xac, 0x1a, 0x27, 0x33, 0xb8, 0xbc, 0x67, 0xa6
        )

        # Create DXGI Factory
        factory = c_void_p()
        hr = dxgi.CreateDXGIFactory1(byref(IID_IDXGIFactory1), byref(factory))
        if hr != 0:
            return False, 0, "Failed to create DXGI Factory"

        # Get primary adapter
        adapter = c_void_p()
        # EnumAdapters is at vtable offset 7
        EnumAdapters = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, c_uint, POINTER(c_void_p))
        vtable = ctypes.cast(factory, POINTER(c_void_p)).contents
        enum_func = EnumAdapters(ctypes.cast(vtable, POINTER(c_void_p))[7])

        hr = enum_func(factory, 0, byref(adapter))
        if hr != 0:
            return False, 0, "No display adapter found"

        # For now, return basic info
        # Full HDR detection would require IDXGIOutput6 which is more complex
        return True, 1000, "HDR capability check - adapter found"

    except Exception as e:
        logger.warning(f"[D3D11-HDR] DXGI check failed: {e}")
        return False, 0, str(e)


def configure_window_for_hdr(hwnd):
    """
    Configure a window for HDR output using DWM.
    This hints to Windows that the window contains HDR content.
    """
    import ctypes
    from ctypes import wintypes, byref, c_int

    try:
        dwmapi = ctypes.windll.dwmapi

        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 10 1809+)
        # DWMWA_BORDER_COLOR = 34 (Windows 11)
        # DWMWA_CAPTION_COLOR = 35 (Windows 11)

        # Try to disable window cloaking for better composition
        DWMWA_CLOAK = 13
        value = c_int(0)  # Uncloak
        dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_CLOAK, byref(value), ctypes.sizeof(value))

        # Set window to prefer no redirection
        # This can help with HDR by reducing composition overhead
        DWMWA_EXCLUDED_FROM_PEEK = 12
        value = c_int(1)
        dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_EXCLUDED_FROM_PEEK, byref(value), ctypes.sizeof(value))

        logger.info(f"[D3D11-HDR] Window {hwnd} configured for HDR")
        return True

    except Exception as e:
        logger.warning(f"[D3D11-HDR] Failed to configure window for HDR: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    from PySide6.QtWidgets import QApplication

    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    supported, msg = check_hdr_support()
    print(f"HDR Support: {supported}")
    print(f"Message: {msg}")

    if HAS_RHI_WIDGET and supported:
        widget = FramepackingDisplayWidgetD3D11()
        widget.resize(960, 1102)
        widget.show()
        sys.exit(app.exec())
