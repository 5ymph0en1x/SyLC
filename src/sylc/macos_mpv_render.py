"""Render libmpv video directly into SyLC's Qt OpenGL surface on macOS.

Supports both standard 2D playback (zero-overhead direct FBO rendering) and
AI-based 2D->3D real-time stereoscopic conversion (Depth Anything V3 + OpenGL DIBR).
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLBuffer,
)

logger = logging.getLogger(__name__)

# Fullscreen quad vertices: (x, y, u, v)
QUAD_VERTS = np.array([
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,
    -1.0,  1.0,  0.0, 1.0,
     1.0,  1.0,  1.0, 1.0,
], dtype=np.float32)

# Standard OpenGL numeric constants (PySide6 QOpenGLFunctions does not expose C enum defines)
GL_FLOAT = 0x1406
GL_TRIANGLE_STRIP = 0x0005
GL_FRAMEBUFFER = 0x8D40
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1
GL_COLOR_BUFFER_BIT = 0x00004000
GL_BLEND = 0x0BE2
GL_CULL_FACE = 0x0B44
GL_DEPTH_TEST = 0x0B71
GL_SCISSOR_TEST = 0x0C11

# -----------------------------------------------------------------------------
# GLSL Shaders: Core Profile (OpenGL 3.2+ on macOS)
# -----------------------------------------------------------------------------
VERTEX_SHADER_CORE = """
#version 150 core
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"""

FRAGMENT_SHADER_CORE = """
#version 150 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_source_tex;
uniform sampler2D u_depth_tex;
uniform float u_strength;
uniform float u_convergence;
uniform int u_depth_view;
uniform int u_stereo_mode;

// Unpack 16-bit nearness from RG channels (0.0 = far at infinity, 1.0 = near)
float get_nearness(vec2 uv) {
    vec4 d = texture(u_depth_tex, uv);
    return d.r * 0.996108948 + d.g * 0.00389105;
}

// Disparity curve matching synth3d.hlsl
float disp_for_nearness(float nearness) {
    float d = nearness - u_convergence;
    float span = d >= 0.0 ? max(0.08, 1.0 - u_convergence) : max(0.08, u_convergence);
    float z = clamp(abs(d) / span, 0.0, 1.0);
    float knee = d >= 0.0 ? 0.68 : 0.76;
    float over = max(0.0, z - knee);
    float softened = z <= knee ? z : knee + over / (1.0 + 1.60 * over);
    return (d < 0.0 ? -1.0 : 1.0) * softened * span *
           u_strength;
}

// Backward DIBR: 3 fixed-point search iterations for eye coordinate
vec4 warp_eye(vec2 uv, float eyeSign) {
    float xs = uv.x;
    for (int i = 0; i < 3; ++i) {
        float n = get_nearness(vec2(xs, uv.y));
        xs = uv.x - eyeSign * 0.5 * disp_for_nearness(n);
    }
    xs = clamp(xs, 0.0, 1.0);
    return texture(u_source_tex, vec2(xs, uv.y));
}

// Turbo colormap for depth inspection
vec3 colormap_turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    vec4 kR = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    vec4 kG = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    vec4 kB = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    vec2 kR2 = vec2(-152.94239396, 59.28637943);
    vec2 kG2 = vec2(4.27729857, 2.82956604);
    vec2 kB2 = vec2(-89.90310912, 27.34824973);

    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;

    return vec3(
        dot(v4, kR) + dot(v2, kR2),
        dot(v4, kG) + dot(v2, kG2),
        dot(v4, kB) + dot(v2, kB2)
    );
}

void main() {
    if (u_depth_view != 0) {
        float n = get_nearness(v_uv);
        fragColor = vec4(colormap_turbo(n), 1.0);
        return;
    }

    if (u_stereo_mode == 0) {
        // Anaglyph Red-Cyan (Left = Red, Right = Cyan)
        vec4 left = warp_eye(v_uv, 1.0);
        vec4 right = warp_eye(v_uv, -1.0);
        fragColor = vec4(left.r, right.g, right.b, 1.0);
    } else if (u_stereo_mode == 1) {
        // Side-by-Side (SBS)
        if (v_uv.x < 0.5) {
            vec2 l_uv = vec2(v_uv.x * 2.0, v_uv.y);
            fragColor = warp_eye(l_uv, 1.0);
        } else {
            vec2 r_uv = vec2((v_uv.x - 0.5) * 2.0, v_uv.y);
            fragColor = warp_eye(r_uv, -1.0);
        }
    } else if (u_stereo_mode == 2) {
        // Top-and-Bottom (TAB)
        if (v_uv.y < 0.5) {
            vec2 l_uv = vec2(v_uv.x, v_uv.y * 2.0);
            fragColor = warp_eye(l_uv, 1.0);
        } else {
            vec2 r_uv = vec2(v_uv.x, (v_uv.y - 0.5) * 2.0);
            fragColor = warp_eye(r_uv, -1.0);
        }
    } else {
        // MultiView / Mono pass-through (Left eye)
        fragColor = warp_eye(v_uv, 1.0);
    }
}
"""

# -----------------------------------------------------------------------------
# GLSL Shaders: Legacy Compatibility Profile (OpenGL 2.1)
# -----------------------------------------------------------------------------
VERTEX_SHADER_LEGACY = """
#version 120
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"""

FRAGMENT_SHADER_LEGACY = """
#version 120
varying vec2 v_uv;
uniform sampler2D u_source_tex;
uniform sampler2D u_depth_tex;
uniform float u_strength;
uniform float u_convergence;
uniform int u_depth_view;
uniform int u_stereo_mode;

float get_nearness(vec2 uv) {
    vec4 d = texture2D(u_depth_tex, uv);
    return d.r * 0.996108948 + d.g * 0.00389105;
}

float disp_for_nearness(float nearness) {
    float d = nearness - u_convergence;
    float span = d >= 0.0 ? max(0.08, 1.0 - u_convergence) : max(0.08, u_convergence);
    float z = clamp(abs(d) / span, 0.0, 1.0);
    float knee = d >= 0.0 ? 0.68 : 0.76;
    float over = max(0.0, z - knee);
    float softened = z <= knee ? z : knee + over / (1.0 + 1.60 * over);
    return (d < 0.0 ? -1.0 : 1.0) * softened * span *
           u_strength;
}

vec4 warp_eye(vec2 uv, float eyeSign) {
    float xs = uv.x;
    for (int i = 0; i < 3; ++i) {
        float n = get_nearness(vec2(xs, uv.y));
        xs = uv.x - eyeSign * 0.5 * disp_for_nearness(n);
    }
    xs = clamp(xs, 0.0, 1.0);
    return texture2D(u_source_tex, vec2(xs, uv.y));
}

vec3 colormap_turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    vec4 kR = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    vec4 kG = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    vec4 kB = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    vec2 kR2 = vec2(-152.94239396, 59.28637943);
    vec2 kG2 = vec2(4.27729857, 2.82956604);
    vec2 kB2 = vec2(-89.90310912, 27.34824973);

    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;

    return vec3(
        dot(v4, kR) + dot(v2, kR2),
        dot(v4, kG) + dot(v2, kG2),
        dot(v4, kB) + dot(v2, kB2)
    );
}

void main() {
    if (u_depth_view != 0) {
        float n = get_nearness(v_uv);
        gl_FragColor = vec4(colormap_turbo(n), 1.0);
        return;
    }

    if (u_stereo_mode == 0) {
        vec4 left = warp_eye(v_uv, 1.0);
        vec4 right = warp_eye(v_uv, -1.0);
        gl_FragColor = vec4(left.r, right.g, right.b, 1.0);
    } else if (u_stereo_mode == 1) {
        if (v_uv.x < 0.5) {
            vec2 l_uv = vec2(v_uv.x * 2.0, v_uv.y);
            gl_FragColor = warp_eye(l_uv, 1.0);
        } else {
            vec2 r_uv = vec2((v_uv.x - 0.5) * 2.0, v_uv.y);
            gl_FragColor = warp_eye(r_uv, -1.0);
        }
    } else if (u_stereo_mode == 2) {
        if (v_uv.y < 0.5) {
            vec2 l_uv = vec2(v_uv.x, v_uv.y * 2.0);
            gl_FragColor = warp_eye(l_uv, 1.0);
        } else {
            vec2 r_uv = vec2(v_uv.x, (v_uv.y - 0.5) * 2.0);
            gl_FragColor = warp_eye(r_uv, -1.0);
        }
    } else {
        gl_FragColor = warp_eye(v_uv, 1.0);
    }
}
"""


class MacOSMpvVideoWidget(QOpenGLWidget):
    frame_ready = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._render_context = None
        self.setMouseTracking(True)
        self.frame_ready.connect(self.update, Qt.ConnectionType.QueuedConnection)

        # 2D->3D synth3d parameters
        self.synth3d_enabled: bool = False
        self.synth3d_strength: float = 2.5        # Disparity budget in % of width
        self.synth3d_convergence: float = 0.5     # Convergence plane (0..1)
        self.synth3d_depth_view: bool = False     # View depth map directly
        self.synth3d_model_path: str = ""
        self.synth3d_side: int = 518
        self.stereo_mode: str = 'mvc'

        # Inference engine
        self._engine = None

        # OpenGL resources for 2D->3D offscreen rendering
        self._source_fbo: Optional[QOpenGLFramebufferObject] = None
        self._probe_fbo: Optional[QOpenGLFramebufferObject] = None
        self._depth_texture: Optional[QOpenGLTexture] = None
        self._shader_program: Optional[QOpenGLShaderProgram] = None
        self._vbo: Optional[QOpenGLBuffer] = None
        self._vao = None
        self._gl_initialized: bool = False
        self._depth_upload_count: int = 0
        self._shader_interface_logged: bool = False

    def attach_player(self, player, mpv_module):
        """Create the render context while Qt's OpenGL context is current."""
        self.detach_player()
        self.makeCurrent()
        try:
            context = self.context()
            if context is None:
                raise RuntimeError("Qt OpenGL context is unavailable")

            def get_proc_address(_ctx, name):
                return int(context.getProcAddress(name.decode('ascii')) or 0)

            self._get_proc_address = mpv_module.MpvGlGetProcAddressFn(get_proc_address)
            self._render_context = mpv_module.MpvRenderContext(
                player, 'opengl',
                opengl_init_params={'get_proc_address': self._get_proc_address},
            )
            # libmpv calls this on its render thread; the signal queues update()
            # onto Qt's GUI thread where paintGL owns the OpenGL context.
            self._render_context.update_cb = self.frame_ready.emit
        finally:
            self.doneCurrent()
        self.update()

    def detach_player(self):
        self.stop_synth3d()
        render_context = self._render_context
        if render_context is None:
            return
        self._render_context = None
        render_context.update_cb = None
        self.makeCurrent()
        try:
            render_context.free()
        finally:
            self.doneCurrent()
            self._get_proc_address = None

    # =========================================================================
    # 2D->3D Synthesis Control API
    # =========================================================================

    def start_synth3d(self, model_path: str, side: int = 518) -> None:
        """Start or update depth engine with given model and grid side."""
        self.synth3d_model_path = model_path
        self.synth3d_side = side
        self.synth3d_enabled = True

        if self._engine is None:
            from sylc.macos_synth3d_engine import MacOSSynth3DEngine
            self._engine = MacOSSynth3DEngine()

        self._engine.start(model_path, side)
        logger.info("[2D3D macOS] Synth3D activated with model=%s side=%d",
                    os.path.basename(model_path), side)
        self.update()

    def stop_synth3d(self) -> None:
        """Stop depth inference engine and revert to normal 2D rendering."""
        self.synth3d_enabled = False
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        self.update()

    def synth3d_status(self) -> str:
        """Return engine status for UI diagnostics."""
        if not self.synth3d_enabled or self._engine is None:
            return "Engine: off"
        gpu_depth = "yes" if self._depth_texture is not None else "waiting"
        return (f"{self._engine.status()} · GPU depth: {gpu_depth} "
                f"({self._depth_upload_count}) · strength: {self.synth3d_strength:.1f}%")

    # =========================================================================
    # OpenGL Pipeline & Rendering
    # =========================================================================

    def _compile_shader_pair(self, vert_src: str, frag_src: str) -> Optional[QOpenGLShaderProgram]:
        program = QOpenGLShaderProgram(self)
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vert_src):
            logger.debug("[2D3D macOS] Vertex shader rejected: %s", program.log())
            return None
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, frag_src):
            logger.debug("[2D3D macOS] Fragment shader rejected: %s", program.log())
            return None
        if not program.link():
            logger.debug("[2D3D macOS] Shader link rejected: %s", program.log())
            return None
        return program

    def _init_gl_resources(self) -> bool:
        """Initialize shaders, VBO, VAO, and textures for stereo warping."""
        if self._gl_initialized:
            return True

        # Try modern Core profile shader (macOS Core Profile 3.2+), fallback to legacy GLSL 120
        modern_gl = self.context().format().majorVersion() >= 3
        program = (self._compile_shader_pair(VERTEX_SHADER_CORE, FRAGMENT_SHADER_CORE)
                   if modern_gl else None)
        if program is None:
            logger.info("[2D3D macOS] Using compatibility GLSL 120 shader")
            program = self._compile_shader_pair(VERTEX_SHADER_LEGACY, FRAGMENT_SHADER_LEGACY)
        if program is None:
            logger.error("[2D3D macOS] Failed to compile and link shader program")
            return False

        self._shader_program = program

        # Create fullscreen quad VBO
        vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        if not vbo.create():
            logger.error("[2D3D macOS] Failed to create VBO")
            return False
        vbo.bind()
        vbo.allocate(QUAD_VERTS.tobytes(), QUAD_VERTS.nbytes)
        vbo.release()
        self._vbo = vbo

        # Create VAO (required on macOS OpenGL Core profile)
        try:
            from PySide6.QtOpenGL import QOpenGLVertexArrayObject
            vao = QOpenGLVertexArrayObject(self)
            if vao.create():
                vao.bind()
                vbo.bind()
                if not program.bind():
                    raise RuntimeError("failed to bind shader while configuring VAO")
                try:
                    if (program.attributeLocation("a_pos") < 0
                            or program.attributeLocation("a_uv") < 0):
                        raise RuntimeError("2D->3D shader has no quad attributes")
                    program.enableAttributeArray("a_pos")
                    program.setAttributeBuffer("a_pos", GL_FLOAT, 0, 2, 16)
                    program.enableAttributeArray("a_uv")
                    program.setAttributeBuffer("a_uv", GL_FLOAT, 8, 2, 16)
                finally:
                    program.release()
                vao.release()
                vbo.release()
                self._vao = vao
                logger.info("[2D3D macOS] VAO created and configured")
        except Exception:
            logger.debug("[2D3D macOS] VAO initialization skipped (fallback to direct VBO)")
            self._vao = None

        self._gl_initialized = True
        logger.info("[2D3D macOS] OpenGL shader & quad resources initialized successfully")
        return True

    def paintGL(self):
        render_context = self._render_context
        if render_context is None:
            return

        scale = self.devicePixelRatioF()
        w = max(1, round(self.width() * scale))
        h = max(1, round(self.height() * scale))

        # -----------------------------------------------------------------
        # NORMAL 2D PLAYBACK (direct zero-copy to Qt default framebuffer)
        # -----------------------------------------------------------------
        if not self.synth3d_enabled or self._engine is None:
            render_attempted = False
            try:
                render_context.update()
                # Mark the attempt before crossing into libmpv. Its wrapper can
                # raise after mpv has accepted the frame; that frame must still
                # receive report_swap() or delivery can stall while audio runs.
                render_attempted = True
                render_context.render(
                    opengl_fbo={
                        'fbo': self.defaultFramebufferObject(),
                        'w': w,
                        'h': h,
                    },
                    flip_y=True,
                )
            except Exception:
                logger.exception("[MPV] Failed to render 2D video into macOS widget")
            finally:
                if render_attempted:
                    try:
                        render_context.report_swap()
                    except Exception:
                        logger.exception("[MPV] Failed to report the 2D swap")
            return

        # -----------------------------------------------------------------
        # 2D->3D STEREOSCOPIC PIPELINE
        # -----------------------------------------------------------------
        render_attempted = False
        try:
            if not self._init_gl_resources():
                # Fallback to direct 2D rendering so video never hangs
                try:
                    render_context.update()
                    render_attempted = True
                    render_context.render(
                        opengl_fbo={'fbo': self.defaultFramebufferObject(), 'w': w, 'h': h},
                        flip_y=True,
                    )
                except Exception:
                    logger.exception("[2D3D macOS] Fallback 2D render failed")
                return

            # 1. Ensure source FBO matches display viewport
            if self._source_fbo is None or self._source_fbo.width() != w or self._source_fbo.height() != h:
                if self._source_fbo is not None:
                    del self._source_fbo
                self._source_fbo = QOpenGLFramebufferObject(w, h)

            # 2. Ensure probe FBO matches model grid side
            side = self.synth3d_side
            if self._probe_fbo is None or self._probe_fbo.width() != side or self._probe_fbo.height() != side:
                if self._probe_fbo is not None:
                    del self._probe_fbo
                self._probe_fbo = QOpenGLFramebufferObject(side, side)

            # 3. Render video into offscreen source FBO (right-side up)
            render_context.update()
            render_attempted = True
            render_context.render(
                opengl_fbo={
                    'fbo': self._source_fbo.handle(),
                    'w': w,
                    'h': h,
                },
                # Match the direct 2D path. Without this, the offscreen texture
                # is vertically inverted before the stereo shader samples it.
                flip_y=True,
            )

            # 4. Asynchronously send downscaled frame probe to depth inference worker
            if self._engine.is_ready_for_frame():
                try:
                    # The scissor left by mpv also affects framebuffer blits.
                    self.context().functions().glDisable(GL_SCISSOR_TEST)
                    QOpenGLFramebufferObject.blitFramebuffer(self._probe_fbo, self._source_fbo)
                    probe_img = self._probe_fbo.toImage()
                    self._engine.submit_qimage(probe_img)
                except Exception:
                    try:
                        probe_img = self._source_fbo.toImage()
                        self._engine.submit_qimage(probe_img)
                    except Exception:
                        pass

            # 5. Check if worker has delivered a new stabilized depth map
            depth_data = self._engine.get_latest_depth()
            if depth_data is not None:
                packed_bytes, dw, dh = depth_data
                # toImage() gave the model top-down raster rows. Restore GL's
                # bottom-up row order so depth and the source FBO share UVs.
                depth_pixels = np.ascontiguousarray(
                    np.frombuffer(packed_bytes, dtype=np.uint8)
                    .reshape(dh, dw, 4)[::-1])
                if (self._depth_texture is None or self._depth_texture.width() != dw
                        or self._depth_texture.height() != dh):
                    if self._depth_texture is not None:
                        self._depth_texture.destroy()
                    self._depth_texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
                    self._depth_texture.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
                    self._depth_texture.setSize(dw, dh)
                    self._depth_texture.setMipLevels(1)
                    self._depth_texture.allocateStorage(
                        QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
                    # Decoding RG16 is a linear weighted sum, so filtering
                    # channels before decoding correctly interpolates depth.
                    self._depth_texture.setMinMagFilters(
                        QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
                    self._depth_texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
                # The QImage overload allocates storage again on every update.
                # Upload pixels into the existing allocation instead. Keep the
                # owning array alive until this synchronous transfer returns.
                self._depth_texture.setData(
                    QOpenGLTexture.PixelFormat.RGBA,
                    QOpenGLTexture.PixelType.UInt8,
                    int(depth_pixels.ctypes.data))
                self._depth_upload_count += 1

            # 6. Render DIBR stereo warped output to Qt's default framebuffer
            funcs = self.context().functions()
            funcs.glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
            funcs.glViewport(0, 0, w, h)
            # libmpv owns the context immediately before this pass and may
            # leave state that clips, blends, culls, or depth-rejects our quad.
            funcs.glDisable(GL_BLEND)
            funcs.glDisable(GL_CULL_FACE)
            funcs.glDisable(GL_DEPTH_TEST)
            funcs.glDisable(GL_SCISSOR_TEST)
            funcs.glColorMask(True, True, True, True)

            if not self._shader_program.bind():
                raise RuntimeError("failed to bind 2D->3D shader program")
            if not self._shader_interface_logged:
                attributes = {
                    name: self._shader_program.attributeLocation(name)
                    for name in ("a_pos", "a_uv")
                }
                uniforms = {
                    name: self._shader_program.uniformLocation(name)
                    for name in (
                        "u_source_tex", "u_depth_tex", "u_strength",
                        "u_convergence", "u_depth_view", "u_stereo_mode",
                    )
                }
                logger.info("[2D3D macOS] Shader interface attributes=%s uniforms=%s",
                            attributes, uniforms)
                if any(location < 0 for location in uniforms.values()):
                    raise RuntimeError("Missing required stereo shader uniform")
                self._uniform_locations = uniforms
                self._shader_interface_logged = True
            locations = self._uniform_locations
            if self._vao is not None:
                self._vao.bind()
            else:
                self._vbo.bind()
                self._shader_program.enableAttributeArray("a_pos")
                self._shader_program.setAttributeBuffer("a_pos", GL_FLOAT, 0, 2, 16)
                self._shader_program.enableAttributeArray("a_uv")
                self._shader_program.setAttributeBuffer("a_uv", GL_FLOAT, 8, 2, 16)

            # Texture 0: Source frame
            funcs.glActiveTexture(GL_TEXTURE0)
            funcs.glBindTexture(GL_TEXTURE_2D, self._source_fbo.texture())
            funcs.glUniform1i(locations["u_source_tex"], 0)

            # Texture 1: Depth map
            funcs.glActiveTexture(GL_TEXTURE1)
            if self._depth_texture is not None:
                funcs.glBindTexture(GL_TEXTURE_2D, self._depth_texture.textureId())
            else:
                funcs.glBindTexture(GL_TEXTURE_2D, 0)
            funcs.glUniform1i(locations["u_depth_tex"], 1)

            # Strength & convergence parameters
            has_depth = self._depth_texture is not None
            strength_val = float(self.synth3d_strength * 0.01) if has_depth else 0.0
            # Use numeric locations and explicitly typed GL entry points.
            # The installed PySide6 named scalar wrappers reject byte names.
            funcs.glUniform1f(locations["u_strength"], strength_val)
            funcs.glUniform1f(
                locations["u_convergence"], float(self.synth3d_convergence))
            funcs.glUniform1i(
                locations["u_depth_view"], 1 if (self.synth3d_depth_view and has_depth) else 0)

            # Stereo presentation mode mapping
            # 0: Anaglyph Red-Cyan, 1: Side-by-Side, 2: Top-and-Bottom, 3: MultiView/Mono
            mode_str = (self.stereo_mode or '').lower()
            if mode_str in ('sbs', 'glasses'):
                mode_int = 1
            elif mode_str == 'tab':
                mode_int = 2
            elif mode_str in ('anaglyph', 'mvc'):
                # On macOS standard screens, MultiView defaults to Anaglyph Red-Cyan
                mode_int = 0
            else:
                mode_int = 3

            funcs.glUniform1i(locations["u_stereo_mode"], mode_int)

            # Draw fullscreen quad
            funcs.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            if self._vao is not None:
                self._vao.release()
            else:
                self._vbo.release()
            self._shader_program.release()

        except Exception:
            logger.exception("[2D3D macOS] Error in paintGL stereo rendering")
            # Preserve the frame already rendered by mpv if stereo composition
            # fails. Do not request a second mpv render for the same frame.
            try:
                funcs = self.context().functions()
                funcs.glUseProgram(0)
                if self._vao is not None:
                    self._vao.release()
                funcs.glDisable(GL_SCISSOR_TEST)
                if self._source_fbo is not None and self._source_fbo.isValid():
                    funcs.glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
                    # Qt resolves a null target to this widget's default FBO.
                    QOpenGLFramebufferObject.blitFramebuffer(None, self._source_fbo)
            except Exception:
                logger.exception("[2D3D macOS] Failed to recover the source frame")
        finally:
            if render_attempted:
                try:
                    render_context.report_swap()
                except Exception:
                    logger.exception("[2D3D macOS] Failed to report the mpv swap")

    # =========================================================================
    # Cleanup & Teardown
    # =========================================================================

    def cleanup_gl(self):
        """Release OpenGL FBOs, textures, VAO, and shader buffers."""
        self.makeCurrent()
        try:
            if self._depth_texture is not None:
                self._depth_texture.destroy()
                self._depth_texture = None
            if self._vao is not None:
                self._vao.destroy()
                self._vao = None
            if self._vbo is not None:
                self._vbo.destroy()
                self._vbo = None
            self._shader_program = None
            self._source_fbo = None
            self._probe_fbo = None
            self._gl_initialized = False
            self._depth_upload_count = 0
            self._shader_interface_logged = False
        finally:
            self.doneCurrent()

    def closeEvent(self, event):
        self.stop_synth3d()
        self.cleanup_gl()
        super().closeEvent(event)
