# -*- coding: utf-8 -*-
"""Pure scheduling and presentation policies for native video playback."""

import logging
import multiprocessing
import sys


logger = logging.getLogger(__name__)


_EDGE264_WORKER_SATURATION = 6


def _physical_core_count():
    """Physical cores, or None when the platform will not report them.

    ``multiprocessing.cpu_count()`` counts SMT siblings. Decoding does not
    scale with siblings the way it scales with cores, and on a power-limited
    APU the sibling count overstates the usable budget by 2x -- which is why
    the worker policy below budgets in physical cores.
    """
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetLogicalProcessorInformationEx.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL

        relation_processor_core = 0
        length = wintypes.DWORD(0)
        # First call fails with ERROR_INSUFFICIENT_BUFFER and reports the size.
        kernel32.GetLogicalProcessorInformationEx(
            relation_processor_core, None, ctypes.byref(length))
        if not length.value:
            return None
        buffer = (ctypes.c_ubyte * length.value)()
        if not kernel32.GetLogicalProcessorInformationEx(
                relation_processor_core, buffer, ctypes.byref(length)):
            return None

        # A sequence of variable-length records; one record per physical core.
        # Only the Size field (second DWORD) is needed to walk it.
        cores = 0
        offset = 0
        while offset + 8 <= length.value:
            size = int.from_bytes(bytes(buffer[offset + 4:offset + 8]), 'little')
            if size <= 0:
                break
            cores += 1
            offset += size
        return cores or None
    except Exception as exc:
        logger.debug("[CPU] Physical core probe failed: %s", exc)
        return None

def _recommended_edge264_threads():
    """Choose the automatic edge264 worker count for this machine.

    Budgets in physical cores, reserving capacity for GUI, audio, presentation
    and -- when 2D->3D is on -- the depth engine. ``SYLC_EDGE264_THREADS``
    remains the authoritative expert override inside MVCDecoderThread; this is
    the automatic default only.

    The previous policy was a step ladder that returned 4 for every machine
    with more than 8 logical CPUs. On the table above that costs 6 % of
    throughput and 44 % of p99 versus the saturation point -- and p99 is the
    figure Z2A_VALIDATION.md calls decisive. The Ryzen Z2 A result is
    deliberately preserved: 4 physical cores still yield 3 workers.
    """
    logical = max(1, int(multiprocessing.cpu_count() or 1))
    physical = _physical_core_count()
    if physical is None or physical < 1:
        # No reliable topology: assume SMT rather than over-committing a
        # small CPU, and never exceed the old ladder's answer.
        physical = max(1, logical // 2) if logical > 4 else logical

    # One core for GUI+audio on small parts, two once there is room -- the
    # presenter and the depth engine both need to run while decode does.
    reserve = 1 if physical <= 4 else 2
    # Floor of two so a dual-core still overlaps demux with decode.
    budget = max(2, physical - reserve)
    workers = max(1, min(_EDGE264_WORKER_SATURATION, budget))

    logger.info(
        "[CPU] edge264 workers=%d (physical=%d logical=%d reserve=%d)",
        workers, physical, logical, reserve)
    return workers

def _edge264_startup_timeout_ms(video_info):
    """Bound the black-screen window while preserving slower optical MVC startup."""
    info = video_info if isinstance(video_info, dict) else {}
    return 25000 if info.get('is_3d') else 12000

def _select_stereo_presentation_targets(embedded, framepacking_window, active,
                                        eye_windows=None):
    """Return ``[(widget, vsync), ...]`` for one decoded stereo frame.

    With `eye_windows` set (Dual Projector), the two eye outputs REPLACE the
    framepack window: they are the same picture cut in two, so presenting both
    would upload every frame twice for nothing. The left eye is the timing
    authority; the right eye and the main-window preview follow with interval 0,
    because two vsync-blocking swapchains on two projectors can halve the
    decoder cadence -- the same hazard the framepack/preview pair already
    documents.
    """
    selected = []
    seen = set()

    def _visible(widget):
        try:
            return widget is not None and widget.isVisible()
        except Exception:
            return False

    def _add(widget, vsync):
        if widget is None or id(widget) in seen:
            return
        seen.add(id(widget))
        selected.append((widget, bool(vsync)))

    eye_widgets = []
    for window in (eye_windows or ()):
        widget = getattr(window, 'display_widget', None)
        if _visible(window) and widget is not None:
            eye_widgets.append(widget)

    if eye_widgets:
        for index, widget in enumerate(eye_widgets):
            _add(widget, index == 0)
        if _visible(embedded):
            _add(embedded, False)
        return selected

    fp_widget = (getattr(framepacking_window, 'display_widget', None)
                 if framepacking_window is not None else None)
    fp_visible = _visible(framepacking_window) and fp_widget is not None

    # Present the timing authority first. The main-window copy follows with
    # interval 0, so the second swapchain cannot halve the decoder cadence.
    if fp_visible:
        _add(fp_widget, True)
    if _visible(embedded):
        _add(embedded, not fp_visible)

    # During construction/visibility transitions, keep the active renderer warm.
    if not selected:
        _add(active, True)
    return selected


__all__ = [
    '_EDGE264_WORKER_SATURATION', '_physical_core_count',
    '_recommended_edge264_threads', '_edge264_startup_timeout_ms',
    '_select_stereo_presentation_targets',
]
