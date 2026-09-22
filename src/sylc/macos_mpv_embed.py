"""macOS helper for embedding mpv's standalone window into PySide6 QWidget.

On macOS, mpv's macvk (Vulkan via MoltenVK) backend creates its own Cocoa
NSWindow.

This module provides Cocoa runtime helpers to:
1. Locate mpv's NSWindow.
2. Reparent its CAMetalLayer contentView into the QWidget's native NSView.
3. Size the reparented view to match video_widget bounds.
4. Pass mouse events through to parent_view so Qt handles clicks/double-clicks.
5. Hide and order out the empty mpv NSWindow completely.
"""

import sys
import logging
import ctypes
import ctypes.util

logger = logging.getLogger(__name__)


class NSPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class NSSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class NSRect(ctypes.Structure):
    _fields_ = [("origin", NSPoint), ("size", NSSize)]


_passthrough_class = None
_hit_test_cb = None


def _get_or_create_passthrough_class(base_class, objc):
    """Creates a subclass whose hitTest: returns nil so clicks fall through to parent_view."""
    global _passthrough_class, _hit_test_cb
    if _passthrough_class:
        return _passthrough_class
    try:
        name = b"SyLCPassthroughVideoView"
        existing = objc.objc_getClass(name)
        if existing:
            _passthrough_class = existing
            return existing

        new_cls = objc.objc_allocateClassPair(base_class, name, 0)
        if not new_cls:
            return base_class

        def hit_test_impl(self_ptr, sel_ptr, point):
            return None  # Let mouse events fall through to parent_view (video_widget)

        HIT_TEST_PROTO = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, NSPoint)
        _hit_test_cb = HIT_TEST_PROTO(hit_test_impl)
        objc.class_addMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        objc.class_addMethod.restype = ctypes.c_bool
        objc.class_addMethod(
            new_cls,
            objc.sel_registerName(b"hitTest:"),
            ctypes.cast(_hit_test_cb, ctypes.c_void_p),
            b"@:@{NSPoint=dd}"
        )
        objc.objc_registerClassPair(new_cls)
        _passthrough_class = new_cls
        return new_cls
    except Exception as e:
        logger.debug(f"[MACOS-EMBED] Could not create passthrough class: {e}")
        return base_class


def update_macos_mpv_frame(video_widget, content_view=None) -> bool:
    """Updates the reparented mpv view frame to match video_widget bounds."""
    if sys.platform != 'darwin' or video_widget is None:
        return False
    cv = content_view or getattr(video_widget, '_mpv_content_view', None)
    if not cv:
        return False
    try:
        objc_lib = ctypes.util.find_library('objc')
        if not objc_lib:
            return False
        objc = ctypes.cdll.LoadLibrary(objc_lib)
        set_frame_proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, NSRect)
        set_frame = set_frame_proto(objc.objc_msgSend)
        w = max(1.0, float(video_widget.width()))
        h = max(1.0, float(video_widget.height()))
        set_frame(cv, objc.sel_registerName(b"setFrame:"), NSRect(NSPoint(0.0, 0.0), NSSize(w, h)))
        return True
    except Exception as e:
        logger.debug(f"[MACOS-EMBED] Could not update frame: {e}")
        return False


def reparent_macos_mpv_view(video_widget) -> bool:
    """Reparent mpv's Cocoa contentView into video_widget's NSView.
    
    Returns True if successfully reparented, False otherwise.
    Safe no-op on non-macOS platforms.
    """
    if sys.platform != 'darwin' or video_widget is None:
        return False

    # If already reparented, just ensure frame is updated
    if getattr(video_widget, '_mpv_reparented', False):
        update_macos_mpv_frame(video_widget)
        return True

    try:
        import ctypes
        import ctypes.util

        objc_lib = ctypes.util.find_library('objc')
        if not objc_lib:
            return False

        objc = ctypes.cdll.LoadLibrary(objc_lib)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.object_getClassName.restype = ctypes.c_char_p
        objc.object_getClassName.argtypes = [ctypes.c_void_p]

        parent_winid = int(video_widget.winId())
        if not parent_winid:
            return False
        parent_view = ctypes.c_void_p(parent_winid)

        NSApp = objc.objc_msgSend(
            objc.objc_getClass(b"NSApplication"),
            objc.sel_registerName(b"sharedApplication")
        )
        if not NSApp:
            return False

        windows = objc.objc_msgSend(NSApp, objc.sel_registerName(b"windows"))
        if not windows:
            return False

        count = objc.objc_msgSend(windows, objc.sel_registerName(b"count"))
        parent_window = objc.objc_msgSend(parent_view, objc.sel_registerName(b"window"))

        for i in range(count):
            nswin = objc.objc_msgSend(
                windows,
                objc.sel_registerName(b"objectAtIndex:"),
                ctypes.c_ulong(i)
            )
            if not nswin or nswin == parent_window:
                continue

            cname = objc.object_getClassName(nswin).decode('utf-8', 'ignore')
            # SyLC overlay tool windows use QNSWindow. mpv creates a window with class Window or NSWindow
            if cname in ('Window', 'NSWindow') or 'mpv' in cname.lower():
                delegate = objc.objc_msgSend(nswin, objc.sel_registerName(b"delegate"))
                del_cname = objc.object_getClassName(delegate).decode('utf-8', 'ignore') if delegate else ''
                if 'QNS' in del_cname:
                    continue

                content_view = objc.objc_msgSend(nswin, objc.sel_registerName(b"contentView"))
                if not content_view:
                    continue

                # Make the standalone mpv window 100% invisible immediately
                set_alpha_proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double)
                set_alpha = set_alpha_proto(objc.objc_msgSend)
                set_alpha(nswin, objc.sel_registerName(b"setAlphaValue:"), ctypes.c_double(0.0))
                objc.objc_msgSend(nswin, objc.sel_registerName(b"orderOut:"), None)

                # Add content_view as subview of video_widget's NSView
                objc.objc_msgSend(
                    parent_view,
                    objc.sel_registerName(b"addSubview:"),
                    content_view
                )

                # Size content_view to fill video_widget exactly
                set_frame_proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, NSRect)
                set_frame = set_frame_proto(objc.objc_msgSend)
                w = max(1.0, float(video_widget.width()))
                h = max(1.0, float(video_widget.height()))
                set_frame(content_view, objc.sel_registerName(b"setFrame:"), NSRect(NSPoint(0.0, 0.0), NSSize(w, h)))

                # NSViewWidthSizable(2) | NSViewHeightSizable(16) = 18
                objc.objc_msgSend(
                    content_view,
                    objc.sel_registerName(b"setAutoresizingMask:"),
                    ctypes.c_ulong(18)
                )

                # Configure hitTest so mouse clicks pass through to parent_view (video_widget)
                try:
                    base_cls = objc.object_getClass(content_view)
                    passthrough_cls = _get_or_create_passthrough_class(base_cls, objc)
                    if passthrough_cls and passthrough_cls != base_cls:
                        objc.object_setClass.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                        objc.object_setClass.restype = ctypes.c_void_p
                        objc.object_setClass(content_view, passthrough_cls)
                except Exception as e:
                    logger.debug(f"[MACOS-EMBED] Could not set passthrough class on content view: {e}")

                video_widget._mpv_content_view = content_view
                video_widget._mpv_reparented = True
                logger.info("[MACOS-EMBED] Successfully reparented MPV view into video widget")
                return True

    except Exception as e:
        logger.warning(f"[MACOS-EMBED] Error during Cocoa reparenting: {e}")

    return False
