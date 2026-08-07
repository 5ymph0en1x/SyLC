# -*- coding: utf-8 -*-
r"""
Blu-ray / optical-disc ISO archiver for SyLC.

Creates a byte-perfect image of the optical *volume* (exactly what Windows reproduces
when you later mount the resulting .iso) by reading the raw volume device \\.\<L>:
sequentially. No admin rights are required to read an optical volume.

Robustness for a flaky drive: a failed bulk read falls back to sector-by-sector reads
with retries; truly unreadable 2048-byte sectors are either zero-filled (if the user
opts in) or abort the copy with the exact bad offset.

This module is self-contained: the Win32 raw reader, the worker thread, the themed
"disc burning inside-out" loading animation, and the configuration/progress dialog.

Pure ctypes + PySide6; Windows-only.
"""
import os
import ctypes
import time
import hashlib
from collections import deque
from ctypes import wintypes

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QRadialGradient,
                           QConicalGradient, QFont, QPainterPath)
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                               QLineEdit, QPushButton, QCheckBox, QFileDialog, QWidget,
                               QFrame, QMessageBox, QSizePolicy)

SECTOR = 2048                      # optical logical sector size
CHUNK = 2048 * 2048                # 4 MiB bulk read (multiple of SECTOR)
SECTOR_RETRIES = 4                 # per-sector retries before declaring it bad

# ---------------------------------------------------------------------------
# Win32 raw volume access (ctypes)
# ---------------------------------------------------------------------------
_k32 = ctypes.WinDLL('kernel32', use_last_error=True)

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
FILE_SHARE_WRITE = 0x2
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE = ctypes.c_void_p(-1).value
FILE_BEGIN = 0
DRIVE_CDROM = 5
IOCTL_DISK_GET_LENGTH_INFO = 0x0007405C

ERR_NOT_READY = 21
ERR_ACCESS_DENIED = 5
ERR_INVALID_PARAMETER = 87

_k32.CreateFileW.restype = wintypes.HANDLE
_k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
_k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                  ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
_k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                 ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
_k32.GetDriveTypeW.restype = wintypes.UINT
_k32.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       wintypes.LPWSTR, wintypes.DWORD]
_k32.GetDiskFreeSpaceExW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_ulonglong),
                                     ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)]


def error_text(code):
    """Human-readable Win32 error, with friendly notes for the common optical cases."""
    friendly = {
        ERR_NOT_READY: "No disc in the drive (or it is still spinning up).",
        ERR_ACCESS_DENIED: "Access denied.",
        1117: "Device I/O error (the drive could not read the disc).",
        23: "Data error / CRC (unreadable sector).",
    }
    if code in friendly:
        return friendly[code]
    try:
        return ctypes.FormatError(code).strip() or f"Win32 error {code}"
    except Exception:
        return f"Win32 error {code}"


def list_optical_drives():
    """Return the drive letters (e.g. ['J']) whose type is CD/DVD/BD-ROM.
    Note: a Windows-mounted ISO also reports as CDROM."""
    out = []
    import string
    for c in string.ascii_uppercase:
        try:
            if _k32.GetDriveTypeW(f"{c}:\\") == DRIVE_CDROM:
                out.append(c)
        except Exception:
            pass
    return out


def _volume_label(letter):
    try:
        name = ctypes.create_unicode_buffer(261)
        ok = _k32.GetVolumeInformationW(f"{letter}:\\", name, 260, None, None, None, None, 0)
        if ok and name.value:
            return name.value
    except Exception:
        pass
    return ""


def _open_raw(letter, no_buffering=True):
    r"""Open \\.\<L>: for raw reading. Prefer FILE_FLAG_NO_BUFFERING (no OS-cache pollution
    for a multi-GB image) but fall back to buffered access, which is more permissive across
    drive states / privilege levels. Our sector-aligned reads work in either mode."""
    modes = []
    if no_buffering:
        modes.append(FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_NO_BUFFERING)
    modes.append(FILE_FLAG_SEQUENTIAL_SCAN)  # buffered fallback
    last_err = 0
    for flags in modes:
        h = _k32.CreateFileW(f"\\\\.\\{letter}:", GENERIC_READ,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, flags, None)
        if h != INVALID_HANDLE and h is not None:
            return h, 0
        last_err = ctypes.get_last_error()
    return None, last_err


def _volume_length(h):
    buf = ctypes.create_string_buffer(8)
    ret = wintypes.DWORD(0)
    ok = _k32.DeviceIoControl(h, IOCTL_DISK_GET_LENGTH_INFO, None, 0,
                              buf, 8, ctypes.byref(ret), None)
    if not ok:
        return None, ctypes.get_last_error()
    return int.from_bytes(buf.raw[:8], 'little'), 0


def probe_volume(letter):
    """Inspect an optical volume. Returns a dict:
       {ok, length, label, error_code, error}. length is the exact byte size to image."""
    info = {"ok": False, "length": 0, "label": _volume_label(letter),
            "error_code": 0, "error": ""}
    h, err = _open_raw(letter, no_buffering=True)
    if not h:
        info["error_code"] = err
        info["error"] = error_text(err)
        return info
    try:
        # Prefer the filesystem volume size. IOCTL_DISK_GET_LENGTH_INFO over-reports by the
        # drive's lead-out padding on some discs (reading into it just yields a clean EOF),
        # while GetDiskFreeSpaceEx gives the exact mountable extent — including the UDF backup
        # anchor in the last sector — i.e. exactly what mounting the resulting .iso reproduces.
        length = None
        total = ctypes.c_ulonglong(0)
        if _k32.GetDiskFreeSpaceExW(f"{letter}:\\", None, ctypes.byref(total), None) and total.value:
            length = int(total.value)
            length -= length % SECTOR  # whole sectors only
        if not length:
            ln, lerr = _volume_length(h)   # fallback: raw device length
            if ln:
                length = ln
            else:
                info["error_code"] = lerr
                info["error"] = error_text(lerr)
                return info
        info["ok"] = length > 0
        info["length"] = length
        if not info["ok"]:
            info["error"] = "Reported a zero-length volume."
        return info
    finally:
        _k32.CloseHandle(h)


def free_space(path):
    """Free bytes available on the volume that would hold `path` (a dir or file path)."""
    try:
        d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)) or "."
        free = ctypes.c_ulonglong(0)
        if _k32.GetDiskFreeSpaceExW(d, ctypes.byref(free), None, None):
            return int(free.value)
    except Exception:
        pass
    return -1


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def human_size(n):
    if n is None or n < 0:
        return "—"
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024 or unit == "TiB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.2f} {unit}"
        f /= 1024


def human_rate(bps):
    if not bps or bps <= 0:
        return "—"
    return human_size(bps) + "/s"


def human_time(seconds):
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN guard
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# Worker thread — reads the source, writes the .iso, emits progress
# ---------------------------------------------------------------------------
class DiscImageWorker(QThread):
    """Images an optical volume (src_kind='volume', src=letter) or copies an existing
    image/file (src_kind='file', src=path) to dest_path, emitting progress."""
    progress = Signal(object)        # dict: bytes_done, bytes_total, rate_bps, eta_s, bad_sectors
    done = Signal(bool, str)         # success, message

    def __init__(self, src_kind, src, dest_path, total, skip_bad=False, verify=False, parent=None):
        super().__init__(parent)
        self.src_kind = src_kind
        self.src = src
        self.dest_path = dest_path
        self.total = int(total)
        self.skip_bad = bool(skip_bad)
        self.verify = bool(verify)
        self._stop = False
        self.bad_sectors = 0
        self._h = None
        self._fsrc = None
        self._buf = ctypes.create_string_buffer(CHUNK + 4096)
        self._aligned = (ctypes.addressof(self._buf) + 4095) & ~4095

    def request_stop(self):
        self._stop = True

    # ---- raw volume reads ----
    def _set_ptr(self, off):
        _k32.SetFilePointerEx(self._h, ctypes.c_longlong(off), None, FILE_BEGIN)

    def _read_sector(self, off):
        for _ in range(SECTOR_RETRIES):
            if self._stop:
                raise _Cancelled()
            self._set_ptr(off)
            got = wintypes.DWORD(0)
            ok = _k32.ReadFile(self._h, ctypes.c_void_p(self._aligned), SECTOR, ctypes.byref(got), None)
            if ok and got.value == SECTOR:
                return ctypes.string_at(self._aligned, SECTOR)
            time.sleep(0.02)
        return None

    def _recover(self, offset, length):
        """Bulk read failed — read sector-by-sector; zero-fill bad sectors if allowed."""
        out = bytearray()
        for i in range(length // SECTOR):
            sec = self._read_sector(offset + i * SECTOR)
            if sec is None:
                if self.skip_bad:
                    out += b"\x00" * SECTOR
                    self.bad_sectors += 1
                else:
                    raise IOError(f"Unreadable sector at offset {offset + i * SECTOR} "
                                  f"({human_size(offset + i * SECTOR)} into the disc).")
            else:
                out += sec
        return bytes(out)

    def _read_block_volume(self, offset, length):
        """Fill up to `length` bytes from `offset`, looping over legal partial reads
        (ReadFile may return fewer bytes than requested). Returns the bytes read — possibly
        fewer at a clean end-of-media. A hard error triggers sector-by-sector recovery."""
        out = bytearray()
        while len(out) < length:
            if self._stop:
                raise _Cancelled()
            pos = offset + len(out)
            want = length - len(out)
            self._set_ptr(pos)
            got = wintypes.DWORD(0)
            ok = _k32.ReadFile(self._h, ctypes.c_void_p(self._aligned), want, ctypes.byref(got), None)
            if ok and got.value > 0:
                out += ctypes.string_at(self._aligned, got.value)
            elif ok and got.value == 0:
                break  # clean end-of-media
            else:
                out += self._recover(pos, want)  # zero-fills bad sectors if allowed, else raises
        return bytes(out)

    def _read_block_file(self, offset, length):
        self._fsrc.seek(offset)
        data = self._fsrc.read(length)
        return data

    # ---- main loop ----
    def run(self):
        out = None
        hasher = hashlib.sha256() if self.verify else None
        win = deque()  # (t, bytes_done) sliding window for rate
        last_emit = 0.0
        try:
            if self.src_kind == 'volume':
                self._h, err = _open_raw(self.src, no_buffering=True)
                if not self._h:
                    self.done.emit(False, error_text(err))
                    return
                read_block = self._read_block_volume
            else:
                self._fsrc = open(self.src, 'rb', buffering=0)
                read_block = self._read_block_file

            out = open(self.dest_path, 'wb', buffering=0)
            offset = 0
            t0 = time.monotonic()
            win.append((t0, 0))
            while offset < self.total:
                if self._stop:
                    raise _Cancelled()
                to_read = min(CHUNK, self.total - offset)
                data = read_block(offset, to_read)
                if not data:
                    break  # clean end-of-media (expected exactly at the volume size)
                out.write(data)
                if hasher:
                    hasher.update(data)
                offset += len(data)

                now = time.monotonic()
                win.append((now, offset))
                while len(win) > 1 and now - win[0][0] > 3.0:
                    win.popleft()
                if now - last_emit >= 0.15 or offset >= self.total:
                    last_emit = now
                    dt = now - win[0][0]
                    rate = (offset - win[0][1]) / dt if dt > 0 else 0
                    eta = (self.total - offset) / rate if rate > 0 else -1
                    self.progress.emit({
                        "bytes_done": offset, "bytes_total": self.total,
                        "rate_bps": rate, "eta_s": eta, "bad_sectors": self.bad_sectors,
                    })

            out.flush()
            os.fsync(out.fileno())
            out.close()
            out = None
            msg = "Image created successfully."
            if self.bad_sectors:
                msg += f" {self.bad_sectors} unreadable sector(s) replaced with zeros."
            if hasher:
                msg += f"\nSHA-256: {hasher.hexdigest()}"
            self.done.emit(True, msg)

        except _Cancelled:
            self._discard(out)
            self.done.emit(False, "Cancelled — partial file deleted.")
        except Exception as e:
            self._discard(out)
            self.done.emit(False, f"Failed: {e}")
        finally:
            if self._h:
                _k32.CloseHandle(self._h)
                self._h = None
            if self._fsrc:
                try:
                    self._fsrc.close()
                except Exception:
                    pass
                self._fsrc = None

    def _discard(self, out):
        """Close and delete the partial destination file (cleanup on stop/error)."""
        try:
            if out:
                out.close()
        except Exception:
            pass
        try:
            if self.dest_path and os.path.exists(self.dest_path):
                os.remove(self.dest_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Themed loading animation — a Blu-ray disc "read" inside-out, with a rotating
# laser-sweep highlight. Deliberately NOT a horizontal progress bar.
# ---------------------------------------------------------------------------
class DiscFillAnimation(QWidget):
    """Optical disc that fills radially from the hub outward as progress advances
    (mirroring how the laser tracks an optical spiral), with a rotating specular
    sweep for liveliness and the percentage in the centre."""
    CYAN = QColor(0, 200, 255)
    BLUE = QColor(0, 130, 210)
    VIOLET = QColor(150, 90, 255)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frac = 0.0
        self._angle = 0.0
        self._done = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(33)  # ~30 fps

    def start(self):
        self._done = False
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        self._timer.stop()
        self.update()

    def set_progress(self, frac):
        self._frac = max(0.0, min(1.0, frac))
        self.update()

    def set_done(self, ok=True):
        self._done = ok
        self._frac = 1.0 if ok else self._frac
        self._timer.stop()
        self.update()

    def _tick(self):
        self._angle = (self._angle + 4.0) % 360.0
        self.update()

    @staticmethod
    def _annulus(cx, cy, r_in, r_out):
        pp = QPainterPath()
        pp.addEllipse(QPointF(cx, cy), r_out, r_out)
        pp.addEllipse(QPointF(cx, cy), r_in, r_in)
        pp.setFillRule(Qt.FillRule.OddEvenFill)
        return pp

    def paintEvent(self, event):
        p = QPainter(self)
        if not p.isActive():
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        Rmax = min(w, h) / 2.0 - 6.0      # outermost halo edge — kept INSIDE the widget so the
        if Rmax <= 16:                    # ring is a full smooth circle (never clipped to 4 sides)
            return
        R = Rmax * 0.82                   # disc radius; the soft halo lives between R and Rmax
        r_in = R * 0.34
        active = self._timer.isActive()

        # --- soft outer halo, fully within bounds → perfectly round, unbroken ---
        halo = QRadialGradient(cx, cy, Rmax)
        a = 80 if active else 42
        f0 = R / Rmax
        halo.setColorAt(max(0.0, f0 - 0.10), QColor(0, 200, 255, 0))
        halo.setColorAt((f0 + 1.0) / 2.0, QColor(0, 196, 255, a))
        halo.setColorAt(1.0, QColor(0, 196, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(QPointF(cx, cy), Rmax, Rmax)

        # --- disc body (dark, subtly glossy via off-centre radial) ---
        body = QRadialGradient(cx - R * 0.28, cy - R * 0.28, R * 1.7)
        body.setColorAt(0.0, QColor(44, 51, 64))
        body.setColorAt(0.55, QColor(22, 26, 35))
        body.setColorAt(1.0, QColor(10, 12, 18))
        p.setBrush(QBrush(body))
        p.drawEllipse(QPointF(cx, cy), R, R)

        # --- faint concentric grooves ---
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 12), 1.0))
        rings = 16
        for i in range(1, rings):
            rr = r_in + (R - r_in) * (i / rings)
            p.drawEllipse(QPointF(cx, cy), rr, rr)

        # --- "read" region: iridescent annulus filling from the hub outward ---
        outer = r_in + (R - r_in) * self._frac
        if outer > r_in + 0.5:
            p.save()
            p.setClipPath(self._annulus(cx, cy, r_in, outer))
            fill = QRadialGradient(cx, cy, R)
            fill.setColorAt(0.0, QColor(0, 225, 255, 235))
            fill.setColorAt(0.55, QColor(0, 150, 230, 215))
            fill.setColorAt(1.0, QColor(150, 95, 255, 205))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(fill))
            p.drawEllipse(QPointF(cx, cy), outer, outer)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(225, 250, 255, 235), 2.0))
            p.drawEllipse(QPointF(cx, cy), outer, outer)
            p.restore()

        # --- rotating specular sweep (the "laser") across the disc face ---
        if active:
            p.save()
            p.setClipPath(self._annulus(cx, cy, r_in, R))
            sweep = QConicalGradient(cx, cy, -self._angle)
            sweep.setColorAt(0.00, QColor(255, 255, 255, 0))
            sweep.setColorAt(0.05, QColor(255, 255, 255, 80))
            sweep.setColorAt(0.11, QColor(255, 255, 255, 0))
            sweep.setColorAt(1.00, QColor(255, 255, 255, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(sweep))
            p.drawEllipse(QPointF(cx, cy), R, R)
            p.restore()

        # --- glossy sheen highlight (upper-left) for a polished finish ---
        p.save()
        p.setClipPath(self._annulus(cx, cy, r_in, R))
        sheen = QRadialGradient(cx - R * 0.35, cy - R * 0.45, R * 1.1)
        sheen.setColorAt(0.0, QColor(255, 255, 255, 34))
        sheen.setColorAt(0.4, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(sheen))
        p.drawEllipse(QPointF(cx, cy), R, R)
        p.restore()

        # --- crisp outer rim ---
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 200, 255, 110), 1.4))
        p.drawEllipse(QPointF(cx, cy), R, R)

        # --- hub ring + centre hole ---
        p.setBrush(QBrush(QColor(14, 17, 24)))
        p.setPen(QPen(QColor(0, 205, 255, 200), 2.2))
        p.drawEllipse(QPointF(cx, cy), r_in, r_in)
        p.setBrush(QBrush(QColor(6, 7, 10)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r_in * 0.42, r_in * 0.42)

        # --- centre readout ---
        if self._done:
            p.setPen(QPen(QColor(60, 230, 160), 4.0, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            s = r_in * 0.5
            p.drawLine(QPointF(cx - s * 0.6, cy), QPointF(cx - s * 0.1, cy + s * 0.5))
            p.drawLine(QPointF(cx - s * 0.1, cy + s * 0.5), QPointF(cx + s * 0.7, cy - s * 0.5))
        else:
            p.setPen(QColor(238, 249, 255))
            f = QFont("Segoe UI", max(10, int(R * 0.19)), QFont.Weight.Bold)
            p.setFont(f)
            p.drawText(QRectF(cx - R, cy - R, R * 2, R * 2),
                       Qt.AlignmentFlag.AlignCenter, f"{int(self._frac * 100)}%")


# ---------------------------------------------------------------------------
# Configuration + progress dialog
# ---------------------------------------------------------------------------
_DIALOG_QSS = """
QDialog { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
          stop:0 #0e141f, stop:1 #080b11); }
QLabel { background: transparent; color: #c9d6e2; }
QLabel#title { color: #eaf6ff; font-size: 18px; font-weight: 800; }
QLabel#subtitle { color: #6f7d8c; font-size: 12px; }
QLabel#stat { color: #7c8b9a; font-size: 11px; }
QLabel#statval { color: #eaf6ff; font-size: 20px; font-weight: 800; }
QLabel#source { color: #c9d6e2; font-size: 13px; }
QFrame#card { background: rgba(90,170,255,0.045);
              border: 1px solid rgba(0,190,255,0.22); border-radius: 16px; }
QFrame#configCard { background: rgba(120,170,255,0.028);
              border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; }
QLineEdit { background: #0f141d; border: 1px solid #263243; border-radius: 9px;
            padding: 9px 11px; color: #eaf6ff; selection-background-color: #00b4f0; }
QLineEdit:focus { border: 1px solid #00b4f0; }
QCheckBox { background: transparent; color: #b9c6d4; spacing: 9px; }
QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px;
                       border: 1px solid #3a4655; background: #0f141d; }
QCheckBox::indicator:checked { background: #00b4f0; border: 1px solid #00b4f0; }
QPushButton { background: #182030; border: 1px solid #2b3849; border-radius: 10px;
              padding: 10px 20px; color: #d7e3ee; font-weight: 600; }
QPushButton:hover { border: 1px solid #3d8bdd; color: #ffffff; }
QPushButton:disabled { color: #56636f; border-color: #1e2733; background: #121822; }
QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                      stop:0 #16c0ff, stop:1 #0089cf); border: none; color: #00141f; }
QPushButton#primary:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                      stop:0 #36ccff, stop:1 #0a9ade); }
QPushButton#primary:disabled { background: #16202c; color: #56636f; }
QPushButton#danger { background: #2a1518; border: 1px solid #5a2a30; color: #ff9a9a; }
QPushButton#danger:hover { border: 1px solid #d8504f; color: #ffffff; }
"""


class DiscArchiveDialog(QDialog):
    """One window, two phases: (1) configure the copy, (2) show live throughput / ETA /
    size with the disc animation. Playback is locked by the host while a copy runs."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.worker = None
        self._source = None        # dict from main_window._resolve_archive_source()
        self._closing_after_stop = False
        self._locked = False

        self.setWindowTitle("Archive Blu-ray to ISO")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setStyleSheet(_DIALOG_QSS)

        self._build_ui()
        self._refresh_source()

    # ---- UI construction ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)

        title = QLabel("Archive Blu-ray to ISO image")
        title.setObjectName("title")
        sub = QLabel("Byte-perfect copy of the Blu-ray disc for archival.")
        sub.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(sub)

        # ---- CONFIG CARD ----
        self.config_page = QFrame()
        self.config_page.setObjectName("configCard")
        cfg = QVBoxLayout(self.config_page)
        cfg.setContentsMargins(18, 16, 18, 16)
        cfg.setSpacing(12)

        self.source_label = QLabel("Source: …")
        self.source_label.setObjectName("source")
        self.source_label.setWordWrap(True)
        cfg.addWidget(self.source_label)

        dest_row = QHBoxLayout()
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Destination .iso file…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        dest_row.addWidget(self.dest_edit, 1)
        dest_row.addWidget(browse)
        cfg.addLayout(dest_row)

        self.skip_cb = QCheckBox("Skip unreadable sectors (fill with zeros)")
        self.verify_cb = QCheckBox("Compute a SHA-256 checksum (archive verification)")
        self.verify_cb.setChecked(True)
        cfg.addWidget(self.skip_cb)
        cfg.addWidget(self.verify_cb)

        self.cfg_note = QLabel("")
        self.cfg_note.setObjectName("subtitle")
        self.cfg_note.setWordWrap(True)
        cfg.addWidget(self.cfg_note)
        root.addWidget(self.config_page)

        # ---- PROGRESS CARD ----
        self.progress_page = QFrame()
        self.progress_page.setObjectName("card")
        prg = QVBoxLayout(self.progress_page)
        prg.setContentsMargins(18, 18, 18, 18)
        prg.setSpacing(16)
        self.anim = DiscFillAnimation()
        self.anim.setMinimumSize(260, 260)
        prg.addWidget(self.anim, 1, Qt.AlignmentFlag.AlignCenter)

        stats = QGridLayout()
        stats.setHorizontalSpacing(26)
        self.val_rate = self._stat(stats, 0, "Read speed")
        self.val_eta = self._stat(stats, 1, "Time remaining")
        self.val_size = self._stat(stats, 2, "Copied / Total")
        prg.addLayout(stats)

        self.prg_note = QLabel("")
        self.prg_note.setObjectName("subtitle")
        self.prg_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prg_note.setWordWrap(True)
        prg.addWidget(self.prg_note)
        root.addWidget(self.progress_page)
        self.progress_page.hide()

        # ---- SHARED FOOTER ----
        footer = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_source)
        footer.addWidget(self.refresh_btn)
        footer.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.hide()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.hide()
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.start_btn)
        footer.addWidget(self.stop_btn)
        footer.addWidget(self.close_btn)
        root.addLayout(footer)

    def _stat(self, grid, col, caption):
        cap = QLabel(caption)
        cap.setObjectName("stat")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QLabel("—")
        val.setObjectName("statval")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(val, 0, col)
        grid.addWidget(cap, 1, col)
        return val

    # ---- source detection ----
    def _refresh_source(self):
        src = self.main_window._resolve_archive_source()
        self._source = src
        if not src or not src.get("found"):
            self.source_label.setText("Source: <span style='color:#ff9a9a'>no optical drive detected</span>")
            self.cfg_note.setText("Insert a Blu-ray into the drive, then click Refresh.")
            self.start_btn.setEnabled(False)
            return
        kind = src.get("kind")
        label = src.get("label") or ("ISO image" if kind == "file" else "Optical disc")
        where = src.get("drive") or src.get("iso_path") or ""
        if not src.get("ready"):
            self.source_label.setText(
                f"Source: <b>{label}</b> ({where}) — "
                f"<span style='color:#ff9a9a'>{src.get('error', 'not ready')}</span>")
            self.cfg_note.setText("Insert the disc and wait until it is ready, then Refresh.")
            self.start_btn.setEnabled(False)
            return
        total = src.get("length", 0)
        self.source_label.setText(
            f"Source: <b>{label}</b> ({where}) — <b>{human_size(total)}</b>"
            + ("  ·  already-mounted image (file copy)" if kind == "file" else ""))
        self.cfg_note.setText("")
        self.start_btn.setEnabled(True)
        if not self.dest_edit.text().strip():
            safe = "".join(ch for ch in (label or "BluRay") if ch.isalnum() or ch in " -_").strip() or "BluRay"
            default_dir = os.path.expanduser("~")
            self.dest_edit.setText(os.path.join(default_dir, safe + ".iso"))

    def _browse(self):
        start = self.dest_edit.text().strip() or os.path.expanduser("~/BluRay.iso")
        path, _ = QFileDialog.getSaveFileName(self, "Save the ISO image", start,
                                              "Disc image (*.iso)")
        if path:
            if not path.lower().endswith(".iso"):
                path += ".iso"
            self.dest_edit.setText(path)

    # ---- start / progress / stop ----
    def _start(self):
        src = self._source
        if not src or not src.get("ready"):
            self._refresh_source()
            return
        dest = self.dest_edit.text().strip()
        if not dest:
            QMessageBox.warning(self, "Destination", "Choose a destination location.")
            return
        if not dest.lower().endswith(".iso"):
            dest += ".iso"
            self.dest_edit.setText(dest)
        ddir = os.path.dirname(os.path.abspath(dest))
        if not os.path.isdir(ddir):
            QMessageBox.warning(self, "Destination", "The destination folder does not exist.")
            return
        total = src.get("length", 0)
        # don't write the image onto the source disc, and check free space
        if os.path.splitdrive(os.path.abspath(dest))[0].rstrip(":").upper() == (src.get("drive") or "").upper():
            QMessageBox.warning(self, "Destination", "Choose a destination other than the source disc.")
            return
        fs = free_space(dest)
        if 0 <= fs < total:
            if QMessageBox.question(
                    self, "Disk space",
                    f"Free space ({human_size(fs)}) is less than the image size "
                    f"({human_size(total)}). Continue anyway?") != QMessageBox.StandardButton.Yes:
                return
        if os.path.exists(dest):
            if QMessageBox.question(self, "Overwrite",
                                    "The file already exists. Replace it?") != QMessageBox.StandardButton.Yes:
                return

        # lock playback (host stops the player + blocks new playback)
        try:
            self.main_window._begin_archive_lock()
            self._locked = True
        except Exception:
            pass

        self.config_page.hide()
        self.progress_page.show()
        self.refresh_btn.hide()
        self.cancel_btn.hide()
        self.start_btn.hide()
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("Stop")
        self.stop_btn.show()
        self.adjustSize()
        self.anim.set_progress(0.0)
        self.anim.start()
        self.prg_note.setText("")

        self.worker = DiscImageWorker(
            src.get("kind", "volume"),
            src.get("drive") if src.get("kind") != "file" else src.get("iso_path"),
            dest, total, skip_bad=self.skip_cb.isChecked(), verify=self.verify_cb.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_progress(self, st):
        total = st.get("bytes_total", 1) or 1
        done = st.get("bytes_done", 0)
        self.anim.set_progress(done / total)
        self.val_rate.setText(human_rate(st.get("rate_bps", 0)))
        self.val_eta.setText(human_time(st.get("eta_s", -1)))
        self.val_size.setText(f"{human_size(done)} / {human_size(total)}")
        bad = st.get("bad_sectors", 0)
        if bad:
            self.prg_note.setText(f"⚠ {bad} unreadable sector(s) replaced with zeros")

    def _on_done(self, ok, msg):
        self.anim.set_done(ok)
        if self._locked:
            try:
                self.main_window._end_archive_lock()
            except Exception:
                pass
            self._locked = False
        self.worker = None
        self.stop_btn.hide()
        self.close_btn.show()
        self.val_eta.setText("—")
        if ok:
            self.val_rate.setText("Done")
        self.prg_note.setText(msg)
        if self._closing_after_stop:
            self.accept()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping…")
            self.prg_note.setText("Stopping, cleaning up the partial file…")
            self.worker.request_stop()

    # ---- closing safety ----
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Copy in progress",
                                    "A copy is in progress. Stop it and close?") \
                    != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._closing_after_stop = True
            self.worker.request_stop()
            event.ignore()  # _on_done will accept() once cleanup finishes
            return
        if self._locked:
            try:
                self.main_window._end_archive_lock()
            except Exception:
                pass
            self._locked = False
        super().closeEvent(event)
