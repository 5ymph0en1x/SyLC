# trt_runtime.py
"""GPU detection and TensorRT-runtime status, for the depth-model dialog.

STDLIB ONLY, and NO Qt import -- the same rule as model_fetcher.py, for the same
two reasons: this module is compiled into the frozen player, so a third-party
dependency here would have to be bundled too; and keeping Qt out means the whole
detection/status path is testable without a QApplication. The Qt layer lives in
model_download_dialog.py.

The GPU architecture is DETECTED, never asked. It selects exactly one thing --
the per-architecture kernel blob inside the TensorRT wheel -- and a wrong answer
costs either a failed 70-minute engine compile or a runtime that builds nothing.

`runtime_status` REPRODUCES the gate in `SyLC_3D_Player._synth3d_ort_dir()`; it
is deliberately not a second, more permissive opinion about what counts as a
usable TensorRT install. An incomplete or unverified assembly does not fail
gracefully -- it takes the process down with a hard native abort during the
first real engine build, which is the whole reason the `.trt_verified` marker
exists. If the two ever disagree, this module must be the one that is wrong.
"""
from __future__ import annotations

import ctypes
import glob
import os
from dataclasses import dataclass

# CUresult. The CUDA driver API returns 0 for success and never raises; every
# call below is checked, because "no NVIDIA driver on this machine" is an
# ordinary case for a video player, not an error worth propagating.
CUDA_SUCCESS = 0

# CUdevice_attribute values from cuda.h. 75/76 are the compute-capability major
# and minor -- the only two attributes this module needs, and the reason it can
# talk to the driver directly instead of depending on a CUDA toolkit.
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

# cuDeviceGetName truncates to the buffer it is given. Real names are ~30 bytes
# ("NVIDIA GeForce RTX 4090"); 256 is what the CUDA samples use.
_NAME_BYTES = 256

# The version of the pinned TensorRT wheel, for the message shown to a user
# whose card is too old. See tensorrt-cu13==10.16.1.11.
TENSORRT_VERSION = "10.16"

# Read out of the pinned wheel's OWN central directory -- these are the
# `nvinfer_builder_resource_sm<N>_10.dll` blobs actually shipped inside
# tensorrt_cu13_libs-10.16.1.11-py3-none-win_amd64.whl, not a list copied from
# documentation that a later wheel could quietly contradict:
#
#   75  RTX 20 series, GTX 16 series
#   80  A100
#   86  RTX 30 series
#   89  RTX 40 series
#   90  H100
#   100 Blackwell datacenter
#   120 RTX 50 series
#
# A card outside this set has no builder blob, so TensorRT can compile nothing
# for it -- the PTX JIT fallback is not a path this feature offers.
SUPPORTED_SM = {75, 80, 86, 89, 90, 100, 120}

# Status.state values. Strings rather than an Enum so a caller (and a test) can
# compare against a literal without importing anything.
NO_GPU = "no_gpu"
UNSUPPORTED_SM = "unsupported_sm"
NOT_INSTALLED = "not_installed"
INCOMPLETE = "incomplete"
UNVERIFIED = "unverified"
READY = "ready"


@dataclass(frozen=True)
class GpuInfo:
    name: str   # verbatim from cuDeviceGetName, e.g. "NVIDIA GeForce RTX 4090"
    sm: int     # compute capability as major*10+minor, e.g. 89

    @property
    def short_name(self):
        """The name with the vendor words the row cannot spare dropped.

        "NVIDIA GeForce RTX 4090" is 23 characters of which 14 are the same on
        every consumer card, and the row that carries it also carries a status
        and an sm number. The full name stays on `name` for logs.
        """
        short = self.name
        for prefix in ("NVIDIA ", "GeForce "):
            if short.startswith(prefix):
                short = short[len(prefix):]
        return short or self.name


@dataclass(frozen=True)
class Status:
    state: str            # one of the six constants above
    message: str          # human sentence, WITHOUT a "TensorRT — " prefix:
                          # the dialog owns its row's label, as it does for the
                          # model packs.
    gpu: GpuInfo | None   # None only when detection found nothing

    @property
    def ready(self):
        return self.state == READY


def detect_gpu():
    """The first CUDA device, or None -- never an exception.

    `nvcuda.dll` ships with every NVIDIA driver, so its absence is exactly the
    signal we want: no driver, no TensorRT, nothing to offer. Any other failure
    (a driver too old for cuInit, a device that refuses an attribute query) is
    reported the same way, because the only decision downstream is whether to
    offer the runtime at all.

    The FIRST device, deliberately: ORT's TensorRT execution provider defaults
    to device 0, so device 0 is the one whose architecture decides which kernel
    blob has to be fetched. Asking about any other card would describe hardware
    the engine build will not use.

    Tests monkeypatch this whole function -- `runtime_status` looks it up
    through the module globals on every call -- so the status logic is testable
    on a machine with no GPU at all.
    """
    # ONE try around the whole body, ending in a bare `except Exception`. The
    # expected failures are an OSError from the DLL load (no NVIDIA driver
    # installed -- the ordinary case for most users of this player) and an
    # AttributeError off Windows, where `ctypes.WinDLL` does not exist at all;
    # but the guarantee this function makes is unconditional, because it is
    # called from the "Depth models…" dialog's CONSTRUCTOR. An exception
    # escaping here would take that whole dialog down, including the model
    # downloads, which have nothing to do with TensorRT.
    try:
        # WinDLL, not CDLL: the CUDA driver API is stdcall on 32-bit Windows.
        driver = ctypes.WinDLL("nvcuda.dll")
        if driver.cuInit(0) != CUDA_SUCCESS:
            # A driver present but unusable: too old for this entry point, a
            # user-mode/kernel-mode version mismatch after an update, or a
            # machine with no CUDA-capable device at all.
            return None
        count = ctypes.c_int(0)
        if driver.cuDeviceGetCount(ctypes.byref(count)) != CUDA_SUCCESS:
            return None
        if count.value < 1:
            # A driver with no device: an NVIDIA driver left behind after the
            # card was removed, or a headless/disabled adapter.
            return None
        device = ctypes.c_int(0)
        if driver.cuDeviceGet(ctypes.byref(device), 0) != CUDA_SUCCESS:
            return None
        buffer = ctypes.create_string_buffer(_NAME_BYTES)
        if driver.cuDeviceGetName(buffer, _NAME_BYTES, device) != CUDA_SUCCESS:
            return None
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        for attribute, out in (
                (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, major),
                (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, minor)):
            if driver.cuDeviceGetAttribute(
                    ctypes.byref(out), attribute, device) != CUDA_SUCCESS:
                return None
        name = buffer.value.decode("utf-8", "replace").strip()
        return GpuInfo(name=name, sm=major.value * 10 + minor.value)
    except Exception:
        # A missing export or a changed signature in some future driver lands
        # here. "No GPU" is the right answer to all of it: the only decision
        # downstream is whether to offer the runtime at all.
        return None


def marker_attests(marker, model_path):
    """True iff a `.trt_verified` marker covers the graph in question.

    A byte-for-byte mirror of `SyLC_3D_Player.synth3d_marker_attests`, which
    cannot be imported here (it lives in the Qt-importing player module). Both
    must stay identical; see this module's docstring for which one yields.

    TensorRT's engine cache is keyed PER GRAPH, so a marker written for
    `da3_base_756.onnx` says nothing about a 518 preset. A marker that names NO
    graph (unreadable, or written by a probe from before model names were
    recorded) attests the directory as a whole: its presence still proves a real
    engine build succeeded against these DLLs. Absent information is not
    negative information.
    """
    try:
        with open(marker, "r", encoding="utf-8", errors="replace") as handle:
            probed = [line.split("=", 1)[1].strip() for line in handle
                      if line.startswith("probe_model=")]
    except OSError:
        return True
    if not probed:
        return True
    return os.path.basename(model_path).lower() in {n.lower() for n in probed}


def runtime_status(ort_dir, model_path=None):
    """What the TensorRT row should say about `ort_dir`.

    `ort_dir` is the candidate TensorRT directory (`ort_tensorrt`), NOT the
    directory the player ends up loading onnxruntime.dll from: the DirectML
    fallback root also holds an onnxruntime.dll, and calling that "an incomplete
    TensorRT install" would be a false statement about a perfectly good default
    install. None means "no runtime directory known".

    `model_path` is the graph the user would actually open. Omitting it checks
    presence and freshness only -- the round-3 behaviour that
    `_synth3d_ort_dir(None)` still has.
    """
    gpu = detect_gpu()
    if gpu is None:
        return Status(NO_GPU, "no NVIDIA GPU detected", None)
    if gpu.sm not in SUPPORTED_SM:
        return Status(
            UNSUPPORTED_SM,
            f"sm{gpu.sm} is not supported by TensorRT {TENSORRT_VERSION}", gpu)

    card = f" ({gpu.short_name}, sm{gpu.sm})"
    if not ort_dir or not os.path.isdir(ort_dir):
        return Status(NOT_INSTALLED, "not installed" + card, gpu)

    # The first two conditions of _synth3d_ort_dir()'s gate: onnxruntime.dll AND
    # at least one nvinfer*.dll must COEXIST, so a half-populated staging
    # leftover is never picked. Split three ways here only to say something
    # more useful than "not installed" about a directory that is half there.
    has_ort = os.path.exists(os.path.join(ort_dir, "onnxruntime.dll"))
    has_nvinfer = bool(glob.glob(os.path.join(ort_dir, "nvinfer*.dll")))
    if not has_ort and not has_nvinfer:
        return Status(NOT_INSTALLED, "not installed" + card, gpu)
    if not (has_ort and has_nvinfer):
        missing = "onnxruntime.dll" if not has_ort else "nvinfer_10.dll"
        return Status(
            INCOMPLETE,
            f"incomplete install — {missing} is missing" + card, gpu)

    # Condition three: the marker exists at all. It is written ONLY after a real
    # engine build plus an inference that passes the value gate -- never on DLL
    # presence -- because that build is precisely what can abort the process.
    marker = os.path.join(ort_dir, ".trt_verified")
    if not os.path.exists(marker):
        return Status(
            UNVERIFIED, "installed but not verified — no engine has been "
                        "built and tested here yet" + card, gpu)

    # Condition four, FRESHNESS: any *.dll newer than the marker means the DLLs
    # were replaced without a matching re-verification, so the marker no longer
    # describes what is on disk and counts as absent. `>=`, matching the player
    # exactly: a marker written in the same filesystem tick as the last DLL copy
    # is the normal successful outcome, not a stale one.
    dlls = glob.glob(os.path.join(ort_dir, "*.dll"))
    newest_dll_mtime = max((os.path.getmtime(p) for p in dlls), default=0.0)
    if os.path.getmtime(marker) < newest_dll_mtime:
        return Status(
            UNVERIFIED, "installed but not verified — the runtime files "
                        "changed after the last verification" + card, gpu)

    # Condition five: the marker must attest THIS graph. Opening a graph the
    # probe never built would pay a full cold engine compile on the first
    # in-playback enable -- minutes -- which is the wait the offline probe
    # exists to absorb, and this feature never builds an engine during playback.
    if model_path is not None and not marker_attests(marker, model_path):
        return Status(
            UNVERIFIED, "installed but not verified for the selected depth "
                        "model" + card, gpu)

    # Two wordings for one state, and the difference is not cosmetic. "active"
    # is a claim about a GRAPH: the player picks this runtime per graph, so it
    # is only true once a model_path has been checked against the marker.
    # A caller that named no graph has verified the DIRECTORY -- presence and
    # freshness -- and must not promise more than it checked, or the row would
    # read "active" for a preset the player will silently run on DirectML.
    return Status(
        READY,
        ("active" if model_path is not None else "installed and verified")
        + card, gpu)
