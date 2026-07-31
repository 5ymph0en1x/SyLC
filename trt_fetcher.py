# trt_fetcher.py
"""Acquires the opt-in TensorRT runtime into a directory, without a toolchain.

STDLIB ONLY, and NO Qt import -- the same rule as model_fetcher.py and
trt_runtime.py, for the same two reasons: this module is compiled into the
frozen player, so a third-party dependency here would have to be bundled too;
and keeping Qt out means the whole download/extract/verify path is testable
without a QApplication. The Qt layer lives in model_download_dialog.py.

It also cannot shell out to `uv`, `pip`, or a scratch venv, which is how
`tools_dev/setup_tensorrt.py` does the same job: none of the three exists in a
portable ZIP install, and `tools_dev/` is not shipped at all. What makes that
avoidable is two properties, both verified against the live services rather than
assumed:

  * `https://pypi.nvidia.com/<package>/` is a plain PEP 503 simple index --
    readable with `urllib`, listing the `.whl` URLs directly. The hrefs are
    RELATIVE, so they must be resolved with `urllib.parse.urljoin`.
  * a wheel is a zip, a `.nupkg` is a zip, and both servers honour HTTP `Range`
    (206). So the central directory can be read from the tail and ONE member
    pulled out of the middle. Enumerating the 1.78 GiB TensorRT wheel costs
    three range requests and about 2 KB.

That second property is what makes this worth its complexity. The TensorRT
wheel ships EIGHT architecture blobs -- sm75, sm80, sm86, sm89, sm90, sm100,
sm120 and a ptx JIT fallback -- and only three architecture-independent DLLs.
Fetching the three plus the ONE blob the detected GPU needs is ~354 MiB over
the wire instead of 1.78 GiB.

--- THE PACKAGE NAMES, WHICH ARE ALL TRAPS ---

`tensorrt-cu13` (what setup_tensorrt.py pins) is a ~15 KB `wheel_stub` SDIST
whose PEP 517 build step downloads the real payload at build time. That
indirection is the entire reason setup_tensorrt.py needs uv. The real payload
is **`tensorrt-cu13-libs`**, which is published on the NVIDIA index as an
ordinary wheel and is what this module reads. Same pinned version, no build
step.

`nvidia-cublas-cu13` does not exist -- it is deprecated upstream in favour of
plain **`nvidia-cublas`**, and the NVIDIA index answers 404 for the suffixed
name (checked). setup_tensorrt.py gets cuBLAS as a transitive dependency of
cuDNN; nothing resolves dependencies for us here, so it is named explicitly.

--- WHY THE VERSIONS ARE PINNED THE WAY THEY ARE ---

TensorRT is pinned EXACTLY, and must stay that way: this ORT build's
`onnxruntime_providers_tensorrt.dll` imports `nvinfer_10.dll` by literal
filename, and the Windows loader matches static imports by name, not by "close
enough" version. TensorRT 11.x ships `nvinfer_11.dll` and does not satisfy it,
no matter how complete the install is.

cuDNN and cuBLAS are pinned too, and the versions are not arbitrary: they are
the ones the engine probe actually passed with. `uv pip install
nvidia-cudnn-cu13` took whatever was newest on the day of that run, and
resolving "newest" again today does NOT reproduce it -- cuDNN has published
9.25.0.15 since, so an unpinned resolver hands the user a combination no engine
build has ever been run against. That is the one thing this whole feature is
built to avoid. The pinned set was confirmed by fetching it and comparing
against the verified install: all four ORT DLLs, all four TensorRT files and
both cuBLAS DLLs came back BYTE-IDENTICAL.

Clear a pin to `None` and the resolver falls back to the newest release whose
MAJOR matches the soname needed (`cudnn64_9.dll` -> cuDNN 9.x,
`cublas64_13.dll` -> cuBLAS 13.x) -- never merely the newest, because a cuDNN
10 would break by exact filename the same way TensorRT 11 does. Either way the
resolved archive's own central directory is then checked for the exact
filenames wanted, so a layout or naming change fails loudly here rather than
producing a runtime that is missing a DLL.

To re-derive a pin after a deliberate upgrade: fetch each 9.x/13.x wheel's
central directory and compare its `file_size` per DLL against a runtime
directory that has passed `--engine-probe`. That is how 9.24.0.43 was
identified -- 10/10 sizes matched, against 3/10 for 9.25.0.15.

--- WHY THE ASSEMBLY IS ATOMIC ---

A half-populated TensorRT directory does not fail gracefully. It takes the
process down with a hard native abort during the first real engine build --
that crash is what the `.trt_verified` marker exists to prevent, and
setup_tensorrt.py's header documents it. So every member is extracted into
`<target>.partial`, each through a `.part` file of its own, and the staging
directory is renamed into place only once every planned file is present at its
exact size. The same `.part`-then-replace discipline as
`model_fetcher._download_one`.

**This module never writes `.trt_verified`.** What it produces is an
UNVERIFIED runtime, which `trt_runtime.runtime_status` will correctly report as
`unverified` -- the honest state until a real engine build has passed the
near/far value gate. Writing that marker without a build would defeat the whole
safety design.
"""
from __future__ import annotations

import contextlib
import io
import os
import posixpath
import re
import shutil
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass

import trt_runtime

# 1 MiB, as in model_fetcher: large enough that the per-chunk progress callback
# and cancel check are free relative to the socket read, small enough that
# Cancel feels immediate.
CHUNK = 1 << 20

# The retry shape is model_fetcher's, and for the same reason: a response that
# ends early with no exception at all was measured against a real CDN roughly
# once in ten transfers, and "start the 1.6 GB over" is the wrong answer to a
# fault that does not reproduce.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.5
RETRY_POLL_S = 0.1
READ_TIMEOUT_S = 60

# Deliberately SEQUENTIAL, unlike model_fetcher's six-way pool. That pool exists
# because HuggingFace throttles per connection -- measured, 1.3 MiB/s on one
# socket against 22.4 MiB/s on eight. Neither service used here does: a single
# stream measured 52.4 MiB/s from pypi.nvidia.com and 36.9 MiB/s from
# globalcdn.nuget.org, which puts the whole ~1.1 GiB transfer near half a minute
# on a fast link. A thread pool would buy nothing and would have to share one
# `_RangeFile` per archive or open several, both of which cost more than they
# save here.

NVIDIA_INDEX = "https://pypi.nvidia.com/{package}/"

# NOT `tensorrt-cu13` -- see the header. That name is the wheel-stub sdist.
TENSORRT_LIBS_PACKAGE = "tensorrt-cu13-libs"
TENSORRT_VERSION = "10.16.1.11"

CUDNN_PACKAGE = "nvidia-cudnn-cu13"
CUDNN_MAJOR = 9                  # cudnn64_9.dll -- the soname, not a preference
CUDNN_VERSION = "9.24.0.43"      # None = newest release with that major

CUBLAS_PACKAGE = "nvidia-cublas"
CUBLAS_MAJOR = 13                # cublas64_13.dll
CUBLAS_VERSION = "13.6.0.2"

WHEEL_PLATFORM = "win_amd64"

# ORT comes from NuGet, and the versionless URL redirects to a ~325 KB META
# package (just .props/.targets) whose nuspec pins an EXACT-version dependency
# on the real platform package. Reading that pin, rather than independently
# resolving "latest", is what keeps onnxruntime.dll and
# onnxruntime_providers_tensorrt.dll from ever coming out of different builds.
# Ported from tools_dev/setup_tensorrt.py's _nuget_meta_version_and_dep.
NUGET_META_ID = "Microsoft.ML.OnnxRuntime.Gpu"
NUGET_WINDOWS_ID = "Microsoft.ML.OnnxRuntime.Gpu.Windows"
NUGET_VERSIONLESS_URL = "https://www.nuget.org/api/v2/package/{id}"
NUGET_VERSIONED_URL = "https://www.nuget.org/api/v2/package/{id}/{version}"

# runtimes/win-x64/native/ in the NuGet package. `onnxruntime_providers_cuda.dll`
# is 255 MiB and the TensorRT EP does not import it by name -- but it IS present
# in the assembly that was actually verified end to end, and setup_tensorrt.py's
# probe checks that it loads. Dropping it would be an untested deviation from a
# known-good configuration whose failure mode is a hard native abort rather than
# an error, which is not a trade worth 255 MiB.
ORT_DLLS = ("onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_tensorrt.dll",
            "onnxruntime_providers_cuda.dll")

# tensorrt_libs/ in the wheel. Architecture-INDEPENDENT: everything else in that
# 1.78 GiB is the eight per-architecture builder blobs.
TENSORRT_DLLS = ("nvinfer_10.dll",
                 "nvinfer_plugin_10.dll",
                 "nvonnxparser_10.dll")

# The one blob that depends on the detected GPU. TensorRT needs it to BUILD an
# engine for that compute capability; it is not touched during provider
# registration, which is exactly why an assembly missing it looks fine right up
# until the first real build.
BUILDER_RESOURCE = "nvinfer_builder_resource_sm{sm}_10.dll"

# nvidia/cudnn/bin/ in the wheel. cudnn64_9.dll is only cuDNN 9's 267 KB
# front-end DISPATCHER; it loads these backend engine DLLs ON DEMAND at the
# first real cudnnCreate(), i.e. at engine-build time. Shipping the dispatcher
# alone is the round-3 crash setup_tensorrt.py's header documents.
CUDNN_DLLS = ("cudnn64_9.dll",
              "cudnn_ops64_9.dll",
              "cudnn_cnn64_9.dll",
              "cudnn_adv64_9.dll",
              "cudnn_graph64_9.dll",
              "cudnn_heuristic64_9.dll",
              "cudnn_engines_precompiled64_9.dll",
              "cudnn_engines_runtime_compiled64_9.dll",
              "cudnn_engines_tensor_ir64_9.dll",
              "cudnn_ext64_9.dll")

# nvidia/cu13/bin/x86_64/ in the wheel. nvblas64_13.dll sits beside them and is
# deliberately NOT taken: it is a BLAS-replacement shim nothing in this runtime
# imports.
CUBLAS_DLLS = ("cublas64_13.dll", "cublasLt64_13.dll")

# Appended to the target directory to form the staging directory. A sibling, not
# a subdirectory, and not %TEMP%: the promotion has to be a same-volume rename,
# and a subdirectory would put a half-built assembly INSIDE the directory the
# player probes.
STAGING_SUFFIX = ".partial"

# Where the previous install is parked for the instant between the two renames.
RETIRED_SUFFIX = ".old"


class TrtFetchError(RuntimeError):
    """The runtime could not be assembled. `ort_tensorrt` is untouched."""


class TrtFetchCancelled(Exception):
    """The caller's cancel event fired. Staging is kept, so a rerun resumes."""


class NotEnoughSpace(TrtFetchError):
    """The destination volume cannot hold the runtime."""


@dataclass(frozen=True)
class Member:
    """One file to pull out of one archive."""
    name: str            # basename as it lands in the runtime directory
    archive_path: str    # full path inside the archive
    size: int            # bytes ON DISK once inflated
    download_size: int   # bytes OVER THE WIRE, i.e. the deflated size


@dataclass(frozen=True)
class Source:
    key: str             # "ort", "tensorrt", "cudnn", "cublas"
    label: str           # human, with the resolved version in it
    url: str
    members: tuple


@dataclass(frozen=True)
class Plan:
    """Exactly which bytes will be fetched, resolved but not yet transferred.

    Separate from the transfer so a dialog can state the real size before
    asking, and so the resolution logic is testable without moving a gigabyte.
    """
    sm: int
    sources: tuple

    @property
    def members(self):
        return tuple(m for source in self.sources for m in source.members)

    @property
    def size(self):
        """Bytes the assembled runtime occupies on disk."""
        return sum(m.size for m in self.members)

    @property
    def download_size(self):
        """Bytes that cross the wire. Smaller -- every member is deflated."""
        return sum(m.download_size for m in self.members)


def staging_dir(target_dir):
    """Where a partial assembly lives. Public: the dialog reports on it."""
    return target_dir.rstrip("\\/") + STAGING_SUFFIX


# --- HTTP plumbing ---------------------------------------------------------

def _request(opener, url, headers=None):
    return opener(urllib.request.Request(url, headers=headers or {}),
                  timeout=READ_TIMEOUT_S)


@contextlib.contextmanager
def _as_fetch_error():
    """Makes the two public entry points raise ONE family of exception.

    Deliberately at the public boundary and not inside `_request`: the reconnect
    loop in `_RangeFile.read` catches OSError to decide what is worth retrying,
    and converting there would turn every recoverable blip into an immediate
    failure.

    Without this, an offline user clicking Install gets a raw `URLError` out of
    `_RangeFile.__init__` -- an OSError, from a module whose documented error
    type is `TrtFetchError` -- and the dialog would have to know that to avoid
    an unhandled exception. Nothing here changes what is on disk; it changes
    only what the caller has to catch.
    """
    try:
        yield
    except (TrtFetchError, TrtFetchCancelled):
        # NotEnoughSpace is a TrtFetchError, so it passes through here too.
        raise
    except OSError as exc:
        raise TrtFetchError(str(exc) or exc.__class__.__name__) from exc


def _backoff(cancel, seconds):
    """Waits between attempts in slices, so Cancel is honoured during the wait.

    Blocking the whole interval would make Cancel land only after the next
    member -- minutes, for a 442 MiB DLL.
    """
    deadline = time.monotonic() + seconds
    while True:
        if cancel is not None and cancel.is_set():
            raise TrtFetchCancelled("cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(RETRY_POLL_S, remaining))


class _RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP `Range`, to hand to `zipfile`.

    Deliberately a FILE rather than a zip parser. The alternative -- reading the
    central directory by hand, then the local header, then inflating -- means
    reimplementing the two traps in that format: a local file header's name and
    extra-field lengths can differ from the central directory's, so the data
    offset cannot be computed from the central entry alone; and members may be
    STORED (method 0) as well as deflated. `zipfile` already handles both, plus
    zip64, plus -- the part that matters most here -- it verifies each member's
    CRC-32 as it inflates and raises `BadZipFile` on a mismatch. Handing it a
    file object gets all of that for free, and the cost is this class.

    The access pattern zipfile produces is almost entirely sequential (tail,
    then the whole central directory, then one forward run per member), so a
    single live response is kept and reused whenever the next read continues
    where the last one stopped. A backward seek costs one new request.

    A connection that drops mid-member is reopened at the current offset and the
    read continues. That is the resume that matters during a run: the inflater's
    state lives in this process and simply keeps being fed. Resume ACROSS runs
    is per member instead, via the staging directory -- deflate state cannot be
    persisted, so there is nothing to resume a half-inflated member from.
    """

    def __init__(self, url, opener, cancel=None):
        self._url = url
        self._opener = opener
        self._cancel = cancel
        self._pos = 0
        self._response = None
        self._response_pos = -1
        self.served = 0       # wire bytes handed out, for progress accounting
        self.requests = 0
        self.size = self._probe_size()

    # -- the file protocol zipfile needs

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, self._pos)
        return self._pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self._pos
        size = min(size, self.size - self._pos)
        if size <= 0:
            return b""
        out = bytearray()
        attempts = 0
        while len(out) < size:
            if self._cancel is not None and self._cancel.is_set():
                raise TrtFetchCancelled(self._url)
            try:
                if self._response is None or self._response_pos != self._pos:
                    self._reopen()
                block = self._response.read(size - len(out))
            except TrtFetchCancelled:
                raise
            except OSError as exc:
                # A reset, a read timeout, a failed reconnect. Every one of them
                # is an OSError, and every one is worth another attempt from the
                # offset already reached rather than from zero.
                self._drop_response()
                attempts += 1
                if attempts >= MAX_ATTEMPTS:
                    raise TrtFetchError(f"{self._url}: {exc}") from exc
                _backoff(self._cancel, RETRY_BACKOFF_S)
                continue
            if not block:
                # A clean EOF short of what the central directory promised: no
                # exception, just a body that stopped. Reconnect and ask for the
                # rest.
                self._drop_response()
                attempts += 1
                if attempts >= MAX_ATTEMPTS:
                    raise TrtFetchError(
                        f"{self._url}: connection ended {size - len(out)} bytes "
                        f"early at offset {self._pos}")
                _backoff(self._cancel, RETRY_BACKOFF_S)
                continue
            out += block
            self._pos += len(block)
            self._response_pos = self._pos
            self.served += len(block)
            # Reset on PROGRESS, not per read: a long member crossing several
            # unrelated blips must not be killed by a cumulative count, while a
            # link that has stopped delivering entirely still dies after
            # MAX_ATTEMPTS.
            attempts = 0
        return bytes(out)

    def close(self):
        self._drop_response()
        super().close()

    # -- internals

    def _probe_size(self):
        """Total length, from the Content-Range of a one-byte request.

        A server that answers 200 here is answering with the WHOLE archive, and
        every later ranged read would silently be served from offset 0 -- which
        is not a slow path, it is a corrupt one. Refuse instead.
        """
        with _request(self._opener, self._url, {"Range": "bytes=0-0"}) as response:
            if getattr(response, "status", 200) != 206:
                raise TrtFetchError(
                    f"{self._url}: server ignored an HTTP Range request "
                    f"(status {getattr(response, 'status', '?')}); this "
                    f"acquisition reads single members out of large archives "
                    f"and cannot work without ranged reads")
            headers = getattr(response, "headers", None)
            content_range = headers.get("Content-Range") if headers else None
        if not content_range or "/" not in content_range:
            raise TrtFetchError(
                f"{self._url}: 206 response carried no usable Content-Range")
        try:
            return int(content_range.rsplit("/", 1)[1])
        except ValueError as exc:
            raise TrtFetchError(
                f"{self._url}: unparsable Content-Range {content_range!r}") from exc

    def _reopen(self):
        self._drop_response()
        response = _request(self._opener, self._url,
                            {"Range": f"bytes={self._pos}-"})
        status = getattr(response, "status", 200)
        if self._pos and status != 206:
            response.close()
            raise TrtFetchError(
                f"{self._url}: server ignored Range at offset {self._pos} "
                f"(status {status}) -- the bytes it is sending are not the "
                f"bytes that were asked for")
        self._response = response
        self._response_pos = self._pos
        self.requests += 1

    def _drop_response(self):
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None
            self._response_pos = -1


# --- index and version resolution ------------------------------------------

_HREF = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def _index_wheels(package, opener):
    """[(version, absolute_url)] for `package`'s win_amd64 wheels.

    The hrefs on a PEP 503 page are RELATIVE and carry a `#sha256=` fragment.
    The fragment is the hash of the WHOLE archive, which is of no use here --
    nothing ever downloads a whole archive. Per-member CRC-32 out of the central
    directory is what integrity rests on instead; see `_extract_member`.
    """
    url = NVIDIA_INDEX.format(package=package)
    with _request(opener, url) as response:
        html = response.read().decode("utf-8", "replace")
        # The response's own URL, when it has one: a redirect would make the
        # requested URL the wrong base for a relative href.
        base = getattr(response, "url", None) or url
    wheels = []
    for href in _HREF.findall(html):
        # defrag: the `#sha256=` is dropped rather than carried, so the URL this
        # module reports in a message is the URL it actually requested --
        # `urllib.request` splits the fragment off anyway.
        absolute = urllib.parse.urldefrag(urllib.parse.urljoin(base, href)).url
        filename = posixpath.basename(urllib.parse.urlparse(absolute).path)
        if not filename.lower().endswith(f"-{WHEEL_PLATFORM}.whl"):
            continue
        version = _wheel_version(filename)
        if version:
            wheels.append((version, absolute))
    return wheels


def _wheel_version(filename):
    """The version out of `{name}-{version}-{python}-{abi}-{platform}.whl`.

    An optional build tag can sit between name and python tag, but it never
    displaces the version, which is always the second field.
    """
    parts = filename[:-len(".whl")].split("-")
    return parts[1] if len(parts) >= 5 else None


def _version_key(version):
    """A sort key for the plain dotted-numeric versions these indexes carry.

    NOT a PEP 440 implementation, and deliberately not: every release of these
    four packages is `N.N.N[.N]`. Anything else -- a pre-release, a local
    version -- returns None and is SKIPPED rather than mis-ordered, which fails
    towards "did not offer a version" instead of "offered the wrong one".
    """
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


def _resolve_wheel(package, opener, pinned=None, major=None):
    """(version, url) for the wheel this runtime wants.

    `pinned` wins outright. Otherwise the newest release whose major version is
    `major` -- see the header for why the major is a constraint and not just a
    preference.
    """
    wheels = _index_wheels(package, opener)
    if not wheels:
        raise TrtFetchError(
            f"{package}: the index at {NVIDIA_INDEX.format(package=package)} "
            f"lists no {WHEEL_PLATFORM} wheel")
    if pinned is not None:
        for version, url in wheels:
            if version == pinned:
                return version, url
        raise TrtFetchError(
            f"{package}=={pinned} is not on the index; it lists "
            f"{', '.join(sorted({v for v, _ in wheels}))}")
    best = None
    for version, url in wheels:
        key = _version_key(version)
        if key is None or (major is not None and key[0] != major):
            continue
        if best is None or key > best[0]:
            best = (key, version, url)
    if best is None:
        raise TrtFetchError(
            f"{package}: the index lists no {major}.x release, so the DLL "
            f"names this runtime imports by exact filename cannot be satisfied")
    return best[1], best[2]


def _nuget_pinned_ort_version(opener):
    """The exact `Gpu.Windows` version the ORT meta package's nuspec pins.

    Ported from `tools_dev/setup_tensorrt.py::_nuget_meta_version_and_dep`. The
    meta package is ~325 KB of .props/.targets, so it is read whole rather than
    ranged -- there is nothing to save.
    """
    url = NUGET_VERSIONLESS_URL.format(id=NUGET_META_ID)
    with _request(opener, url) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        nuspec = next((n for n in archive.namelist() if n.endswith(".nuspec")),
                      None)
        if nuspec is None:
            raise TrtFetchError(f"{NUGET_META_ID}: no .nuspec in the package")
        root = ET.fromstring(archive.read(nuspec))
    for element in root.iter():
        if element.tag.endswith("dependency") \
                and element.get("id") == NUGET_WINDOWS_ID:
            version = element.get("version")
            if version:
                return version
    raise TrtFetchError(
        f"{NUGET_META_ID}'s nuspec has no pinned dependency on "
        f"{NUGET_WINDOWS_ID} -- the NuGet package layout changed and this "
        f"needs a look before any runtime is assembled from it")


# --- planning ---------------------------------------------------------------

def _plan_source(key, label, url, wanted, opener, cancel=None):
    """Reads one archive's central directory and locates `wanted` inside it.

    Members are matched by BASENAME, case-insensitively, first match wins --
    the same rule `setup_tensorrt._harvest` uses when it walks a site-packages
    tree, and for the same reason: the directory a wheel puts its DLLs under has
    moved between package versions before, and the filename is the part the
    Windows loader actually cares about.
    """
    handle = _RangeFile(url, opener, cancel)
    try:
        with zipfile.ZipFile(handle) as archive:
            index = {}
            for info in archive.infolist():
                if info.is_dir():
                    continue
                index.setdefault(posixpath.basename(info.filename).lower(), info)
            members, missing = [], []
            for name in wanted:
                info = index.get(name.lower())
                if info is None:
                    missing.append(name)
                    continue
                members.append(Member(name=name, archive_path=info.filename,
                                      size=info.file_size,
                                      download_size=info.compress_size))
            if missing:
                raise TrtFetchError(
                    f"{label}: {url} does not contain "
                    f"{', '.join(missing)}. Assembling a runtime without them "
                    f"would produce a directory that aborts the process on the "
                    f"first engine build rather than failing gracefully.")
    finally:
        handle.close()
    return Source(key=key, label=label, url=url, members=tuple(members))


def resolve_plan(sm, opener=None, cancel=None):
    """Exactly which members will be fetched for compute capability `sm`.

    Network cost is four small reads -- three PEP 503 pages and the ORT meta
    nuspec -- plus three range requests per archive for its central directory.
    Nothing large moves here.

    Raises `TrtFetchError` or `TrtFetchCancelled`, and nothing else: see
    `_as_fetch_error`.
    """
    with _as_fetch_error():
        return _resolve_plan(sm, opener, cancel)


def _resolve_plan(sm, opener, cancel):
    opener = opener or urllib.request.urlopen
    if sm not in trt_runtime.SUPPORTED_SM:
        # The wheel is the real authority and would refuse below anyway, with a
        # perfectly clear message. This check just answers before four network
        # round trips, and keeps the offered set identical to the one the status
        # row greys out.
        raise TrtFetchError(
            f"sm{sm} is not supported by TensorRT {TENSORRT_VERSION}: the "
            f"wheel ships builder blobs only for "
            f"{', '.join(f'sm{s}' for s in sorted(trt_runtime.SUPPORTED_SM))}")

    ort_version = _nuget_pinned_ort_version(opener)
    trt_version, trt_url = _resolve_wheel(
        TENSORRT_LIBS_PACKAGE, opener, pinned=TENSORRT_VERSION)
    cudnn_version, cudnn_url = _resolve_wheel(
        CUDNN_PACKAGE, opener, pinned=CUDNN_VERSION, major=CUDNN_MAJOR)
    cublas_version, cublas_url = _resolve_wheel(
        CUBLAS_PACKAGE, opener, pinned=CUBLAS_VERSION, major=CUBLAS_MAJOR)

    sources = (
        _plan_source("ort", f"ONNX Runtime {ort_version}",
                     NUGET_VERSIONED_URL.format(id=NUGET_WINDOWS_ID,
                                                version=ort_version),
                     ORT_DLLS, opener, cancel),
        _plan_source("tensorrt", f"TensorRT {trt_version}", trt_url,
                     TENSORRT_DLLS + (BUILDER_RESOURCE.format(sm=sm),),
                     opener, cancel),
        _plan_source("cudnn", f"cuDNN {cudnn_version}", cudnn_url,
                     CUDNN_DLLS, opener, cancel),
        _plan_source("cublas", f"cuBLAS {cublas_version}", cublas_url,
                     CUBLAS_DLLS, opener, cancel),
    )
    return Plan(sm=sm, sources=sources)


# --- extraction -------------------------------------------------------------

def _present(directory, member):
    """Presence AND exact size -- the same rule as `model_fetcher._installed`.

    Exact size rather than a hash: the CRC-32 was already checked, against the
    archive's own central directory, at the moment this file was written. Size
    here is only asking "did that finish", and a rerun must not re-transfer a
    gigabyte to re-answer a question the `.part` rename already answered.

    Asked of the STAGING directory it means "already fetched"; asked of the
    TARGET it means "already installed". Same question, two directories.
    """
    try:
        return os.path.getsize(
            os.path.join(directory, member.name)) == member.size
    except OSError:
        return False


def _free_bytes(path):
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def _extract_member(archive, handle, member, stage, report, cancel):
    """Inflates one member into `stage`, via a `.part` of its own.

    `report(wire_bytes_for_this_member)` is called per chunk.

    Two independent checks, and both matter. `zipfile` verifies the member's
    CRC-32 from the central directory as it inflates and raises `BadZipFile` on
    a mismatch -- that is what catches bytes that arrived wrong. The size check
    below catches the other shape, a stream that simply stopped: an inflater fed
    a truncated deflate block can end without complaint, and a short
    `nvinfer_10.dll` under its real name is precisely the half-populated
    assembly that aborts the process instead of failing.

    The `.part` file is what keeps a failure of either kind from ever appearing
    under the member's real name.
    """
    part = os.path.join(stage, member.name + ".part")
    final = os.path.join(stage, member.name)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        before = handle.served
        try:
            with archive.open(member.archive_path) as source, \
                    open(part, "wb") as out:
                written = 0
                while True:
                    if cancel is not None and cancel.is_set():
                        raise TrtFetchCancelled(member.name)
                    block = source.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
                    if report is not None:
                        # Clamped: `served` also covers the member's local
                        # header and any read-ahead, and a progress number that
                        # can exceed its own denominator is worse than one that
                        # arrives a few hundred bytes short.
                        report(min(handle.served - before, member.download_size))
            if written != member.size:
                raise TrtFetchError(
                    f"{member.name}: extracted {written} bytes, expected "
                    f"{member.size}")
        except TrtFetchCancelled:
            raise
        except (TrtFetchError, zipfile.BadZipFile, zlib.error, EOFError,
                OSError) as exc:
            # Five families, and all five were reached by damaging a REAL
            # transfer of a real wheel member rather than reasoned about:
            #   TrtFetchError   the size check above, and the reconnect giving up
            #   BadZipFile      `Bad CRC-32 for file ...` -- bytes arrived wrong
            #   zlib.error      `invalid distance too far back` -- the inflater
            #                   choking on a stream that was cut mid-block, which
            #                   is what a truncated transfer actually produces
            #   EOFError        the other truncation shape zipfile raises, when
            #                   the stream ends before the end-of-stream marker
            #   OSError         the local write, and every network fault
            # Leaving zlib.error and EOFError out is not a missing retry, it is
            # an UNCAUGHT exception that skips the `.part` cleanup below and
            # leaves debris in the staging directory.
            #
            # Nothing here is worth resuming ONTO -- a bad CRC does not say
            # which bytes were wrong, and a half-inflated file has no offset to
            # continue from. Drop it and re-extract.
            _unlink(part)
            if attempt == MAX_ATTEMPTS:
                # `zipfile` raises EOFError BARE for a stream that stopped, so
                # str() on it is empty; the class name is then the only thing
                # that says anything at all about what went wrong.
                raise TrtFetchError(
                    f"{member.name}: {exc or exc.__class__.__name__}") from exc
            _backoff(cancel, RETRY_BACKOFF_S)
            continue
        os.replace(part, final)
        return


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _promote(stage, target_dir):
    """Puts the finished staging directory in place, via renames only.

    The previous install is renamed aside first and removed only once the new
    one is in place, so there is no instant in which `target_dir` exists and is
    half populated -- the state that aborts the process. It can briefly not
    exist at all, which merely reads as `not_installed`.

    The engine cache and any `.trt_verified` in the old directory go with it,
    deliberately: both describe the DLLs being replaced. TensorRT keys its
    engines by graph and by TensorRT version, so once the DLLs change the old
    cache no longer applies, and a marker that outlived its DLLs would be caught
    by the freshness rule anyway -- but leaving it to that would be relying on
    mtimes to undo a mistake made here.

    That reasoning holds ONLY when the DLLs actually change. Reaching this
    function at an unchanged version would throw away a valid cache -- ~2.9 GB
    and roughly 70 minutes of compilation -- for nothing, so the guard against
    it lives in `acquire_runtime`, which returns before staging anything when
    every planned member is already present at its exact size. This function is
    the destructive step by design; it is not the one that decides whether
    destruction is warranted.
    """
    retired = target_dir.rstrip("\\/") + RETIRED_SUFFIX
    try:
        if os.path.isdir(retired):
            shutil.rmtree(retired, ignore_errors=True)
        if os.path.isdir(target_dir):
            os.replace(target_dir, retired)
        else:
            retired = None
        try:
            os.replace(stage, target_dir)
        except OSError:
            if retired:
                # Put the old runtime back rather than leave the user with
                # none. The staging directory survives either way.
                os.replace(retired, target_dir)
            raise
    except OSError as exc:
        raise TrtFetchError(
            f"could not put {target_dir} in place: {exc}. The most likely "
            f"cause is that the current runtime is still loaded by this "
            f"process -- close and reopen the player, then retry; the "
            f"downloaded files are kept in {stage}.") from exc
    if retired:
        # ignore_errors: a DLL still mapped by some process cannot be deleted,
        # and a leftover .old directory is harmless -- the next run clears it.
        shutil.rmtree(retired, ignore_errors=True)


def acquire_runtime(target_dir, sm, *, plan=None, progress=None, cancel=None,
                    opener=None, force=False):
    """Assembles a complete, UNVERIFIED TensorRT runtime at `target_dir`.

    Returns the basenames the runtime consists of, sorted. Re-running after a
    cancel or a failure re-fetches only what is missing: the staging directory
    persists and completed members are kept.

    Re-running against a target that ALREADY holds this exact plan is a no-op.
    Nothing is downloaded and nothing is renamed -- see `_promote` for what a
    needless promotion would cost.

    `progress(name, have, size, done, total)` -- every number is bytes OVER THE
    WIRE, never bytes on disk. The two differ by each member's deflate ratio
    (51% for nvinfer_10.dll, 75% for cublasLt64_13.dll), so a bar that mixed
    them would stall and then jump. `Plan.size` is the on-disk figure, and it is
    what the free-space check uses.

    `force` re-acquires even when the target already matches -- for repairing an
    install whose files are the right SIZE but the wrong bytes, which is the one
    fault the no-op check cannot see. It costs the engine cache, so it is opt-in.

    Raises `TrtFetchError` (including `NotEnoughSpace`) or `TrtFetchCancelled`,
    and nothing else. On EVERY one of those paths `target_dir` is exactly as it
    was: the transfer happens in a sibling staging directory and the target is
    touched only by `_promote`, after the last thing that can fail.

    Writes no `.trt_verified`: see the module docstring.
    """
    with _as_fetch_error():
        return _acquire_runtime(target_dir, sm, plan, progress, cancel, opener,
                                force)


def _acquire_runtime(target_dir, sm, plan, progress, cancel, opener, force):
    opener = opener or urllib.request.urlopen
    plan = plan or resolve_plan(sm, opener=opener, cancel=cancel)
    # Once, here: `staging_dir` and `_promote` both derive sibling names by
    # appending to this string, and a trailing separator would make them derive
    # DIFFERENT ones from the same argument.
    target_dir = target_dir.rstrip("\\/")

    if not force and all(_present(target_dir, m) for m in plan.members):
        # ALREADY INSTALLED, at exactly this plan. Returning here is not an
        # optimisation, it is the difference between a harmless second click and
        # a destructive one: `_promote` replaces the directory wholesale, so
        # without this check a user who reinstalls at the same pinned version
        # would pay a 1.35 GB download to be handed back the same DLLs, minus
        # `.trt_verified` and minus the compiled engine cache -- ~2.9 GB and
        # roughly 70 minutes of compilation on the author's machine.
        #
        # Discarding those is CORRECT when the DLLs change, because TensorRT
        # keys its engines by graph and by TensorRT version and the old ones
        # would no longer apply. It is pure loss when the DLLs are identical,
        # which is precisely the case this check identifies: every planned
        # member already present at its exact size means the plan is what is on
        # disk, so the engines built against it are still valid.
        #
        # A stale staging directory, if any, is deliberately left alone rather
        # than deleted here -- a call that reports "nothing to do" should not
        # also remove a gigabyte. `discard_staging` is the caller's tool for it.
        return sorted(m.name for m in plan.members)

    stage = staging_dir(target_dir)
    os.makedirs(stage, exist_ok=True)

    # Keyed by (source, name) rather than by the Member itself: two archives
    # could in principle carry a same-named file, and the identity that matters
    # is "this member of this source".
    pending = {(source.key, member.name): member
               for source in plan.sources for member in source.members
               if not _present(stage, member)}
    needed = sum(m.size for m in pending.values())
    free = _free_bytes(stage)
    if free < needed:
        raise NotEnoughSpace(
            f"{needed / 1e9:.2f} GB needed in {stage}, {free / 1e9:.2f} GB free")

    total = sum(m.download_size for m in pending.values())
    state = {"done": 0}

    def make_report(member):
        # `contributed` is this member's ABSOLUTE share of the numerator, and
        # max() keeps it from walking backwards when a retry re-extracts the
        # member from its start -- a bar that goes back reads as a fault rather
        # than as the retry it is.
        best = [0]

        def report(value):
            best[0] = max(best[0], value)
            progress(member.name, best[0], member.download_size,
                     state["done"] + best[0], total)
        return report

    for source in plan.sources:
        wanted = [m for m in source.members
                  if (source.key, m.name) in pending]
        if not wanted:
            continue
        handle = _RangeFile(source.url, opener, cancel)
        try:
            with zipfile.ZipFile(handle) as archive:
                for member in wanted:
                    _extract_member(
                        archive, handle, member, stage,
                        make_report(member) if progress is not None else None,
                        cancel)
                    state["done"] += member.download_size
        finally:
            handle.close()

    missing = [m.name for m in plan.members if not _present(stage, m)]
    if missing:
        # Belt and braces over `_extract_member`'s own size check: promoting a
        # directory that is short one DLL is the exact failure this whole
        # staging dance exists to make impossible.
        raise TrtFetchError(
            f"assembly incomplete, {target_dir} left untouched -- missing "
            f"{', '.join(sorted(missing))}")
    _promote(stage, target_dir)
    return sorted(m.name for m in plan.members)


def discard_staging(target_dir):
    """Throws away a partial assembly. Returns True if there was one."""
    stage = staging_dir(target_dir)
    if not os.path.isdir(stage):
        return False
    shutil.rmtree(stage, ignore_errors=True)
    return not os.path.isdir(stage)


# --- command line -----------------------------------------------------------
#
# The dialog is the intended caller, but this file also lives in the GitHub
# repository, where "no v5.2.0 user can obtain TensorRT" is the problem being
# fixed -- and there, running this module directly IS the fix, with no uv, no
# pip and no scratch venv.

def _main(argv):
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    target = next((a for a in argv if not a.startswith("--")),
                  os.path.join(here, "ort_tensorrt"))
    sm = None
    if "--sm" in argv:
        sm = int(argv[argv.index("--sm") + 1])
    if sm is None:
        gpu = trt_runtime.detect_gpu()
        if gpu is None:
            print("[trt_fetcher] no NVIDIA GPU detected, and the architecture "
                  "decides which builder blob to fetch. Pass --sm <N> to "
                  "override.")
            return 1
        print(f"[trt_fetcher] detected {gpu.name} -> sm{gpu.sm}")
        sm = gpu.sm

    print(f"[trt_fetcher] resolving the plan for sm{sm} ...")
    try:
        plan = resolve_plan(sm)
    except TrtFetchError as exc:
        print(f"[trt_fetcher] {exc}")
        return 2
    for source in plan.sources:
        print(f"    {source.label}")
        for member in source.members:
            print(f"        {member.name:<44} "
                  f"{member.download_size / 2 ** 20:8.1f} MiB over the wire, "
                  f"{member.size / 2 ** 20:8.1f} MiB on disk")
    print(f"[trt_fetcher] {plan.download_size / 2 ** 20:.0f} MiB to download, "
          f"{plan.size / 2 ** 20:.0f} MiB on disk, into {target}")
    if "--plan-only" in argv:
        return 0

    state = {"line": ""}

    def progress(name, have, size, done, total):
        line = (f"    {name:<44} {have / 2 ** 20:7.1f}/{size / 2 ** 20:.1f} MiB"
                f"   [{100.0 * done / max(total, 1):5.1f}% overall]")
        if line != state["line"]:
            state["line"] = line
            print(line, end="\r", flush=True)

    try:
        written = acquire_runtime(target, sm, plan=plan, progress=progress)
    except TrtFetchCancelled:
        print("\n[trt_fetcher] cancelled; rerun to resume.")
        return 3
    except TrtFetchError as exc:
        print(f"\n[trt_fetcher] FAILED: {exc}")
        return 2
    print(f"\n[trt_fetcher] wrote {len(written)} files into {target}.")
    print("[trt_fetcher] This runtime is UNVERIFIED: no .trt_verified marker "
          "was written, and the player will keep using DirectML until a real "
          "engine build has passed the value gate.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
