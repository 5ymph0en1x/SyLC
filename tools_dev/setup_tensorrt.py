# tools_dev/setup_tensorrt.py
#
# Round 3, Task 3 (inference throughput): acquires a LOCAL, opt-in ONNX
# Runtime + TensorRT execution-provider runtime into the project root's
# ort_tensorrt\ for the author's RTX 4090. This is a pure runtime-acquisition
# script -- it never touches DepthEngine, never packages anything
# (ort_tensorrt/ is NOT in build_exe_v510.bat's whitelist -- verified by
# grep, see BUILD.md), and never WRITES into the project venv (.venv,
# Python 3.14) -- it may READ from it (see "project-venv fallback" below)
# but never installs into or modifies it.
#
# --- ROUND 2 REWRITE: the acquisition is now COMPLETE. Read this if you're
#     wondering why this looks nothing like round 1's version. ---
#
# Round 1 concluded BLOCKED: `pip install tensorrt-cu12`/`tensorrt` on
# Windows only ever resolves to source-only or ancient-beta packages, so
# round 1's plain-pip-with---only-binary=:all: approach never got real
# nvinfer*.dll binaries. That conclusion was WRONG in one specific way, found
# during a live re-review (2026-07-29, later the same day): the *bare*
# `tensorrt` (or cu12) meta-packages are indeed dead ends on Windows, but
# **`tensorrt-cu13`** is a different story. `tensorrt-cu13` / `tensorrt-cu13-
# libs` are published on PyPI as tiny (~15-18KB) `wheel_stub`-backed sdists --
# NOT real content, just a PEP 517 build backend
# (https://pypi.org/project/wheel-stub/) whose `build_wheel()` step reaches
# out to `https://pypi.nvidia.com/` at BUILD TIME and downloads the real,
# huge (2+ GB), platform-matched wheel from NVIDIA's own index. Round 1's
# `--only-binary=:all:` flag on every pip invocation makes pip refuse to even
# attempt building ANY sdist -- which silently defeats this exact mechanism.
# There is no static wheel-listing shortcut that reveals this: it only shows
# up if you actually let the sdist build run. (The user independently found
# this by running plain `uv pip install --upgrade tensorrt-cu13` in their own
# terminal -- see task-3-report.md's "fix round 2" section for the full
# attribution/verification trail.)
#
# A SECOND wrinkle, found while reproducing this: `tensorrt-cu13`'s "latest"
# version (11.1.0.106, TensorRT 11.x) produces `nvinfer_11.dll` /
# `nvonnxparser_11.dll` -- but the ORT 1.28.0 GPU build fetched by Step 1
# below has `onnxruntime_providers_tensorrt.dll` STATICALLY IMPORTING
# `nvinfer_10.dll` / `nvonnxparser_10.dll` by EXACT filename (confirmed via
# `_pe_import_dll_names()`, a real PE-import-table read, not a guess). The
# Windows loader matches static imports by literal filename, not by "close
# enough" version -- so TensorRT 11.x DLLs sitting right next to
# onnxruntime_providers_tensorrt.dll do NOT satisfy its `nvinfer_10.dll`
# import, no matter how complete they are. `tensorrt-cu13` ALSO publishes
# 10.x-line releases (10.13.2.6 through 10.16.1.11 -- CUDA 13 support was
# backported/paralleled onto TensorRT's 10.x series before 11.x shipped);
# TENSORRT_VERSION below is pinned to the latest of those (10.16.1.11),
# empirically verified (see next paragraph) to produce the exact
# `nvinfer_10.dll` name ORT 1.28.0 wants.
#
# A THIRD wrinkle, the one that actually explains round 1's confusing dead
# end: even with byte-for-byte correct, version-matched DLLs assembled, a
# bare `LoadLibraryExW` of `onnxruntime_providers_tensorrt.dll` in isolation
# STILL fails, with WinError 1114 (DLL init routine failed) -- reproduced
# consistently across FOUR different TensorRT 10.x point releases
# (10.14.1.48, 10.15.1.29, 10.16.0.72, 10.16.1.11), all with matching
# cublas64_13.dll/cudnn64_9.dll present. This is NOT a missing-dependency
# signature (that would be error 126, ERROR_MOD_NOT_FOUND) -- something in
# this DLL's own static initialization touches state that behaves
# differently when the DLL is force-loaded standalone versus loaded the way
# ONNX Runtime actually loads it internally (on demand, from inside a live
# OrtEnv/session-creation call sequence, quite possibly after some CUDA/
# driver state ORT itself sets up first). PROOF that "bare LoadLibraryExW"
# was a methodologically invalid gate for this specific DLL: calling the
# REAL ORT C API in the exact sequence ORT itself uses --
# `OrtGetApiBase()->GetApi(24)` -> `CreateEnv` -> `CreateSessionOptions` ->
# `OrtSessionOptionsAppendExecutionProvider_Tensorrt(options, 0)` -- with the
# IDENTICAL files sitting in the IDENTICAL directory, SUCCEEDS CLEANLY. See
# `_real_tensorrt_session_test()` below: this, not a bare DLL load, is now
# the actual completeness gate. The old bare-load + PE-import dependency walk
# is KEPT as a diagnostic (still useful for naming exactly which file is
# missing when the real test fails for a genuinely-incomplete assembly), but
# it no longer decides pass/fail on its own.
#
# --- WHAT THIS SCRIPT ACTUALLY DOES (round 2) ---
#
# 1. NuGet: fetches the ONNX Runtime GPU build's native DLLs, UNCHANGED from
#    round 1. `Microsoft.ML.OnnxRuntime.Gpu`'s versionless URL redirects to a
#    thin meta-package (~325KB, just .props/.targets); its .nuspec pins an
#    exact-version dependency on `Microsoft.ML.OnnxRuntime.Gpu.Windows` (the
#    real platform package, runtimes/win-x64/native/*.dll). This script
#    fetches the meta package first to read that pinned version, then
#    fetches `Gpu.Windows` AT THAT EXACT VERSION (verified: 1.28.0, and
#    confirmed via NuGet's own version listing to be the true latest --
#    there is no newer ORT release that might target TensorRT 11.x instead).
#
# 2. TensorRT + cuDNN/cuBLAS via `uv` (the proven route), PRIMARY:
#      `uv venv <scratch>` (a disposable venv, %TEMP%-side, never the
#      project .venv) + `uv pip install --python <scratch> tensorrt-cu13==<pinned>`
#      + `uv pip install --python <scratch> nvidia-cudnn-cu13` (this ALSO
#      transitively installs `nvidia-cublas` -- a modern, unsuffixed package
#      name; `nvidia-cublas-cu13` is EXPLICITLY DEPRECATED upstream and its
#      own build step tells you to use plain `nvidia-cublas` instead --
#      confirmed by actually attempting it and reading the deprecation
#      notice in the build failure output).
#    FALLBACK if `uv` isn't on PATH: the same disposable-embeddable-CPython-
#    3.12 recipe from `tools_dev/convert_fp16.py`, but WITHOUT
#    `--only-binary=:all:` for this specific package (that flag is exactly
#    what defeats the wheel-stub sdist bootstrap -- see above).
#    FALLBACK if network/index access fails entirely but the user has
#    already run `uv pip install tensorrt-cu13` themselves in the project
#    venv (as happened here): READ-ONLY harvest from the project venv's
#    `.venv\Lib\site-packages\tensorrt_libs\` etc. This path only ever
#    COPIES files out; it never writes to `.venv` in any way.
#
# 3. Harvest is MINIMAL, not "whole tensorrt_libs directory": empirically
#    verified (a 9-file test directory: the 3 ORT DLLs + these 6) that the
#    real session test passes WITHOUT any of the ~2 GB of
#    `nvinfer_builder_resource_*_10.dll` (per-GPU-architecture precompiled
#    kernels, needed only when actually BUILDING a TensorRT engine for a
#    specific SM target -- not for provider registration) or `nvrtc*`/
#    `nvblas*.dll` (JIT/BLAS-replacement extras, unused by this provider's
#    import table). Harvested: nvinfer_10.dll, nvinfer_plugin_10.dll,
#    nvonnxparser_10.dll, cudnn64_9.dll, cublas64_13.dll, cublasLt64_13.dll.
#    NOTE FOR T4: if actual engine *building* on the RTX 4090 (compute
#    capability 8.9, "sm89") fails for a missing precompiled-kernel resource,
#    the fix is `nvinfer_builder_resource_sm89_10.dll` from this exact same
#    `tensorrt-cu13==10.16.1.11` install (already cached by this script) --
#    deliberately not pre-included here to keep the default footprint modest
#    and because whether it's actually needed (TensorRT can often JIT PTX as
#    a fallback) wasn't tested by this task.
#
# 4. Assemble into a staging directory that MERGES rather than wipes (round-1
#    review fix, unchanged): manually-placed files survive reruns.
#
# 5. Probe: GPU-build symbol checks (unchanged) + the REAL session test
#    described above as the TensorRT-EP gate. On a full pass, promote
#    staging -> `ort_tensorrt\` (same-volume rename). On failure, leave
#    `ort_tensorrt\` untouched and print exactly what failed.
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_DIR = os.path.join(ROOT, ".ort_tensorrt_staging")
FINAL_DIR = os.path.join(ROOT, "ort_tensorrt")
PROJECT_VENV_SITE_PACKAGES = os.path.join(ROOT, ".venv", "Lib", "site-packages")

# Pre-built TensorRT engines published alongside the ONNX packs. These are
# sm89 (Ada / RTX 40-series) and were built by TensorRT 10.16.1.11 against the
# ORT build this script fetches -- TensorRT engines are not portable across
# compute capabilities or TensorRT versions, which is why the directory name
# states both. On any other GPU TensorRT simply rebuilds, as it always has.
#
# THIS DOES NOT REPLACE THE PROBE. `.trt_verified` is still written only after
# a real local engine build plus an inference that passes the near/far value
# gate on THIS machine. Downloaded engines make that probe fast (a cold compile
# of all 21 graphs costs roughly 70 minutes); they are never taken on trust.
HF_ENGINE_REPO = "Symphoenix/sylc_TRT"
HF_ENGINE_REVISION = "b1e70aab24abde0ecf82a1dce8c6689c1c71f935"
TRT_ENGINE_PREFIX = "trt/sm89-trt10.16.1.11/"
TRT_CACHE_DIRNAME = "trt_cache"


def _hf_list_entries(repo, revision, prefix):
    """File entries under `prefix`, from the HuggingFace tree API.

    Returns the raw entries rather than bare paths because each one already
    carries what verification needs: `size`, and for LFS-tracked files an
    `lfs.oid` that IS the content's SHA-256. Verified against the live API on
    2026-07-31 — an engine entry reads:

        {"type": "file", "size": 54466204, "path": "...engine",
         "oid": "89072045...",            # git blob id, NOT the content hash
         "lfs": {"oid": "759ca003...", "size": 54466204, "pointerSize": 133}}

    Note `oid` and `lfs.oid` are different things: the top-level one is the git
    pointer's blob id. Only `lfs.oid` hashes the real bytes. Small files
    (the 21-byte `.profile` companions) are not LFS-tracked and carry no `lfs`
    key at all, which is why the check below is conditional.
    """
    url = (f"https://huggingface.co/api/models/{repo}/tree/{revision}"
           f"?recursive=true")
    with urllib.request.urlopen(url) as response:
        entries = json.loads(response.read().decode("utf-8"))
    return [e for e in entries
            if e.get("type") == "file" and e.get("path", "").startswith(prefix)]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_engines(dest_dir, opener=None):
    """Downloads the published sm89 engine cache into `dest_dir`.

    Returns the number of files written. Existing files are left alone, so a
    rerun after an interruption only fetches what is missing.

    Every file is size-checked, and SHA-256-checked when the listing carries an
    LFS oid, BEFORE it is promoted from `.part` to its real name. Without that,
    a transfer that closes early WITHOUT raising -- a flaky proxy or CDN, which
    `shutil.copyfileobj` returning normally does not rule out -- would publish a
    truncated engine under its real name, and every later run would skip it
    forever, since presence is the only "already have it" test. A poisoned cache
    entry cannot crash playback (the local probe still gates `.trt_verified`),
    but it would burn a multi-minute probe and read as a TensorRT fault rather
    than as a bad download.
    """
    opener = opener or urllib.request.urlopen
    os.makedirs(dest_dir, exist_ok=True)
    written = 0
    for entry in _hf_list_entries(HF_ENGINE_REPO, HF_ENGINE_REVISION,
                                   TRT_ENGINE_PREFIX):
        path = entry["path"]
        name = path[len(TRT_ENGINE_PREFIX):]
        if not name or "/" in name:
            continue
        target = os.path.join(dest_dir, name)
        if os.path.exists(target):
            continue
        url = (f"https://huggingface.co/{HF_ENGINE_REPO}/resolve/"
               f"{HF_ENGINE_REVISION}/{path}")
        part = target + ".part"
        print(f"[setup_tensorrt] fetching engine {name}")
        with opener(urllib.request.Request(url)) as response, \
                open(part, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)

        expected_size = entry.get("size")
        actual_size = os.path.getsize(part)
        if expected_size is not None and actual_size != expected_size:
            os.remove(part)
            raise RuntimeError(f"{name}: got {actual_size} bytes, "
                               f"expected {expected_size}")
        expected_sha = (entry.get("lfs") or {}).get("oid")
        if expected_sha:
            actual_sha = _sha256_file(part)
            if actual_sha != expected_sha:
                os.remove(part)
                raise RuntimeError(f"{name}: sha256 {actual_sha}, "
                                   f"expected {expected_sha}")
        os.replace(part, target)
        written += 1
    print(f"[setup_tensorrt] {written} engine file(s) fetched into {dest_dir}")
    return written


# --- T4 round-3 "playback never dies" guard ---------------------------------
# Registration-only (the probe above) is NOT sufficient proof that a real
# TensorRT engine BUILD will succeed: it does not exercise CreateSession, so
# it cannot catch native runtime gaps like the cuDNN9 modular-DLL one this
# round found (a hard process crash, not a graceful OrtStatus error -- see
# task-4-report.md, "The crash"). TRT_VERIFIED_MARKER is written ONLY after a
# REAL engine build + inference actually completes AND passes the fix-round-1
# VALUE GATE (_trt_engine_probe.py's engine mode now asserts the same
# near-vs-far polarity test_depth_engine.py checks -- a clean build under
# trt_fp16_enable=1 is not proof of numerically correct output).
# _synth3d_ort_dir() in SyLC_3D_Player.py requires this marker's presence AND
# freshness (no *.dll in the directory newer than it) before ever preferring
# ort_tensorrt\ -- no marker, no TRT: this closes the realistic crash path at
# acquisition time, not every conceivable one (a driver/GPU change after a
# fresh marker is a narrower case left for a future round).
# ROUND 4: the player now also requires the marker to NAME the graph it is
# about to open (one `probe_model=` line per attested graph, written below),
# because the depth presets ship 518 exports alongside the 756 ones and
# TensorRT's cache is per graph -- a marker proving `da3_base_756.onnx` builds
# says nothing about whether `da3_base_518.onnx` would compile in seconds or in
# minutes. A marker naming NO graph keeps its round-3 meaning (the directory as
# a whole), which is why this script must never write one: see
# _run_engine_probe's empty-graph guard.
TRT_VERIFIED_MARKER = ".trt_verified"
# The engine-probe reuses tests/synth3d/_trt_engine_probe.py's "engine" mode
# (a fresh, registry-independent DepthEngine via mvc_demuxer_cpp.depth_infer_test)
# rather than re-implementing the C-API call sequence a second time here --
# that file is already proven (this round) to correctly surface a native
# engine-build crash as a subprocess failure instead of taking the caller down.
ENGINE_PROBE_SCRIPT = os.path.join(ROOT, "tests", "synth3d", "_trt_engine_probe.py")
# The probe must build the engine for every graph the PLAYER can actually load
# -- this list mirrors SyLC_3D_Player.SYNTH3D_DEPTH_PRESETS entry for entry,
# (candidate models, inference grid) per depth preset, and
# tests/synth3d/test_trt_optin.py::test_engine_probe_candidates_match_player
# pins the two equal. TensorRT's engine cache is keyed PER GRAPH: compiling one
# model's engine does NOT warm another's, so probing a model the player never
# opens leaves the real default cold and makes the user's FIRST in-playback
# enable pay a fresh multi-minute compile -- exactly the wait this verification
# step is supposed to have already absorbed. (Round 3's "smallest model = fastest
# compile" rationale for probing Small was empirically wrong on this graph
# family: Base compiled in 202.3s vs Small's 232.7s. TensorRT's optimizer, not
# parameter count, dominates compile time here.)
#
# The GRID travels with the models because it is half of the identity of the
# graph being built: a fixed-shape 518 export opened at 756 fails DepthEngine
# init outright, and even for the one dynamic-axes export the input shape is
# what TensorRT builds its optimization profile from. Round 3's flat filename
# tuple could not express that, which is the whole reason this is now a
# per-preset structure.
ENGINE_PROBE_MODEL_CANDIDATES = (
    (("da3_base_756.onnx", "da3_small_756.onnx", "da3_small.onnx"), 756),
    (("da3_base_518.onnx", "da3_small_518.onnx"),                   518),
    (("da3_small_518.onnx",),                                       518),
)
# Adaptive fixed rectangles are not user-facing presets, but the player can
# open them after matte detection. They therefore need their own cold engine
# build and marker attestation just like every square preset graph.
ENGINE_PROBE_ADAPTIVE_MODEL_GRIDS = (
    ("da3_base_756x406.onnx", 756, 406),
    ("da3_base_756x378.onnx", 756, 378),
    ("da3_base_756x350.onnx", 756, 350),
    ("da3_base_756x322.onnx", 756, 322),
    ("da3_small_756x406.onnx", 756, 406),
    ("da3_small_756x378.onnx", 756, 378),
    ("da3_small_756x350.onnx", 756, 350),
    ("da3_small_756x322.onnx", 756, 322),
    ("da3_base_518x280.onnx", 518, 280),
    ("da3_base_518x266.onnx", 518, 266),
    ("da3_base_518x238.onnx", 518, 238),
    ("da3_base_518x210.onnx", 518, 210),
    ("da3_small_518x280.onnx", 518, 280),
    ("da3_small_518x266.onnx", 518, 266),
    ("da3_small_518x238.onnx", 518, 238),
    ("da3_small_518x210.onnx", 518, 210),
)
# Per-GRAPH budget. Round 3 measured 202.3s (Base) / 232.7s (Small) for a cold
# 756 compile, so the old 300s covered a single graph with little to spare;
# this probe now builds several in a row and one of them is the dynamic-axes
# export, whose optimization profile TensorRT has to work harder for.
ENGINE_PROBE_TIMEOUT_S = 600.0


def _engine_probe_graphs():
    """[(model_path, width, height), ...]: every loadable graph on disk.

    EVERY candidate, not just each preset's first choice, because the player's
    gate (`SyLC_3D_Player.synth3d_marker_attests`) is checked against the model
    a preset RESOLVES to, and that resolution reads the disk at enable time --
    while the marker only has to stay newer than the *.dll files to remain
    valid. Delete `da3_base_756.onnx` from an installed copy and Quality
    resolves to `da3_small_756.onnx` without invalidating anything, so a marker
    naming only the first choices would silently drop that preset to DirectML.
    Attesting a candidate that is never resolved costs an offline compile;
    failing to attest one that IS resolved costs the user the multi-minute
    in-playback wait this whole mechanism exists to prevent.

    Order is the presets' own display order, de-duplicated by (name, grid) --
    `da3_small_518.onnx` is both Balanced's fallback and Performance's only
    model, and it is ONE graph, hence one engine and one probe.
    """
    graphs, seen = [], set()
    for candidates, side in ENGINE_PROBE_MODEL_CANDIDATES:
        for name in candidates:
            key = (name.lower(), side)
            if key in seen:
                continue
            path = os.path.join(ROOT, "models", name)
            if os.path.exists(path):
                seen.add(key)
                graphs.append((path, side, side))
    for name, width, height in ENGINE_PROBE_ADAPTIVE_MODEL_GRIDS:
        key = (name.lower(), width, height)
        if key in seen:
            continue
        path = os.path.join(ROOT, "models", name)
        if os.path.exists(path):
            seen.add(key)
            graphs.append((path, width, height))
    return graphs

CACHE_DIR = os.path.join(tempfile.gettempdir(), "sylc_tensorrt_setup")
NUGET_CACHE = os.path.join(CACHE_DIR, "nuget")
PY312_EMBED_DIR = os.path.join(CACHE_DIR, "py312_embed")  # pip fallback only
UV_SCRATCH_VENV_DIR = os.path.join(CACHE_DIR, "uv_scratch_venv")  # primary

NUGET_META_ID = "Microsoft.ML.OnnxRuntime.Gpu"
NUGET_WINDOWS_ID = "Microsoft.ML.OnnxRuntime.Gpu.Windows"
NUGET_VERSIONLESS_URL = "https://www.nuget.org/api/v2/package/{id}"
NUGET_VERSIONED_URL = "https://www.nuget.org/api/v2/package/{id}/{version}"

PY312_EMBED_URL = "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# Pinned to the LATEST TensorRT 10.x-line release published under the cu13
# CUDA-13 packaging line (10.13.2.6 .. 10.16.1.11 exist; 11.x also exists but
# produces nvinfer_11.dll, which does NOT satisfy this ORT build's exact
# nvinfer_10.dll import -- see header). Empirically verified 2026-07-29: real
# ORT C-API session test (CreateEnv -> CreateSessionOptions ->
# AppendExecutionProvider_Tensorrt) succeeds with this exact version.
TENSORRT_PACKAGE = "tensorrt-cu13"
TENSORRT_VERSION = "10.16.1.11"
CUDNN_PACKAGE = "nvidia-cudnn-cu13"  # transitively installs nvidia-cublas too

# Symbols proving the fetched onnxruntime.dll is genuinely the GPU build
# (not e.g. a plain CPU or DirectML build re-fetched by mistake).
SYM_API_BASE = b"OrtGetApiBase"
SYM_TENSORRT_EP = b"OrtSessionOptionsAppendExecutionProvider_Tensorrt"
SYM_CUDA_EP = b"OrtSessionOptionsAppendExecutionProvider_CUDA"

LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008

# Field indices into the (flat, function-pointer-array) `struct OrtApi`,
# parsed from this project's vendored header
# (mvc_realtime_demuxer/third_party/onnxruntime/include/onnxruntime_c_api.h,
# ORT_API_VERSION 24). ORT's own ABI stability guarantee is that new API
# functions are ONLY EVER APPENDED to the end of this struct, never
# reordered or removed -- so these early-struct offsets are stable across
# ORT releases as long as we request GetApi(24) specifically (a newer
# onnxruntime.dll still answers older requested versions for exactly this
# reason). Re-derive with the small parser in task-3-report.md's fix-round-2
# section if this ever needs re-verifying against a different vendored
# header.
ORT_API_VERSION = 24
IDX_CREATE_STATUS = 0
IDX_GET_ERROR_MESSAGE = 2
IDX_CREATE_ENV = 3
IDX_CREATE_SESSION_OPTIONS = 10
IDX_RELEASE_ENV = 92
IDX_RELEASE_STATUS = 93
IDX_RELEASE_SESSION_OPTIONS = 100

# DLLs that are always resolvable via the system/CRT redistributable and are
# never something WE need to bundle -- skipped by the PE-import dependency
# walk (diagnostic only, see header) so it only reports on DLLs actually
# relevant to this runtime.
_SYSTEM_OR_CRT_PREFIXES = ("api-ms-win-", "kernel32", "msvcp140", "vcruntime140",
                           "ucrtbase", "user32", "advapi32", "ntdll")

# Minimal harvest set (see header point 3). Maps final filename -> a list of
# basename patterns to search for recursively under a site-packages tree
# (case-insensitive startswith), first match wins.
#
# --- T4 round-3 fix (2026-07-29, user-directed "repair at all costs"):
# real engine-BUILD testing (never exercised by the registration-only probe
# below) proved this set was still incomplete in two ways that manifest as a
# HARD PROCESS CRASH, not a graceful OrtStatus error (see
# .superpowers/sdd/2026-07-29-synth3d-round3-perf/task-4-report.md, "The
# crash"):
#   1. cudnn64_9.dll above is only cuDNN 9's thin front-end DISPATCHER (267
#      KB). It loads these modular backend engine DLLs ON DEMAND, at first
#      real cudnnCreate() -- i.e. at TensorRT engine build time, never at
#      provider registration. All confirmed present in the SAME
#      nvidia-cudnn-cu13 install this script already performs (found cached
#      in the uv scratch venv from the ORIGINAL run, never harvested).
#   2. nvinfer_builder_resource_sm89_10.dll: T3's own flagged watch-item
#      (RTX 4090, compute capability 8.9 = "sm89") -- needed for a real
#      engine BUILD, not for provider registration.
HARVEST_PATTERNS = {
    "nvinfer_10.dll": ["nvinfer_10.dll"],
    "nvinfer_plugin_10.dll": ["nvinfer_plugin_10.dll"],
    "nvonnxparser_10.dll": ["nvonnxparser_10.dll"],
    "cudnn64_9.dll": ["cudnn64_9.dll"],
    "cublas64_13.dll": ["cublas64_13.dll"],
    "cublasLt64_13.dll": ["cublasLt64_13.dll"],
    "cudnn_ops64_9.dll": ["cudnn_ops64_9.dll"],
    "cudnn_cnn64_9.dll": ["cudnn_cnn64_9.dll"],
    "cudnn_adv64_9.dll": ["cudnn_adv64_9.dll"],
    "cudnn_graph64_9.dll": ["cudnn_graph64_9.dll"],
    "cudnn_heuristic64_9.dll": ["cudnn_heuristic64_9.dll"],
    "cudnn_engines_precompiled64_9.dll": ["cudnn_engines_precompiled64_9.dll"],
    "cudnn_engines_runtime_compiled64_9.dll": ["cudnn_engines_runtime_compiled64_9.dll"],
    "cudnn_engines_tensor_ir64_9.dll": ["cudnn_engines_tensor_ir64_9.dll"],
    "cudnn_ext64_9.dll": ["cudnn_ext64_9.dll"],
    "nvinfer_builder_resource_sm89_10.dll": ["nvinfer_builder_resource_sm89_10.dll"],
}


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.LoadLibraryExW.restype = ctypes.c_void_p
    k.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    k.GetProcAddress.restype = ctypes.c_void_p
    k.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    k.FreeLibrary.argtypes = [ctypes.c_void_p]
    k.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
    return k


def _download(url, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[setup_tensorrt] cached: {label} -> {dest} "
              f"({os.path.getsize(dest)/1e6:.1f} MB)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[setup_tensorrt] downloading {label}\n    {url}")
    t0 = time.time()
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    print(f"[setup_tensorrt]   -> {os.path.getsize(dest)/1e6:.1f} MB in "
          f"{time.time()-t0:.1f}s")


def _nuget_meta_version_and_dep(pkg_id, dep_id):
    """Downloads pkg_id's versionless nupkg (cached), parses its .nuspec,
    and returns (own_version, dep_id's pinned version)."""
    dest = os.path.join(NUGET_CACHE, f"{pkg_id}.meta.nupkg.zip")
    _download(NUGET_VERSIONLESS_URL.format(id=pkg_id), dest, f"{pkg_id} (meta)")
    with zipfile.ZipFile(dest) as z:
        nuspec_name = next(n for n in z.namelist() if n.endswith(".nuspec"))
        xml_bytes = z.read(nuspec_name)
    root = ET.fromstring(xml_bytes)
    ns = {"n": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    meta = root.find("n:metadata", ns) if ns else root.find("metadata")
    own_version = meta.findtext("n:version" if ns else "version", namespaces=ns)
    dep_version = None
    for dep in meta.iter():
        if dep.tag.endswith("dependency") and dep.get("id") == dep_id:
            dep_version = dep.get("version")
            break
    if dep_version is None:
        raise RuntimeError(f"{pkg_id}'s nuspec has no dependency on {dep_id} "
                            f"-- NuGet package layout changed, needs a look")
    return own_version, dep_version


def _nuget_fetch_windows_native(version):
    """Fetches Microsoft.ML.OnnxRuntime.Gpu.Windows at an EXACT version
    (pinned from the meta package's own nuspec, not independently resolved)
    and extracts runtimes/win-x64/native/*.dll into a cache dir. Returns
    {dll_name: path}."""
    dest = os.path.join(NUGET_CACHE, f"{NUGET_WINDOWS_ID}.{version}.nupkg.zip")
    url = NUGET_VERSIONED_URL.format(id=NUGET_WINDOWS_ID, version=version)
    _download(url, dest, f"{NUGET_WINDOWS_ID} {version}")

    extract_dir = os.path.join(NUGET_CACHE, f"{NUGET_WINDOWS_ID}.{version}.extracted")
    native_dir = os.path.join(extract_dir, "runtimes", "win-x64", "native")
    if not os.path.isdir(native_dir):
        with zipfile.ZipFile(dest) as z:
            z.extractall(extract_dir)
    dlls = {}
    for name in os.listdir(native_dir):
        if name.lower().endswith(".dll"):
            dlls[name] = os.path.join(native_dir, name)
    return dlls


# --- TensorRT acquisition: uv (primary), pip/embeddable-python (fallback),
#     project-venv read-only harvest (last resort) --------------------------

def _uv_exe():
    return shutil.which("uv")


def _setup_uv_scratch_venv(uv_exe):
    """Disposable venv via `uv venv` -- %TEMP%-side, never the project
    .venv. uv manages its own Python distribution (no embeddable-zip
    download needed for this path). Idempotent: reuses an existing scratch
    venv if present."""
    python_exe = os.path.join(UV_SCRATCH_VENV_DIR, "Scripts", "python.exe")
    if os.path.exists(python_exe):
        print(f"[setup_tensorrt] reusing uv scratch venv: {UV_SCRATCH_VENV_DIR}")
        return python_exe
    print(f"[setup_tensorrt] creating uv scratch venv: {UV_SCRATCH_VENV_DIR}")
    r = subprocess.run([uv_exe, "venv", UV_SCRATCH_VENV_DIR, "--python", "3.12"],
                        capture_output=True, text=True)
    print(f"[setup_tensorrt]   rc={r.returncode}\n" +
          "\n".join(f"    {l}" for l in (r.stdout + r.stderr).strip().splitlines()))
    if r.returncode != 0 or not os.path.exists(python_exe):
        raise RuntimeError("uv venv creation failed")
    return python_exe


def _uv_pip_install(uv_exe, python_exe, package):
    cmd = [uv_exe, "pip", "install", "--python", python_exe, package]
    print(f"[setup_tensorrt] uv pip install --python <scratch> {package} ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
    print(f"[setup_tensorrt]   rc={r.returncode}\n" +
          "\n".join(f"    {l}" for l in tail))
    return r.returncode == 0


def _setup_embeddable_python():
    """FALLBACK ONLY (used when `uv` isn't on PATH). Disposable venv recipe
    (verbatim from tools_dev/convert_fp16.py's header): a genuine CPython
    3.12 EMBEDDABLE distribution, never installed into G:\\SyLC-main\\.venv."""
    python_exe = os.path.join(PY312_EMBED_DIR, "python.exe")
    if os.path.exists(python_exe):
        r = subprocess.run([python_exe, "-m", "pip", "--version"],
                            capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[setup_tensorrt] reusing embeddable Python: {python_exe}")
            return python_exe
    embed_zip = os.path.join(CACHE_DIR, "py312_embed.zip")
    _download(PY312_EMBED_URL, embed_zip, "CPython 3.12.8 embeddable")
    os.makedirs(PY312_EMBED_DIR, exist_ok=True)
    with zipfile.ZipFile(embed_zip) as z:
        z.extractall(PY312_EMBED_DIR)
    pth_files = [f for f in os.listdir(PY312_EMBED_DIR) if f.endswith("._pth")]
    pth_path = os.path.join(PY312_EMBED_DIR, pth_files[0])
    with open(pth_path) as f:
        content = f.read()
    with open(pth_path, "w") as f:
        f.write(content.replace("#import site", "import site"))
    get_pip = os.path.join(CACHE_DIR, "get-pip.py")
    _download(GET_PIP_URL, get_pip, "get-pip.py")
    subprocess.run([python_exe, get_pip], check=True)
    return python_exe


def _pip_install_allow_sdist(python_exe, package):
    """FALLBACK ONLY. Deliberately NO --only-binary=:all: -- that flag is
    exactly what defeats tensorrt-cu13's wheel-stub sdist bootstrap (see
    header). This lets pip build the tiny stub sdist, whose build step
    fetches the real wheel from pypi.nvidia.com."""
    cmd = [python_exe, "-m", "pip", "install", package]
    print(f"[setup_tensorrt] pip install {package} (sdist builds allowed) ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
    print(f"[setup_tensorrt]   rc={r.returncode}\n" +
          "\n".join(f"    {l}" for l in tail))
    return r.returncode == 0


def _site_packages_of(python_exe):
    r = subprocess.run(
        [python_exe, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _harvest(site_packages):
    """Recursively finds each file in HARVEST_PATTERNS under site_packages.
    Returns {final_name: source_path} for whatever was found (partial
    results are fine -- the probe reports exactly what's still missing)."""
    found = {}
    if not os.path.isdir(site_packages):
        return found
    remaining = dict(HARVEST_PATTERNS)
    for dirpath, _dirs, files in os.walk(site_packages):
        if not remaining:
            break
        lower_files = {f.lower(): f for f in files}
        for final_name, patterns in list(remaining.items()):
            for pat in patterns:
                if pat.lower() in lower_files:
                    found[final_name] = os.path.join(dirpath, lower_files[pat.lower()])
                    del remaining[final_name]
                    break
    return found


# --- --sdk-dir: user-provided official SDK, highest-priority route ---------
#
# T4 round-3 (2026-07-29, user directive): the user may hand this script a
# path to an officially-downloaded NVIDIA TensorRT SDK zip, already extracted
# (e.g. C:\TensorRT-11.1.0.106). This is intentionally NOT hardcoded to any
# one TensorRT major version: the exact filenames actually required are
# determined from the REAL PE import table of the onnxruntime_providers_
# tensorrt.dll this run just fetched via NuGet (the same _pe_import_dll_names
# reader used by the diagnostic dependency walk below) -- so this route is a
# durable, version-checked mechanism, not a one-off. If the SDK's own DLLs
# are a different major version than what THIS ORT build imports by exact
# name (e.g. SDK ships nvinfer_11.dll, ORT needs nvinfer_10.dll), it reports
# that mismatch explicitly and harvests nothing for those names -- never
# silently substitutes an incompatible version.
def _required_tensorrt_dep_names(trt_ep_dll_path):
    """Returns the sorted list of non-system/CRT DLL names that
    onnxruntime_providers_tensorrt.dll actually imports (excluding its own
    sibling onnxruntime_providers_shared.dll, which NuGet already provides)."""
    imported = _pe_import_dll_names(trt_ep_dll_path)
    return sorted({
        n for n in imported
        if not _is_system_or_crt(n) and n.lower() != "onnxruntime_providers_shared.dll"
    })


def _harvest_from_sdk_dir(sdk_dir, required_names):
    """Recursively searches sdk_dir (e.g. an extracted TensorRT SDK zip --
    its DLLs live under bin\\, not lib\\, which only holds .lib import
    libraries) for exact-filename (case-insensitive) matches among
    required_names. Returns (found: {name: path}, missing: [name, ...])."""
    found = {}
    if not sdk_dir or not os.path.isdir(sdk_dir):
        return found, sorted(required_names)
    remaining = set(required_names)
    for dirpath, _dirs, files in os.walk(sdk_dir):
        if not remaining:
            break
        lower_files = {f.lower(): f for f in files}
        for name in list(remaining):
            if name.lower() in lower_files:
                found[name] = os.path.join(dirpath, lower_files[name.lower()])
                remaining.discard(name)
    return found, sorted(remaining)


def _acquire_tensorrt_dlls():
    """Returns {final_name: source_path} for the 6-file minimal TensorRT/
    cuBLAS/cuDNN set, trying uv -> pip-allow-sdist -> project-venv-readonly,
    in that order. Never raises; returns whatever it managed to find (the
    probe's own dependency walk reports precisely what's still missing)."""
    uv_exe = _uv_exe()
    if uv_exe:
        try:
            print(f"[setup_tensorrt] uv found: {uv_exe}")
            python_exe = _setup_uv_scratch_venv(uv_exe)
            ok_trt = _uv_pip_install(uv_exe, python_exe,
                                      f"{TENSORRT_PACKAGE}=={TENSORRT_VERSION}")
            ok_cudnn = _uv_pip_install(uv_exe, python_exe, CUDNN_PACKAGE)
            print(f"[setup_tensorrt] uv route: tensorrt={ok_trt} cudnn(+cublas)={ok_cudnn}")
            site_packages = _site_packages_of(python_exe)
            harvested = _harvest(site_packages)
            if len(harvested) == len(HARVEST_PATTERNS):
                return harvested
            print(f"[setup_tensorrt] uv route incomplete "
                  f"({len(harvested)}/{len(HARVEST_PATTERNS)} found) -- "
                  f"falling through to next route")
        except Exception as e:
            print(f"[setup_tensorrt] uv route failed: {e} -- falling through")
    else:
        print("[setup_tensorrt] uv not found on PATH -- skipping primary route")

    print("[setup_tensorrt] trying pip (embeddable Python, sdist builds allowed) ...")
    try:
        python_exe = _setup_embeddable_python()
        ok_trt = _pip_install_allow_sdist(python_exe, f"{TENSORRT_PACKAGE}=={TENSORRT_VERSION}")
        ok_cudnn = _pip_install_allow_sdist(python_exe, CUDNN_PACKAGE)
        print(f"[setup_tensorrt] pip route: tensorrt={ok_trt} cudnn(+cublas)={ok_cudnn}")
        site_packages = _site_packages_of(python_exe)
        harvested = _harvest(site_packages)
        if len(harvested) == len(HARVEST_PATTERNS):
            return harvested
        print(f"[setup_tensorrt] pip route incomplete "
              f"({len(harvested)}/{len(HARVEST_PATTERNS)} found) -- "
              f"falling through to project-venv read-only harvest")
    except Exception as e:
        print(f"[setup_tensorrt] pip route failed: {e} -- falling through")

    print(f"[setup_tensorrt] trying READ-ONLY harvest from {PROJECT_VENV_SITE_PACKAGES} "
          f"(never modifies .venv) ...")
    return _harvest(PROJECT_VENV_SITE_PACKAGES)


# --- PE import-table reader (diagnostic dependency walk) -------------------

def _pe_import_dll_names(dll_path):
    """Pure-stdlib PE import-table reader: returns the list of DLL names a
    PE image imports, straight from its Import Directory. No `pefile`
    dependency. Used for the DIAGNOSTIC dependency walk (see header: this is
    no longer the pass/fail gate, but still useful for naming exactly which
    file is missing)."""
    with open(dll_path, "rb") as f:
        data = f.read()
    if data[:2] != b"MZ":
        raise ValueError(f"{dll_path}: not a PE file (bad DOS signature)")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError(f"{dll_path}: bad NT signature")
    coff_off = e_lfanew + 4
    _machine, num_sections = struct.unpack_from("<HH", data, coff_off)
    opt_hdr_off = coff_off + 20
    opt_magic = struct.unpack_from("<H", data, opt_hdr_off)[0]
    is_pe32_plus = (opt_magic == 0x20B)
    data_dir_off = opt_hdr_off + (112 if is_pe32_plus else 96)
    import_dir_rva, _import_dir_size = struct.unpack_from("<II", data, data_dir_off + 8)
    if import_dir_rva == 0:
        return []
    opt_hdr_size = struct.unpack_from("<H", data, coff_off + 16)[0]
    section_table_off = opt_hdr_off + opt_hdr_size
    sections = []
    for i in range(num_sections):
        off = section_table_off + i * 40
        virt_size, virt_addr = struct.unpack_from("<II", data, off + 8)
        raw_ptr = struct.unpack_from("<I", data, off + 20)[0]
        sections.append((virt_addr, virt_size, raw_ptr))

    def rva_to_offset(rva):
        for virt_addr, virt_size, raw_ptr in sections:
            if virt_addr <= rva < virt_addr + max(virt_size, 1):
                return raw_ptr + (rva - virt_addr)
        raise ValueError(f"RVA {rva:#x} not in any section")

    def read_cstr(offset):
        end = data.index(b"\x00", offset)
        return data[offset:end].decode("ascii", errors="replace")

    names = []
    descriptor_off = rva_to_offset(import_dir_rva)
    i = 0
    while True:
        entry_off = descriptor_off + i * 20
        entry = data[entry_off:entry_off + 20]
        if len(entry) < 20 or entry == b"\x00" * 20:
            break
        name_rva = struct.unpack_from("<I", entry, 12)[0]
        names.append(read_cstr(rva_to_offset(name_rva)))
        i += 1
    return names


def _is_system_or_crt(dll_name):
    lname = dll_name.lower()
    return any(lname.startswith(p) for p in _SYSTEM_OR_CRT_PREFIXES)


# --- staging assembly (merge, not wipe -- round-1 review fix) --------------

def _assemble_stage(ort_dlls, extra_dlls):
    """MERGES this run's freshly-fetched DLLs into STAGE_DIR -- does NOT
    wipe it first. Only the specific files this run itself provides are
    overwritten; anything else already present survives untouched. Manifest
    reflects the FULL current directory contents."""
    os.makedirs(STAGE_DIR, exist_ok=True)
    for name, path in {**ort_dlls, **extra_dlls}.items():
        shutil.copy2(path, os.path.join(STAGE_DIR, name))
    manifest = []
    for name in os.listdir(STAGE_DIR):
        full = os.path.join(STAGE_DIR, name)
        if os.path.isfile(full):
            manifest.append((name, os.path.getsize(full)))
    return manifest


# --- the REAL completeness gate: a live ORT C-API session test ------------

def _real_tensorrt_session_test(runtime_dir):
    """Authoritative TensorRT-EP completeness gate (see header point 5 for
    why the old bare-LoadLibraryExW test was a false negative for this
    specific DLL). Exercises the exact call sequence ONNX Runtime itself
    uses: OrtGetApiBase()->GetApi(24) -> CreateEnv -> CreateSessionOptions ->
    OrtSessionOptionsAppendExecutionProvider_Tensorrt(options, device_id=0).
    Returns (success: bool, message: str)."""
    k = _kernel32()
    k.SetDllDirectoryW(runtime_dir)  # so onnxruntime.dll's own on-demand
    # provider loads find sibling nvinfer_10.dll etc. in runtime_dir.

    ort_path = os.path.join(runtime_dir, "onnxruntime.dll")
    h_ort = k.LoadLibraryExW(ort_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
    if not h_ort:
        return False, f"onnxruntime.dll failed to load: WinError {ctypes.get_last_error()}"

    get_api_base = k.GetProcAddress(h_ort, SYM_API_BASE)
    trt_ep_func = k.GetProcAddress(h_ort, SYM_TENSORRT_EP)
    if not get_api_base or not trt_ep_func:
        return False, "OrtGetApiBase or OrtSessionOptionsAppendExecutionProvider_Tensorrt not exported"

    get_api_base_fn = ctypes.CFUNCTYPE(ctypes.c_void_p)(get_api_base)
    api_base_ptr = get_api_base_fn()
    if not api_base_ptr:
        return False, "OrtGetApiBase() returned NULL"

    # struct OrtApiBase { const OrtApi*(*GetApi)(uint32_t); const char*(*GetVersionString)(void); }
    get_api_fnptr = ctypes.cast(api_base_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    get_api_fn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)(get_api_fnptr)
    api_ptr = get_api_fn(ORT_API_VERSION)
    if not api_ptr:
        return False, f"OrtApiBase::GetApi({ORT_API_VERSION}) returned NULL"

    api_array = ctypes.cast(api_ptr, ctypes.POINTER(ctypes.c_void_p))

    def api_fn(idx, restype, *argtypes):
        ptr = api_array[idx]
        if not ptr:
            raise RuntimeError(f"OrtApi field {idx} is NULL")
        return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)

    try:
        create_env = api_fn(IDX_CREATE_ENV, ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))
        create_session_options = api_fn(IDX_CREATE_SESSION_OPTIONS, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_void_p))
        get_error_message = api_fn(IDX_GET_ERROR_MESSAGE, ctypes.c_char_p, ctypes.c_void_p)
        release_env = api_fn(IDX_RELEASE_ENV, None, ctypes.c_void_p)
        release_status = api_fn(IDX_RELEASE_STATUS, None, ctypes.c_void_p)
        release_session_options = api_fn(IDX_RELEASE_SESSION_OPTIONS, None, ctypes.c_void_p)
        append_trt = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int)(trt_ep_func)
    except RuntimeError as e:
        return False, str(e)

    env = ctypes.c_void_p()
    ORT_LOGGING_LEVEL_WARNING = 2
    status = create_env(ORT_LOGGING_LEVEL_WARNING, b"trt_setup_probe", ctypes.byref(env))
    if status:
        msg = get_error_message(status)
        release_status(status)
        return False, f"CreateEnv failed: {msg.decode('utf-8', 'replace') if msg else '?'}"

    session_options = ctypes.c_void_p()
    status = create_session_options(ctypes.byref(session_options))
    if status:
        msg = get_error_message(status)
        release_status(status)
        release_env(env)
        return False, f"CreateSessionOptions failed: {msg.decode('utf-8', 'replace') if msg else '?'}"

    status = append_trt(session_options, 0)
    ok = not status
    msg = ""
    if status:
        raw = get_error_message(status)
        msg = raw.decode("utf-8", "replace") if raw else "(no message)"
        release_status(status)

    release_session_options(session_options)
    release_env(env)
    return ok, (msg if not ok else "AppendExecutionProvider_Tensorrt succeeded")


# --- diagnostic-only bare-load dependency walk ------------------------------

def _diagnostic_dependency_walk(runtime_dir, manifest_names):
    """Only invoked when _real_tensorrt_session_test() fails, to help name
    exactly which file is missing. NOT the pass/fail gate (see header)."""
    k = _kernel32()
    trt_path = os.path.join(runtime_dir, "onnxruntime_providers_tensorrt.dll")
    if not os.path.exists(trt_path):
        return [], ["onnxruntime_providers_tensorrt.dll itself is missing from " + runtime_dir]
    try:
        imported = _pe_import_dll_names(trt_path)
    except Exception as e:
        return [], [f"PE import-table read failed: {e}"]
    walk = []
    for dep_name in imported:
        if _is_system_or_crt(dep_name):
            continue
        present = dep_name in manifest_names
        loadable = False
        if present:
            h = k.LoadLibraryExW(os.path.join(runtime_dir, dep_name), None,
                                  LOAD_WITH_ALTERED_SEARCH_PATH)
            loadable = bool(h)
            if h:
                k.FreeLibrary(h)
        walk.append({"name": dep_name, "present": present, "loadable": loadable})
    missing = [w["name"] for w in walk if not w["present"]]
    issues = []
    if missing:
        issues.append("MISSING native dependencies: " + ", ".join(missing))
    return walk, issues


def _probe(manifest_names, runtime_dir):
    """Runs the completeness probe against `runtime_dir`. Read-only: never
    creates, deletes, or modifies anything in runtime_dir. The TensorRT-EP
    gate is `_real_tensorrt_session_test()` (see header); the old bare-load
    dependency walk only runs as a diagnostic when that fails."""
    k = _kernel32()
    report = {"issues": []}

    ort_path = os.path.join(runtime_dir, "onnxruntime.dll")
    h_ort = k.LoadLibraryExW(ort_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
    report["onnxruntime_loads"] = bool(h_ort)
    if not h_ort:
        report["issues"].append(
            f"onnxruntime.dll failed to load: WinError {ctypes.get_last_error()}")
        return report

    def has_symbol(sym):
        return bool(k.GetProcAddress(h_ort, sym))

    report["api_base_found"] = has_symbol(SYM_API_BASE)
    report["tensorrt_symbol_found"] = has_symbol(SYM_TENSORRT_EP)
    report["cuda_symbol_found"] = has_symbol(SYM_CUDA_EP)
    if not report["tensorrt_symbol_found"]:
        report["issues"].append(
            "OrtSessionOptionsAppendExecutionProvider_Tensorrt not exported "
            "-- this onnxruntime.dll is not a GPU/TensorRT-EP build")
    k.FreeLibrary(h_ort)

    shared_path = os.path.join(runtime_dir, "onnxruntime_providers_shared.dll")
    h_shared = k.LoadLibraryExW(shared_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
    report["providers_shared_loads"] = bool(h_shared)
    if h_shared:
        k.FreeLibrary(h_shared)
    else:
        report["issues"].append(
            f"onnxruntime_providers_shared.dll failed to load: "
            f"WinError {ctypes.get_last_error()}")

    cuda_path = os.path.join(runtime_dir, "onnxruntime_providers_cuda.dll")
    if os.path.exists(cuda_path):
        h_cuda = k.LoadLibraryExW(cuda_path, None, LOAD_WITH_ALTERED_SEARCH_PATH)
        report["providers_cuda_loads"] = bool(h_cuda)
        if h_cuda:
            k.FreeLibrary(h_cuda)
        else:
            report["providers_cuda_error"] = ctypes.get_last_error()
    else:
        report["providers_cuda_loads"] = None

    if report["tensorrt_symbol_found"]:
        ok, msg = _real_tensorrt_session_test(runtime_dir)
        report["tensorrt_ep_registers"] = ok
        report["tensorrt_ep_message"] = msg
        if not ok:
            report["issues"].append(f"TensorRT EP registration FAILED: {msg}")
            walk, walk_issues = _diagnostic_dependency_walk(runtime_dir, manifest_names)
            if walk:
                report["dependency_walk"] = walk
            report["issues"].extend(walk_issues)
    else:
        report["tensorrt_ep_registers"] = False

    return report


def _print_report(report):
    print("\n--- completeness probe ---")
    for key in ("onnxruntime_loads", "api_base_found", "tensorrt_symbol_found",
                "cuda_symbol_found", "providers_shared_loads",
                "providers_cuda_loads"):
        if key in report:
            print(f"  {key}: {report[key]}")
    if "providers_cuda_error" in report:
        print(f"  providers_cuda_error: WinError {report['providers_cuda_error']} "
              f"({ctypes.WinError(report['providers_cuda_error']).strerror})")
    if "tensorrt_ep_registers" in report:
        print(f"  tensorrt_ep_registers (REAL session-API gate): "
              f"{report['tensorrt_ep_registers']}  -- {report.get('tensorrt_ep_message', '')}")
    if "dependency_walk" in report:
        print("  diagnostic dependency_walk (bare-load, NOT the gate -- see script header):")
        for w in report["dependency_walk"]:
            state = "present+loadable" if w["loadable"] else (
                "present but NOT loadable" if w["present"] else "MISSING")
            print(f"    {w['name']}: {state}")
    if report["issues"]:
        print("  ISSUES:")
        for issue in report["issues"]:
            print(f"    - {issue}")
    else:
        print("  no issues -- runtime is complete")


def _ort_version_string(runtime_dir):
    """Best-effort onnxruntime.dll version string for the marker file
    (OrtApiBase::GetVersionString -- the second function pointer in
    OrtApiBase, same technique as tests/synth3d/test_ort_runtime.py)."""
    try:
        k = _kernel32()
        h = k.LoadLibraryExW(os.path.join(runtime_dir, "onnxruntime.dll"), None,
                              LOAD_WITH_ALTERED_SEARCH_PATH)
        if not h:
            return "?"
        get_api_base = ctypes.CFUNCTYPE(ctypes.c_void_p)(
            k.GetProcAddress(h, SYM_API_BASE))
        base_ptr = get_api_base()
        get_version = ctypes.CFUNCTYPE(ctypes.c_char_p)(
            ctypes.cast(base_ptr, ctypes.POINTER(ctypes.c_void_p))[1])
        version = get_version()
        k.FreeLibrary(h)
        return version.decode("utf-8", "replace") if version else "?"
    except Exception as e:
        return f"?({e})"


def _one_line(text):
    """Flattens a child process's multi-line output into a single marker line.

    The marker is parsed LINE BY LINE by the player (`probe_model=` lines) and
    by the tests, so an embedded newline would turn the tail of a detail into
    what looks like a top-level field -- round 3's single-graph marker really
    did end with a bare `RESULT=OK ...` line for exactly that reason."""
    return " | ".join(l.strip() for l in str(text).splitlines() if l.strip())


# What a PASSING graph contributes to the marker: the probe's own two verdict
# lines, nothing else. A real TensorRT build writes dozens of WARNING lines to
# the child's stderr ("Was not able to infer kOPT value(s) for tensor ..." is
# emitted once per RoPE Cast node on the dynamic-axes export alone), which
# would bury the value-gate numbers under kilobytes of console noise in a file
# whose whole job is to be read at a glance. A FAILING graph keeps its output
# verbatim -- there the noise IS the diagnosis, and it never reaches a marker
# anyway, since a failure writes none.
_PROBE_VERDICT_PREFIXES = ("VALUE_GATE", "RESULT=")


def _probe_verdict(out):
    kept = [l.strip() for l in out.splitlines()
            if l.strip().startswith(_PROBE_VERDICT_PREFIXES)]
    return " | ".join(kept) if kept else _one_line(out)


def _run_one_engine_probe(runtime_dir, model, width, height, timeout_s):
    """One graph: a REAL TensorRT engine build + inference (via
    tests/synth3d/_trt_engine_probe.py's "engine" mode -- a fresh,
    registry-independent DepthEngine, in ITS OWN child process so a native
    crash here can never take this script down) against runtime_dir.
    Returns a result dict."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, ENGINE_PROBE_SCRIPT, "engine", model, runtime_dir,
             "--width", str(width), "--height", str(height)],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {"name": os.path.basename(model), "side": width, "height": height,
                "ok": False,
                "elapsed": elapsed,
                "detail": f"TIMEOUT after {elapsed:.1f}s (no result within budget)"}
    elapsed = time.time() - t0
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    ok = (proc.returncode == 0 and "RESULT=OK" in out)
    return {"name": os.path.basename(model), "side": width, "height": height,
            "ok": ok,
            "elapsed": elapsed,
            "detail": _probe_verdict(out) if ok else (
                f"child process did not complete cleanly "
                f"(returncode={proc.returncode}): {out}")}


def _run_engine_probe(runtime_dir, timeout_s=ENGINE_PROBE_TIMEOUT_S):
    """Probes EVERY graph `_engine_probe_graphs()` found, one child process
    each. Returns (ok: bool, total_elapsed_s: float, results: [dict, ...]).

    A failure of ANY graph fails the whole run (no marker -- see main()). The
    marker format could in principle attest just the graphs that did build, but
    a failed build does not come labelled: "this one graph uses an op TensorRT
    cannot place" and "this TensorRT assembly is broken" look identical from
    here, and the second is the case that ends in a hard native abort during
    playback (round 3's cuDNN9 finding). The conservative reading is the safe
    one, and it costs nothing anyone is waiting on -- DirectML keeps working
    exactly as it does on every machine without this opt-in runtime.

    The remaining graphs are still probed rather than short-circuited, so one
    run reports every problem instead of revealing them one rerun at a time --
    and since each engine that DID build stays in the on-disk cache, a rerun
    after a fix only pays for what actually failed.
    """
    if not os.path.exists(ENGINE_PROBE_SCRIPT):
        return False, 0.0, [{"name": "-", "side": 0, "ok": False, "elapsed": 0.0,
                             "detail": f"probe script not found: {ENGINE_PROBE_SCRIPT}"}]
    graphs = _engine_probe_graphs()
    if not graphs:
        # Never write an "attests nothing" marker: the player reads a marker
        # with no probe_model= line as the round-3 whole-directory attestation
        # (see synth3d_marker_attests), so an empty probe would hand TensorRT
        # every graph without having built a single engine.
        return False, 0.0, [{"name": "-", "side": 0, "ok": False, "elapsed": 0.0,
                             "detail": f"no depth-preset model found under "
                                       f"{os.path.join(ROOT, 'models')} -- nothing to "
                                       f"verify, and a marker attesting no graph at "
                                       f"all would re-open the whole directory"}]
    print(f"[setup_tensorrt] engine-probe graphs ({len(graphs)}, every depth-preset "
          f"and adaptive candidate present on disk):")
    for path, width, height in graphs:
        print(f"    {os.path.basename(path)} @ {width}x{height}")
    results, t0 = [], time.time()
    for i, (path, width, height) in enumerate(graphs, 1):
        print(f"[setup_tensorrt] [{i}/{len(graphs)}] building "
              f"{os.path.basename(path)} @ {width}x{height} ...", flush=True)
        r = _run_one_engine_probe(
            runtime_dir, path, width, height, timeout_s)
        print(f"[setup_tensorrt] [{i}/{len(graphs)}] {'OK' if r['ok'] else 'FAILED'} "
              f"in {r['elapsed']:.1f}s -- {_one_line(r['detail'])}", flush=True)
        results.append(r)
    return all(r["ok"] for r in results), time.time() - t0, results


def _write_verified_marker(runtime_dir, results):
    """Writes the marker attesting EVERY graph in `results`.

    One `probe_model=` line per attested graph: that is what the player's
    per-graph gate reads (SyLC_3D_Player.synth3d_marker_attests), and a graph
    absent from this list gets the DirectML runtime instead of a cold
    in-playback engine compile. The per-graph evidence sits alongside in
    `probe_detail_<n>=` lines rather than replacing them, so the marker still
    records the actual value-gate numbers each engine produced."""
    marker_path = os.path.join(runtime_dir, TRT_VERIFIED_MARKER)
    lines = [
        f"verified_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"onnxruntime_version={_ort_version_string(runtime_dir)}",
        f"tensorrt_wheel_pinned={TENSORRT_VERSION}",
    ]
    lines += [f"probe_model={r['name']}" for r in results]
    lines.append(f"probe_elapsed_s={sum(r['elapsed'] for r in results):.1f}")
    lines += [f"probe_detail_{i}={r['name']} grid={r['side']}x"
              f"{r.get('height', r['side'])} "
              f"elapsed_s={r['elapsed']:.1f} {_one_line(r['detail'])}"
              for i, r in enumerate(results, 1)]
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return marker_path


def _remove_stale_marker(runtime_dir):
    marker_path = os.path.join(runtime_dir, TRT_VERIFIED_MARKER)
    if os.path.exists(marker_path):
        os.remove(marker_path)
        return marker_path
    return None


MANUAL_ROUTE = """
[setup_tensorrt] Automated acquisition failed on this run (network/index
unreachable, or `uv`/pip both unavailable). Manual fallback:

  1. `uv pip install --upgrade tensorrt-cu13=={version}` in ANY venv (a
     scratch one is fine) -- this is the exact proven route. If `uv` is not
     installed, `pip install tensorrt-cu13=={version}` (WITHOUT
     --only-binary=:all:) works too, just slower.
  2. `uv pip install nvidia-cudnn-cu13` in the same venv (pulls in
     nvidia-cublas transitively).
  3. Copy these 6 files from that venv's site-packages into {stage}:
     tensorrt_libs/nvinfer_10.dll, tensorrt_libs/nvinfer_plugin_10.dll,
     tensorrt_libs/nvonnxparser_10.dll, nvidia/cudnn/bin/cudnn64_9.dll,
     nvidia/cu13/bin/x86_64/cublas64_13.dll,
     nvidia/cu13/bin/x86_64/cublasLt64_13.dll (exact sub-paths may shift
     between package versions -- search by filename if not found there).
  4. Re-run this script (no flags) -- files placed in {stage} are MERGED
     with the ORT DLLs this script fetches itself and auto-promoted to
     {final} the moment the probe passes. Or run
     `setup_tensorrt.py --verify-manual` against a directly-populated
     {final} for a read-only, zero-network check.
""".format(version=TENSORRT_VERSION, stage=STAGE_DIR, final=FINAL_DIR)


def main():
    argv = sys.argv[1:]
    verify_manual = "--verify-manual" in argv or "--probe-only" in argv
    engine_probe = "--engine-probe" in argv
    fetch_engines_flag = "--fetch-engines" in argv
    sdk_dir = None
    if "--sdk-dir" in argv:
        idx = argv.index("--sdk-dir")
        if idx + 1 < len(argv):
            sdk_dir = argv[idx + 1]
        else:
            print("[setup_tensorrt] --sdk-dir requires a path argument"); return 4

    print(f"[setup_tensorrt] cache dir: {CACHE_DIR}")
    print(f"[setup_tensorrt] staging dir: {STAGE_DIR}")
    print(f"[setup_tensorrt] final dir:   {FINAL_DIR}")
    if sdk_dir:
        print(f"[setup_tensorrt] --sdk-dir:    {sdk_dir} (highest-priority route)")
    print()

    if verify_manual:
        print(f"[setup_tensorrt] --verify-manual: read-only probe of {FINAL_DIR} "
              f"as-is. No downloads, no staging, nothing is ever deleted or "
              f"modified in this mode.\n")
        if not os.path.isdir(FINAL_DIR):
            print(f"[setup_tensorrt] {FINAL_DIR} does not exist -- nothing to "
                  f"verify. Populate it per BUILD.md's manual route and rerun.")
            return 1
        names = {n for n in os.listdir(FINAL_DIR)
                  if os.path.isfile(os.path.join(FINAL_DIR, n))}
        report = _probe(names, FINAL_DIR)
        _print_report(report)
        if not report["issues"]:
            print(f"\n[setup_tensorrt] VERIFY-MANUAL: PASS -- {FINAL_DIR} is a "
                  f"complete runtime.")
            return 0
        print(f"\n[setup_tensorrt] VERIFY-MANUAL: FAIL -- see ISSUES above. "
              f"Nothing in {FINAL_DIR} was modified.")
        return 2

    if fetch_engines_flag:
        if HF_ENGINE_REVISION == "PENDING_UPLOAD":
            print("[setup_tensorrt] engine cache not published yet "
                  "-- skipping --fetch-engines")
        else:
            print(f"[setup_tensorrt] --fetch-engines: downloading the published "
                  f"sm89 / TensorRT 10.16.1.11 engine cache. RTX 40-series only; "
                  f"harmless but useless on any other GPU. This does NOT skip "
                  f"the probe below -- the marker is still only written after a "
                  f"real local build passes the value gate.\n")
            fetch_engines(os.path.join(FINAL_DIR, TRT_CACHE_DIRNAME))

    if engine_probe:
        print(f"[setup_tensorrt] --engine-probe: REAL TensorRT engine build + "
              f"inference against {FINAL_DIR} as-is, ONCE PER DEPTH-PRESET GRAPH "
              f"(registration-only completeness above is NOT sufficient proof a real "
              f"build succeeds -- see task-4-report.md, \"The crash\"). Each build runs "
              f"isolated in a child process: a native crash there becomes a reported "
              f"FAILURE, never takes this script down. Budget MINUTES PER GRAPH on a "
              f"cold engine cache.\n")
        if not os.path.isdir(FINAL_DIR):
            print(f"[setup_tensorrt] {FINAL_DIR} does not exist -- nothing to probe.")
            return 1
        ok, elapsed, results = _run_engine_probe(FINAL_DIR)
        print(f"\n[setup_tensorrt] engine probe: {'OK' if ok else 'FAILED'} -- "
              f"{sum(1 for r in results if r['ok'])}/{len(results)} graphs in "
              f"{elapsed:.1f}s")
        for r in results:
            print(f"    {'OK    ' if r['ok'] else 'FAILED'} {r['name']} @ "
                  f"{r['side']}x{r.get('height', r['side'])} "
                  f"{r['elapsed']:.1f}s -- {_one_line(r['detail'])}")
        if ok:
            marker_path = _write_verified_marker(FINAL_DIR, results)
            print(f"\n[setup_tensorrt] ENGINE-PROBE: PASS -- wrote {marker_path} "
                  f"attesting {len(results)} graph(s). _synth3d_ort_dir() will now "
                  f"prefer {FINAL_DIR} for each of them.")
            return 0
        removed = _remove_stale_marker(FINAL_DIR)
        if removed:
            print(f"[setup_tensorrt] removed STALE marker {removed} (this run's "
                  f"probe failed -- a previous verification no longer applies).")
        print(f"\n[setup_tensorrt] ENGINE-PROBE: FAIL -- no marker written. "
              f"_synth3d_ort_dir() will NOT prefer {FINAL_DIR} (falls back to "
              f"DirectML) until a rerun of --engine-probe passes.")
        return 2

    print("=== Step 1: NuGet Microsoft.ML.OnnxRuntime.Gpu[.Windows] ===")
    meta_version, windows_version = _nuget_meta_version_and_dep(
        NUGET_META_ID, NUGET_WINDOWS_ID)
    print(f"[setup_tensorrt] {NUGET_META_ID} meta version={meta_version}; "
          f"pinned dependency {NUGET_WINDOWS_ID}=={windows_version}")
    ort_dlls = _nuget_fetch_windows_native(windows_version)
    print(f"[setup_tensorrt] ORT native DLLs: " +
          ", ".join(f"{n} ({os.path.getsize(p)/1e6:.1f}MB)"
                     for n, p in sorted(ort_dlls.items())))

    sdk_dlls = {}
    if sdk_dir:
        print(f"\n=== Step 1b: --sdk-dir acquisition route ({sdk_dir}) ===")
        trt_ep_path = ort_dlls.get("onnxruntime_providers_tensorrt.dll")
        if not trt_ep_path:
            print("[setup_tensorrt] onnxruntime_providers_tensorrt.dll was not among "
                  "the fetched ORT DLLs -- cannot determine the exact filenames THIS "
                  "ORT build requires; skipping --sdk-dir this run.")
        else:
            required = _required_tensorrt_dep_names(trt_ep_path)
            print(f"[setup_tensorrt] this ORT build's onnxruntime_providers_tensorrt.dll "
                  f"requires (real PE import table, not a guess): {', '.join(required)}")
            sdk_dlls, sdk_missing = _harvest_from_sdk_dir(sdk_dir, required)
            if sdk_dlls:
                print(f"[setup_tensorrt] --sdk-dir provided {len(sdk_dlls)}/{len(required)}: "
                      f"{', '.join(sorted(sdk_dlls))}")
            if sdk_missing:
                print(f"[setup_tensorrt] --sdk-dir did NOT provide: "
                      f"{', '.join(sdk_missing)} -- if this is a full miss, the SDK is "
                      f"very likely a different TensorRT MAJOR VERSION than this ORT "
                      f"build imports by exact filename (Windows matches static "
                      f"imports by literal name, not \"close enough\" version); falling "
                      f"through to the uv/pip/project-venv routes below for these.")

    print(f"\n=== Step 2: TensorRT {TENSORRT_VERSION} + cuDNN/cuBLAS (uv route) ===")
    extra_dlls = _acquire_tensorrt_dlls()
    print(f"[setup_tensorrt] harvested {len(extra_dlls)}/{len(HARVEST_PATTERNS)}: " +
          (", ".join(sorted(extra_dlls)) if extra_dlls else "(none)"))
    missing_harvest = set(HARVEST_PATTERNS) - set(extra_dlls)
    if missing_harvest:
        print(f"[setup_tensorrt] NOT harvested: {', '.join(sorted(missing_harvest))}")

    if sdk_dlls:
        overridden = sorted(set(sdk_dlls) & set(extra_dlls))
        if overridden:
            print(f"[setup_tensorrt] --sdk-dir (highest priority) overrides the uv/pip/"
                  f"venv route for: {', '.join(overridden)}")
        extra_dlls = {**extra_dlls, **sdk_dlls}

    print("\n=== Step 3: assemble staging dir + completeness probe ===")
    manifest = _assemble_stage(ort_dlls, extra_dlls)
    total_size = sum(sz for _, sz in manifest)
    print(f"[setup_tensorrt] staged {len(manifest)} files, "
          f"{total_size/1e9:.3f} GB total, in {STAGE_DIR}")

    report = _probe({n for n, _ in manifest}, STAGE_DIR)
    _print_report(report)

    if not report["issues"]:
        print(f"\n[setup_tensorrt] PROBE PASSED -- promoting {STAGE_DIR} -> {FINAL_DIR}")
        if os.path.isdir(FINAL_DIR):
            shutil.rmtree(FINAL_DIR)
        os.replace(STAGE_DIR, FINAL_DIR)  # same-volume rename, near-atomic
        print(f"[setup_tensorrt] DONE. {FINAL_DIR} is ready.")
        return 0

    print(MANUAL_ROUTE)
    print(f"[setup_tensorrt] BLOCKED -- {FINAL_DIR} was NOT created/touched. "
          f"DirectML-only behavior is unaffected. Partial assembly left at "
          f"{STAGE_DIR} for inspection (safe to delete).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
