# trt_engines.py
"""Turns an UNVERIFIED TensorRT runtime into a verified one, and writes the
`.trt_verified` marker that lets the player use it.

STDLIB ONLY at module level, and NO Qt -- the same rule as model_fetcher.py,
trt_runtime.py and trt_fetcher.py, for the same two reasons: this module is
compiled into the frozen player, so a third-party dependency here would have to
be bundled too; and keeping Qt out means the whole planning/downloading/marker
path is testable without a QApplication. The Qt layer lives in
model_download_dialog.py.

`numpy` and `mvc_demuxer_cpp` ARE imported -- but only inside `probe_main`,
which never runs in the player's own process (see below). The parent process
that plans, downloads and writes the marker touches neither.

This is the piece whose absence is why no v5.2.0 user can enable TensorRT.
`tools_dev/setup_tensorrt.py --engine-probe` shells out to
`tests/synth3d/_trt_engine_probe.py`, `tests/` is not distributed, and
`_run_engine_probe` refuses outright when that script is missing -- so no marker
is ever written and the player keeps refusing the runtime. What that script does
is reproduced here, in a file that ships.

--- WHY EVERY ENGINE BUILD RUNS IN A CHILD PROCESS ---

An incomplete or incompatible TensorRT assembly does not raise. It takes the
process down with a hard native abort -- observed verbatim as "Invalid handle.
Cannot load symbol cudnnCreate", no Python traceback, no exception to catch.
setup_tensorrt.py's header documents that crash and
tests/synth3d/_trt_engine_probe.py exists precisely so it becomes a reported
subprocess failure instead of a dead pytest host. Here the host is the PLAYER,
so the stake is a user's playback session rather than a test run, and "playback
never dies for 3D" makes the isolation non-negotiable.

In a frozen build there is no Python interpreter to spawn: `sys.executable` IS
the player's own exe. So the child is the player, re-invoked with
`PROBE_ARGV_FLAG` -- a hidden argv token its `__main__` block answers by
running `probe_main` and exiting BEFORE any Qt or GUI initialisation. That is
the pattern `SYLC_EXPORT_SELFTEST` already establishes in SyLC_3D_Player.py for
the release build's export self-test. From a source checkout the same command
is `[sys.executable, <this file>, PROBE_ARGV_FLAG, ...]`, i.e. this module run
as a script, which needs no player at all.

The child reports through a FILE, not only through stdout, and that is
deliberate: the frozen player is a GUI-subsystem binary, and SYLC_EXPORT_SELFTEST
writes its verdict to a file for the same reason. Its stdout is captured too and
used as the fallback diagnosis, because the ABSENCE of both a result file and a
`RESULT=` line is exactly the signature of a native abort.

--- WHAT `.trt_verified` COSTS TO EARN ---

A real engine build plus an inference that passes the near/far VALUE GATE, per
graph, on this machine. Roughly 200 s per graph on a cold cache and 21 graphs,
so about 70 minutes -- which is what the published engine cache is for: when the
detected architecture and the pinned TensorRT version match a published set, the
engines are downloaded and the probe then finds them in the cache and finishes
in seconds.

**Downloaded engines never replace the probe.** They only make it fast. The
marker is written by `verify_runtime` and by nothing else, after every graph has
actually been built and has actually produced numerically correct output on this
GPU.

A failure on ANY graph means NO marker, which is setup_tensorrt.py's existing
rule and is kept for its reason: "this one graph uses an op TensorRT cannot
place" and "this TensorRT assembly is broken" look identical from outside, and
the second is the one that ends in a hard native abort during playback. The
conservative reading is the safe one, and it costs nothing anyone is waiting on
-- DirectML keeps working exactly as it does on every machine without this
opt-in runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from sylc import trt_fetcher

# The hidden argv token the frozen player answers by running `probe_main`.
# SyLC_3D_Player.py's `__main__` block matches this literal before it does
# anything else; tests/models/test_trt_engines.py pins the two equal.
PROBE_ARGV_FLAG = "--sylc-trt-engine-probe"

# Pre-built engines published alongside the ONNX packs. TensorRT engines are
# portable across NEITHER compute capability NOR TensorRT version, which is why
# the directory names state both and why the table below is keyed on both.
#
# The version half of that key is `trt_fetcher.TENSORRT_VERSION`, deliberately:
# move the pin and every published set stops matching, so a user on the new
# version compiles locally instead of being handed engines no runtime here can
# deserialise. That is the correct failure -- slow, not broken.
HF_ENGINE_REPO = "Symphoenix/sylc_TRT"
HF_ENGINE_REVISION = "b1e70aab24abde0ecf82a1dce8c6689c1c71f935"
PUBLISHED_ENGINE_SETS = {
    (89, "10.16.1.11"): "trt/sm89-trt10.16.1.11/",   # Ada / RTX 40-series
}

# What may be copied into the engine cache. The published directory also holds a
# README.md, and `setup_tensorrt.fetch_engines` -- which rejects only entries
# containing a "/" -- would put it there. ORT looks its cache entries up by
# exact filename so a stray README is inert, but the cache directory is one the
# player owns, and "43 files fetched" for 42 cache entries is a count that lies.
# Allow-list the two suffixes ORT actually writes instead.
ENGINE_SUFFIXES = (".engine", ".profile")

TRT_CACHE_DIRNAME = "trt_cache"
TRT_VERIFIED_MARKER = ".trt_verified"

HF_TREE_URL = "https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

# The HuggingFace tree API paginates and advertises the next page in a
# `Link: <url>; rel="next"` header. `setup_tensorrt._hf_list_entries` reads one
# page and stops -- which happens to be enough for this repository TODAY (70
# entries came back on one page, measured 2026-07-31, no Link header at all),
# and would silently truncate the day a second architecture is published. A
# truncated listing does not fail: it fetches fewer engines and the probe then
# pays a full cold compile for the ones that were dropped, which reads as
# TensorRT being slow rather than as a bug here.
_LINK_NEXT = re.compile(r"<([^>]+)>\s*;\s*rel\s*=\s*\"?next\"?", re.IGNORECASE)
MAX_TREE_PAGES = 100      # a cursor loop must end even if the server misbehaves

CHUNK = 1 << 20

# Per-GRAPH budget, from setup_tensorrt.ENGINE_PROBE_TIMEOUT_S. Round 3 measured
# 202.3 s (Base) / 232.7 s (Small) for a cold 756 compile, so 300 s covered a
# single graph with little to spare; the dynamic-axes export is slower still.
PROBE_TIMEOUT_S = 600.0

# What a cold local compile costs per graph, for the confirmation the user is
# asked to accept. Measured on the author's 4090 across the 21 graphs the marker
# in ort_tensorrt/ records: the four square presets came back in seconds against
# an already-warm cache, the sixteen adaptive rectangles at 156-213 s each.
ESTIMATED_COMPILE_S_PER_GRAPH = 200.0

# How often `run_child` reports elapsed time while a build runs. 70 minutes with
# no feedback reads as a hang, and a single graph alone is over three minutes.
TICK_S = 1.0

# --- the graphs to build ----------------------------------------------------
#
# A MIRROR of SyLC_3D_Player.SYNTH3D_DEPTH_PRESETS and
# SYNTH3D_ADAPTIVE_MODEL_GRIDS, which cannot be imported here: they live in the
# Qt-importing player module, and this module must stay loadable in the probe
# child before any GUI exists. `setup_tensorrt.py` keeps its own mirror for the
# same reason and `test_trt_optin.py` pins it to the player;
# tests/models/test_trt_engines.py pins THIS one to both.
#
# The probe must build every graph the PLAYER can open, not just each preset's
# first choice: the player's gate is checked against the model a preset RESOLVES
# to, and that resolution reads the disk at enable time. Delete
# `da3_base_756.onnx` from an installed copy and Quality resolves to
# `da3_small_756.onnx` without invalidating anything, so a marker naming only
# first choices would silently drop that preset to DirectML. Attesting a
# candidate that is never resolved costs an offline compile; failing to attest
# one that IS resolved costs the user the multi-minute in-playback wait this
# whole mechanism exists to prevent.
DEPTH_PRESETS = (
    (("da3_base_756.onnx", "da3_small_756.onnx", "da3_small.onnx"), 756),
    (("da3_base_518.onnx", "da3_small_518.onnx"),                   518),
    (("da3_small_518.onnx",),                                       518),
)
# Adaptive fixed rectangles are not user-facing presets, but the player opens
# them after matte detection. TensorRT's cache is keyed per graph, so each needs
# its own cold build and its own marker line just like every square preset.
ADAPTIVE_MODEL_GRIDS = (
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


class TrtEngineError(RuntimeError):
    """Engines could not be acquired or verified. No marker was written."""


class TrtEngineCancelled(Exception):
    """The caller's cancel event fired. Every engine already built is kept."""


@dataclass(frozen=True)
class Graph:
    """One ONNX graph to build a TensorRT engine for, at ONE input geometry.

    The grid travels with the model because it is half of the graph's identity:
    a fixed-shape 518 export opened at 756 fails DepthEngine init outright, and
    even for the one dynamic-axes export the input shape is what TensorRT builds
    its optimization profile from.
    """
    path: str
    width: int
    height: int

    @property
    def name(self):
        return os.path.basename(self.path)


@dataclass(frozen=True)
class EngineFile:
    """One published cache entry, with what verification needs already in hand."""
    name: str            # basename as it lands in trt_cache/
    path: str            # full path inside the repository
    size: int
    sha256: str | None   # the LFS oid; None for entries git tracks directly

    @property
    def url(self):
        return HF_RESOLVE_URL.format(repo=HF_ENGINE_REPO,
                                     revision=HF_ENGINE_REVISION,
                                     path=urllib.parse.quote(self.path))


@dataclass(frozen=True)
class ProbeResult:
    name: str
    width: int
    height: int
    ok: bool
    elapsed: float
    detail: str          # one line, marker-safe
    ort_version: str     # "?" when the child could not read it


@dataclass(frozen=True)
class Verification:
    ok: bool
    results: tuple
    marker_path: str | None    # None whenever `ok` is False -- see verify_runtime
    elapsed: float

    @property
    def failed(self):
        return tuple(r for r in self.results if not r.ok)


# --- planning ----------------------------------------------------------------

def probe_graphs(models_dirs):
    """Every loadable graph on disk, in the presets' own display order.

    `models_dirs` is searched in order per NAME, so an install-directory copy
    wins over a stale per-user one -- the same precedence
    `SyLC_3D_Player._synth3d_models_dirs()` documents. De-duplicated by
    (name, grid): `da3_small_518.onnx` is both Balanced's fallback and
    Performance's only model, and it is ONE graph, hence one engine and one
    probe.
    """
    if isinstance(models_dirs, str):
        models_dirs = (models_dirs,)
    graphs, seen = [], set()

    def add(name, width, height):
        key = (name.lower(), width, height)
        if key in seen:
            return
        for directory in models_dirs:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                seen.add(key)
                graphs.append(Graph(path=path, width=width, height=height))
                return

    for candidates, side in DEPTH_PRESETS:
        for name in candidates:
            add(name, side, side)
    for name, width, height in ADAPTIVE_MODEL_GRIDS:
        add(name, width, height)
    return tuple(graphs)


def published_prefix(sm, version=None):
    """The repository prefix holding prebuilt engines for `sm`, or None."""
    return PUBLISHED_ENGINE_SETS.get(
        (sm, version if version is not None else trt_fetcher.TENSORRT_VERSION))


def cache_dir(ort_dir):
    """Where DepthEngine points TensorRT's `trt_engine_cache_path`.

    Pinned by depth_engine.cpp, which appends exactly this name to `ort_dir`
    and creates it: putting engines anywhere else would leave the cache cold.
    """
    return os.path.join(ort_dir, TRT_CACHE_DIRNAME)


# --- the published engine cache ----------------------------------------------

def _next_page_url(headers, current):
    link = None
    if headers is not None:
        try:
            link = headers.get("Link")
        except Exception:
            link = None
    if not link:
        return None
    match = _LINK_NEXT.search(link)
    if not match:
        return None
    return urllib.parse.urljoin(current, match.group(1))


def _hf_list_entries(repo, revision, prefix, opener=None):
    """File entries under `prefix`, across EVERY page of the tree API.

    Ported from `setup_tensorrt._hf_list_entries`, plus the pagination it
    omits. The raw entries are returned rather than bare paths because each one
    already carries what verification needs: `size`, and for LFS-tracked files
    an `lfs.oid` that IS the content's SHA-256. Verified against the live API on
    2026-07-31 -- an engine entry reads:

        {"type": "file", "size": 54466204, "path": "...engine",
         "oid": "89072045...",            # git blob id, NOT the content hash
         "lfs": {"oid": "759ca003...", "size": 54466204, "pointerSize": 133}}

    `oid` and `lfs.oid` are different things: the top-level one is the git
    pointer's blob id and hashing the file will never reproduce it. Only
    `lfs.oid` hashes the real bytes. The 21-byte `.profile` companions are not
    LFS-tracked and carry no `lfs` key at all.
    """
    opener = opener or urllib.request.urlopen
    url = HF_TREE_URL.format(repo=repo, revision=revision)
    entries, seen_urls = [], set()
    for _page in range(MAX_TREE_PAGES):
        if url is None or url in seen_urls:
            break
        seen_urls.add(url)
        with opener(urllib.request.Request(url)) as response:
            payload = json.loads(response.read().decode("utf-8"))
            url = _next_page_url(getattr(response, "headers", None), url)
        entries += [e for e in payload
                    if e.get("type") == "file"
                    and e.get("path", "").startswith(prefix)]
    return entries


def list_published_engines(sm, *, version=None, opener=None):
    """The published cache entries for `sm`, or () when none are published.

    Only files sitting DIRECTLY under the prefix and carrying one of
    `ENGINE_SUFFIXES` -- which is what keeps the published README out of a
    directory the player owns.
    """
    prefix = published_prefix(sm, version)
    if prefix is None:
        return ()
    files = []
    for entry in _hf_list_entries(HF_ENGINE_REPO, HF_ENGINE_REVISION, prefix,
                                  opener):
        path = entry.get("path", "")
        name = path[len(prefix):]
        if not name or "/" in name:
            continue
        if not name.lower().endswith(ENGINE_SUFFIXES):
            continue
        files.append(EngineFile(
            name=name, path=path, size=int(entry.get("size") or 0),
            sha256=(entry.get("lfs") or {}).get("oid")))
    return tuple(files)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _free_bytes(path):
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def fetch_engines(dest_dir, entries, *, progress=None, cancel=None, opener=None):
    """Downloads the published engines into `dest_dir`. Returns names written.

    Existing files are left alone, so a rerun after an interruption fetches only
    what is missing -- and so a locally COMPILED engine, which is just as valid
    as a published one for the same graph hash, is never replaced.

    Every file is size-checked, and SHA-256-checked when the listing carries an
    LFS oid, BEFORE it is promoted from `.part` to its real name. Without that,
    a transfer that closes early WITHOUT raising -- a flaky proxy or CDN, which
    `shutil.copyfileobj` returning normally does not rule out -- would publish a
    truncated engine under its real name, and every later run would skip it
    forever, since presence is the only "already have it" test. A poisoned cache
    entry cannot crash playback (the local probe still gates `.trt_verified`),
    but it would burn a multi-minute probe and read as a TensorRT fault rather
    than as a bad download.

    `progress(message, done, total)` -- bytes, over the whole set.
    """
    opener = opener or urllib.request.urlopen
    os.makedirs(dest_dir, exist_ok=True)
    wanted = [e for e in entries
              if not os.path.exists(os.path.join(dest_dir, e.name))]
    needed = sum(e.size for e in wanted)
    if needed:
        free = _free_bytes(dest_dir)
        if free < needed:
            raise TrtEngineError(
                f"the prebuilt engines need {needed / 1e9:.2f} GB in "
                f"{dest_dir} and only {free / 1e9:.2f} GB is free")
    written, done = [], 0
    for entry in wanted:
        if cancel is not None and cancel.is_set():
            raise TrtEngineCancelled(entry.name)
        target = os.path.join(dest_dir, entry.name)
        part = target + ".part"
        try:
            with opener(urllib.request.Request(entry.url)) as response, \
                    open(part, "wb") as handle:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise TrtEngineCancelled(entry.name)
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    if progress is not None:
                        progress(f"Downloading prebuilt engines — "
                                 f"{entry.name}", min(done, needed), needed)
            actual = os.path.getsize(part)
            if entry.size and actual != entry.size:
                raise TrtEngineError(
                    f"{entry.name}: got {actual} bytes, expected {entry.size}")
            if entry.sha256:
                digest = _sha256_file(part)
                if digest != entry.sha256:
                    raise TrtEngineError(
                        f"{entry.name}: sha256 {digest}, expected {entry.sha256}")
        except BaseException:
            # Including TrtEngineCancelled and any OSError from the socket: a
            # `.part` must never survive the call that created it, or the next
            # run inherits debris it cannot judge.
            _unlink(part)
            raise
        os.replace(part, target)
        written.append(entry.name)
    return tuple(written)


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def acquire_engines(ort_dir, sm, *, entries=None, progress=None, cancel=None,
                    opener=None):
    """FAST ROUTE. Downloads prebuilt engines when a published set matches.

    Returns the names written -- () when nothing is published for this
    (architecture, TensorRT version) pair, which is the ordinary case for every
    non-Ada GPU and is not an error: `verify_runtime` then compiles locally.

    This NEVER writes `.trt_verified`, and could not: the marker is written by
    `verify_runtime` and by nothing else, after a real build on this machine.
    """
    if entries is None:
        entries = list_published_engines(sm, opener=opener)
    if not entries:
        return ()
    return fetch_engines(cache_dir(ort_dir), entries, progress=progress,
                         cancel=cancel, opener=opener)


# --- the child process -------------------------------------------------------

def probe_command(graph, ort_dir, result_path):
    """The argv that runs ONE engine build in a child process.

    Frozen, `sys.executable` is the player itself and there is no interpreter to
    spawn, so it is re-invoked with `PROBE_ARGV_FLAG`; from a source checkout
    the same flag is handed to this file run as a script. Both forms end in
    `probe_main`, which is why the flag is passed in either case.
    """
    argv = [PROBE_ARGV_FLAG,
            "--model", graph.path,
            "--ort-dir", ort_dir,
            "--width", str(graph.width),
            "--height", str(graph.height),
            "--result", result_path]
    if is_frozen():
        return [sys.executable] + argv
    return [sys.executable, os.path.abspath(__file__)] + argv


def is_frozen():
    """True inside the Nuitka-built player.

    `__compiled__` is the marker `SyLC_3D_Player._setup_nuitka_paths` trusts
    first, and it exists only in a Nuitka build.
    """
    try:
        import __compiled__          # noqa: F401  -- Nuitka only
    except ImportError:
        return False
    return True


def _creation_flags():
    # Without this the frozen GUI player flashes a console window for each of
    # 21 children. Absent off Windows, hence the getattr.
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_child(command, timeout_s, tick=None):
    """Runs `command` to completion. Returns (returncode, output).

    `returncode` is None when the child had to be killed for exceeding
    `timeout_s`; the caller reads that as a failure, never as a result.

    Never raises for anything the child does -- a native abort is an ordinary
    non-zero (or negative) return code here, which is the entire point of
    running the build out of process.

    `tick(elapsed_seconds)` is called about once a second while the child runs.
    Output is drained by `communicate` throughout, including across every
    timeout retry, because a real TensorRT build writes dozens of warnings and a
    child blocked on a full pipe would look exactly like a hung build.
    """
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", creationflags=_creation_flags())
    except OSError as exc:
        return None, f"could not start the engine probe: {exc}"

    started = time.monotonic()
    deadline = started + timeout_s
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            proc.kill()
            try:
                output, _ = proc.communicate(timeout=30)
            except Exception:
                output = ""
            return None, (f"TIMEOUT after {timeout_s:.0f}s "
                          f"(no result within budget): {output or ''}".strip())
        try:
            output, _ = proc.communicate(timeout=min(TICK_S, remaining))
        except subprocess.TimeoutExpired:
            if tick is not None:
                tick(time.monotonic() - started)
            continue
        return proc.returncode, output or ""


# What a PASSING graph contributes to the marker: the probe's own two verdict
# lines, nothing else. A real TensorRT build writes dozens of WARNING lines
# ("Was not able to infer kOPT value(s) for tensor ..." is emitted once per RoPE
# Cast node on the dynamic-axes export alone), which would bury the value-gate
# numbers under kilobytes of console noise in a file whose whole job is to be
# read at a glance. A FAILING graph keeps its output verbatim -- there the noise
# IS the diagnosis, and it never reaches a marker anyway, since a failure writes
# none. Ported from setup_tensorrt._probe_verdict.
_PROBE_VERDICT_PREFIXES = ("VALUE_GATE", "RESULT=")


def _one_line(text):
    """Flattens multi-line output onto one line.

    The marker is parsed LINE BY LINE by the player (`probe_model=` lines), so
    an embedded newline would turn the tail of a detail into what looks like a
    top-level field.
    """
    return " | ".join(l.strip() for l in str(text).splitlines() if l.strip())


def _probe_verdict(output):
    kept = [l.strip() for l in output.splitlines()
            if l.strip().startswith(_PROBE_VERDICT_PREFIXES)]
    return " | ".join(kept) if kept else _one_line(output)


def _read_payload(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def probe_graph(graph, ort_dir, *, timeout_s=PROBE_TIMEOUT_S, runner=None,
                tick=None):
    """Builds and tests ONE graph's engine, out of process. Never raises.

    The verdict is taken from the child's result FILE first and from its stdout
    only as a fallback, because the frozen player is a GUI-subsystem binary
    whose stdout is not guaranteed to reach us. Neither one arriving, with a
    non-zero return code, is the signature of a native abort -- and it is
    reported as a failure of THIS graph, exactly like a graceful one.
    """
    runner = runner or run_child
    staging = tempfile.mkdtemp(prefix="sylc_trt_probe_")
    started = time.monotonic()
    try:
        result_path = os.path.join(staging, "result.json")
        returncode, output = runner(
            probe_command(graph, ort_dir, result_path), timeout_s, tick)
        payload = _read_payload(result_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    elapsed = time.monotonic() - started
    output = output or ""
    version = str((payload or {}).get("ort_version") or "?")

    def result(ok, detail):
        return ProbeResult(name=graph.name, width=graph.width,
                           height=graph.height, ok=ok, elapsed=elapsed,
                           detail=_one_line(detail), ort_version=version)

    if returncode is None:
        # Either the child never started or it blew the per-graph budget. Both
        # are failures of this graph and neither is an exception here.
        return result(False, output or "the engine probe did not complete")
    if payload is not None:
        verdict = str(payload.get("result") or "")
        detail = str(payload.get("detail") or verdict)
        if returncode == 0 and verdict == "OK":
            return result(True, detail)
        return result(False, f"{detail} (returncode={returncode})")
    if any(f"RESULT={name}" in output
           for name in ("OK", "EXCEPTION", "VALUE_GATE_FAILED", "TIMEOUT")):
        # The child stayed alive long enough to diagnose itself but could not
        # write its result file. Its own words are still the answer: reaching a
        # RESULT= line at all means the build and the value gate both ran to a
        # verdict. The file is a robustness channel for a GUI-subsystem binary,
        # not a second gate -- treating a printed RESULT=OK as a failure would
        # deny TensorRT to a machine whose temp directory merely misbehaved.
        return result(returncode == 0 and "RESULT=OK" in output,
                      _probe_verdict(output))
    return result(False,
                  f"the engine probe process exited ABNORMALLY "
                  f"(returncode={returncode}, no result file and no graceful "
                  f"RESULT= line -- this is a native crash, not a Python "
                  f"exception): {output}")


# --- the marker --------------------------------------------------------------

def write_verified_marker(ort_dir, results):
    """Writes the marker attesting EVERY graph in `results`.

    Byte-format-compatible with `setup_tensorrt._write_verified_marker`: one
    file, two writers, and the player's per-graph gate
    (`SyLC_3D_Player.synth3d_marker_attests`) reads the `probe_model=` lines
    from either. A graph absent from that list gets the DirectML runtime instead
    of a cold in-playback engine compile.

    `results` must not be empty. A marker naming NO graph attests the DIRECTORY
    as a whole -- that is its documented round-3 meaning -- so an empty one
    would hand TensorRT every graph without a single engine having been built.
    """
    if not results:
        raise TrtEngineError(
            "refusing to write a marker that attests no graph: the player "
            "reads a marker with no probe_model= line as attesting the whole "
            "directory")
    marker_path = os.path.join(ort_dir, TRT_VERIFIED_MARKER)
    version = next((r.ort_version for r in results
                    if r.ort_version and r.ort_version != "?"), "?")
    lines = [
        f"verified_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"onnxruntime_version={version}",
        f"tensorrt_wheel_pinned={trt_fetcher.TENSORRT_VERSION}",
    ]
    lines += [f"probe_model={r.name}" for r in results]
    lines.append(f"probe_elapsed_s={sum(r.elapsed for r in results):.1f}")
    lines += [f"probe_detail_{i}={r.name} grid={r.width}x{r.height} "
              f"elapsed_s={r.elapsed:.1f} {r.detail}"
              for i, r in enumerate(results, 1)]
    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return marker_path


def remove_stale_marker(ort_dir):
    """Deletes a marker a failed run has just contradicted. Returns its path."""
    marker_path = os.path.join(ort_dir, TRT_VERIFIED_MARKER)
    if os.path.exists(marker_path):
        try:
            os.remove(marker_path)
        except OSError:
            return None
        return marker_path
    return None


# --- the whole verification --------------------------------------------------

def _duration(seconds):
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def verify_runtime(ort_dir, graphs, *, timeout_s=PROBE_TIMEOUT_S, progress=None,
                   cancel=None, runner=None):
    """Builds every graph's engine out of process, then writes the marker.

    The ONLY writer of `.trt_verified` in the shipped player. Downloaded engines
    make this fast; they never stand in for it.

    `progress(message, done, total)` -- `done`/`total` count GRAPHS, and the
    message names the graph, its position and the elapsed time, because 21
    builds at ~200 s each is not a spinner's worth of waiting.

    Cancel is honoured BETWEEN graphs. A build already running is deliberately
    not killed: TensorRT writes its cache entry in place, and a half-written
    `.engine` under its real name would be picked up as valid by every later
    run. Every engine that did build stays in the cache, so a cancelled run
    resumes cheaply rather than starting over.

    A failure on ANY graph writes no marker and REMOVES a previous one, which is
    setup_tensorrt.py's rule: an incompatible assembly and one unsupported op
    look identical from here, and a marker that a fresh run has just
    contradicted no longer describes what is on disk.
    """
    if not graphs:
        raise TrtEngineError(
            "no depth model is installed, so there is nothing to verify -- and "
            "a marker attesting no graph at all would re-open the whole "
            "directory to TensorRT without a single engine having been built")
    started = time.monotonic()
    results = []
    total = len(graphs)
    for index, graph in enumerate(graphs, 1):
        if cancel is not None and cancel.is_set():
            raise TrtEngineCancelled(
                f"cancelled after {index - 1} of {total} graph(s)")

        def report(elapsed_in_graph, _index=index, _graph=graph):
            if progress is not None:
                progress(
                    f"Building {_graph.name} at {_graph.width}x{_graph.height} "
                    f"— graph {_index} of {total}, "
                    f"{_duration(elapsed_in_graph)} on this one, "
                    f"{_duration(time.monotonic() - started)} in total",
                    _index - 1, total)

        report(0.0)
        result = probe_graph(graph, ort_dir, timeout_s=timeout_s, runner=runner,
                             tick=report)
        results.append(result)
        if progress is not None:
            progress(f"{'Built' if result.ok else 'FAILED'} {graph.name} "
                     f"({index} of {total}) in {_duration(result.elapsed)}",
                     index, total)
    ok = all(r.ok for r in results)
    if ok:
        marker_path = write_verified_marker(ort_dir, results)
    else:
        remove_stale_marker(ort_dir)
        marker_path = None
    return Verification(ok=ok, results=tuple(results), marker_path=marker_path,
                        elapsed=time.monotonic() - started)


# --- the child's own work ----------------------------------------------------
#
# Everything below runs ONLY in the probe child. `numpy` and `mvc_demuxer_cpp`
# are imported here rather than at module level so the parent -- and the test
# suite -- can use everything above with neither installed.

def _rgb_scene():
    """The probe scene, construction-identical to test_depth_engine.py's.

    That identity is what makes the value gate below meaningful: the same
    zeros/linspace/square layout means that test's own polarity slices apply
    verbatim.
    """
    import numpy as np
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = np.linspace(40, 120, 480, dtype=np.uint8)[:, None, None]
    img[280:460, 200:440] = (200, 180, 160)
    return img


def _ort_version(ort_dir):
    """Best-effort onnxruntime.dll version string, for the marker.

    Read in the CHILD, not in the player: this loads a DLL out of the directory
    an install may be about to replace, and a mapped DLL is a directory Windows
    will not let `trt_fetcher._promote` rename. The child exits seconds later,
    so the handle costs nothing there.
    """
    try:
        import ctypes
        LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p
        kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p,
                                            ctypes.c_uint32]
        kernel32.GetProcAddress.restype = ctypes.c_void_p
        kernel32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
        handle = kernel32.LoadLibraryExW(
            os.path.join(ort_dir, "onnxruntime.dll"), None,
            LOAD_WITH_ALTERED_SEARCH_PATH)
        if not handle:
            return "?"
        try:
            # struct OrtApiBase { const OrtApi*(*GetApi)(uint32_t);
            #                     const char*(*GetVersionString)(void); }
            get_api_base = ctypes.CFUNCTYPE(ctypes.c_void_p)(
                kernel32.GetProcAddress(handle, b"OrtGetApiBase"))
            base = get_api_base()
            get_version = ctypes.CFUNCTYPE(ctypes.c_char_p)(
                ctypes.cast(base, ctypes.POINTER(ctypes.c_void_p))[1])
            raw = get_version()
        finally:
            kernel32.FreeLibrary(handle)
        return raw.decode("utf-8", "replace") if raw else "?"
    except Exception:
        return "?"


def _probe_engine(model, ort_dir, width, height):
    """One real engine build + inference + value gate. Returns a payload dict.

    Ported from tests/synth3d/_trt_engine_probe.py's "engine" mode -- a fresh,
    registry-independent DepthEngine via `mvc_demuxer_cpp.depth_infer_test`,
    with no NativeRenderer and no SharedDepthService involved.
    """
    import mvc_demuxer_cpp
    started = time.monotonic()
    try:
        kwargs = {"side": width}
        if height != width:
            kwargs.update(grid_width=width, grid_height=height)
        depth = mvc_demuxer_cpp.depth_infer_test(
            model, ort_dir, _rgb_scene(), **kwargs)
    except Exception as exc:
        return {"result": "EXCEPTION",
                "elapsed": time.monotonic() - started,
                "detail": f"RESULT=EXCEPTION elapsed="
                          f"{time.monotonic() - started:.3f} message={exc}"}
    elapsed = time.monotonic() - started

    # VALUE GATE: a clean engine BUILD is not proof of a numerically correct
    # one -- trt_fp16_enable=1 could run a fp32-trained artifact on fp16 kernels
    # and silently produce garbage. A near (bright square) region must score
    # higher than a far (gradient sky) region, or this is NOT a passing result
    # regardless of the engine having built without error.
    #
    # The slices were derived in 756x756 output space. Each axis is scaled
    # independently so a rectangular graph probes the same source regions.
    sx, sy = width / 756.0, height / 756.0
    square = float(depth[int(450 * sy):int(710 * sy),
                         int(245 * sx):int(510 * sx)].mean())
    sky = float(depth[int(70 * sy):int(300 * sy),
                      int(130 * sx):int(625 * sx)].mean())
    gate = f"VALUE_GATE sq={square:.4f} sky={sky:.4f}"
    if not (square > sky):
        return {"result": "VALUE_GATE_FAILED", "elapsed": elapsed,
                "sq": square, "sky": sky,
                "detail": f"{gate} | RESULT=VALUE_GATE_FAILED "
                          f"elapsed={elapsed:.3f} sq={square:.4f} "
                          f"sky={sky:.4f} (polarity inverted/degenerate -- the "
                          f"engine built cleanly but produced numerically wrong "
                          f"output)"}
    return {"result": "OK", "elapsed": elapsed, "sq": square, "sky": sky,
            "shape": list(depth.shape),
            "detail": f"{gate} | RESULT=OK elapsed={elapsed:.3f} "
                      f"shape={tuple(depth.shape)} sq={square:.4f} "
                      f"sky={sky:.4f}"}


def _flag(argv, name, default=None):
    """Pops `--name VALUE` / `--name=VALUE` out of `argv` (in place)."""
    for index, argument in enumerate(argv):
        if argument == name:
            if index + 1 >= len(argv):
                return default
            value = argv[index + 1]
            del argv[index:index + 2]
            return value
        if argument.startswith(name + "="):
            value = argument.split("=", 1)[1]
            del argv[index]
            return value
    return default


def probe_main(argv):
    """The CHILD entry point. One engine build, one verdict, then exit.

    Called from `SyLC_3D_Player.__main__` when `PROBE_ARGV_FLAG` is on the
    command line -- before any Qt or GUI initialisation -- and from this file's
    own `__main__`. Never call it in a process that has anything else to do:
    the whole reason it exists is that the call it makes can abort the process.

    Returns a process exit code, and writes the same verdict to `--result` as
    JSON. Prints it too, which is what makes a source-checkout run readable.
    """
    argv = list(argv)
    while PROBE_ARGV_FLAG in argv:
        argv.remove(PROBE_ARGV_FLAG)
    model = _flag(argv, "--model", "")
    ort_dir = _flag(argv, "--ort-dir", "")
    result_path = _flag(argv, "--result")
    try:
        width = int(_flag(argv, "--width", "756"))
        height = int(_flag(argv, "--height", str(width)))
    except ValueError:
        width = height = 0
    if width <= 0 or height <= 0:
        payload = {"result": "EXCEPTION", "elapsed": 0.0,
                   "detail": "RESULT=EXCEPTION elapsed=0.000 "
                             "message=grid must be positive"}
    elif not model or not ort_dir:
        payload = {"result": "EXCEPTION", "elapsed": 0.0,
                   "detail": "RESULT=EXCEPTION elapsed=0.000 "
                             "message=--model and --ort-dir are required"}
    else:
        try:
            payload = _probe_engine(model, ort_dir, width, height)
        except BaseException as exc:
            # A MemoryError, a KeyboardInterrupt, an ImportError for numpy in a
            # deployment that somehow lacks it: whatever it is, the parent needs
            # a verdict rather than an empty file it has to read as a crash.
            payload = {"result": "EXCEPTION", "elapsed": 0.0,
                       "detail": f"RESULT=EXCEPTION elapsed=0.000 "
                                 f"message={exc.__class__.__name__}: {exc}"}
        payload["ort_version"] = _ort_version(ort_dir)

    if result_path:
        try:
            with open(result_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            # The parent falls back to stdout, and to "no verdict at all" being
            # the native-crash signature. Nothing here is worth dying for.
            pass
    print(payload.get("detail", ""), flush=True)
    return 0 if payload.get("result") == "OK" else 1


# --- command line ------------------------------------------------------------
#
# The dialog is the intended caller of everything above, but this file also
# lives in the GitHub repository, where "no v5.2.0 user can obtain TensorRT" is
# the problem being fixed -- and there, running this module directly IS the fix,
# with no uv, no pip and no scratch venv.
#
#   python trt_engines.py --probe                 # verify ort_tensorrt/
#   python trt_engines.py --probe --no-download   # local compile only
#
# and, internally, `python trt_engines.py --sylc-trt-engine-probe ...` for one
# graph, which is what `probe_command` builds.

def _main(argv):
    if PROBE_ARGV_FLAG in argv:
        return probe_main(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    ort_dir = _flag(argv, "--ort-dir", os.path.join(here, "ort_tensorrt"))
    models_dir = _flag(argv, "--models-dir", os.path.join(here, "models"))
    if not os.path.isdir(ort_dir):
        print(f"[trt_engines] {ort_dir} does not exist. Run trt_fetcher.py "
              f"first to assemble the runtime.")
        return 1

    from sylc import trt_runtime
    gpu = trt_runtime.detect_gpu()
    if gpu is None:
        print("[trt_engines] no NVIDIA GPU detected.")
        return 1
    print(f"[trt_engines] detected {gpu.name} -> sm{gpu.sm}")

    graphs = probe_graphs((models_dir,))
    if not graphs:
        print(f"[trt_engines] no depth model found under {models_dir}.")
        return 1
    print(f"[trt_engines] {len(graphs)} graph(s) to verify:")
    for graph in graphs:
        print(f"    {graph.name} @ {graph.width}x{graph.height}")

    if "--no-download" not in argv:
        try:
            entries = list_published_engines(gpu.sm)
        except OSError as exc:
            print(f"[trt_engines] could not reach the engine repository "
                  f"({exc}); compiling locally instead.")
            entries = ()
        if entries:
            size = sum(e.size for e in entries)
            print(f"[trt_engines] {len(entries)} prebuilt engine file(s), "
                  f"{size / 2 ** 30:.2f} GiB, for sm{gpu.sm} / TensorRT "
                  f"{trt_fetcher.TENSORRT_VERSION}")
            written = acquire_engines(
                ort_dir, gpu.sm, entries=entries,
                progress=lambda m, d, t: print(
                    f"    {m} [{100.0 * d / max(t, 1):5.1f}%]",
                    end="\r", flush=True))
            print(f"\n[trt_engines] {len(written)} engine file(s) fetched.")
        else:
            print(f"[trt_engines] no prebuilt engines published for sm{gpu.sm} "
                  f"/ TensorRT {trt_fetcher.TENSORRT_VERSION} -- compiling "
                  f"locally, roughly "
                  f"{_duration(len(graphs) * ESTIMATED_COMPILE_S_PER_GRAPH)}.")

    verification = verify_runtime(
        ort_dir, graphs, progress=lambda m, d, t: print(f"[trt_engines] {m}"))
    if verification.ok:
        print(f"\n[trt_engines] PASS -- wrote {verification.marker_path} "
              f"attesting {len(verification.results)} graph(s) in "
              f"{_duration(verification.elapsed)}.")
        return 0
    print("\n[trt_engines] FAIL -- no marker written. The player will keep "
          "using DirectML.")
    for result in verification.failed:
        print(f"    {result.name} @ {result.width}x{result.height}: "
              f"{result.detail}")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
