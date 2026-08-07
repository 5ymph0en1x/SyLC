# model_fetcher.py
"""Fetches the 2D->3D depth-model packs from HuggingFace.

STDLIB ONLY, and NO Qt import. This module is compiled into the frozen player
(`--include-module=model_fetcher`), so a third-party dependency here would have
to be bundled too; and keeping Qt out means the whole download/verify path is
testable without a QApplication. The Qt layer lives in model_download_dialog.py.

The manifest pins a HuggingFace COMMIT SHA rather than `main`: with `main`, any
later edit upstream would make every installed client's SHA-256 check fail with
no way to tell a corrupted transfer from a legitimate new export.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass

# 1 MiB. Large enough that the per-chunk progress callback and cancel check are
# free relative to the socket read, small enough that Cancel feels immediate.
CHUNK = 1 << 20

# HuggingFace throttles per CONNECTION, not per client. Measured against the
# real repository on a 10 Gbps link: one socket doing 1 MiB reads sustained
# 1.3 MiB/s, while eight sockets moved the same 48 MiB at 22.4 MiB/s. That is
# 17x, and a sequential loop leaves all of it on the table -- the Base pack
# takes ~48 minutes on one connection and a few on several.
#
# Six rather than eight: it captures most of the gain, and this code runs on
# every installation of the player against a public CDN, so the polite shape is
# the one to ship. There are 10 files in the largest pack, so six is also enough
# to keep the pool busy for nearly the whole transfer.
DEFAULT_WORKERS = 6

# Three attempts per file, not one. Measured against the real repository, a
# response ended 14 801 bytes early with no network exception at all -- a clean
# EOF short of the body -- roughly once in ten file downloads. The size check
# caught it, so nothing corrupt was published, but it failed the whole pack. Over
# a 20-file, 4.61 GB acquisition that lands on real users, most likely on their
# first try, and "Download failed, start over" is the wrong answer to a fault
# that does not reproduce.
MAX_ATTEMPTS = 3

# Long enough to ride out a transient, short enough that the dialog does not read
# as hung between attempts.
RETRY_BACKOFF_S = 1.5

# Cancel is polled at this interval during a backoff instead of being slept
# through: the combined stop signal only promises `is_set()`, so there is no
# single event to block on, and a user who presses Cancel should not wait out
# the wait.
RETRY_POLL_S = 0.1

# Cancel is only checked BETWEEN chunk reads, so a socket that goes quiet rather
# than closing would block read() forever and make Cancel unresponsive -- the one
# failure mode the chunk size alone cannot protect against. 60 s is generous for
# a 1 MiB chunk on any connection worth resuming over, and short enough that a
# dead link surfaces as an error instead of a hang.
READ_TIMEOUT_S = 60

HF_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


class ModelFetchError(RuntimeError):
    """A download completed but did not produce the expected bytes."""


class ModelFetchCancelled(Exception):
    """The caller's cancel event fired mid-transfer."""


class NotEnoughSpace(ModelFetchError):
    """The destination volume cannot hold the pack."""


class LocalWriteError(ModelFetchError):
    """The `.part` could not be written -- a fault on this machine, not in transit.

    Its own class purely so the retry can tell it apart. Transfers are retried on
    OSError, because a reset or a timeout is exactly what a second attempt
    exists for; but the OSError raised by writing to a volume that just filled is
    the one shape that will NOT have healed in a second and a half, and retrying
    it means re-transferring megabytes to hit the same wall and delaying the real
    message to the user.
    """


class _AnyEvent:
    """`is_set()` over several events, with the same shape as one.

    A transfer has to stop for two unrelated reasons: the user pressed Cancel,
    or a sibling file in the same pack failed and there is no point finishing.
    Neither event can express the other, and `_download_one` only ever asks
    `is_set()` -- so combining them here keeps the per-chunk check a single
    question and keeps the pool invisible to the download itself.
    """
    __slots__ = ("_events",)

    def __init__(self, *events):
        self._events = tuple(e for e in events if e is not None)

    def is_set(self):
        return any(event.is_set() for event in self._events)


@dataclass(frozen=True)
class ModelFile:
    path: str      # repo-relative, e.g. "onnx/small/da3_small_756.onnx"
    name: str      # local basename under models/
    size: int
    sha256: str


@dataclass(frozen=True)
class Pack:
    key: str
    label: str
    size: int
    files: tuple


@dataclass(frozen=True)
class Manifest:
    repo: str
    revision: str
    packs: dict

    def url_for(self, model_file):
        return HF_URL.format(repo=self.repo, revision=self.revision,
                             path=model_file.path)


@dataclass(frozen=True)
class PackStatus:
    key: str
    label: str
    installed: int
    total: int
    missing_bytes: int

    @property
    def complete(self):
        return self.installed == self.total


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    packs = {}
    for key, entry in raw["packs"].items():
        files = tuple(
            ModelFile(path=f["path"], name=f["name"], size=int(f["bytes"]),
                      sha256=f["sha256"])
            for f in entry["files"])
        packs[key] = Pack(key=key, label=entry["label"],
                          size=int(entry["bytes"]), files=files)
    return Manifest(repo=raw["repo"], revision=raw["revision"], packs=packs)


def _installed(models_dir, model_file):
    """Presence AND exact size -- deliberately NOT a hash.

    This runs every time the AI menu opens; hashing 4.6 GB there would freeze
    the UI. The hash is the download-time gate, where it belongs. A file of the
    exact expected size that is nonetheless wrong can only come from a
    deliberate substitution, which this check was never the defence against.
    """
    full = os.path.join(models_dir, model_file.name)
    try:
        return os.path.getsize(full) == model_file.size
    except OSError:
        return False


def pack_status(manifest, models_dir):
    out = {}
    for key, pack in manifest.packs.items():
        present = [f for f in pack.files if _installed(models_dir, f)]
        out[key] = PackStatus(
            key=key, label=pack.label, installed=len(present),
            total=len(pack.files),
            missing_bytes=sum(f.size for f in pack.files if f not in present))
    return out


def _resumable_bytes(models_dir, model_file):
    """Bytes of this file already on disk that a resume will actually keep.

    Mirrors _download_one's own rule exactly: a `.part` at or past the full size
    is discarded rather than resumed, so it counts for nothing. Any other answer
    here would make the space check disagree with what the download then does.
    """
    try:
        have = os.path.getsize(
            os.path.join(models_dir, model_file.name + ".part"))
    except OSError:
        return 0
    return have if have < model_file.size else 0


def _free_bytes(path):
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_one(url, dest, model_file, report, cancel, opener, charged_from):
    """Fetches one file. `report(have, contributed)` is called per chunk.

    `contributed` is this file's ABSOLUTE share of the pack numerator, never a
    delta: several of these run at once, so there is no "bytes before me" to add
    to, and the caller sums the latest value from each instead.

    `charged_from` is the offset the pack total charged this file FROM, and it
    is passed in rather than measured here for a reason: a retry after a short
    read resumes from a LONGER `.part` than the first attempt did. Measuring
    locally, the retry would report only the bytes of its own attempt against a
    total that charged it all of them -- the bar would drop back and then stop
    short of 100%. It is the same number the space check subtracted, so the two
    ends of the fraction cannot drift apart.
    """
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have >= model_file.size:
        # A .part at or past full size is not a resume point -- the previous run
        # died before verification, or the file changed upstream. Start over.
        os.remove(part)
        have = 0

    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")

    with opener(request, timeout=READ_TIMEOUT_S) as response:
        mode = "ab"
        if have and getattr(response, "status", 200) != 206:
            # The server ignored Range and is sending the whole body.
            have = 0
            mode = "wb"
        with open(part, mode) as handle:
            while True:
                if cancel is not None and cancel.is_set():
                    raise ModelFetchCancelled(model_file.name)
                block = response.read(CHUNK)
                if not block:
                    break
                try:
                    handle.write(block)
                except OSError as exc:
                    # The ONE OSError in this loop that is ours, not the
                    # network's. Everything else here -- the read above, the
                    # connect before it -- is a transfer fault and gets another
                    # attempt; a failed write does not. See LocalWriteError.
                    raise LocalWriteError(
                        f"{model_file.name}: cannot write "
                        f"{os.path.basename(part)}: {exc}") from exc
                have += len(block)
                if report is not None:
                    # max(): a server that ignored `Range` reset `have` to 0 and
                    # is rewriting the bytes we already had. Those repeats are
                    # not pack progress -- this file contributes nothing until
                    # the transfer passes `charged_from`, and because that
                    # baseline is fixed for the whole file, retries included, it
                    # then tops out at exactly `size - charged_from`: the amount
                    # the total charged it, no matter how many attempts it took.
                    report(have, max(0, have - charged_from))

    actual = os.path.getsize(part)
    if actual != model_file.size:
        # The connection ended early -- a clean EOF short of the body, with no
        # exception to catch. The `.part` is deliberately KEPT: the bytes that
        # did arrive came off a byte-accurate stream, so a retry should ask for
        # the REST. A 375 MB model that dies at 90% must resume at 90%, which is
        # what the resume machinery was built for. If those bytes are in fact
        # wrong, the SHA-256 below is what will say so.
        #
        # A `.part` that came out LONGER than the file is not a resume point
        # either, but it needs no special case here: the next attempt's own
        # `have >= size` rule discards it.
        raise ModelFetchError(
            f"{model_file.name}: got {actual} bytes, expected {model_file.size}")
    digest = _sha256(part)
    if digest != model_file.sha256:
        # Provably wrong bytes, and no way to know WHICH. Unlike a short read
        # there is nothing here worth resuming onto, so this one does go.
        os.remove(part)
        raise ModelFetchError(
            f"{model_file.name}: sha256 {digest}, expected {model_file.sha256}")
    os.replace(part, dest)


def _backoff(cancel, seconds, model_file):
    """Waits between attempts, in slices, so Cancel is honoured during the wait.

    Blocking the full interval would make Cancel land only after the next whole
    file -- minutes, for a 375 MB model.
    """
    deadline = time.monotonic() + seconds
    while True:
        if cancel is not None and cancel.is_set():
            raise ModelFetchCancelled(model_file.name)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(RETRY_POLL_S, remaining))


def _download_with_retries(url, dest, model_file, report, cancel, opener,
                           charged_from):
    """`_download_one`, retried on the transient faults it can survive."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _download_one(url, dest, model_file, report, cancel, opener,
                          charged_from)
            return
        except (NotEnoughSpace, LocalWriteError):
            # MUST come first: both are ModelFetchError subclasses and would
            # otherwise be swallowed by the clause below. Neither will have
            # fixed itself in a second and a half -- one is a verdict on the
            # volume, the other a fault on the very write a retry repeats.
            raise
        except (ModelFetchError, OSError):
            # Two separate families, deliberately both named: ModelFetchError
            # derives from RuntimeError, and the network faults do not derive
            # from it at all. A clean EOF short of the body arrives as the
            # former; a connection reset, a read timeout and a failed connect
            # arrive as ConnectionResetError, TimeoutError and URLError, every
            # one of them an OSError. All are the same event to a user -- the
            # transfer did not finish -- and all are worth another attempt.
            #
            # ModelFetchCancelled is in neither family, so it is still never
            # caught here and never retried: the user pressed Cancel.
            if attempt == MAX_ATTEMPTS:
                raise
            _backoff(cancel, RETRY_BACKOFF_S, model_file)


def download_pack(manifest, pack_key, dest_dir, progress=None, cancel=None,
                  opener=None, *, workers=DEFAULT_WORKERS):
    """Downloads every missing file of `pack_key` into `dest_dir`.

    Up to `workers` files transfer concurrently -- see DEFAULT_WORKERS for why
    that is where the time goes. Returns the basenames actually fetched, in
    manifest order. Already-present files are skipped, so this is safe to re-run
    after a cancel or a failure.
    """
    opener = opener or urllib.request.urlopen
    pack = manifest.packs[pack_key]
    missing = [f for f in pack.files if not _installed(dest_dir, f)]
    if not missing:
        return []
    # Bytes still to WRITE, not what the files weigh: a half-finished `.part`
    # already occupies its share of the volume. Charging the full size would
    # refuse a resume that fits -- precisely the tight-disk case where resuming
    # instead of restarting is what saves the user.
    #
    # Measured ONCE, here, BEFORE any transfer starts: after a file completes
    # its `.part` is gone, so asking _resumable_bytes again would answer 0 and
    # silently re-inflate the very total this subtraction removes. The same
    # values are handed to each download as its `charged_from`, so the numerator
    # and the denominator are not merely computed by the same rule -- they are
    # the same numbers, and no retry can make them drift.
    resumable = {f: _resumable_bytes(dest_dir, f) for f in missing}
    needed = sum(f.size - resumable[f] for f in missing)

    os.makedirs(dest_dir, exist_ok=True)
    free = _free_bytes(dest_dir)
    if free < needed:
        raise NotEnoughSpace(
            f"{needed/1e9:.2f} GB needed, {free/1e9:.2f} GB free")

    # One file failing has to stop the other five -- otherwise a SHA-256
    # mismatch on the first file still costs the user the full download of the
    # rest before it surfaces. The caller's Cancel has to stop all six.
    abort = threading.Event()
    stop = _AnyEvent(cancel, abort)

    # {name: bytes that file has contributed so far}. Summing the LATEST
    # absolute value from each in-flight transfer is what replaces the
    # sequential `base_done` running total; see _download_one for why deltas
    # would double-count a server that answers 200 to a `Range` request.
    #
    # The lock is held across the callback, not just across the map write: were
    # it released in between, two threads could compute their sums in one order
    # and deliver them in the other, and the progress bar would jump backwards.
    contributed = {}
    lock = threading.Lock()

    def make_report(model_file):
        def report(have, absolute):
            with lock:
                contributed[model_file.name] = absolute
                progress(model_file.name, have, model_file.size,
                         sum(contributed.values()), needed)
        return report

    def run(model_file):
        # Checked before the request, not only inside the read loop: a file the
        # pool has not started yet must not open a connection at all once a
        # sibling has failed or the user has cancelled.
        if stop.is_set():
            raise ModelFetchCancelled(model_file.name)
        _download_with_retries(
            manifest.url_for(model_file),
            os.path.join(dest_dir, model_file.name), model_file,
            make_report(model_file) if progress is not None else None,
            stop, opener, resumable[model_file])

    errors, fetched = [], set()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(workers, len(missing)))) as pool:
        futures = {pool.submit(run, f): f for f in missing}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except BaseException as exc:      # noqa: BLE001 - re-raised below
                errors.append(exc)
                abort.set()
            else:
                fetched.add(futures[future].name)

    if errors:
        # A cancellation caused BY a failure must not mask the failure: the
        # dialog would say "Cancelled — resume any time" for a corrupted
        # download. The caller's own Cancel still wins when it is what fired.
        if cancel is not None and cancel.is_set():
            raise next((e for e in errors
                        if isinstance(e, ModelFetchCancelled)), errors[0])
        raise next((e for e in errors
                    if not isinstance(e, ModelFetchCancelled)), errors[0])
    return [f.name for f in missing if f.name in fetched]
