# model_download_dialog.py
"""Qt front end for model_fetcher, trt_fetcher and trt_engines.

Separate from SyLC_3D_Player.py on purpose: that file is already ~10k lines, and
this dialog has one job. It knows nothing about the player -- it takes a
manifest path and the directories to work in, and reports what it did.
"""
import os
import threading

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QVBoxLayout)

from sylc import model_fetcher
from sylc import trt_engines
from sylc import trt_fetcher
from sylc import trt_runtime


def _human(n):
    # Binary units: this is what Windows Explorer shows for the same byte
    # counts, and what every SyLC document quotes. A decimal split would make
    # the dialog the lone outlier, disagreeing with the user's own file manager.
    return f"{n/2**30:.2f} GB" if n >= 2**30 else f"{n/2**20:.0f} MB"


def _install_summary(runtime_wire, runtime_disk, engine_bytes, engine_files,
                     graphs):
    """What the user is agreeing to, in bytes and in minutes.

    Built here rather than in the worker's message because it is the one place
    the two halves of the operation are visible at once: a runtime download that
    is measured in gigabytes, and a verification that is measured either in
    gigabytes (the published engines) or in HOURS (a local compile). Quoting
    only the first would understate the wait by a factor of a hundred.
    """
    lines = []
    if runtime_wire:
        lines.append(f"Download the TensorRT runtime: {_human(runtime_wire)} "
                     f"over the network, {_human(runtime_disk)} on disk.")
    if engine_files:
        lines.append(f"Download {engine_files} prebuilt engine file(s) for this "
                     f"GPU: {_human(engine_bytes)}.")
        lines.append(f"Then build and test all {graphs} graph(s) against them, "
                     f"which takes a few minutes.")
    else:
        estimate = trt_engines._duration(
            graphs * trt_engines.ESTIMATED_COMPILE_S_PER_GRAPH)
        lines.append(f"Compile a TensorRT engine for each of {graphs} graph(s) "
                     f"on this machine. No prebuilt engines are published for "
                     f"this GPU and TensorRT version, so this takes roughly "
                     f"{estimate} and cannot be made faster.")
    lines.append("")
    lines.append("Each engine is built in a separate process, so a TensorRT "
                 "fault is reported rather than taking the player down. You can "
                 "cancel between graphs; everything already built is kept.")
    return "\n".join(lines)


class _Worker(QObject):
    # qint64, NOT int. PySide6 marshals a bare `int` signal argument to a 32-bit
    # C int, and the Base pack is 3 938 318 270 bytes -- over that limit before a
    # single chunk is written. The emit does not raise: it SUCCEEDS and delivers
    # -356 649 026, so the progress bar computes a negative percentage and stays
    # frozen at 0% for the whole 3.67 GB while reporting "-0.36 GB", with no
    # error anywhere. Small (1 006 501 134) stays under 2^31, so the failure
    # lands only on the pack this feature exists to serve.
    progress = Signal(str, "qint64", "qint64")  # file name, pack done, pack total
    done = Signal(bool, str)                    # ok, message

    def __init__(self, manifest, pack_key, dest_dir):
        super().__init__()
        self._manifest = manifest
        self._pack_key = pack_key
        self._dest = dest_dir
        self.cancel = threading.Event()

    def run(self):
        try:
            fetched = model_fetcher.download_pack(
                self._manifest, self._pack_key, self._dest,
                progress=lambda name, fd, ft, pd, pt:
                    self.progress.emit(name, pd, pt),
                cancel=self.cancel)
        except model_fetcher.ModelFetchCancelled:
            self.done.emit(False, "Cancelled — resume any time, "
                                  "already-downloaded files are kept.")
        except model_fetcher.NotEnoughSpace as exc:
            self.done.emit(False, f"Not enough disk space: {exc}")
        except Exception as exc:                       # network, disk, HTTP
            self.done.emit(False, f"Download failed: {exc}")
        else:
            self.done.emit(True, f"{len(fetched)} file(s) installed.")


class _TrtWorker(QObject):
    """Acquire the runtime, get engines, verify, write the marker -- or none.

    One worker for the whole chain because the three steps share one cancel and
    one progress path, and because stopping between them would leave the user
    holding exactly the state stage 2 called a dead end: a complete runtime the
    player still refuses, because nothing has proved an engine can be built
    against it.

    Nothing here writes `.trt_verified`. Only `trt_engines.verify_runtime` does,
    and only after every graph has really been built and has really produced
    numerically correct output on this GPU.
    """
    progress = Signal(str, "qint64", "qint64")   # message, done, total
    confirm = Signal(str)                        # summary; answered by answer()
    done = Signal(bool, str)

    def __init__(self, ort_dir, models_dirs):
        super().__init__()
        self._ort_dir = ort_dir
        self._models_dirs = tuple(models_dirs or ())
        self.cancel = threading.Event()
        self._answered = threading.Event()
        self._accepted = False

    def answer(self, accepted):
        """Called on the GUI thread once the confirmation has been shown."""
        self._accepted = bool(accepted)
        self._answered.set()

    def _report(self, message, done=0, total=1):
        self.progress.emit(message, int(done), int(total))

    def _wait_for_answer(self):
        # Polled rather than a bare wait() so a Cancel or a dialog close during
        # the confirmation is honoured instead of parking this thread forever.
        while not self._answered.wait(0.1):
            if self.cancel.is_set():
                return False
        return self._accepted and not self.cancel.is_set()

    def run(self):
        try:
            ok, message = self._run()
        except trt_fetcher.TrtFetchCancelled:
            self.done.emit(False, "Cancelled — rerun to resume; every file "
                                  "already downloaded is kept.")
        except trt_engines.TrtEngineCancelled as exc:
            self.done.emit(False, f"Cancelled ({exc}) — rerun to resume; every "
                                  f"engine already built is kept.")
        except trt_fetcher.NotEnoughSpace as exc:
            self.done.emit(False, f"Not enough disk space: {exc}")
        except (trt_fetcher.TrtFetchError, trt_engines.TrtEngineError) as exc:
            self.done.emit(False, str(exc))
        except Exception as exc:                       # network, disk, HTTP
            self.done.emit(False, f"TensorRT setup failed: {exc}")
        else:
            self.done.emit(ok, message)

    def _run(self):
        self._report("Checking the current runtime…")
        status = trt_runtime.runtime_status(self._ort_dir)
        if status.gpu is None or status.state == trt_runtime.UNSUPPORTED_SM:
            return False, status.message

        # Resolve only what is actually missing. An `unverified` runtime is
        # COMPLETE -- stage 2 assembled it and stopped, by design -- so asking
        # the user to accept a 1.3 GB download to reach a directory that is
        # already on disk would be a false statement about the work ahead.
        plan = None
        if status.state in (trt_runtime.NOT_INSTALLED, trt_runtime.INCOMPLETE):
            self._report("Resolving the TensorRT runtime…")
            plan = trt_fetcher.resolve_plan(status.gpu.sm, cancel=self.cancel)

        graphs = trt_engines.probe_graphs(self._models_dirs)
        if not graphs:
            return False, ("No depth model is installed, so there is nothing "
                           "to build an engine for. Download Small or Base "
                           "first.")

        # The fast route, and it is only ever a shortcut: a published set turns
        # ~70 minutes of compilation into a download plus a few minutes of
        # verification. Its absence is not an error -- it is the ordinary case
        # for every GPU but Ada today.
        self._report("Looking for prebuilt engines…")
        try:
            entries = trt_engines.list_published_engines(status.gpu.sm)
        except Exception:
            # Unreachable repository, changed API, anything: the local compile
            # route still works and is the whole reason it exists. Failing the
            # install here would deny TensorRT over an optional optimisation.
            entries = ()
        cache = trt_engines.cache_dir(self._ort_dir)
        missing = [e for e in entries
                   if not os.path.exists(os.path.join(cache, e.name))]

        self.confirm.emit(_install_summary(
            plan.download_size if plan else 0, plan.size if plan else 0,
            sum(e.size for e in missing), len(missing), len(graphs)))
        if not self._wait_for_answer():
            return False, "Cancelled."

        if plan is not None:
            trt_fetcher.acquire_runtime(
                self._ort_dir, status.gpu.sm, plan=plan, cancel=self.cancel,
                progress=lambda name, have, size, done, total: self._report(
                    f"Downloading the TensorRT runtime — {name}", done, total))
        elif trt_fetcher.discard_staging(self._ort_dir):
            # Nothing to acquire, and a staging directory from an abandoned or
            # superseded run is sitting beside a complete install. Stage 2
            # deliberately leaves that judgement to the caller -- "a call that
            # reports nothing to do should not also remove a gigabyte" -- and
            # this is the caller, at the one moment the answer is unambiguous.
            self._report("Removed a leftover partial download.")

        if missing:
            trt_engines.acquire_engines(
                self._ort_dir, status.gpu.sm, entries=missing,
                progress=self._report, cancel=self.cancel)

        verification = trt_engines.verify_runtime(
            self._ort_dir, graphs, progress=self._report, cancel=self.cancel)
        if verification.ok:
            return True, (f"TensorRT verified for {len(verification.results)} "
                          f"graph(s) in "
                          f"{trt_engines._duration(verification.elapsed)}.")
        failed = ", ".join(r.name for r in verification.failed)
        return False, (f"No marker was written: {len(verification.failed)} of "
                       f"{len(verification.results)} graph(s) failed to build "
                       f"({failed}). The player keeps using DirectML.")


class ModelDownloadDialog(QDialog):
    def __init__(self, manifest_path, models_dir, parent=None, ort_dir=None,
                 models_dirs=None, playback_active=False):
        super().__init__(parent)
        self.setWindowTitle("2D→3D depth models")
        self.setMinimumWidth(560)
        self._models_dir = models_dir
        # The candidate `ort_tensorrt` directory, resolved by the player and
        # handed over the same way models_dir is -- this dialog knows nothing
        # about the player. None means "no runtime directory known", which the
        # status row reports from GPU detection alone.
        self._ort_dir = ort_dir
        # Every directory the player searches for depth models, because the
        # engine probe must build every graph the player can OPEN, and the
        # player looks in three places (next to the exe, the source tree, the
        # per-user fallback) while downloads land in exactly one of them.
        self._models_dirs = tuple(models_dirs or (models_dir,))
        # An engine build saturates the GPU for minutes at a time. Modality
        # keeps the user out of the player's controls but does not stop a
        # decode, so the player says whether a file is loaded and the TensorRT
        # action refuses while one is -- "playback never dies for 3D".
        self._playback_active = bool(playback_active)
        self._manifest = model_fetcher.load_manifest(manifest_path)
        # The manifest pins a commit SHA. Until the model repository is actually
        # published the placeholder is still in there, and every URL built from
        # it resolves to nothing -- so a Download click would spend the user's
        # click on "Download failed: HTTP Error 404: Not Found", which reads as
        # a broken project rather than as an unfinished one. Say so up front and
        # keep the buttons off (refresh() honours this too).
        self._unpublished = self._manifest.revision == "PENDING_UPLOAD"
        self._thread = None
        self._worker = None

        self._layout = QVBoxLayout(self)
        self._layout.addWidget(QLabel(
            "The depth models are downloaded separately so the player itself "
            "stays small. Small alone makes all three presets work; Base is the "
            "quality upgrade."))
        self._rows = {}
        for key, pack in self._manifest.packs.items():
            row = QHBoxLayout()
            label = QLabel()
            button = QPushButton("Download")
            button.clicked.connect(lambda _c=False, k=key: self._start(k))
            row.addWidget(label, 1)
            row.addWidget(button)
            self._layout.addLayout(row)
            self._rows[key] = (label, button)

        # A third row, below the packs, for the optional TensorRT runtime.
        trt_row = QHBoxLayout()
        self._trt_label = QLabel()
        self._trt_button = QPushButton("Install")
        self._trt_button.clicked.connect(self._start_trt)
        trt_row.addWidget(self._trt_label, 1)
        trt_row.addWidget(self._trt_button)
        self._layout.addLayout(trt_row)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setVisible(False)
        self._layout.addWidget(self._bar)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._layout.addWidget(self._status)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._cancel)
        # Stage 2's staging directory survives a cancel on purpose, so a rerun
        # re-fetches only what is missing rather than restarting a 1.3 GB
        # transfer. Throwing it away is therefore the user's decision, not a
        # side effect of cancelling -- which is what this button is.
        self._discard_button = QPushButton("Discard partial download")
        self._discard_button.setVisible(False)
        self._discard_button.clicked.connect(self._discard_staging)
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._discard_button)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(self._close_button)
        self._layout.addLayout(buttons)

        self.refresh()
        if self._unpublished:
            self._status.setText(
                "The model repository has not been published yet — this build "
                "cannot download them.")

    def refresh(self):
        # Two independent reasons a pack cannot be fetched right now. refresh()
        # runs after every state change, so it -- not __init__ -- is what has to
        # keep the buttons off for an unpublished manifest.
        blocked = self._thread is not None or self._unpublished
        for key, status in model_fetcher.pack_status(
                self._manifest, self._models_dir).items():
            label, button = self._rows[key]
            pack = self._manifest.packs[key]
            if status.complete:
                label.setText(f"{pack.label} — installed ({_human(pack.size)})")
                button.setText("Re-download")
            else:
                label.setText(
                    f"{pack.label} — {status.installed}/{status.total} files, "
                    f"{_human(status.missing_bytes)} to download")
                button.setText("Download")
            button.setEnabled(not blocked)
        self._refresh_trt()

    def _refresh_trt(self):
        """Re-reads the TensorRT runtime's state from disk.

        Deliberately in refresh() rather than only in __init__: an install
        writes INTO that directory and the row has to follow it without a second
        mechanism.
        """
        status = trt_runtime.runtime_status(self._ort_dir)
        self._trt_label.setText(f"TensorRT — {status.message}")
        # Greyed only for the two verdicts the user can do nothing about. The
        # rest are states an install will move out of, so they stay live text.
        actionable = status.state not in (trt_runtime.NO_GPU,
                                          trt_runtime.UNSUPPORTED_SM)
        self._trt_label.setEnabled(actionable)
        # The full device name, which the row itself shortens to keep one line.
        self._trt_label.setToolTip(status.gpu.name if status.gpu else "")

        # Three verbs for three different amounts of work, because they are not
        # the same offer. "Install" downloads ~1.3 GB and then verifies;
        # "Verify" only builds engines against a runtime already on disk;
        # "Re-verify" re-attests a working install, which is what a newly
        # downloaded model pack needs, since the marker names graphs and a graph
        # it does not name gets DirectML.
        self._trt_button.setVisible(actionable)
        self._trt_button.setText(
            "Verify" if status.state == trt_runtime.UNVERIFIED else
            "Re-verify" if status.state == trt_runtime.READY else "Install")
        self._trt_button.setEnabled(
            actionable and self._thread is None and not self._playback_active)
        self._trt_button.setToolTip(
            "Close the current video first — building TensorRT engines uses "
            "the GPU for minutes at a time." if self._playback_active else "")

        staged = self._ort_dir and os.path.isdir(
            trt_fetcher.staging_dir(self._ort_dir))
        self._discard_button.setVisible(bool(staged) and self._thread is None)

    def _start(self, pack_key):
        if self._thread is not None or self._unpublished:
            return
        self._begin(_Worker(self._manifest, pack_key, self._models_dir),
                    self._on_progress)

    def _start_trt(self):
        if self._thread is not None or self._playback_active:
            return
        worker = _TrtWorker(self._ort_dir, self._models_dirs)
        worker.confirm.connect(self._on_confirm)
        self._begin(worker, self._on_trt_progress)

    def _begin(self, worker, progress_slot):
        self._worker = worker
        self._thread = QThread(self)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(progress_slot)
        worker.done.connect(self._on_done)
        self._bar.setValue(0)
        self._bar.setVisible(True)
        self._cancel_button.setVisible(True)
        self._close_button.setEnabled(False)
        self.refresh()
        self._thread.start()

    def _on_confirm(self, summary):
        """Runs on the GUI thread; the worker is parked until it is answered."""
        worker = self._worker
        if worker is None:
            return
        answer = QMessageBox.question(
            self, "Set up TensorRT", summary,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        worker.answer(answer == QMessageBox.StandardButton.Ok)

    def _on_progress(self, name, done, total):
        # Pack-level only. Six files now transfer at once, so `name` changes
        # several times a second and a per-file label reads as a glitch rather
        # than as progress. The bar and the byte count are the honest signal;
        # the worker still sends the name (the signal shape is unchanged) and
        # nothing here needs it.
        self._bar.setValue(int(done * 100 / total) if total else 0)
        self._status.setText(f"{_human(done)} of {_human(total)}")

    def _on_trt_progress(self, message, done, total):
        # The message IS the report here, and it has to be: the three phases
        # count different things (bytes, then bytes again, then graphs), and a
        # bar alone cannot say which of 21 engine builds is running or how long
        # it has been running for.
        self._bar.setValue(int(done * 100 / total) if total else 0)
        self._status.setText(message)

    def _on_done(self, ok, message):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        self._bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._close_button.setEnabled(True)
        self._status.setText(message)
        self.refresh()

    def _cancel(self):
        worker = self._worker
        if worker is not None:
            worker.cancel.set()
            # A worker parked on its confirmation would otherwise never see the
            # cancel: it is waiting on an answer, not on the event.
            if hasattr(worker, "answer"):
                worker.answer(False)
        self._status.setText(
            "Cancelling — an engine build already running finishes first, so "
            "its work is not thrown away…")

    def _discard_staging(self):
        if self._thread is not None or not self._ort_dir:
            return
        if trt_fetcher.discard_staging(self._ort_dir):
            self._status.setText("Partial download discarded.")
        self.refresh()

    def reject(self):
        # Closing mid-download would leave the worker writing into a dialog that
        # no longer exists. Cancel and let _on_done close it.
        if self._thread is not None:
            self._cancel()
            return
        super().reject()
