# -*- coding: utf-8 -*-
"""Thread-safe, session-aware seek orchestration for the SyLC player.

The queue deliberately knows only the narrow PlayerWindow protocol it needs:
media-session identity, decoder/player state, and three seek request handlers.
Keeping that boundary explicit prevents the main window from also owning the
seek state machine and makes this timing-sensitive component testable alone.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal


logger = logging.getLogger(__name__)


def should_resume_after_sync(is_user_seek, was_playing):
    """Resolve transport state for the shared MVC startup/seek handshake."""
    return bool(was_playing) if is_user_seek else True


class SeekState(Enum):
    """States of the seek state machine."""

    IDLE = auto()
    SEEKING = auto()
    COOLDOWN = auto()


@dataclass
class SeekRequest:
    """One immutable-in-practice seek request owned by a media session."""

    target_time: float
    timestamp: float
    is_mvc: bool
    session_id: int = 0
    resume_after: bool = True


class RobustSeekQueue(QObject):
    """Coalesce and serialize seeks without touching mpv off the Qt thread.

    Expected parent protocol:

    * ``_media_session_id`` and optional ``_media_single_shot``;
    * ``mvc_mode_active``, ``mvc_decoder_thread``, ``player`` and ``is_playing``;
    * ``_on_seek_queue_pause_request(bool)``;
    * ``_on_seek_queue_mpv_seek(float)``;
    * ``_on_seek_queue_decoder_seek(float)``.
    """

    request_mpv_pause = Signal(bool)
    request_mpv_seek = Signal(float)
    request_decoder_seek = Signal(float)
    seek_started = Signal(float)
    seek_completed = Signal()
    request_enqueued = Signal(object)
    finish_enqueued = Signal(object)

    # The decoder side of a seek measures ~25 ms (headless bench). These
    # delays still coalesce slider events while avoiding a perceptible floor.
    DEBOUNCE_DELAY_MS = 60
    COOLDOWN_PERIOD_MS = 100
    # A cold optical SSIF can legitimately need several seconds to re-pair
    # base and dependent views; timeout only a genuine hang.
    SEEK_TIMEOUT_MS = 45000

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self._parent = parent_window
        self._lock = threading.Lock()
        self._state = SeekState.IDLE
        self._pending_request: Optional[SeekRequest] = None
        self._active_request: Optional[SeekRequest] = None
        self._current_seek_start: float = 0.0
        self._last_seek_completed: float = 0.0

        self._debounce_timer: Optional[QTimer] = None
        self._timeout_timer: Optional[QTimer] = None
        self._cooldown_timer: Optional[QTimer] = None

        self._seeks_requested = 0
        self._seeks_executed = 0
        self._seeks_coalesced = 0
        self._timeouts = 0

        self.request_mpv_pause.connect(self._parent._on_seek_queue_pause_request)
        self.request_mpv_seek.connect(self._parent._on_seek_queue_mpv_seek)
        self.request_decoder_seek.connect(self._parent._on_seek_queue_decoder_seek)
        self.request_enqueued.connect(self._handle_request)
        self.finish_enqueued.connect(self._handle_seek_finished)

        logger.info("[SEEK-QUEUE] RobustSeekQueue initialized with Qt signals")

    def _ensure_timers(self):
        if self._debounce_timer is None:
            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._on_debounce_expired)

        if self._timeout_timer is None:
            self._timeout_timer = QTimer(self)
            self._timeout_timer.setSingleShot(True)
            self._timeout_timer.timeout.connect(self._on_timeout)

        if self._cooldown_timer is None:
            self._cooldown_timer = QTimer(self)
            self._cooldown_timer.setSingleShot(True)
            self._cooldown_timer.timeout.connect(self._on_cooldown_expired)

    def _session_single_shot(self, delay_ms, callback, session_id):
        scheduler = getattr(self._parent, '_media_single_shot', None)
        if scheduler is not None:
            scheduler(delay_ms, callback, session_id)
        else:
            QTimer.singleShot(delay_ms, callback)

    def _request_is_current(self, request):
        return (request is not None
                and request.session_id
                == getattr(self._parent, '_media_session_id', 0))

    def invalidate_session(self):
        """Discard every delayed action owned by the previous media session."""
        with self._lock:
            for timer in (
                    self._timeout_timer,
                    self._debounce_timer,
                    self._cooldown_timer):
                if timer is not None:
                    timer.stop()
            self._pending_request = None
            self._active_request = None
            self._state = SeekState.IDLE

    @property
    def resume_after_seek(self):
        """Playback intent captured before the queue's technical pause."""
        request = self._active_request or self._pending_request
        return request.resume_after if request is not None else False

    def request_seek(self, target_time: float, is_mvc: bool = True,
                     resume_after=None):
        self._seeks_requested += 1
        if resume_after is None:
            resume_after = bool(getattr(self._parent, 'is_playing', False))
        request = SeekRequest(
            target_time=target_time,
            timestamp=time.monotonic(),
            is_mvc=is_mvc,
            session_id=getattr(self._parent, '_media_session_id', 0),
            resume_after=bool(resume_after),
        )
        self.request_enqueued.emit(request)

    def _handle_request(self, request: SeekRequest):
        if request.session_id != getattr(self._parent, '_media_session_id', 0):
            return
        self._ensure_timers()

        with self._lock:
            logger.info(
                "[SEEK-QUEUE] Request to %.2fs, state=%s",
                request.target_time, self._state.name)

            if self._state == SeekState.IDLE:
                self._pending_request = request
                self._state = SeekState.SEEKING
                self._debounce_timer.start(self.DEBOUNCE_DELAY_MS)

            elif self._state == SeekState.SEEKING:
                if self._pending_request:
                    self._seeks_coalesced += 1
                    logger.info(
                        "[SEEK-QUEUE] Coalesced: %.2fs -> %.2fs",
                        self._pending_request.target_time,
                        request.target_time)
                self._pending_request = request
                if self._debounce_timer.isActive():
                    self._debounce_timer.start(self.DEBOUNCE_DELAY_MS)

            elif self._state == SeekState.COOLDOWN:
                self._pending_request = request
                logger.info(
                    "[SEEK-QUEUE] Queued during cooldown: %.2fs",
                    request.target_time)

    def _on_debounce_expired(self):
        request = None
        with self._lock:
            if self._pending_request is None:
                self._state = SeekState.IDLE
                return
            request = self._pending_request
            self._pending_request = None
            self._current_seek_start = time.monotonic()

        if self._request_is_current(request):
            self._execute_seek(request)
            self._timeout_timer.start(self.SEEK_TIMEOUT_MS)
        else:
            with self._lock:
                self._state = SeekState.IDLE

    def _execute_seek(self, request: SeekRequest):
        if not self._request_is_current(request):
            return
        self._active_request = request
        self._seeks_executed += 1
        target = request.target_time
        logger.info(
            "[SEEK-QUEUE] Executing seek to %.2fs (#%d)",
            target, self._seeks_executed)

        try:
            self.seek_started.emit(target)

            if request.is_mvc and self._parent.mvc_mode_active:
                self.request_mpv_pause.emit(True)
                self._session_single_shot(
                    50, lambda: self._do_mvc_seek(target), request.session_id)
            else:
                self.request_mpv_seek.emit(target)
                self._session_single_shot(
                    300,
                    lambda: self.notify_seek_finished(request.session_id),
                    request.session_id)

        except Exception as exc:
            logger.error("[SEEK-QUEUE] Seek execution failed: %s", exc)
            self._force_reset_state()

    def _do_mvc_seek(self, target_time: float):
        """Seek only the MVC decoder while mpv remains paused and idle.

        On optical media the audio and video readers share one head. Seeking
        them to different locations concurrently causes severe head thrashing.
        The decoder therefore scans to its actual IDR first; PlayerWindow's
        ``seekIDRFound`` handler then aligns and resumes mpv atomically.
        """
        try:
            logger.info(
                "[SEEK-QUEUE] MVC seek: decoder scans "
                "(MPV stays paused/idle) at %.3fs",
                target_time)

            try:
                player = getattr(self._parent, 'player', None)
                if player is not None:
                    player.pause = True
                    player['demuxer-readahead-secs'] = 1
                    player['demuxer-max-bytes'] = '8MiB'
                    player['demuxer-max-back-bytes'] = '4MiB'
            except Exception as exc:
                logger.warning(
                    "[SEEK-QUEUE] MPV read-ahead clamp skipped: %s", exc)

            self.request_decoder_seek.emit(target_time)
        except Exception as exc:
            logger.error("[SEEK-QUEUE] MVC seek failed: %s", exc)
            self._force_reset_state()

    def notify_seek_finished(self, session_id=None):
        """Queue decoder completion on the object's owning Qt thread."""
        owner = (getattr(self._parent, '_media_session_id', 0)
                 if session_id is None else session_id)
        self.finish_enqueued.emit(owner)

    def _handle_seek_finished(self, session_id=None):
        if (session_id is not None
                and session_id != getattr(self._parent, '_media_session_id', 0)):
            return
        with self._lock:
            logger.info(
                "[SEEK-QUEUE] Seek finished, state=%s", self._state.name)

            if self._timeout_timer and self._timeout_timer.isActive():
                self._timeout_timer.stop()

            self._last_seek_completed = time.monotonic()

        # PlayerWindow restores the user state captured before the queue's
        # technical pause. There must be exactly one pause authority here.
        self.seek_completed.emit()
        self._active_request = None

        with self._lock:
            if self._pending_request:
                self._state = SeekState.COOLDOWN
                self._cooldown_timer.start(self.COOLDOWN_PERIOD_MS)
            else:
                self._state = SeekState.IDLE
                logger.info("[SEEK-QUEUE] Back to IDLE")

    def _on_cooldown_expired(self):
        request = None
        with self._lock:
            logger.info("[SEEK-QUEUE] Cooldown expired")
            if self._pending_request:
                request = self._pending_request
                self._pending_request = None
                self._state = SeekState.SEEKING
                self._current_seek_start = time.monotonic()
            else:
                self._state = SeekState.IDLE

        if self._request_is_current(request):
            self._execute_seek(request)
            self._timeout_timer.start(self.SEEK_TIMEOUT_MS)
        elif request is not None:
            with self._lock:
                self._state = SeekState.IDLE

    def _on_timeout(self):
        self._timeouts += 1
        logger.error("[SEEK-QUEUE] TIMEOUT! (total: %d)", self._timeouts)
        self._force_reset_state()

    def _force_reset_state(self):
        pending = None
        with self._lock:
            logger.warning("[SEEK-QUEUE] Forcing state reset to IDLE")

            if self._timeout_timer:
                self._timeout_timer.stop()
            if self._debounce_timer:
                self._debounce_timer.stop()
            if self._cooldown_timer:
                self._cooldown_timer.stop()

            self._state = SeekState.IDLE
            pending = self._pending_request
            self._pending_request = None

        self.request_mpv_pause.emit(not self.resume_after_seek)
        self.seek_completed.emit()
        self._active_request = None

        if pending:
            logger.info(
                "[SEEK-QUEUE] Re-requesting pending seek to %.2fs",
                pending.target_time)
            self._session_single_shot(
                200,
                lambda: self.request_seek(
                    pending.target_time, pending.is_mvc,
                    resume_after=pending.resume_after),
                pending.session_id)

    def is_busy(self) -> bool:
        with self._lock:
            return self._state != SeekState.IDLE


__all__ = [
    'RobustSeekQueue', 'SeekRequest', 'SeekState',
    'should_resume_after_sync',
]
