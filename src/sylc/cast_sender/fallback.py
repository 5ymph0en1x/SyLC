"""SyLC Cast -- adaptive-quality fallback policy (pure decision, no I/O).

`FallbackLadder` is the sender's decision brain for keeping the stream alive when the
link degrades. It is deliberately PURE: it never touches the encoder, the transport, or
the clock. Given feedback each cycle -- how deep the transport's send queue is, and
whether the receiving client actually underran its buffer -- it returns the encoder
reconfiguration the controller (Task 12) should apply via
``NativeRenderer.cast_reconfigure(mode, bitrate_bps)``, or ``None`` to hold steady.

The ladder, highest quality first::

    0  lossless      (wired / USB-C only -- ~gigabits for 4K SBS)
    1  cbr 500 Mbps  (visually lossless -- the Wi-Fi start rung)
    2  cbr 300 Mbps
    3  cbr 200 Mbps  <- the "CBR floor": as low as pure queue backpressure may push
    4  cbr_lowres    (drop resolution -- the controller performs the actual downscale;
                      reachable only on Wi-Fi in v1)

Policy in one paragraph: a single 500-ms sample is diagnostic noise, not permission to
reconfigure a live encoder. Step DOWN only after ``PRESSURE_STEPS_FOR_DOWN`` consecutive
pressure samples; step UP after ``STABLE_STEPS_FOR_UP`` stable samples; never climb above
the starting rung. USB never falls into the v1 pseudo-low-resolution 100-Mbps rung because
that rung does not actually reduce resolution and made a transient decoder event a
permanent quality cap.

The class is fully unit-testable on any host (see tests/cast/test_fallback_ladder.py).
"""
from __future__ import annotations


class FallbackLadder:
    """Adaptive-quality step policy. Stateful but side-effect-free.

    Parameters
    ----------
    start_mode : str
        ``"lossless"`` (start at rung 0) or ``"cbr"`` (start at rung 1, the top CBR).
        ``"balanced"`` starts at the production 500 Mbps rung. Any other value is
        treated as ``"cbr"``.
    wired : bool
        ``True`` for a USB-C link: starts at the caller's rung and lets queue
        backpressure fall only as far as the CBR floor (rung 3); dropping resolution
        (rung 4) is never selected. ``False`` (Wi-Fi) lets sustained pressure
        walk all the way to the bottom rung.
    """

    # Rungs, highest quality first. (mode, bitrate_bps); lossless ignores the bitrate.
    RUNGS: tuple[tuple[str, int], ...] = (
        ("lossless",   0),
        ("cbr",        500_000_000),
        ("cbr",        300_000_000),
        ("cbr",        200_000_000),
        ("cbr_lowres", 100_000_000),
    )

    # The receiver reports only AUs genuinely waiting for an input buffer. Three
    # queued AUs still fit its four-AU pool; four means the whole pool is occupied.
    HIGH_WATER_MARK: int = 3
    # Zero or one pending AU is a healthy asynchronous MediaCodec pipeline.
    LOW_WATER_MARK: int = 1
    # Feedback arrives every 500 ms. Require persistence in both directions:
    # 1.5 s before reducing quality, 6 s before raising it again.
    PRESSURE_STEPS_FOR_DOWN: int = 3
    STABLE_STEPS_FOR_UP: int = 12

    # Rung index of the wired CBR floor.
    _CBR_FLOOR_INDEX: int = 3

    # "balanced" is the production auto mode.  500 Mbps is visually lossless
    # and deterministic on modern Wi-Fi 6 links.
    _START_INDEX = {"lossless": 0, "cbr": 1, "balanced": 1}

    def __init__(self, start_mode: str, wired: bool):
        self._start_index = self._START_INDEX.get(start_mode, 1)
        self._index = self._start_index
        self._wired = bool(wired)
        # USB always stops at 200 Mbps. The 100-Mbps v1 rung claims "lowres" but
        # still sends 3840x1080, so using it on a wired link only destroys quality
        # without removing the receiver's actual pixel-decoding workload.
        self._backpressure_floor = self._CBR_FLOOR_INDEX if wired else len(self.RUNGS) - 1
        self._stable = 0            # consecutive stable cycles since the last step
        self._pressure = 0          # consecutive pressure cycles since the last step

    # -- read-only introspection (for the controller / tests) ---------------- #
    @property
    def index(self) -> int:
        """Current rung index (0 = highest quality)."""
        return self._index

    @property
    def start_index(self) -> int:
        """The rung the session started on; the ladder never climbs above it."""
        return self._start_index

    @property
    def current(self) -> dict:
        """The current rung as ``{"mode", "bitrate_bps"}`` (no step, no force_idr)."""
        mode, bitrate = self.RUNGS[self._index]
        return {"mode": mode, "bitrate_bps": bitrate}

    # -- the decision -------------------------------------------------------- #
    def on_feedback(self, send_queue_depth: int, client_underrun: bool) -> dict | None:
        """Fold one feedback sample into the policy.

        Returns the reconfiguration to apply -- ``{"mode", "bitrate_bps", "force_idr"}``
        (``force_idr`` is always ``True``: every rung change must emit an IDR so the
        decoder recovers instantly) -- or ``None`` to hold the current rung.
        """
        # --- DOWN: only sustained pressure is actionable. -------------------
        pressured = bool(client_underrun) or send_queue_depth > self.HIGH_WATER_MARK
        if pressured:
            self._stable = 0
            self._pressure += 1
            if self._pressure >= self.PRESSURE_STEPS_FOR_DOWN:
                self._pressure = 0
                return self._step_down(limit=self._backpressure_floor)
            return None

        # --- UP: only after a sustained stable run, and never above the start rung. ---
        self._pressure = 0
        if send_queue_depth <= self.LOW_WATER_MARK:
            if self._index > self._start_index:
                self._stable += 1
                if self._stable >= self.STABLE_STEPS_FOR_UP:
                    self._stable = 0
                    self._index -= 1
                    return self._step()
            else:
                self._stable = 0     # already at the ceiling: nothing to accumulate toward
            return None

        # --- Dead-band (LOW_WATER < depth <= HIGH_WATER): reset recovery, hold rung. ---
        self._stable = 0
        return None

    # -- helpers ------------------------------------------------------------- #
    def _step_down(self, limit: int) -> dict | None:
        """Advance one rung toward `limit` (inclusive); ``None`` if already at/below it."""
        if self._index < limit:
            self._index += 1
            return self._step()
        return None

    def _step(self) -> dict:
        mode, bitrate = self.RUNGS[self._index]
        return {"mode": mode, "bitrate_bps": bitrate, "force_idr": True}
