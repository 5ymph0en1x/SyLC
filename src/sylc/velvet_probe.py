"""
velvet_probe — read-only timing instrumentation for the "liquid smoothness" push.

Disabled by default (ENABLED=False) => every public call is a one-line no-op,
zero overhead, no behaviour change. Enable with the env var SYLC_VELVET_PROBE=1.

Optional env:
  SYLC_VELVET_PROBE_SECONDS=N   auto-exit (os._exit) after ~N s of probe activity
                                (bounded measurement run); 0/unset = never.
  SYLC_VELVET_PROBE_DUMP=N      dump summary every N s (default 3).
  SYLC_VELVET_PROBE_LOG=path    append summaries to this file (default velvet_probe.log).

What it measures (all distributions reported as mean/p50/p95/p99/max/std in ms):
  au_ms       per-AU decode-thread cost (decode + extract), excl. backpressure wait
  deliver_ms  per-frame plane extraction (get_plane x6 + frame_data build)
  emit_ms     inter-emit interval at the presenter (the player's chosen cadence)
  qlen        presentation_queue depth sampled at each emit (decode head-room)
  present_ms  inter-present interval at the GUI (actual D3D11/vsync cadence)
  counters    drops (GUI 1-slot overwrite), holds (V12 early-hold), bulkdrops (V12 late)

Revert: delete this file + grep 'velvet_probe' call sites.
"""
import os
import sys
import time
import threading

ENABLED = os.environ.get('SYLC_VELVET_PROBE', '') not in ('', '0', 'false', 'False', 'no')

now = time.perf_counter

_AUTO_SECONDS = 0.0
_DUMP_EVERY = 3.0
_LOG_PATH = 'velvet_probe.log'
try:
    _AUTO_SECONDS = float(os.environ.get('SYLC_VELVET_PROBE_SECONDS', '0') or 0)
    _DUMP_EVERY = float(os.environ.get('SYLC_VELVET_PROBE_DUMP', '3') or 3)
    _LOG_PATH = os.environ.get('SYLC_VELVET_PROBE_LOG', 'velvet_probe.log')
except Exception:
    pass

_lock = threading.Lock()
_series = {}        # name -> list[float]
_last_tick = {}     # name -> last perf_counter
_counters = {}      # name -> cumulative int
_counter_marks = {} # name -> last-dumped cumulative int
_started = False
_start_t = None


def _series_for(name):
    s = _series.get(name)
    if s is None:
        s = []
        _series[name] = s
    return s


def _ensure():
    global _started, _start_t
    if _started:
        return
    with _lock:
        if _started:
            return
        _started = True
        _start_t = now()
    t = threading.Thread(target=_loop, name="velvet-probe", daemon=True)
    t.start()


def record(name, value):
    """Record one value (ms) into the named distribution."""
    if not ENABLED:
        return
    with _lock:
        _series_for(name).append(value)
    _ensure()


def tick(name):
    """Record the interval (ms) since the previous tick(name) into <name>_ms."""
    if not ENABLED:
        return
    t = now()
    with _lock:
        last = _last_tick.get(name)
        _last_tick[name] = t
        if last is not None:
            _series_for(name + '_ms').append((t - last) * 1000.0)
    _ensure()


def incr(name, n=1):
    if not ENABLED:
        return
    with _lock:
        _counters[name] = _counters.get(name, 0) + n
    _ensure()


# ---- convenience wrappers (call sites stay tiny) ----
def on_emit(queue_len):
    if not ENABLED:
        return
    tick('emit')
    record('qlen', float(queue_len))


def on_present():
    if not ENABLED:
        return
    tick('present')


def on_drop(n=1):
    incr('drops', n)


def on_hold():
    incr('holds', 1)


def on_bulkdrop(n):
    incr('bulkdrops', n)


# ---- reporting ----
def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _fmt(name, vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    std = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    return ("  %-11s n=%-5d mean=%7.2f p50=%7.2f p95=%7.2f p99=%7.2f max=%8.2f std=%6.2f"
            % (name, n, mean, _pct(s, 0.5), _pct(s, 0.95), _pct(s, 0.99), s[-1], std))


def _dump():
    with _lock:
        snap = {k: v[:] for k, v in _series.items()}
        for v in _series.values():
            v.clear()
        counters = dict(_counters)
    el = (now() - _start_t) if _start_t else 0.0
    lines = ["==== VELVET PROBE  t=%6.1fs ====" % el]
    for name in sorted(snap):
        line = _fmt(name, snap[name])
        if line:
            lines.append(line)
    if counters:
        parts = []
        for k in sorted(counters):
            d = counters[k] - _counter_marks.get(k, 0)
            _counter_marks[k] = counters[k]
            parts.append("%s +%d (tot %d)" % (k, d, counters[k]))
        lines.append("  counters: " + "   ".join(parts))
    out = "\n".join(lines) + "\n"
    try:
        sys.stderr.write(out)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(out)
    except Exception:
        pass


def _loop():
    while True:
        try:
            time.sleep(_DUMP_EVERY)
        except Exception:
            return
        try:
            _dump()
        except Exception:
            pass
        if _AUTO_SECONDS and _start_t and (now() - _start_t) >= _AUTO_SECONDS:
            try:
                _dump()
            except Exception:
                pass
            try:
                sys.stderr.write("[VELVET PROBE] auto-exit after %.0fs\n" % _AUTO_SECONDS)
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(0)
