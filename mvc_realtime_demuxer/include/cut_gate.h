// CutGate: two-consecutive-frame confirmation for the source-histogram
// scene-cut signal, paired with an instantaneous pass-through for an
// already-authoritative depth-residual cut (see
// .superpowers/sdd/2026-07-29-synth3d-round2, Task 5).
//
// SharedDepthService's histogram-distance detector alone fires on flashes and
// fast fades, not just hard cuts: a single frame's luma histogram can spike
// and fall back within one cycle. Gating it behind a SECOND consecutive
// exceedance keeps a genuine cut detected within two frames while collapsing
// a single-frame spike back to "no cut" before it ever reaches the
// stabilizer's scene-cut OR. The depth residual computed inside
// DepthStabilizer::step() stays instantaneous -- the service never routes it
// through this gate; `depth_cut` exists so a caller (and this file's own
// tests) can express "an authoritative cut already happened this cycle,
// treat it as confirmed and re-arm" in the same call.
//
// Pure math, header-only, no ORT/D3D; single-owner-thread use, same idiom as
// DepthStabilizer (see depth_stabilizer.h).
#pragma once

class CutGate {
public:
    float histogram_threshold = 0.42f;

    // depth_cut: an already-authoritative cut this cycle (e.g. the depth
    // residual test inside DepthStabilizer::step()) -- always confirms
    // immediately and re-arms the histogram confirmation state.
    // histogram_distance: the source-image histogram distance for this
    // cycle; two CONSECUTIVE calls at/above histogram_threshold confirm.
    // Returns true exactly on the cycle the cut is confirmed.
    bool update(bool depth_cut, float histogram_distance) {
        if (depth_cut) {
            pending_ = false;
            return true;
        }
        if (histogram_distance >= histogram_threshold) {
            if (pending_) {
                pending_ = false;  // confirmed: re-arm for the next shot
                return true;
            }
            pending_ = true;
            return false;
        }
        pending_ = false;  // below threshold: drop a single-frame spike
        return false;
    }

    bool pending() const { return pending_; }

private:
    bool pending_ = false;
};
