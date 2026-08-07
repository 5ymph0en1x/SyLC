// SharedDepthService — one ONNX depth session for every Synth3D renderer.
//
// A SyLC presentation can own up to four D3D11 renderers (embedded preview,
// frame-pack output, two projector eyes). Their GPU resources necessarily stay
// device-local, but depth inference must not be duplicated: all surfaces show
// the same source frame and therefore need the same temporally-stabilized map.
//
// This process-wide service owns the ORT session, temporal state and latest
// immutable q16 map. Renderers attach by model/runtime/grid key. One renderer
// holds a short renewable "input leader" lease and performs the width()*height()
// readback;
// every renderer consumes the same published map on its own D3D device. If the
// leader stops presenting, another surface takes over after a bounded lease
// timeout.
//
// The registry keeps at most one unattached service warm. Older idle services
// and failed services are retired by a dedicated reaper thread, so interactive
// disable remains non-blocking even while ORT is building/destroying a session
// without letting preset/aspect switches retain every model for process life.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>   // _dupenv_s/free: SYLC_LOOKAHEAD_DECAY probe (header-inline)
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "depth_engine.h"   // kDefaultDepthSide (the grid this service defaults to)

// CPU optical flow used by the service worker (defined in the .cpp).
// Exposed here so the python test binding can drive the exact production
// code path: the parallel implementation is required to be BIT-EXACT vs
// max_threads=1 (independent rows/nodes/candidates; the global-search min
// is reduced chunk-in-order so the sequential winner always wins).
namespace synth3d_flow {
struct DenseFlow {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> quality;
};
// Round 7 « flot biface » : fusion par pixel du transport CAUSAL
// (forward = prev->cur) et du candidat FUTUR négué (future = next->cur, donc
// -future estime prev->cur sous mouvement localement linéaire), départagés
// par résidu photométrique contre la VRAIE cible du transport (prev). Oracle
// 2026-08-03 (Oblivion) : -43 % de résidu en statique, -90 % en mouvement
// rapide/post-cut. La fiabilité est le max de l'accord symétrique des deux
// directions et de l'auto-confiance résiduelle du candidat retenu — la
// seconde évite de jeter l'historique de la DERNIÈRE frame d'un plan (le
// futur y appartient au plan suivant et désaccorde toujours).
// Sorties par pixel : flot fusionné, fiabilité, masque de mouvement (résidu
// aligné + pénalité d'incertitude, même forme que l'ancien passage FB).
// Bit-exact à tout nombre de threads (écritures par pixel, sommes réduites
// par chunk en ordre).
// Phase 1 « dopage naturel » (04/08) : `codec_hint` est un TROISIÈME
// candidat optionnel — le champ de mouvement du décodeur, rasterisé sur la
// grille. Oracle (oracle_mv_candidate.py, Oblivion x264) : min(causal,
// futur, MV) gagne 20-26 % de résidu sur la biface dans tous les régimes,
// et en motion/fast/looming les MV seuls battent le flot CPU ×2. Le même
// arbitre par résidu photométrique qui départage causal/futur rejette les
// vecteurs faux (intra = qualité 0, plat, sur-échelonné) ; à égalité la
// priorité reste causal > futur > MV (déterminisme). nullptr/vide = la
// biface historique, bit-exacte.
void fuse_bidirectional(const std::vector<float>& prev_luma,
                        const std::vector<float>& cur_luma,
                        const DenseFlow& forward,
                        const DenseFlow& future,
                        int width, int height,
                        float video_time_scale,
                        std::vector<float>& out_x,
                        std::vector<float>& out_y,
                        std::vector<float>& out_reliability,
                        std::vector<float>& out_motion,
                        double& motion_sum, float& mean_flow,
                        int max_threads,
                        const DenseFlow* codec_hint = nullptr);

// Rasterise un champ de vecteurs de blocs DÉCODEUR vers la grille
// d'inférence. `mv_xy` = paires int16 QUART-de-pel (convention H.264) par
// frame d'affichage, dst -> flot production (cur(p) ~ prev(p - flot)) ;
// `valid` (peut être null = tout valide) : 0 = bloc intra/absent ->
// qualité 0, le candidat n'existe pas. `time_scale` = source_dt de
// l'observation / durée d'une frame d'affichage (le worker peut sauter des
// frames). Échelonnage anisotrope source -> grille. Pur, testable
// (_synth3d_rasterize_hints_test).
void rasterize_motion_hints(const int16_t* mv_xy, const uint8_t* valid,
                            int blocks_w, int blocks_h,
                            int source_width, int source_height,
                            int grid_width, int grid_height,
                            float time_scale,
                            std::vector<float>& out_x,
                            std::vector<float>& out_y,
                            std::vector<float>& out_quality);

// Phase 2 (04/08) — divergence moyenne du champ (texels/texel) : positif =
// expansion (marche face caméra, dolly avant, zoom in). Détecte le looming
// DANS les vecteurs, sans dépendre d'un avis externe. Différences
// centrées, sous-échantillonné d'un facteur 2, déterministe.
double mean_divergence(const std::vector<float>& flow_x,
                       const std::vector<float>& flow_y,
                       int width, int height);

// Phase 2 (04/08) — expansion du mouvement aux silhouettes, extraite du
// worker (bit-exacte à l'historique quand directional=false : max 3×3 sur
// la bande de garde). directional=true ajoute la portée ANTI-TRAÎNÉE : les
// fantômes se forment en AMONT du contour qui bouge (le flot y est faible,
// l'EMA garde la vieille arête) ; un texel hérite donc du mouvement d'une
// sonde à 2-3 texels seulement si le flot DE LA SONDE avance de lui vers
// elle (projeté >= 0.75 texel) — « le contenu a bougé de moi vers elle,
// je suis sa traînée ». Cliché en scratch, écriture par texel propre :
// bit-exact à tout nombre de threads.
void expand_boundary_motion(std::vector<float>& motion,
                            const std::vector<float>& boundary,
                            const std::vector<float>& flow_x,
                            const std::vector<float>& flow_y,
                            int width, int height,
                            std::vector<float>& scratch,
                            int max_threads,
                            bool directional);

DenseFlow estimate_flow(const std::vector<float>& source_full,
                        const std::vector<float>& destination_full,
                        int full_width, int full_height,
                        int max_threads = 1);
}  // namespace synth3d_flow

namespace synth3d_aspect {
struct HorizontalBars {
    int top = 0;
    int bottom = 0;
    bool valid = false;
};
HorizontalBars detect_horizontal_letterbox(
    const std::vector<float>& luma, int width, int height);
}  // namespace synth3d_aspect

// Spatial refinement used between raw DA3 inference and temporal fusion.
// The production worker and Python regression binding share this exact path.
namespace synth3d_surface {
void compute_boundary(const std::vector<float>& depth,
                      const std::vector<float>& luma,
                      int width, int height,
                      std::vector<float>& boundary,
                      std::vector<float>& scratch,
                      int max_threads = 1);
void refine_observation(std::vector<float>& depth,
                        const std::vector<float>& luma,
                        const std::vector<float>& confidence,
                        const std::vector<float>& boundary,
                        int width, int height,
                        std::vector<float>& scratch,
                        int max_threads = 1);
// Contour re-anchoring (04/08): the depth discontinuity belongs to the
// dominant IMAGE edge. DA3 dilates silhouettes by 1-3 grid texels, so a
// band of BACKGROUND texels carries the foreground's depth, travels with
// it in the warp, and the disocclusion refills right behind — the fringe
// crawling along a backlit back/head. The whole downstream stack is
// near-biased by design (filament protection), so those texels had no way
// back. For each texel inside a real depth separation: locate the unique
// strong luma edge nearby, take median depth/luma representatives sampled
// BEYOND the contaminated band on each side (local color votes would be
// poisoned by the band's own near-depth/far-color coherence), and snap the
// texel to its own image side's representative — only when its current
// depth clearly belongs to the other side AND its luma votes for its own
// side. Symmetric: also recovers an arm eaten by the background. Runs on
// the published q16 after stabilization; the image edge is temporally
// stable, so this also damps contour breathing. X then Y pass, bit-exact
// at any thread count. SYLC_SYNTH3D_REALIGN=0 disables the worker call.
void realign_contours(std::vector<uint16_t>& depth_q16,
                      const std::vector<float>& luma,
                      int width, int height,
                      std::vector<uint16_t>& scratch,
                      int max_threads = 1);
// Build the packed RGBA16 ownership analysis used by the worker:
//   R = stabilized depth, G = conservative foreground-owned depth,
//   B = stereo safety, A = ownership repair strength.
// The RGB guide is the worker's ImageNet-normalized CHW model input. Repairs
// only ever propagate a supported NEAR layer into an ambiguous image contour;
// the safety channel remains available to flatten anything still unresolved.
// The worker then precomposes this analysis into the renderer's RG16 map.
void build_geometry_map(const std::vector<uint16_t>& depth_q16,
                        const std::vector<float>& rgb_chw,
                        const std::vector<float>& luma,
                        const std::vector<float>& confidence,
                        const std::vector<float>& boundary,
                        int width, int height,
                        std::vector<uint16_t>& geometry_rgba16,
                        std::vector<float>& scratch,
                        int max_threads = 1,
                        double* local_ms = nullptr,
                        double* propagation_ms = nullptr);
}  // namespace synth3d_surface

class SharedDepthService {
public:
    static constexpr int kDefaultSide = kDefaultDepthSide;
    // Six uint16 values per inference-grid pixel. Channels 0-1 are the
    // historical pair precomposed from the four CPU ownership channels
    // (effective depth, effective stereo safety) and feed the RG16 geometry
    // texture. Channels 2-5 are the round-5a transport block for the GPU
    // background plate: flow x/y in grid pixels quantized as (v/128)+0.5,
    // flow reliability, and one reserved slot — a skipped-flow (static)
    // cycle publishes the identity transport at full reliability.
    using GeometryMap = std::vector<uint16_t>;
    static constexpr size_t kGeometryChannels = 6;

    // Immutable worker publication. The CPU path fills `geometry` with the
    // historical six-channel final map. The GPU ownership path instead fills
    // two RGBA16 input planes plus RGBA16 transport; the renderer completes
    // ownership/safety directly into its device-local RG16 warp texture.
    struct GeometryFrame {
        bool gpu_ownership = false;
        GeometryMap geometry;
        GeometryMap surface_rgba16; // depth, luma, confidence, boundary
        GeometryMap rgb_rgba16;     // linear RGB, reserved alpha
        GeometryMap transport_rgba16;
    };

    // Both dimensions are part of the cache key. `height <= 0` preserves the
    // historical square call and resolves to width x width.
    // Acquisition and client attachment are one registry transaction. Keeping
    // them atomic prevents an idle trim from evicting a service in the narrow
    // gap between returning it and attach().
    static std::shared_ptr<SharedDepthService> acquire_attached(
        const std::wstring& model_path, const std::wstring& ort_dir,
        int width, int height, uint64_t client_id);

    // Detach and hand the renderer's owning reference to the registry reaper.
    // The caller's shared_ptr is empty on return; the last ORT/thread teardown
    // can therefore never run on the renderer/presentation thread.
    static void detach_and_release(
        std::shared_ptr<SharedDepthService>& service, uint64_t client_id);

    ~SharedDepthService();
    SharedDepthService(const SharedDepthService&) = delete;
    SharedDepthService& operator=(const SharedDepthService&) = delete;

    void attach(uint64_t client_id);
    void detach(uint64_t client_id);

    // Renderer-side leader protocol. wants_input() is cheap and must be called
    // before drawing/copying the width()*height() prep target. submit() swaps the CHW
    // buffer, exact media PTS and capture fallback into the service. Keeping
    // both clocks beside the pixels is essential: PTS drives content-temporal
    // behavior; worker/capture wall time is only compute observability/fallback.
    bool wants_input(uint64_t client_id, double video_time_ms = -1.0);
    // Diagnostic: a granted tap that ended without a submit. `ring_empty` tells
    // the two causes apart (nothing pending vs a copy still in flight).
    void note_drain_miss(bool ring_empty);
    // Diagnostic: called once per client render pass, so the SERVICE times the
    // interval instead of trusting each client to compute it. inwait_ms says the
    // worker waited; only this says whether it waited because the renderer was
    // slow to come back or because the tap path lost time between passes.
    // LEADER ONLY, by client_id: every attached surface calls process(), so
    // timing all of them measured the interval between passes across surfaces
    // (2 clients at 42 ms read as 21 ms) instead of the period of the one
    // surface that actually feeds the worker.
    void note_pass(uint64_t client_id);
    // Diagnostic (round 8): distinguish the user's request, resource arming,
    // and the FINAL shader decision after every cut/map-state gate. `cut_in`
    // is the decayed advisory in ms (positive = cut ahead).
    void note_plate_state(bool requested, bool armed, bool live, double cut_in);
    // Source VIDEO-PTS interval of the frames the worker consumes (41.7 ms at
    // 24 fps). The renderer needs it to express cut anticipation in FRAMES
    // rather than in a hardcoded millisecond constant that would be wrong at
    // 25 or 60 fps.
    float source_dt_ms() const {
        return source_dt_ms_.load(std::memory_order_acquire);
    }
    bool submit(uint64_t client_id, std::vector<float>& chw,
                std::vector<float>& full_frame_luma,
                double video_time_ms,
                std::chrono::steady_clock::time_point capture_time,
                int source_width = 0, int source_height = 0);
    // Compatibility path for non-renderer callers: no PTS, capture "now".
    bool submit(uint64_t client_id, std::vector<float>& chw) {
        std::vector<float> no_full_frame_luma;
        return submit(
            client_id, chw, no_full_frame_luma, -1.0,
            std::chrono::steady_clock::now(), 0, 0);
    }

    // The inference grid this service was created with. side() remains as a
    // compatibility alias for the horizontal/long-edge dimension.
    int side() const { return width_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Returns an immutable latest map only when its publication sequence is
    // newer than `after_sequence`; null means "keep the local GPU texture".
    // `source_video_ms` is the media PTS of the SOURCE OBSERVATION this map
    // was computed from (-1 when the timed frame path is unavailable): the
    // map's shot identity (04/08). The readback + pipeline overlap put the
    // published map 1-2 frames behind the presented image, so the consumer
    // must be able to ask "does this map belong to the shot on screen?"
    // rather than approximate it with racing wall-clock deadlines.
    std::shared_ptr<const GeometryFrame> snapshot(
        uint64_t after_sequence, uint64_t& sequence,
        double& source_video_ms) const;
    std::shared_ptr<const GeometryFrame> snapshot(
        uint64_t after_sequence, uint64_t& sequence) const {
        double ignored_video_ms = -1.0;
        return snapshot(after_sequence, sequence, ignored_video_ms);
    }

    void notify_seek();
    bool running() const;
    bool failed() const;
    int client_count() const;
    uint64_t instance_id() const { return instance_id_; }
    std::string status() const;
    // Renderer capability handshake. The worker publishes compute inputs only
    // after the attached D3D11 path has created every required shader/view.
    // False keeps the historical CPU map and is a safe driver fallback.
    void set_gpu_ownership_enabled(bool enabled) {
        gpu_ownership_enabled_.store(enabled, std::memory_order_release);
    }

    // Narrow diagnostics used by native regression tests and support logs.
    static void debug_registry_stats(
        size_t& services, size_t& active, size_t& idle);

    // steady_clock ms timestamp of the last effective geometry snap (a
    // confirmed scene cut OR a consumed seek/reset), or -1 if none has
    // happened yet. Read by Synth3D::process() to drive the post-cut/seek
    // ease-out ramp (Task 5); never blocks, safe to call from any thread.
    int64_t last_snap_steady_ms() const;
    // Media PTS of the source observation that caused the last effective snap,
    // or -1 when the timed frame path is unavailable.
    double last_snap_video_ms() const;
    // Per-shot zero-parallax suggestion from the stabilizer (normalized
    // nearness, already smoothed in video time and snapped on cuts). Read by
    // Synth3D::process() when auto-convergence is enabled; never blocks.
    float suggested_convergence() const {
        return suggested_convergence_.load(std::memory_order_acquire);
    }

    // Look-ahead advisory (two-filter scout, spec 2026-08-03): delays in ms
    // from the PRESENTED position to the next cut / motion-storm onset seen
    // in the DECODED future, or <0 for none. Refreshed by the player each UI
    // tick; the worker only honours values fresher than ~500ms so a stalled
    // pump can never pin an imminence. SYLC_LOOKAHEAD=0 disables intake.
    // `cut_pts_ms` (04/08) is the ABSOLUTE media PTS of the same reported
    // cut, or a negative value for none (a PTS is always >= 0, so -1.0 is
    // safe here, unlike the relative delays). It feeds the cross-shot cut
    // boundary list below — state, not another deadline.
    void set_lookahead_advisory(double cut_in_ms, double storm_in_ms,
                                double cut_pts_ms = -1.0);

    // ---- Cross-shot gate (04/08) -------------------------------------
    // The structural gap behind "old-shot contours survive every hard cut":
    // the published map lags the presented image by 1-2 frames and carried
    // no shot identity, so all protection was temporal — four deadlines
    // (decayed advisory, 120 ms hold, 300 ms ramp, worker snap) racing each
    // other, any single late one re-opening the disparity budget while the
    // map still described the DYING shot. These primitives replace the race
    // with the exact state test: does a known cut boundary c separate the
    // map's source observation from the presented frame?
    //
    // note_cut_pts records a cut boundary on the media clock: the scout's
    // absolute pts (ahead of presentation) and the worker's own snap pts
    // (authoritative, later) both land here; duplicates within half a frame
    // merge. notify_seek() purges the list (pre-seek timeline).
    void note_cut_pts(double pts_ms);

    // Codec motion hints (phase 1, 04/08): the decoder's per-frame block
    // motion field, keyed by media pts. Quarter-pel per DISPLAY frame,
    // production flow convention. Kept in a small ring; the worker fetches
    // the entry matching its observation's pts (±21 ms) at prep and
    // rasterizes it as fuse_bidirectional's third candidate. Stale entries
    // simply never match. SYLC_SYNTH3D_MV_HINTS=0 disables intake.
    void note_motion_hints(double pts_ms, double frame_ms,
                           int blocks_w, int blocks_h,
                           int source_width, int source_height,
                           std::vector<int16_t>&& mv_xy,
                           std::vector<uint8_t>&& valid);
    // True while a recorded cut c satisfies map < c <= presented (with the
    // jitter tolerance of cross_shot_gate). The consumer must then present
    // FLAT (zero disparity budget, no plate): warping the new shot with the
    // old shot's geometry is what painted the old silhouettes as
    // disocclusion masks. Releases by STATE — the first map whose source
    // observation is at/after the cut — never by timer. Also prunes
    // boundaries more than ~4 s behind `presented_video_ms` (every surface
    // shows the same source frame, so any surface's clock may prune).
    bool cross_shot(double map_video_ms, double presented_video_ms) const;
    // Latest recorded boundary already reached by this presented frame, or
    // -1.0. Unlike cross_shot(), this deliberately ignores map identity: GPU
    // temporal memories must change shot exactly at T0, even if the worker's
    // new-shot map/snap has not arrived yet. Each renderer tracks the returned
    // PTS independently, so multiple surfaces can purge their local plates.
    double latest_presented_cut(double presented_video_ms) const;
    // Pure decision for ONE boundary, exposed to the python test binding
    // (_synth3d_shot_gate_test) so the exact production math is pinned.
    // The three inputs live on the same media clock but reach it through
    // different roundings (scout push pts, capture pts, present pts):
    // kPtsJitterMs absorbs that without ever spanning a real frame
    // (frame spacing >= ~16 ms at 60 fps). Invalid (<0) clocks never gate:
    // the temporal protections remain the only cover on untimed paths.
    static constexpr double kPtsJitterMs = 4.0;
    static bool cross_shot_gate(double map_video_ms,
                                double presented_video_ms,
                                double cut_pts_ms) {
        if (map_video_ms < 0.0 || presented_video_ms < 0.0 ||
            cut_pts_ms < 0.0) {
            return false;
        }
        return map_video_ms < cut_pts_ms - kPtsJitterMs &&
               presented_video_ms >= cut_pts_ms - kPtsJitterMs;
    }
    // Pure (04/08 round 2): the scene-cut signal handed to
    // DepthStabilizer::step() for a worker observation that CROSSED a
    // recorded cut boundary (same math as the gate: cross_shot_gate(prev
    // observation pts, this observation pts, c)). A boundary is
    // AUTHORITATIVE knowledge — the scout already confirmed the cut in the
    // decoded future — yet step()'s own OR only fires on the depth residual
    // or the histogram distance, and a shot/reverse-shot cut between similar
    // compositions (two faces under the same light) blinds BOTH: no snap,
    // and the EMA blends the old face's contours into the new shot for its
    // half-life ("the effect imprinted into the following shot"). Raising
    // the signal to the stabilizer's scene-cut threshold guarantees the
    // snap; a higher measured distance passes through untouched so the
    // scene= diagnostic stays honest.
    static float boundary_scene_signal(float histogram_distance,
                                       float scene_cut_threshold) {
        return histogram_distance >= scene_cut_threshold
                   ? histogram_distance : scene_cut_threshold;
    }

    // Freshness-checked delay to the next OBSERVED cut (<0 = none/stale).
    // First consumer: Synth3D::process()'s PRE-cut disparity ease-down — the
    // scout dates a cut at the first frame of the NEW shot (T1), so a
    // positive value means "the presented frame still belongs to the dying
    // shot": glide its depth to flat so it meets the post-snap ramp's flat
    // start and the cut carries no stereo jump (author rule 2026-08-03:
    // always anticipate the cut, never merely react at T1).
    // "No event" sentinel. MUST be far outside the hold window: small
    // negative delays are REAL data ("the cut landed ~1 frame ago, hold the
    // flat"), so -1.0 would collide with them and flatten depth forever.
    static constexpr double kLookaheadNone = -1.0e9;

    // Dead-reckoning of the pump-quantized advisory (2026-08-03: the
    // one-frame wavy tear at every cut). The player refreshes the advisory
    // from a 10 Hz QTimer (SyLC_3D_Player.py _playback_timer setInterval(100))
    // while process() presents at the source cadence (41.7 ms at 24 fps), so
    // the stored delay is FROZEN for up to ~100 ms = 2.4 presented frames.
    // At the cut frame T1 the frozen value still reads +0..100 ms ("cut
    // ahead"): the ease-down computes t2 = stale/300 and re-opens up to 55%
    // of the disparity budget, warping the NEW shot's first frame with the
    // OLD shot's map — and cb.temporal_fill (old-shot plate) only cuts at
    // <= 0, so the plate keeps filling the holes with old-shot pixels.
    // Subtracting the advisory's steady-clock age reconstructs the true
    // delay to within pump/position jitter (video time advances at wall
    // rate during playback; while paused the pump keeps re-stamping every
    // 100 ms, bounding the decay error to one pump period). The sentinel
    // never decays. Pure and static so the test binding pins the exact
    // production math (_synth3d_advisory_decay_test).
    static double advisory_decay(double value, int64_t age_ms) {
        if (value <= 0.5 * kLookaheadNone) return kLookaheadNone;
        return value - static_cast<double>(age_ms);
    }

    double lookahead_cut_in_ms() const {
        const int64_t set = lookahead_set_ms_.load(std::memory_order_acquire);
        if (set < 0) return kLookaheadNone;
        const int64_t now =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        const int64_t age = now - set;
        if (age > 500) return kLookaheadNone;
        const double raw = lookahead_cut_ms_.load(std::memory_order_acquire);
        const double corrected = advisory_decay(raw, age);
        // la_seen: the advisory's raw delay flipped positive -> negative, i.e.
        // one cut travelled through the pump. exchange() claims the transition
        // ONCE process-wide, so the count is invariant to the number of
        // reading surfaces AND to pump/present phase (>= one negative store is
        // guaranteed before the scout's 120 ms purge at 10 Hz; the sentinel is
        // excluded so a seek purge does not fake a cut). THIS is the counter
        // two runs may be compared on (~= cuts the scout reported).
        const double prev = la_prev_raw_.exchange(
            raw, std::memory_order_relaxed);
        if (prev > 0.0 && raw <= 0.0 && raw > 0.5 * kLookaheadNone)
            la_seen_.fetch_add(1, std::memory_order_relaxed);
        // la_stale: reads inside [cut landed .. next pump tick] while the RAW
        // value still said "cut ahead" -- the presents the FROZEN advisory
        // rendered with stale positive disparity. The per-read predicate uses
        // only raw and age (pump-written), so the flag cannot change it; but
        // the per-cut count is a small integer (0-3) cut from a lattice
        // (41.7 ms frame grid x 100 ms pump x per-run position/present clock
        // bias): simulated per-run means span 0.47..2.04 per cut with a
        // geometric floor ~0.5. It is therefore NOT comparable across runs --
        // normalize on la_seen, and treat a value below ~0.5/cut as a
        // measurement artifact (counter reset mid-window: service instance
        // changed; or missed reads), not as a flag effect.
        if (raw > 0.0 && corrected <= 0.0)
            la_stale_.fetch_add(1, std::memory_order_relaxed);
        static const bool decay_on = []() {
            char* env = nullptr;   // house idiom (MSVC-safe, cf. SYLC_FLOW_THREADS)
            size_t len = 0;
            bool on = true;
            if (_dupenv_s(&env, &len, "SYLC_LOOKAHEAD_DECAY") == 0 && env) {
                on = env[0] != '0';   // SYLC_LOOKAHEAD_DECAY=0 = frozen advisory
                free(env);
            }
            return on;
        }();
        return decay_on ? corrected : raw;
    }

private:
    enum class State : int { Init = 0, Running = 1, Error = 2, Stopping = 3 };

    SharedDepthService(std::wstring model_path, std::wstring ort_dir,
                       int width, int height);
    void start_worker();
    void worker_main();
    void set_error(const std::string& message);

    std::wstring model_path_;
    std::wstring ort_dir_;
    const int width_;
    const int height_;
    const uint64_t instance_id_;

    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<State> state_{State::Init};
    std::atomic<int> clients_{0};
    std::atomic<bool> reset_stabilizer_{true};

    mutable std::mutex input_mtx_;
    std::condition_variable input_cv_;
    std::vector<float> input_mailbox_;
    std::vector<float> input_full_frame_luma_;
    std::chrono::steady_clock::time_point input_capture_time_{};
    double input_video_time_ms_ = -1.0;
    int input_source_width_ = 0;
    int input_source_height_ = 0;
    bool input_fresh_ = false;
    uint64_t leader_id_ = 0;
    std::chrono::steady_clock::time_point leader_seen_{};

    mutable std::mutex output_mtx_;
    std::shared_ptr<const GeometryFrame> latest_;
    uint64_t output_sequence_ = 0;
    std::chrono::steady_clock::time_point output_time_{};
    // Media PTS of the source observation the published map was computed
    // from — the map's shot identity (04/08). -1 = untimed path.
    double output_video_ms_ = -1.0;

    // Cross-shot cut boundaries on the media clock (04/08): scout advisory
    // intake + worker snaps, purged on seek, pruned by media distance in
    // cross_shot(). Small and bounded (kMaxCutBoundaries).
    static constexpr size_t kMaxCutBoundaries = 16;
    mutable std::mutex cut_pts_mtx_;
    // mutable: cross_shot() prunes long-passed boundaries in place (cache
    // maintenance, observably const).
    mutable std::vector<double> cut_pts_;
    // Presents flattened by the cross-shot gate (diagnostic `gate=` in
    // status). Counted per SURFACE present, so like la_stale it scales with
    // the number of reading surfaces — normalize before comparing runs.
    mutable std::atomic<uint64_t> gate_frames_{0};

    // Codec motion hints ring (see note_motion_hints). Entries are looked
    // up by pts at worker prep; the ring bounds memory and staleness. Multiple
    // renderer clients forward the same decoder packet, so note_motion_hints
    // replaces an existing equal-PTS entry and these remain six SOURCE-frame
    // slots rather than six renderer submissions.
    struct MotionHints {
        double pts_ms = -1.0;
        double frame_ms = 41.7;
        int blocks_w = 0;
        int blocks_h = 0;
        int source_width = 0;
        int source_height = 0;
        std::vector<int16_t> mv_xy;
        std::vector<uint8_t> valid;
    };
    static constexpr size_t kHintRing = 6;
    mutable std::mutex hints_mtx_;
    std::array<MotionHints, kHintRing> hints_;
    size_t hint_write_ = 0;
    bool fetch_motion_hints(double pts_ms, MotionHints& out) const;

    mutable std::mutex meta_mtx_;
    std::string provider_ = "none";
    std::string error_;
    std::atomic<float> fps_{0.0f};
    std::atomic<float> motion_{0.0f};
    // Per-cycle worker stage costs, averaged over the same 2s window as fps_.
    std::atomic<float> flow_ms_{0.0f};   // flow estimation + reprojection + prep
    std::atomic<float> infer_ms_{0.0f};  // engine.infer()
    // Historical aggregate kept for dashboards. It covers temporal state
    // evolution plus contour/ownership construction and final packing.
    std::atomic<float> stab_ms_{0.0f};
    // Fine-grained work diagnostics. These are observational only: no
    // behaviour or scheduling decision may depend on them.
    std::atomic<float> obs_ms_{0.0f};     // confidence + boundary + refine
    std::atomic<float> guard_ms_{0.0f};   // boundary-motion conditioning
    std::atomic<float> reproj_ms_{0.0f};  // stabilizer.reproject()
    std::atomic<float> step_ms_{0.0f};    // temporal controls + step/retry
    std::atomic<float> realign_ms_{0.0f}; // realign_contours()
    std::atomic<float> owner_ms_{0.0f};   // build_geometry_map()
    std::atomic<float> owner_local_ms_{0.0f}; // edge/local ownership analysis
    std::atomic<float> owner_prop_ms_{0.0f};  // six-hop geodesic propagation
    std::atomic<float> pack_ms_{0.0f};    // RG16 + transport publication pack
    std::atomic<bool> gpu_owner_{false};  // worker publishes DirectCompute inputs
    std::atomic<bool> gpu_ownership_enabled_{false};
    std::atomic<float> source_dt_ms_{120.0f}; // source VIDEO-PTS interval
    std::atomic<float> update_dt_ms_{120.0f}; // COMPUTE map-update interval
    // --- wait instrumentation (diagnostic only; no behaviour depends on it) --
    // flow_ms/infer_ms/stab_ms time WORK. A worker cycle can also be spent
    // BLOCKED, and that time was invisible: measured stages summed to ~31 ms
    // inside a measured 51 ms cycle on the edge264 path. These four name the
    // gap. Averaged over the same 2 s window as the stage timers.
    std::atomic<float> inwait_ms_{0.0f};   // blocked waiting for a submission
    std::atomic<float> reswait_ms_{0.0f};  // blocked in InferPipe::wait_result
    std::atomic<float> joinwait_ms_{0.0f}; // blocked joining flow/future tasks
    std::atomic<float> cycle_ms_{0.0f};    // whole iteration, loop top to top
    // The four above close the cycle to the millisecond (work + inwait ==
    // cycle), which is precisely why they cannot say WHY inwait grows. These
    // two name the segment they leave dark, on the client side of the mailbox:
    //   pass_ms_   -- wall-clock period of the render pass that feeds us. If it
    //                 tracks cycle_ms, the renderer sets the cadence and the
    //                 worker is merely following it.
    //   taplat_ms_ -- capture-to-submit latency of one tap: how long a GPU
    //                 readback takes to reach the mailbox, in wall time. One
    //                 render pass is by design; more means copies are landing
    //                 late and each miss costs a whole source interval.
    // Accumulated from client threads, snapshotted by the worker's 2 s window.
    // Sums are integer MICROSECONDS, not doubles: std::atomic<double> has no
    // fetch_add before C++20 and this target is C++17. Microsecond resolution
    // is three orders finer than anything reported here.
    std::atomic<float> pass_ms_{0.0f};
    std::atomic<float> taplat_ms_{0.0f};
    std::atomic<int64_t> last_pass_ns_{0};
    std::atomic<uint64_t> pass_count_{0};
    std::atomic<uint64_t> pass_us_sum_{0};
    std::atomic<uint64_t> taplat_count_{0};
    std::atomic<uint64_t> taplat_us_sum_{0};
    // Tap accounting, SERVICE-side so it survives leader election: only one of
    // the N client surfaces ever feeds, and it is not necessarily the one whose
    // status the player polls. grants-submits == presented frames that were
    // granted a tap but produced no submission (the readback had not landed).
    std::atomic<uint64_t> grants_{0};
    std::atomic<uint64_t> submits_{0};
    // A content PTS may be presented repeatedly while paused. Such presents
    // may redraw but must never advance depth-temporal state.
    std::atomic<double> last_submitted_video_ms_{-1.0};
    std::atomic<uint64_t> duplicate_pts_{0};
    // Why a granted tap produced nothing: the ring held no pending copy at all
    // (empty) versus it held one the GPU had not finished (stalled). The two
    // point at completely different fixes, so they are counted apart.
    std::atomic<uint64_t> drain_empty_{0};
    std::atomic<uint64_t> drain_stalled_{0};
    // Plate-vs-cut accounting. plate_post_ counts presents where the plate was
    // LIVE while the advisory still remembered a cut within the last 500 ms —
    // i.e. after the -150 ms hold released it. plate_cut_in_ is the advisory
    // offset of the most recent such present: the exact distance from the cut
    // at which remembered background is allowed back on screen.
    std::atomic<uint64_t> plate_held_{0};
    std::atomic<uint64_t> plate_post_{0};
    std::atomic<float> plate_cut_in_{0.0f};
    std::atomic<bool> plate_requested_{false};
    std::atomic<bool> plate_armed_{false};
    std::atomic<bool> plate_live_{false};
    std::atomic<float> effective_alpha_{0.0f};
    std::atomic<float> scene_change_{0.0f};
    std::atomic<float> confidence_{1.0f};
    std::atomic<float> flow_{0.0f};
    std::atomic<float> stability_{0.0f};
    std::atomic<float> history_support_{1.0f};
    std::atomic<float> suggested_convergence_{0.5f};
    std::atomic<int> temporal_views_{1};
    // Stable encoded-letterbox observation, expressed in SOURCE pixels.
    // Zero/confidence 0 means no trustworthy horizontal matte was found.
    std::atomic<int> crop_top_{0};
    std::atomic<int> crop_bottom_{0};
    std::atomic<int> crop_source_width_{0};
    std::atomic<int> crop_source_height_{0};
    std::atomic<float> crop_confidence_{0.0f};
    // True after eight agreeing observations of either a matte or no matte.
    // Unlike crop_confidence, this can certify a transition back to 16:9.
    std::atomic<bool> crop_ready_{false};
    std::atomic<uint64_t> cuts_{0};
    std::atomic<uint64_t> cut_boundary_{0};
    std::atomic<uint64_t> cut_source_{0};
    std::atomic<uint64_t> cut_depth_{0};
    std::atomic<float> depth_residual_{0.0f};
    // Ambient residual baseline the adaptive depth-cut gate compares against
    // (depthbase= in the [2D3D] line): tune cut_baseline_gain with evidence.
    std::atomic<float> depth_baseline_{0.0f};
    std::atomic<int64_t> last_snap_ms_{-1};
    std::atomic<double> last_snap_video_ms_{-1.0};
    // Look-ahead advisory intake (see set_lookahead_advisory).
    std::atomic<double> lookahead_cut_ms_{kLookaheadNone};
    std::atomic<double> lookahead_storm_ms_{kLookaheadNone};
    std::atomic<int64_t> lookahead_set_ms_{-1};
    // Advisory reads that were stale-positive across a landed cut (see
    // lookahead_cut_in_ms; phase-sensitive, never compare across runs).
    // mutable: counted from the const accessor on the present path.
    mutable std::atomic<uint64_t> la_stale_{0};
    // Advisory raw-delay sign transitions = cuts that travelled through the
    // pump. Phase- and surface-count-invariant (claimed once via exchange);
    // the honest cross-run witness/denominator for la_stale.
    mutable std::atomic<double> la_prev_raw_{kLookaheadNone};
    mutable std::atomic<uint64_t> la_seen_{0};
};
