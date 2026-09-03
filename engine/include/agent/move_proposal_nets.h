#pragma once

// The move proposal model's two TensorRT engines (roadmap item 3), shared by
// every consumer of a run. The model runs incrementally as a per-turn `cache`
// graph and a per-evidence-iteration `step` graph (model_specs.h); this class
// owns the two NeuralNet<Spec>s and performs the two GPU operations -- each
// under one mutex, because the nets' host staging buffers are shared too:
//
//   run_cache(board_row, moves, cache)   -- the cache graph over one position,
//                                           retaining its full-M handoff tensors
//                                           and raw predictions in `cache`.
//   run_step(cache, evidence, out)       -- the step graph over that cache and
//                                           an evidence set, raw outputs to `out`.
//
// What is retained per position lives in the caller's MoveProposalCache, not
// here: one loaded pair serves many positions at once (the agents of every game
// thread, each holding its own MoveProposalSession), which is what create()
// exists for. The engines are driven DIRECTLY (net.predict() + net.host<T>()),
// not through TrtEvalService, whose row-uniform decode cannot serve the
// static board/g or raw move_enc handoff outputs -- so unlike the position
// family's create(), this sharing brings no cross-caller batching, only one
// engine pair per run instead of one per agent.
//
// Memory: the cache graph's `planes` output is (M, 4 * kSlotsPerCell, 225)
// floats per candidate, allocated at its row bound on device and pinned host,
// which is why the cache spec defaults max_rows to 1024 rather than the
// move-set graph's 4096 (one extra chunk launch per turn past 1024 candidates,
// paid once per turn). The step graph emits no planes at all -- nothing reads a
// conditioned plane -- so its per-row buffers are a few C-wide floats, and it
// keeps the 4096 bound: a step chunk is paid per evidence-loop ITERATION, under
// the shared mutex, and re-runs the position-level fusion each time.

#include "agent/move_proposal_service.h"
#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/trt_util.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace scribblez {
namespace agent {

// One position's retained state between the cache and step graphs: the raw
// per-candidate outputs the step graph's evidence staging gathers from (full
// M, held across chunk boundaries -- the nets' own host buffers hold only
// max_rows rows), and the static board/g handoff tensors. Filled by
// MoveProposalNets::run_cache; owned by the session that encoded the position.
struct MoveProposalCache {
  int num_moves = 0;
  std::vector<float> move_enc;    // (M, C) raw
  std::vector<float> wld;         // (M, 3) raw logits
  std::vector<float> score_diff;  // (M, 2) [mean, std]
  // (M, 4*kSlotsPerCell, 225) footprint probabilities, evidence-channel layout
  // (evidence_staging.h): the ONE retained copy, read only by scored index when
  // a simmed candidate's evidence token is staged.
  std::vector<float> planes;
  std::vector<float> board;  // (225, C) raw
  std::vector<float> g;      // (3C,) raw
};

class MoveProposalNets {
 public:
  struct Params {
    std::string cache_onnx_path;
    std::string step_onnx_path;
    int cuda_device_id = 0;
    // The candidate ceiling per predict() of each graph; candidate sets above
    // it are chunked. Separate bounds because the two graphs' per-row costs
    // differ by orders of magnitude (see the file comment).
    int max_rows = nn::MoveProposalCacheSpec::kDefaultMaxRows;
    int step_max_rows = nn::MoveProposalStepSpec::kDefaultMaxRows;
    // FP32: correctness and the parity contract, not FP16 speed -- the fusion
    // graph's masked_fill / 4D einsum are FP16 hazards gated separately
    // (docs/fp16_safe_serving.md). Callers may override, with that caveat.
    nn::Precision precision = nn::Precision::kFP32;
    bool fast_build = false;
    std::string mount_root = "/workspace/mount";

    // Every field determines the engines that get built or the buffers they
    // allocate, so equality over all of them decides whether two callers may
    // share one loaded pair (create()).
    bool operator==(const Params&) const = default;
  };

  // A loaded pair for `params`, shared: a second call with equal params returns
  // the same still-live instance, so the agents of every game thread of a run
  // drive one engine pair. Builds or loads both graphs' engines, then validates
  // the pair came from one model: they must share the proposal_export_id
  // fingerprint and agree on the trunk channel width C. Throws otherwise. The
  // instance lives as long as its shared_ptr holders.
  static std::shared_ptr<MoveProposalNets> create(const Params& params);

  MoveProposalNets(const MoveProposalNets&) = delete;
  MoveProposalNets& operator=(const MoveProposalNets&) = delete;

  // The trunk channel width C, the padded evidence width E the step graph is
  // specialized to, each graph's per-predict() row ceiling, and the cache
  // graph's board-row widths and input arm (for a caller sizing the encoder
  // row it passes to run_cache).
  int channels() const { return cache_net_.channels(); }
  int max_evidence() const { return nn::kMaxEvidence; }
  int max_rows() const { return cache_net_.max_rows(); }
  int step_max_rows() const { return step_net_.max_rows(); }
  int spatial_planes() const { return cache_net_.spatial_planes(); }
  int scalar_floats() const { return cache_net_.scalar_floats(); }
  bool opp_leave_input() const { return cache_net_.opp_leave_input(); }

  // The cache graph over one position: `board_row` is [spatial | scalar] floats
  // as GameStateEncoder::encode_input writes them; `moves` its candidate set,
  // chunked to max_rows. Fills `cache` (sized here) with the position's
  // retained state. Throws on an empty candidate set.
  void run_cache(const float* board_row, const move_set::MoveFeatureArrays& moves,
                 MoveProposalCache* cache);

  // The step graph over an encoded position and an evidence set: stages the
  // evidence (agent/evidence_staging.h) from `cache`'s per-candidate outputs,
  // copies the cached handoff tensors into the step graph's inputs, runs it
  // chunked to max_rows, and fills `out` with the RAW per-candidate outputs --
  // wld as logits, gain as the graph's softplus'd value -- for the caller to
  // decode outside the lock. Throws on an un-encoded cache or an evidence set
  // wider than max_evidence().
  void run_step(const MoveProposalCache& cache, const EvidenceSet& evidence,
                MoveProposalPredictions* out);

 private:
  explicit MoveProposalNets(const Params& params);
  void load();

  Params params_;
  nn::NeuralNet<nn::MoveProposalCacheSpec> cache_net_;
  nn::NeuralNet<nn::MoveProposalStepSpec> step_net_;
  // Held across a whole run_cache / run_step: staging, every chunk's predict,
  // and the copy-out -- the nets' host buffers are the shared state.
  std::mutex mutex_;
};

}  // namespace agent
}  // namespace scribblez
