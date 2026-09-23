#pragma once

// The move proposal model's two TensorRT engines (docs/roadmap.md item 3),
// shared by every consumer in a run. The model runs as a per-turn `cache`
// graph and a per-evidence-iteration `step` graph (model_specs.h). This class
// owns both NeuralNet<Spec>s and runs them, each call under one mutex because
// the nets' host staging buffers are shared state.
//
// Per-position state lives in the caller's MoveProposalCache, so one loaded
// pair serves every game thread's agent, each through its own
// MoveProposalSession. The engines are driven directly rather than through
// TrtEvalService, whose row-uniform decode cannot return the static board/g or
// raw move_enc handoff tensors. The sharing therefore saves memory (one engine
// pair per run) but brings no cross-caller batching.
//
// Row bounds: the cache graph's `planes` output is 4 * kSlotsPerCell * 225
// floats per candidate, allocated at the row bound on device and in pinned
// host memory, so the cache spec defaults to 1024 rows rather than the
// move-set graph's 4096; a turn with more candidates pays one extra launch.
// The step graph emits no planes, since nothing reads a conditioned plane, so
// its rows are cheap. It keeps the 4096 bound because each extra step chunk
// is paid on every loop iteration, under the shared mutex, and re-runs the
// position-level fusion.

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

// One position's state between the cache and step graphs, filled by
// MoveProposalNets::run_cache: the raw per-candidate outputs for all M
// candidates (the nets' own host buffers hold only one chunk), and the static
// board/g handoff tensors.
struct MoveProposalCache {
  int num_moves = 0;
  std::vector<float> move_enc;    // (M, C) raw
  std::vector<float> wld;         // (M, 3) raw logits
  std::vector<float> score_diff;  // (M, 2) [mean, std]
  // (M, 4*kSlotsPerCell, 225) footprint probabilities in evidence-channel
  // layout (evidence_staging.h), read by scored index when a simmed
  // candidate's evidence token is staged.
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
    // Each graph's row ceiling per predict(); larger candidate sets are
    // chunked. Separate because the graphs' per-row costs differ by orders of
    // magnitude (see the file comment).
    int max_rows = nn::MoveProposalCacheSpec::kDefaultMaxRows;
    int step_max_rows = nn::MoveProposalStepSpec::kDefaultMaxRows;
    // FP32 by default: the fusion graph's masked_fill and 4D einsum have not
    // been validated at reduced precision, unlike the BF16-served value models
    // (docs/plans/fp16_safe_serving.md).
    nn::Precision precision = nn::Precision::kFP32;
    bool fast_build = false;
    std::string mount_root = "/workspace/mount";

    // Every field affects the built engines or their buffers, so create()
    // shares a loaded pair only between callers with equal Params.
    bool operator==(const Params&) const = default;
  };

  // A loaded pair for `params`; a call with equal params while an instance is
  // alive returns that instance. Throws unless both graphs come from one
  // exported model: same proposal_export_id, same trained_max_evidence, same
  // trunk width C.
  static std::shared_ptr<MoveProposalNets> create(const Params& params);

  MoveProposalNets(const MoveProposalNets&) = delete;
  MoveProposalNets& operator=(const MoveProposalNets&) = delete;

  int channels() const { return cache_net_.channels(); }  // trunk width C
  // The padded evidence width the step graph is built for.
  int max_evidence() const { return nn::kMaxEvidence; }
  // The widest evidence set the fusion stage was trained on. A consumer must
  // not condition on wider sets, which the model has never seen.
  int trained_max_evidence() const { return trained_max_evidence_; }
  int max_rows() const { return cache_net_.max_rows(); }
  int step_max_rows() const { return step_net_.max_rows(); }
  int spatial_planes() const { return cache_net_.spatial_planes(); }
  int scalar_floats() const { return cache_net_.scalar_floats(); }
  // The cache graph's board-row layout, for sizing the row passed to run_cache.
  bool opp_leave_input() const { return cache_net_.opp_leave_input(); }

  // Run the cache graph over one position and fill `cache`. `board_row` is
  // [spatial | scalar] floats as GameStateEncoder::encode_input writes them;
  // `moves` is the non-empty candidate set.
  void run_cache(const float* board_row, const move_set::MoveFeatureArrays& moves,
                 MoveProposalCache* cache);

  // Run the step graph over an encoded position and an evidence set, filling
  // `out` with raw outputs (wld as logits; gain already softplus'd by the
  // graph) for the caller to decode outside the lock. Throws on an un-encoded
  // cache or an evidence set wider than max_evidence().
  void run_step(const MoveProposalCache& cache, const EvidenceSet& evidence,
                MoveProposalPredictions* out);

 private:
  explicit MoveProposalNets(const Params& params);
  void load();

  Params params_;
  nn::NeuralNet<nn::MoveProposalCacheSpec> cache_net_;
  nn::NeuralNet<nn::MoveProposalStepSpec> step_net_;
  int trained_max_evidence_ = 0;
  // Held for a whole run_cache / run_step, staging and copy-out included.
  std::mutex mutex_;
};

}  // namespace agent
}  // namespace scribblez
