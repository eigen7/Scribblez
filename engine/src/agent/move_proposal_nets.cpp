#include "agent/move_proposal_nets.h"

#include "agent/evidence_staging.h"
#include "encoding/input_encoder.h"
#include "nn/onnx_metadata.h"
#include "nn/shared_registry.h"
#include "util/exception.h"

#include <algorithm>
#include <cstring>

namespace scribblez {
namespace agent {

// model_specs.h restates the evidence layout to stay free of the agent/sim
// headers; it must match what stage_evidence writes. Checked here, the one
// file that includes both.
static_assert(nn::kEvidencePlanes == evidence::kNumEvidencePlanes);
static_assert(nn::kEvidenceScalars == evidence::kNumEvidenceScalars);
static_assert(kBoardCells == evidence::kEvidencePlaneCells);

namespace {

using nn::GainOutput;
using nn::MoveProposalCacheSpec;
using nn::MoveProposalStepSpec;
using nn::PlanesOutput;
using nn::ScoreDiffOutput;
using nn::WldOutput;

// Copy one chunk of a per-move input tensor from the MoveFeatureArrays field
// its descriptor names, as the move-set graph stages its candidates.
template <typename Tensor>
void stage_move_rows(nn::NeuralNet<MoveProposalCacheSpec>& net,
                     const move_set::MoveFeatureArrays& moves, int start, int rows) {
  using Elem = typename Tensor::Elem;
  const std::vector<Elem>& src = moves.*Tensor::kBatchSource;
  std::memcpy(net.host<Tensor>(), src.data() + size_t(start) * Tensor::kRowElems,
              sizeof(Elem) * size_t(rows) * Tensor::kRowElems);
}

template <typename... Ts>
void stage_move_tensors(nn::NeuralNet<MoveProposalCacheSpec>& net,
                        const move_set::MoveFeatureArrays& moves, int start, int rows,
                        nn::TensorList<Ts...>) {
  (stage_move_rows<Ts>(net, moves, start, rows), ...);
}

void copy_rows(const float* src, int rows, int width, float* dst) {
  std::memcpy(dst, src, sizeof(float) * size_t(rows) * width);
}

// The exporter's stamps tying a cache graph to its step graph: the
// proposal_export_id fingerprint of the exported model ("" if absent) and the
// evidence width the fusion stage trained at (0 if absent).
struct PairStamp {
  std::string export_id;
  int trained_max_evidence;
};

PairStamp read_pair_stamp(const std::string& onnx_path) {
  const nn::OnnxMetadata meta = nn::parse_onnx_metadata(nn::read_file_bytes(onnx_path));
  const auto it = meta.entries.find("proposal_export_id");
  return {it == meta.entries.end() ? std::string() : it->second,
          meta.int_entry("trained_max_evidence", 0)};
}

// One net's params from the shared Params. A function, because NeuralNet is
// neither copyable nor movable and must be constructed in the init list.
template <typename Spec>
nn::NeuralNetParams<Spec> net_params_from(const std::string& onnx_path, int max_rows,
                                          const MoveProposalNets::Params& p) {
  nn::NeuralNetParams<Spec> np;
  np.onnx_path = onnx_path;
  np.cuda_device_id = p.cuda_device_id;
  np.max_rows = max_rows;
  np.precision = p.precision;
  np.fast_build = p.fast_build;
  np.mount_root = p.mount_root;
  return np;
}

}  // namespace

MoveProposalNets::MoveProposalNets(const Params& params)
    : params_(params),
      cache_net_(
        net_params_from<MoveProposalCacheSpec>(params.cache_onnx_path, params.max_rows, params)),
      step_net_(net_params_from<MoveProposalStepSpec>(params.step_onnx_path, params.step_max_rows,
                                                      params)) {}

std::shared_ptr<MoveProposalNets> MoveProposalNets::create(const Params& params) {
  static nn::SharedRegistry<Params, MoveProposalNets> registry;
  return registry.get_or_create(params, [&] {
    std::shared_ptr<MoveProposalNets> nets(new MoveProposalNets(params));
    nets->load();
    return nets;
  });
}

void MoveProposalNets::load() {
  cache_net_.load();
  step_net_.load();

  // Check that the pair came from one exported model. Precision agrees by
  // construction, and the step spec's load-time width check pins the evidence
  // width; a mismatched pair would otherwise differ only in its numbers.
  if (cache_net_.channels() != step_net_.channels()) {
    throw util::CleanException(
      "move proposal cache/step graphs disagree on trunk channels: cache {} vs step {}",
      cache_net_.channels(), step_net_.channels());
  }
  const PairStamp cache = read_pair_stamp(params_.cache_onnx_path);
  const PairStamp step = read_pair_stamp(params_.step_onnx_path);
  if (cache.export_id.empty() || step.export_id.empty()) {
    throw util::CleanException(
      "move proposal graphs carry no proposal_export_id; re-export the pair with "
      "proposal_export.py (cache '{}', step '{}')",
      params_.cache_onnx_path, params_.step_onnx_path);
  }
  if (cache.export_id != step.export_id) {
    throw util::CleanException(
      "move proposal cache/step graphs are from different models (proposal_export_id {} vs {}); "
      "export both from one checkpoint",
      cache.export_id, step.export_id);
  }
  // Refuse a missing stamp rather than assume the padded width.
  if (cache.trained_max_evidence != step.trained_max_evidence) {
    throw util::CleanException(
      "move proposal cache/step graphs disagree on trained_max_evidence ({} vs {})",
      cache.trained_max_evidence, step.trained_max_evidence);
  }
  if (cache.trained_max_evidence < 1 || cache.trained_max_evidence > nn::kMaxEvidence) {
    throw util::CleanException(
      "move proposal graphs carry no usable trained_max_evidence ({}; the step graph pads to "
      "{}); re-export the pair with proposal_export.py",
      cache.trained_max_evidence, nn::kMaxEvidence);
  }
  trained_max_evidence_ = cache.trained_max_evidence;
}

void MoveProposalNets::run_cache(const float* board_row, const move_set::MoveFeatureArrays& moves,
                                 MoveProposalCache* cache) {
  const int m = moves.count;
  if (m < 1) throw util::CleanException("MoveProposalNets::run_cache: empty candidate set");
  std::lock_guard<std::mutex> lock(mutex_);
  const int c = cache_net_.channels();
  const int max_rows = cache_net_.max_rows();

  cache->num_moves = m;
  cache->move_enc.resize(size_t(m) * c);
  cache->wld.resize(size_t(m) * WldOutput::kRowElems);
  cache->score_diff.resize(size_t(m) * ScoreDiffOutput::kRowElems);
  cache->planes.resize(size_t(m) * PlanesOutput::kRowElems);
  cache->board.resize(size_t(kBoardCells) * c);
  cache->g.resize(size_t(3) * c);

  // The board inputs are staged once; the candidates ride the dynamic axis
  // and are re-staged per chunk.
  const size_t spatial_floats = size_t(cache_net_.spatial_planes()) * kBoardCells;
  const size_t scalar_floats = cache_net_.scalar_floats();
  std::memcpy(cache_net_.host<nn::SpatialInput>(), board_row, sizeof(float) * spatial_floats);
  std::memcpy(cache_net_.host<nn::ScalarInput>(), board_row + spatial_floats,
              sizeof(float) * scalar_floats);

  for (int start = 0; start < m; start += max_rows) {
    const int chunk = std::min(max_rows, m - start);
    stage_move_tensors(cache_net_, moves, start, chunk, MoveProposalCacheSpec::MoveInputs{});
    cache_net_.predict(chunk);

    // board/g are position-level and identical for every chunk.
    if (start == 0) {
      copy_rows(cache_net_.host<nn::BoardHandoff>(), 1, kBoardCells * c, cache->board.data());
      copy_rows(cache_net_.host<nn::GHandoff>(), 1, 3 * c, cache->g.data());
    }
    copy_rows(cache_net_.host<nn::MoveEncHandoff>(), chunk, c,
              cache->move_enc.data() + size_t(start) * c);
    copy_rows(cache_net_.host<WldOutput>(), chunk, WldOutput::kRowElems,
              cache->wld.data() + size_t(start) * WldOutput::kRowElems);
    copy_rows(cache_net_.host<ScoreDiffOutput>(), chunk, ScoreDiffOutput::kRowElems,
              cache->score_diff.data() + size_t(start) * ScoreDiffOutput::kRowElems);
    copy_rows(cache_net_.host<PlanesOutput>(), chunk, PlanesOutput::kRowElems,
              cache->planes.data() + size_t(start) * PlanesOutput::kRowElems);
  }
}

void MoveProposalNets::run_step(const MoveProposalCache& cache, const EvidenceSet& evidence,
                                MoveProposalPredictions* out) {
  if (cache.num_moves < 1) {
    throw util::CleanException("MoveProposalNets::run_step over an un-encoded position");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const int m = cache.num_moves;
  const int c = cache_net_.channels();
  const int max_rows = step_net_.max_rows();

  const evidence::CachePredictions predictions{cache.move_enc.data(), cache.wld.data(),
                                               cache.score_diff.data(), cache.planes.data(), c};
  const evidence::EvidenceStagingOutputs staged{
    step_net_.host<nn::EvMoveEncInput>(), step_net_.host<nn::EvObsPlanesInput>(),
    step_net_.host<nn::EvObsScalarsInput>(), step_net_.host<nn::EvMaskInput>()};
  evidence::stage_evidence(evidence.moves, evidence.observations, evidence.scored_indices,
                           predictions, nn::kMaxEvidence, staged);

  // The evidence and the board/g handoff are position-level: staged once and
  // re-sent with each chunk. move_enc is re-staged per chunk.
  copy_rows(cache.board.data(), 1, kBoardCells * c, step_net_.host<nn::BoardHandoff>());
  copy_rows(cache.g.data(), 1, 3 * c, step_net_.host<nn::GHandoff>());

  out->num_moves = m;
  out->wld.resize(size_t(m) * WldOutput::kRowElems);
  out->score_diff.resize(size_t(m) * ScoreDiffOutput::kRowElems);
  out->gain.resize(size_t(m) * GainOutput::kRowElems);

  for (int start = 0; start < m; start += max_rows) {
    const int chunk = std::min(max_rows, m - start);
    copy_rows(cache.move_enc.data() + size_t(start) * c, chunk, c,
              step_net_.host<nn::MoveEncHandoff>());
    step_net_.predict(chunk);

    copy_rows(step_net_.host<WldOutput>(), chunk, WldOutput::kRowElems,
              out->wld.data() + size_t(start) * WldOutput::kRowElems);
    copy_rows(step_net_.host<ScoreDiffOutput>(), chunk, ScoreDiffOutput::kRowElems,
              out->score_diff.data() + size_t(start) * ScoreDiffOutput::kRowElems);
    copy_rows(step_net_.host<GainOutput>(), chunk, GainOutput::kRowElems,
              out->gain.data() + size_t(start) * GainOutput::kRowElems);
  }
}

}  // namespace agent
}  // namespace scribblez
