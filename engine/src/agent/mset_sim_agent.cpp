#include "agent/mset_sim_agent.h"

#include "encoding/input_encoder.h"
#include "util/exception.h"
#include "util/math.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>

// The production constructor -- the only member that references the concrete
// TensorRT-backed service (and thus pulls in CUDA / TensorRT) -- lives in
// mset_sim_agent_factory.cpp, so this translation unit, and the agent's unit
// tests that compile it, carry no GPU dependency.

namespace scribblez {

namespace {

// The dictionary reference the members that need it read before any
// constructor body could check it.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw util::Exception("mset-sim agent: a dictionary is required");
  return *dict;
}

// The runner params `params` describe, over the serialized leaf evaluator
// (null for terminal rollouts).
SimRunner::Params runner_params(const MsetSimAgent::Params& params, nn::PositionEvalService* leaf) {
  SimRunner::Params p = params.sim;
  p.horizon_plies = params.sim_horizon;
  p.leaf_service = leaf;
  return p;
}

}  // namespace

MsetSimAgent::MsetSimAgent(const Params& params, std::unique_ptr<nn::MoveSetEvalService> service,
                           std::unique_ptr<nn::PositionEvalService> leaf_service)
    : Agent(params.thread_id, params.name),
      shortlist_(params.shortlist),
      sim_top_k_(params.sim_top_k),
      rank_objective_(params.rank_objective),
      sim_objective_(params.sim_objective),
      seed_(params.seed),
      service_(std::move(service)),
      spec_(derive_input_spec(require_dict(params.dict), *service_, "mset-sim agent")),
      encoder_(spec_),
      leaf_service_(std::move(leaf_service)),
      leaf_(leaf_service_ ? std::make_unique<nn::SerializedEvalService<nn::PositionEvaluationSpec>>(
                              *leaf_service_)
                          : nullptr),
      runner_(*params.dict, runner_params(params, leaf_.get())),
      endgame_(params.thread_id, params.endgame) {
  validate(params);
  board_row_.resize(size_t(input_floats(spec_)));
}

void MsetSimAgent::validate(const Params& params) {
  if (params.shortlist < 0)
    throw util::CleanException("mset-sim agent: --shortlist must be >= 0 (0 = all moves)");
  if (params.sim_top_k < 1) throw util::CleanException("mset-sim agent: --sim-top-k must be >= 1");
  SimRunner::validate(params.sim);
  if (params.sim_horizon != 0 && params.sim_horizon < SimRunner::kMinHorizonPlies) {
    throw util::CleanException(
      "mset-sim agent: --sim-horizon must be 0 (terminal rollouts) or "
      ">= {}",
      SimRunner::kMinHorizonPlies);
  }
}

uint64_t MsetSimAgent::sim_seed(int ply) const {
  return util::splitmix64(seed_ ^ util::splitmix64(uint64_t(ply)));
}

void MsetSimAgent::begin_game(const BeginGameRequest& req) {
  encoder_ = GameStateEncoder(spec_, req.initial_scores);
  endgame_.begin_game();
  ply_ = 0;
}

void MsetSimAgent::observe_move(const Move& move) {
  encoder_.apply_move(move);
  endgame_.observe_move(move);
  ++ply_;
}

void MsetSimAgent::encode_board_row(const MoveRequest& req, float* dst) const {
  // The contingent input planes read the board's move-generation caches;
  // building them here (a no-op once valid) keeps them lexicon-accurate.
  encoder_.board().ensure_movegen_caches(*spec_.dict);
  // The encoder's active player is this agent's own seat: it has observed every
  // prior move, and this is its turn.
  const int me = encoder_.active_player();
  if (spec_.opp_leave_input) {
    encoder_.encode_input(me, req.my_rack, req.opp_rack, /*apply_flip=*/false, dst);
  } else {
    encoder_.encode_input(me, req.my_rack, /*apply_flip=*/false, dst);
  }
}

void MsetSimAgent::rank_candidates(const MoveRequest& req, const std::vector<Move>& candidates) {
  const int n = candidates.size();
  encode_board_row(req, board_row_.data());
  // The differential the moves resolve is read off the same mirrored encoder
  // that wrote the board row's score-diff feature, so a candidate's resultant
  // differential is exactly that feature plus the move's score -- the plain sum
  // the two representations were designed to share (input_encoder.h).
  const int me = encoder_.active_player();
  move_features_.encode(candidates.data(), n, encoder_.score(me) - encoder_.score(1 - me));
  wld_buf_.resize(size_t(n) * nn::WldOutput::kRowElems);
  score_diff_buf_.resize(size_t(n) * nn::ScoreDiffOutput::kRowElems);
  float* const head_out[] = {wld_buf_.data(), score_diff_buf_.data()};
  service_->evaluate({board_row_.data(), &move_features_}, head_out);

  rank_.resize(size_t(n));
  std::iota(rank_.begin(), rank_.end(), 0);
  std::stable_sort(rank_.begin(), rank_.end(),
                   [&](int a, int b) { return objective(a) > objective(b); });
}

float MsetSimAgent::objective(int i) const {
  return objective_value(wld_buf_.data() + size_t(i) * nn::WldOutput::kRowElems,
                         score_diff_buf_.data() + size_t(i) * nn::ScoreDiffOutput::kRowElems,
                         rank_objective_);
}

MoveDecision MsetSimAgent::make_move(const MoveRequest& req) {
  // The endgame belongs to the exact solver, which needs no candidates of ours.
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> candidates =
    equity_top_k(req, shortlist_ == 0 ? std::numeric_limits<int>::max() : shortlist_);
  // A bag-empty turn the solver declined: the value model is out of its
  // training regime and rollouts have no bag to draw the opponent's
  // replenishments from, so play the static-equity move -- which is what
  // equity_top_k already ranked first.
  if (req.bag_size == 0 || candidates.size() == 1) return candidates.front();

  rank_candidates(req, candidates);

  const int k = std::min(sim_top_k_, int(candidates.size()));
  sim_moves_.clear();
  for (int j = 0; j < k; ++j) sim_moves_.push_back(candidates[size_t(rank_[j])]);
  if (k == 1) return sim_moves_.front();

  const SimPosition pos = sim_position_from(req);

  const std::vector<SimObservation> observations = runner_.run(pos, sim_moves_, sim_seed(ply_));
  // Ties in the observations go to the earlier candidate -- the better model
  // rank, this agent's own ordering.
  return sim_moves_[size_t(best_observation_index(observations, sim_objective_))];
}

}  // namespace scribblez
