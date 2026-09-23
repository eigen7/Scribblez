#include "agent/mset_sim_agent.h"

#include "encoding/input_encoder.h"
#include "util/exception.h"
#include "util/math.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>

// The production constructor and from_spec live in mset_sim_agent_factory.cpp.

namespace scribblez {

namespace {

// Checked in the initializer list, where members dereference the dictionary
// before any constructor body could check it.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw util::Exception("mset-sim agent: a dictionary is required");
  return *dict;
}

}  // namespace

MsetSimAgent::MsetSimAgent(const Params& params, std::unique_ptr<nn::MoveSetEvalService> service,
                           std::shared_ptr<nn::PositionEvalService> leaf_service)
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
      runner_(*params.dict,
              make_runner_params(params.sim, params.sim_horizon, leaf_service_.get())),
      endgame_(params.thread_id, params.endgame) {
  validate(params);
  board_row_.resize(size_t(input_floats(spec_)));
}

void MsetSimAgent::validate(const Params& params) {
  if (params.shortlist < 0)
    throw util::CleanException("mset-sim agent: --shortlist must be >= 0 (0 = all moves)");
  if (params.sim_top_k < 1) throw util::CleanException("mset-sim agent: --sim-top-k must be >= 1");
  SimRunner::validate(params.sim);
  // Only the horizon's range: whether a leaf model accompanies it is checked
  // by from_spec, which alone knows whether a path was given.
  SimRunner::validate_min_horizon("mset-sim agent", params.sim_horizon);
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
  // The cross-check input planes read the board's move-generation caches;
  // building them here is a no-op once they are valid.
  encoder_.board().ensure_movegen_caches(*spec_.dict);
  // The encoder has observed every prior move, so its active player is this
  // agent's seat.
  const int me = encoder_.active_player();
  if (spec_.opp_leave_input) {
    encoder_.encode_input(me, req.my_rack, req.opp_rack, dst);
  } else {
    encoder_.encode_input(me, req.my_rack, dst);
  }
}

void MsetSimAgent::rank_candidates(const MoveRequest& req, const std::vector<Move>& candidates) {
  const int n = candidates.size();
  encode_board_row(req, board_row_.data());
  // Take the score differential from the same encoder that wrote the board
  // row, so each candidate's resulting differential is exactly the row's
  // score-diff feature plus the move's score (input_encoder.h).
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
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> candidates =
    equity_top_k(req, shortlist_ == 0 ? std::numeric_limits<int>::max() : shortlist_);
  // On a bag-empty turn the solver declined, the model is outside its
  // training regime and rollouts have no bag to draw from, so play the
  // static-equity favourite, which equity_top_k ranked first.
  if (req.bag_size == 0 || candidates.size() == 1) return candidates.front();

  rank_candidates(req, candidates);

  const int k = std::min(sim_top_k_, int(candidates.size()));
  sim_moves_.clear();
  for (int j = 0; j < k; ++j) sim_moves_.push_back(candidates[size_t(rank_[j])]);
  if (k == 1) return sim_moves_.front();

  const SimPosition pos = sim_position_from(req);

  const std::vector<SimObservation> observations = runner_.run(pos, sim_moves_, sim_seed(ply_));
  // Ties go to the earlier candidate, the better model rank.
  return sim_moves_[size_t(best_observation_index(observations, sim_objective_))];
}

}  // namespace scribblez
