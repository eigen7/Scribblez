#include "agent/sim_agent.h"

#include "lexicon/dictionary.h"
#include "util/exception.h"
#include "util/math.h"

#include <optional>

// from_spec and options_help live in sim_agent_factory.cpp.

namespace scribblez {

namespace {

// Checked in the initializer list, where the SimRunner member dereferences it
// before any constructor body could look.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw util::Exception("sim agent: a dictionary is required");
  return *dict;
}

}  // namespace

SimAgent::SimAgent(const Params& params, std::shared_ptr<nn::PositionEvalService> leaf_service)
    : Agent(params.thread_id, params.name),
      top_k_(params.top_k),
      objective_(params.objective),
      seed_(params.seed),
      leaf_service_(std::move(leaf_service)),
      runner_(require_dict(params.dict),
              make_runner_params(params.sim, params.sim_horizon, leaf_service_.get())),
      endgame_(params.thread_id, params.endgame) {
  if (top_k_ < 1) throw util::CleanException("sim agent: --top-k must be >= 1");
}

uint64_t SimAgent::sim_seed(int ply) const {
  return util::splitmix64(seed_ ^ util::splitmix64(uint64_t(ply)));
}

void SimAgent::begin_game(const BeginGameRequest& /*req*/) {
  endgame_.begin_game();
  ply_ = 0;
}

void SimAgent::observe_move(const Move& move) {
  endgame_.observe_move(move);
  ++ply_;
}

MoveDecision SimAgent::make_move(const MoveRequest& req) {
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> candidates = equity_top_k(req, top_k_);
  // Rollouts need a bag to draw from, so a bag-empty turn the solver declined
  // plays the static-equity favourite, which equity_top_k ranked first.
  if (req.bag_size == 0 || candidates.size() == 1) return candidates.front();

  const SimPosition pos = sim_position_from(req);

  const std::vector<SimObservation> observations = runner_.run(pos, candidates, sim_seed(ply_));
  return candidates[size_t(best_observation_index(observations, objective_))];
}

}  // namespace scribblez
