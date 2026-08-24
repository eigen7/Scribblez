#include "agent/sim_agent.h"

#include "lexicon/dictionary.h"
#include "util/exception.h"
#include "util/math.h"

#include <optional>

// from_spec and options_help -- the only members that can pull in the
// concrete TensorRT-backed leaf service (--leaf-model) -- live in
// sim_agent_factory.cpp, so this translation unit, and the agent's unit
// tests that compile it, carry no GPU dependency.

namespace scribblez {

namespace {

// Checked in the initializer list, where the SimRunner member dereferences it
// before any constructor body could look.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw util::Exception("sim agent: a dictionary is required");
  return *dict;
}

// The runner params `params` describe, over the leaf evaluator (null for
// terminal rollouts; EvalService serializes the sim threads' calls itself).
SimRunner::Params runner_params(const SimAgent::Params& params, nn::PositionEvalService* leaf) {
  SimRunner::Params p = params.sim;
  p.horizon_plies = params.sim_horizon;
  p.leaf_service = leaf;
  return p;
}

}  // namespace

SimAgent::SimAgent(const Params& params, std::unique_ptr<nn::PositionEvalService> leaf_service)
    : Agent(params.thread_id, params.name),
      top_k_(params.top_k),
      objective_(params.objective),
      seed_(params.seed),
      leaf_service_(std::move(leaf_service)),
      runner_(require_dict(params.dict), runner_params(params, leaf_service_.get())),
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
  // The endgame belongs to the exact solver, which needs no candidates of ours.
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> candidates = equity_top_k(req, top_k_);
  // Rollouts need a bag to draw the opponent's replenishments from, so a
  // bag-empty turn the solver declined falls back to the static-equity move --
  // which is what equity_top_k already ranked first.
  if (req.bag_size == 0 || candidates.size() == 1) return candidates.front();

  const SimPosition pos = sim_position_from(req);

  const std::vector<SimObservation> observations = runner_.run(pos, candidates, sim_seed(ply_));
  return candidates[size_t(best_observation_index(observations, objective_))];
}

}  // namespace scribblez
