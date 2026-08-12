#include "agent/neural_sim_agent.h"

#include "util/math.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

// The production constructor -- the only member that references the concrete
// nn::NNEvaluationService (and thus pulls in CUDA / TensorRT) -- lives in
// neural_sim_agent_factory.cpp, so this translation unit, and the agent's unit
// tests that compile it, carry no GPU dependency.

namespace scribblez {

namespace {

// Salt separating the drop-decision stream from the rollout-seed stream, so
// the two deterministic per-ply draws never correlate.
constexpr uint64_t kDropStreamSalt = 0x9e3779b97f4a7c15ULL;

// The dictionary reference the CandidateEvaluator member needs before any
// constructor body could check it.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw std::runtime_error("neural-sim agent: a dictionary is required");
  return *dict;
}

}  // namespace

NeuralSimAgent::NeuralSimAgent(const Params& params, std::unique_ptr<nn::EvalService> service,
                               int max_batch)
    : Agent(params.thread_id, params.name),
      shortlist_(params.shortlist),
      sim_top_k_(params.sim_top_k),
      rank_objective_(params.rank_objective),
      sim_objective_(params.sim_objective),
      drop_best_prob_(params.drop_best_prob),
      seed_(params.seed),
      evaluator_(require_dict(params.dict), std::move(service), max_batch),
      runner_(*params.dict, params.sim),
      endgame_(params.thread_id, params.endgame) {
  validate(params);
}

void NeuralSimAgent::validate(const Params& params) {
  if (params.shortlist < 0)
    throw std::runtime_error("neural-sim agent: --shortlist must be >= 0 (0 = all moves)");
  if (params.sim_top_k < 1) throw std::runtime_error("neural-sim agent: --sim-top-k must be >= 1");
  if (params.drop_best_prob < 0.0 || params.drop_best_prob > 1.0)
    throw std::runtime_error("neural-sim agent: --drop-best-prob must be in [0, 1]");
  SimRunner::validate(params.sim);
}

uint64_t NeuralSimAgent::sim_seed(int ply) const {
  return util::splitmix64(seed_ ^ util::splitmix64(static_cast<uint64_t>(ply)));
}

bool NeuralSimAgent::drop_best(int ply) const {
  if (drop_best_prob_ <= 0.0) return false;
  const uint64_t draw =
    util::splitmix64(seed_ ^ kDropStreamSalt ^ util::splitmix64(static_cast<uint64_t>(ply)));
  // The top 53 bits as a uniform double in [0, 1).
  return static_cast<double>(draw >> 11) * 0x1.0p-53 < drop_best_prob_;
}

void NeuralSimAgent::begin_game(const BeginGameRequest& req) {
  evaluator_.begin_game(req);
  endgame_.begin_game();
  ply_ = 0;
}

void NeuralSimAgent::observe_move(const Move& move) {
  evaluator_.observe_move(move);
  endgame_.observe_move(move);
  ++ply_;
}

void NeuralSimAgent::rank_candidates(const MoveRequest& req, const std::vector<Move>& candidates) {
  const int n = static_cast<int>(candidates.size());
  rank_.resize(static_cast<size_t>(n));
  std::iota(rank_.begin(), rank_.end(), 0);
  evaluator_.evaluate(req, candidates, rank_, n);
  const std::vector<nn::Eval>& evals = evaluator_.evals();
  std::stable_sort(rank_.begin(), rank_.end(), [&](int a, int b) {
    return objective_value(evals[a], rank_objective_) > objective_value(evals[b], rank_objective_);
  });
}

MoveDecision NeuralSimAgent::make_move(const MoveRequest& req) {
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

  // The sim set: the model's top K, minus its #1 on a drop turn (the injected
  // recall miss), which shifts the window down one so K candidates still sim.
  const int n = static_cast<int>(candidates.size());
  const int first = drop_best(ply_) ? 1 : 0;
  const int k = std::min(sim_top_k_, n - first);
  sim_moves_.clear();
  for (int j = 0; j < k; ++j)
    sim_moves_.push_back(candidates[static_cast<size_t>(rank_[first + j])]);
  if (k == 1) return sim_moves_.front();

  const SimPosition pos = sim_position_from(req);

  const std::vector<SimObservation> observations = runner_.run(pos, sim_moves_, sim_seed(ply_));
  // Ties in the observations go to the earlier candidate -- the better model
  // rank, this agent's own ordering.
  return sim_moves_[static_cast<size_t>(best_observation_index(observations, sim_objective_))];
}

}  // namespace scribblez
