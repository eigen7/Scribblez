#include "agent/neural_agent.h"

#include "lexicon/hasty_equity.h"
#include "util/math.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>

// The production constructor -- the only member that references the concrete
// nn::NNEvaluationService (and thus pulls in CUDA / TensorRT) -- lives in
// neural_agent_factory.cpp, so this translation unit, and the agent's unit
// tests that compile it, carry no GPU dependency.

namespace scribblez {

NeuralAgent::NeuralAgent(const Params& params, std::unique_ptr<nn::EvalService> service,
                         int max_batch)
    : Agent(params.thread_id, params.name),
      top_k_(params.top_k),
      objective_(params.objective),
      temperature_(params.temperature),
      evaluator_(*params.dict, std::move(service), max_batch),
      endgame_(params.thread_id, params.endgame),
      rng_(params.seed) {
  init();
}

void NeuralAgent::init() {
  if (top_k_ < 0) throw std::runtime_error("neural agent: --top-k must be >= 0 (0 = all moves)");
  if (temperature_ < 0.0) throw std::runtime_error("neural agent: --temperature must be >= 0");
}

void NeuralAgent::begin_game() {
  evaluator_.begin_game();
  endgame_.begin_game();
}

void NeuralAgent::observe_move(const Move& move) {
  evaluator_.observe_move(move);
  endgame_.observe_move(move);
}

// --- make_move()'s per-turn pipeline ---------------------------------------
// make_move() (at the bottom of this file) drives three stages:
//   select_candidates()    -- prune the legal plays to the set worth scoring
//   evaluator_.evaluate()  -- encode each candidate's post-move position from
//                             the agent's POV and run the model on the batch
//   select_index()         -- pick one candidate from the model's scores
// The rest (candidate_equities, greedy_equity_index, encode_candidate) are
// shared helpers those stages call.

std::vector<double> NeuralAgent::candidate_equities(const MoveRequest& req,
                                                    const std::vector<Move>& plays) const {
  return HastyEquity::instance().equities(plays, req.board, req.bag_size, req.opp_rack,
                                          req.my_rack);
}

int NeuralAgent::greedy_equity_index(const MoveRequest& req, const std::vector<Move>& plays) const {
  const std::vector<double> equities = candidate_equities(req, plays);
  return static_cast<int>(std::max_element(equities.begin(), equities.end()) - equities.begin());
}

int NeuralAgent::select_candidates(const MoveRequest& req, const std::vector<Move>& plays) {
  const int n = static_cast<int>(plays.size());
  cand_idx_.resize(static_cast<size_t>(n));
  std::iota(cand_idx_.begin(), cand_idx_.end(), 0);

  // All-moves mode (top_k_ == 0) or fewer plays than the cap: evaluate every
  // play, without consulting static equity.
  if (top_k_ == 0 || n <= top_k_) return n;

  // Otherwise keep the top_k_ plays by HastyBot static equity.
  const std::vector<double> equities = candidate_equities(req, plays);
  std::partial_sort(cand_idx_.begin(), cand_idx_.begin() + top_k_, cand_idx_.end(),
                    [&](int a, int b) { return equities[a] > equities[b]; });
  cand_idx_.resize(static_cast<size_t>(top_k_));
  return top_k_;
}

void NeuralAgent::encode_candidate(const Move& mv, const Rack& my_rack, int my_seat,
                                   const Rack& opp_leave, float* dst) const {
  evaluator_.encode_candidate(mv, my_rack, my_seat, opp_leave, dst);
}

int NeuralAgent::select_index(int k) {
  const std::vector<nn::Eval>& evals = evaluator_.evals();
  if (temperature_ <= 0.0 || k == 1) {
    int best = 0;
    for (int j = 1; j < k; ++j)
      if (objective_value(evals[j], objective_) > objective_value(evals[best], objective_))
        best = j;
    return best;
  }

  if (static_cast<int>(obj_values_.size()) < k) obj_values_.resize(static_cast<size_t>(k));
  for (int j = 0; j < k; ++j) obj_values_[j] = objective_value(evals[j], objective_);
  return sampler_.sample(obj_values_, k, temperature_, rng_);
}

// TODO: use the WMP/shadow-play based play-generation that the streaming data generation code
// uses in order to generate just the top-K plays efficiently
MoveDecision NeuralAgent::make_move(const MoveRequest& req) {
  // The endgame belongs to the exact solver, which needs no move list of ours.
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> plays = generate_legal_plays(req);
  if (plays.empty()) return Move::pass();

  // A bag-empty position the solver declined: the value model is out of its
  // training regime and ranks these worse than static equity, so play the
  // greedy HastyBot equity move instead of consulting the model.
  if (req.bag_size == 0) {
    return plays[static_cast<size_t>(greedy_equity_index(req, plays))];
  }

  const int k = select_candidates(req, plays);
  evaluator_.evaluate(req, plays, cand_idx_, k);
  return plays[static_cast<size_t>(cand_idx_[select_index(k)])];
}

}  // namespace scribblez
