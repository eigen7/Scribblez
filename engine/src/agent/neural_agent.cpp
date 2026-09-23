#include "agent/neural_agent.h"

#include "sim/sim_runner.h"
#include "util/exception.h"
#include "util/math.h"

#include <limits>
#include <numeric>
#include <optional>

// from_spec lives in neural_agent_factory.cpp.

namespace scribblez {

NeuralAgent::NeuralAgent(const Params& params, std::shared_ptr<nn::PositionEvalService> service,
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
  if (top_k_ < 0) throw util::CleanException("neural agent: --top-k must be >= 0 (0 = all moves)");
  if (temperature_ < 0.0) throw util::CleanException("neural agent: --temperature must be >= 0");
}

void NeuralAgent::begin_game(const BeginGameRequest& req) {
  evaluator_.begin_game(req);
  endgame_.begin_game();
}

void NeuralAgent::observe_move(const Move& move) {
  evaluator_.observe_move(move);
  endgame_.observe_move(move);
}

void NeuralAgent::encode_candidate(const Move& mv, const Rack& my_rack, int my_seat,
                                   const Rack& opp_leave, float* dst) const {
  evaluator_.encode_candidate(mv, my_rack, my_seat, opp_leave, dst);
}

int NeuralAgent::select_index(int k) {
  if (temperature_ <= 0.0 || k == 1) {
    int best = 0;
    for (int j = 1; j < k; ++j) {
      if (objective(j) > objective(best)) best = j;
    }
    return best;
  }

  if (int(obj_values_.size()) < k) obj_values_.resize(size_t(k));
  for (int j = 0; j < k; ++j) obj_values_[j] = objective(j);
  return sampler_.sample(obj_values_, k, temperature_, rng_);
}

float NeuralAgent::objective(int i) const {
  return objective_value(evaluator_.wld_row(i), evaluator_.score_diff_row(i), objective_);
}

// TODO: with a positive top_k, generate only the top-K plays under a
// shadow-play bound (as hasty_best_move_wmp does for the argmax) instead of
// every legal play.
MoveDecision NeuralAgent::make_move(const MoveRequest& req) {
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> candidates =
    equity_top_k(req, top_k_ == 0 ? std::numeric_limits<int>::max() : top_k_);
  // On a bag-empty turn the solver declined, the model is outside its
  // training regime and ranks worse than static equity, so play the
  // static-equity favourite, which equity_top_k ranked first.
  if (req.bag_size == 0 || candidates.size() == 1) return candidates.front();

  const int k = candidates.size();
  cand_idx_.resize(size_t(k));
  std::iota(cand_idx_.begin(), cand_idx_.end(), 0);
  evaluator_.evaluate(req, candidates, cand_idx_, k);
  return candidates[size_t(select_index(k))];
}

}  // namespace scribblez
