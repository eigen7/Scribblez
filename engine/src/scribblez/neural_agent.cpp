#include "scribblez/neural_agent.h"

#include "scribblez/glyph.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/sampling.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

// The production constructor and make_service() -- the only members that
// reference the concrete nn::NNEvaluationService (and thus pull in CUDA /
// TensorRT) -- live in neural_agent_factory.cpp, so this translation unit, and
// the agent's unit tests that compile it, carry no GPU dependency.

namespace scribblez {

using binlog::kInputFloats;

NeuralAgent::NeuralAgent(int thread_id, const std::string& name,
                         std::unique_ptr<nn::EvalService> service, int top_k, Objective objective,
                         double temperature, uint64_t seed, int max_batch)
    : Agent(thread_id, name),
      top_k_(top_k),
      objective_(objective),
      temperature_(temperature),
      max_batch_(max_batch),
      service_(std::move(service)),
      rng_(seed) {
  init();
}

void NeuralAgent::init() {
  if (top_k_ < 0) throw std::runtime_error("neural agent: --top-k must be >= 0 (0 = all moves)");
  if (max_batch_ < 1) throw std::runtime_error("neural agent: max batch must be >= 1");
  if (temperature_ < 0.0) throw std::runtime_error("neural agent: --temperature must be >= 0");
  input_buf_.resize(static_cast<size_t>(max_batch_) * kInputFloats);
}

void NeuralAgent::begin_game() { encoder_ = binlog::GameStateEncoder(); }

void NeuralAgent::observe_move(const Move& move) { encoder_.apply_move(move); }

// --- make_move()'s per-turn pipeline ---------------------------------------
// make_move() (at the bottom of this file) drives three stages, each a method
// below:
//   select_candidates()   -- prune the legal plays to the set worth scoring
//   evaluate_candidates() -- encode each candidate's post-move position from the
//                            agent's POV and run the model on the batch
//   select_index()        -- pick one candidate from the model's scores
// The rest (objective_value, candidate_equities, greedy_equity_index,
// encode_candidate) are shared helpers those stages call.

float NeuralAgent::objective_value(const nn::Eval& e) const {
  return objective_ == Objective::kScoreDiff ? e.score_diff_mean : e.win_prob;
}

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
                                   float* dst) const {
  Rack leave = my_rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());

  binlog::GameStateEncoder post = encoder_;
  post.apply_move(mv);
  post.encode_input(my_seat, leave, /*apply_flip=*/false, dst);
}

void NeuralAgent::evaluate_candidates(const MoveRequest& req, const std::vector<Move>& plays,
                                      int k) {
  if (static_cast<int>(eval_buf_.size()) < k) eval_buf_.resize(static_cast<size_t>(k));

  // The encoder's active player is this agent's seat (it has observed every
  // prior move). Each candidate is scored from a post-move copy of the encoder,
  // in chunks no larger than the model's batch limit.
  const int my_seat = encoder_.active_player();
  int done = 0;
  while (done < k) {
    const int chunk = std::min(max_batch_, k - done);
    for (int j = 0; j < chunk; ++j) {
      const Move& mv = plays[static_cast<size_t>(cand_idx_[done + j])];
      encode_candidate(mv, req.my_rack, my_seat,
                       input_buf_.data() + static_cast<size_t>(j) * kInputFloats);
    }
    service_->evaluate(input_buf_.data(), chunk, eval_buf_.data() + done);
    done += chunk;
  }
}

int NeuralAgent::select_index(int k) {
  if (temperature_ <= 0.0 || k == 1) {
    int best = 0;
    for (int j = 1; j < k; ++j)
      if (objective_value(eval_buf_[j]) > objective_value(eval_buf_[best])) best = j;
    return best;
  }

  if (static_cast<int>(obj_values_.size()) < k) obj_values_.resize(static_cast<size_t>(k));
  for (int j = 0; j < k; ++j) obj_values_[j] = objective_value(eval_buf_[j]);
  return softmax_sample(obj_values_, k, temperature_, rng_, weights_);
}

Move NeuralAgent::make_move(const MoveRequest& req) {
  const std::vector<Move> plays = generate_legal_plays(req);
  if (plays.empty()) return Move::pass();

  // Endgame (empty bag): the value model is out of its training regime and
  // ranks these positions worse than static equity, so play the greedy HastyBot
  // equity move instead of consulting the model.
  if (req.bag_size == 0) {
    return plays[static_cast<size_t>(greedy_equity_index(req, plays))];
  }

  const int k = select_candidates(req, plays);
  evaluate_candidates(req, plays, k);
  return plays[static_cast<size_t>(cand_idx_[select_index(k)])];
}

}  // namespace scribblez
