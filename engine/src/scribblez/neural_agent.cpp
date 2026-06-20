#include "scribblez/neural_agent.h"

#include "scribblez/glyph.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/nn/nn_evaluation_service.h"
#include "scribblez/sampling.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace scribblez {

using binlog::kInputFloats;

NeuralAgent::NeuralAgent(int thread_id, const std::string& name, const std::string& onnx_path,
                         int top_k, Objective objective, const nn::NeuralNetParams& net_params,
                         double temperature, uint64_t seed)
    : Agent(thread_id, name),
      top_k_(top_k),
      objective_(objective),
      temperature_(temperature),
      max_batch_(net_params.max_batch_size),
      service_(make_service(net_params, onnx_path)),
      rng_(seed) {
  init();
}

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

std::unique_ptr<nn::EvalService> NeuralAgent::make_service(const nn::NeuralNetParams& net_params,
                                                           const std::string& onnx_path) {
  auto svc = std::make_unique<nn::NNEvaluationService>(net_params);
  svc->load(onnx_path);
  return svc;
}

void NeuralAgent::begin_game(std::array<int, 2> initial_scores) {
  encoder_ = binlog::GameStateEncoder(initial_scores);
}

void NeuralAgent::observe_move(const Move& move) { encoder_.apply_move(move); }

float NeuralAgent::objective_value(const nn::Eval& e) const {
  return objective_ == Objective::kScoreDiff ? e.score_diff_mean : e.win_prob;
}

int NeuralAgent::select_candidates(const MoveRequest& req) {
  const int n = static_cast<int>(req.legal_plays.size());
  cand_idx_.resize(static_cast<size_t>(n));
  std::iota(cand_idx_.begin(), cand_idx_.end(), 0);

  // All-moves mode (top_k_ == 0) or fewer plays than the cap: evaluate every
  // play, without consulting static equity.
  if (top_k_ == 0 || n <= top_k_) return n;

  // Otherwise keep the top_k_ plays by HastyBot static equity.
  const HastyEquity& eq = HastyEquity::instance();
  const std::vector<double> equities =
    eq.equities(req.legal_plays, req.board, req.bag_size, req.opp_rack, req.my_rack);
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

void NeuralAgent::evaluate_candidates(const MoveRequest& req, int k) {
  if (static_cast<int>(eval_buf_.size()) < k) eval_buf_.resize(static_cast<size_t>(k));

  // The encoder's active player is this agent's seat (it has observed every
  // prior move). Each candidate is scored from a post-move copy of the encoder,
  // in chunks no larger than the model's batch limit.
  const int my_seat = encoder_.active_player();
  int done = 0;
  while (done < k) {
    const int chunk = std::min(max_batch_, k - done);
    for (int j = 0; j < chunk; ++j) {
      const Move& mv = req.legal_plays[static_cast<size_t>(cand_idx_[done + j])];
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
  if (req.legal_plays.empty()) return Move::pass();
  const int k = select_candidates(req);
  evaluate_candidates(req, k);
  return req.legal_plays[static_cast<size_t>(cand_idx_[select_index(k)])];
}

}  // namespace scribblez
