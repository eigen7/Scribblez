#include "scribblez/neural_top_k_agent.h"

#include "scribblez/glyph.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/nn/nn_evaluation_service.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace scribblez {

using binlog::kInputFloats;

NeuralTopKAgent::NeuralTopKAgent(int thread_id, const std::string& name,
                                 const std::string& onnx_path, int top_k, Objective objective,
                                 const nn::NeuralNetParams& net_params, double temperature,
                                 uint64_t seed)
    : Agent(thread_id, name),
      top_k_(top_k),
      objective_(objective),
      temperature_(temperature),
      rng_(seed) {
  init_buffers();
  auto svc = std::make_unique<nn::NNEvaluationService>(net_params);
  svc->load(onnx_path);
  service_ = std::move(svc);
}

NeuralTopKAgent::NeuralTopKAgent(int thread_id, const std::string& name,
                                 std::unique_ptr<nn::EvalService> service, int top_k,
                                 Objective objective, double temperature, uint64_t seed)
    : Agent(thread_id, name),
      top_k_(top_k),
      objective_(objective),
      temperature_(temperature),
      service_(std::move(service)),
      rng_(seed) {
  init_buffers();
}

void NeuralTopKAgent::init_buffers() {
  if (top_k_ < 1) throw std::runtime_error("neural agent: --top-k must be >= 1");
  if (temperature_ < 0.0) throw std::runtime_error("neural agent: --temperature must be >= 0");
  input_buf_.resize(static_cast<size_t>(top_k_) * kInputFloats);
  eval_buf_.resize(top_k_);
  weights_.resize(top_k_);
  top_idx_.reserve(top_k_);
}

void NeuralTopKAgent::begin_game(std::array<int, 2> initial_scores) {
  encoder_ = binlog::GameStateEncoder(initial_scores);
}

void NeuralTopKAgent::observe_move(const Move& move) { encoder_.apply_move(move); }

float NeuralTopKAgent::objective_value(const nn::Eval& e) const {
  return objective_ == Objective::kScoreDiff ? e.score_diff_mean : e.win_prob;
}

int NeuralTopKAgent::select_top_k(const std::vector<double>& equities) {
  const int n = static_cast<int>(equities.size());
  const int k = std::min(top_k_, n);
  top_idx_.resize(n);
  std::iota(top_idx_.begin(), top_idx_.end(), 0);
  std::partial_sort(top_idx_.begin(), top_idx_.begin() + k, top_idx_.end(),
                    [&](int a, int b) { return equities[a] > equities[b]; });
  top_idx_.resize(k);
  return k;
}

void NeuralTopKAgent::encode_candidate(const Move& mv, const Rack& my_rack, int my_seat,
                                       float* dst) const {
  Rack leave = my_rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());

  binlog::GameStateEncoder post = encoder_;
  post.apply_move(mv);
  post.encode_input(my_seat, leave, /*apply_flip=*/false, dst);
}

int NeuralTopKAgent::select_index(int k) {
  if (temperature_ <= 0.0 || k == 1) {
    int best = 0;
    for (int j = 1; j < k; ++j)
      if (objective_value(eval_buf_[j]) > objective_value(eval_buf_[best])) best = j;
    return best;
  }

  // Softmax sample, shifting by the max for numerical stability.
  double max_v = objective_value(eval_buf_[0]);
  for (int j = 1; j < k; ++j) max_v = std::max<double>(max_v, objective_value(eval_buf_[j]));
  double sum = 0.0;
  for (int j = 0; j < k; ++j) {
    weights_[j] = std::exp((objective_value(eval_buf_[j]) - max_v) / temperature_);
    sum += weights_[j];
  }
  double r = std::uniform_real_distribution<double>(0.0, sum)(rng_);
  double acc = 0.0;
  for (int j = 0; j < k; ++j) {
    acc += weights_[j];
    if (r <= acc) return j;
  }
  return k - 1;
}

Move NeuralTopKAgent::make_move(const MoveRequest& req) {
  if (req.legal_plays.empty()) return Move::pass();

  const HastyEquity& eq = HastyEquity::instance();
  const std::vector<double> equities =
    eq.equities(req.legal_plays, req.board, req.bag_size, req.opp_rack, req.my_rack);
  const int k = select_top_k(equities);

  // The encoder's active player is this agent's seat (it has observed every
  // prior move). Each candidate is scored from a post-move copy of the encoder.
  const int my_seat = encoder_.active_player();
  for (int j = 0; j < k; ++j) {
    const Move& mv = req.legal_plays[top_idx_[j]];
    encode_candidate(mv, req.my_rack, my_seat,
                     input_buf_.data() + static_cast<size_t>(j) * kInputFloats);
  }

  service_->evaluate(input_buf_.data(), k, eval_buf_.data());
  return req.legal_plays[top_idx_[select_index(k)]];
}

}  // namespace scribblez
