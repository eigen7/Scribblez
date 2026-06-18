#pragma once

#include "scribblez/agent.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/nn/eval_service.h"
#include "scribblez/nn/neural_net.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// An agent that combines HastyBot's move generator with the post-move value
// model M_post (roadmap Phase 3.2). On its turn it:
//   1. Ranks the legal plays by HastyBot static equity and keeps the top K.
//   2. Applies each candidate to a copy of its tracked GameStateEncoder and
//      encodes the resulting post-move position from its own POV.
//   3. Batch-evaluates the K positions with the model and plays the move whose
//      post-move state has the highest predicted win probability.
//
// The agent maintains a GameStateEncoder that mirrors the real game via the
// begin_game() / observe_move() hooks (the encoder's placement-plane features
// depend on both players' most-recent moves, which make_move() alone cannot
// see). On a position with no legal plays it passes, matching HastyBot.
class NeuralTopKAgent : public Agent {
 public:
  // Which model head drives move selection among the top-K candidates.
  //   kScoreDiff -- highest expected final score differential (the ScoreDiff
  //                 head's mean on the post-move position). The default.
  //   kWinProb   -- highest P(win) + 0.5*P(draw) from the WLD head.
  enum class Objective { kScoreDiff, kWinProb };

  // Production constructor: builds an NNEvaluationService from `net_params` and
  // loads `onnx_path` into it.
  NeuralTopKAgent(int thread_id, const std::string& name, const std::string& onnx_path, int top_k,
                  Objective objective, const nn::NeuralNetParams& net_params);

  // Test/injection constructor: takes an already-constructed evaluator (real or
  // a scripted stub). Performs no model loading and touches no GPU.
  NeuralTopKAgent(int thread_id, const std::string& name,
                  std::unique_ptr<nn::EvalService> service, int top_k, Objective objective);

  Move make_move(const MoveRequest& req) override;
  void begin_game(std::array<int, 2> initial_scores) override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=neural [options]"` tokens (after the factory
  // strips --type and --name). Requires --model=<path.onnx>; optional
  // --top-k=K, --batch-size=N, --precision={FP16,FP32}, --cuda-device=D.
  // Throws std::runtime_error on bad input. `name` is the resolved display name.
  static std::unique_ptr<NeuralTopKAgent> from_spec(const std::vector<std::string>& tokens,
                                                    int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

  // Encode the post-move position for candidate `mv` into `dst` (kInputFloats
  // floats), exactly as make_move() does it: it copies the agent's tracked
  // encoder, applies `mv`, and encodes from `my_seat`'s POV with the rack that
  // remains after playing `mv` (no diagonal flip). Public so the encoding the
  // model actually sees can be checked against an independent replay.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, float* dst) const;

 private:
  // Shared construction: size the scratch buffers for top_k_ candidates.
  void init_buffers();

  // Indices of the (up to) top_k_ legal plays by static equity, written into
  // top_idx_ in descending-equity order; returns the count selected.
  int select_top_k(const std::vector<double>& equities);

  // The selection score for an eval under the configured objective.
  float objective_value(const nn::Eval& e) const;

  int top_k_;
  Objective objective_;
  std::unique_ptr<nn::EvalService> service_;
  binlog::GameStateEncoder encoder_;

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<int> top_idx_;
  std::vector<float> input_buf_;       // top_k_ rows x kInputFloats
  std::vector<nn::Eval> eval_buf_;     // top_k_ evals
};

}  // namespace scribblez
