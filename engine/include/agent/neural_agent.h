#pragma once

#include "agent/agent.h"
#include "encoding/game_state_encoder.h"
#include "nn/eval_service.h"
#include "nn/neural_net.h"
#include "util/math.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

// Position evaluation model agent. On its turn it picks a candidate set of
// legal plays, encodes each resulting post-move position from its own POV,
// batch-evaluates them with the model, and selects among them by the configured
// objective -- greedily at temperature 0, else by a
// softmax(objective / temperature) sample.
//
// top_k == 0 evaluates every legal play, keeping the move distribution
// independent of HastyBot and so yielding the most diverse self-play data, at
// the price of putting every play through the GPU. A positive top_k keeps the
// best plays by HastyBot static equity instead -- cheaper, and a safety valve
// against the blank explosion of positions with thousands of plays.
//
// In the endgame the agent bypasses the model and plays the greedy HastyBot
// move: the value model never trains on bag-empty positions and evaluates them
// poorly, whereas static equity is the strong heuristic there.
//
// The agent's GameStateEncoder mirrors the real game through begin_game() /
// observe_move(), since its placement-plane features depend on both players'
// most-recent moves, which make_move() alone cannot see.
class NeuralAgent : public Agent {
 public:
  // Which model head ranks the candidates: the ScoreDiff head's predicted mean
  // final differential, or P(win) + 0.5*P(draw) from the WLD head.
  enum class Objective { kScoreDiff, kWinProb };

  // `dict` is required and must outlive the agent; `seed` is read only when
  // temperature is positive.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    int top_k = 0;
    Objective objective = Objective::kWinProb;
    double temperature = 0.0;
    uint64_t seed = 0;
  };

  NeuralAgent(const Params& params, const nn::NeuralNetParams& net_params);

  // Takes an already-constructed evaluator (real or a scripted stub), loading
  // no model and touching no GPU. `max_batch` bounds one evaluate() call.
  NeuralAgent(const Params& params, std::unique_ptr<nn::EvalService> service, int max_batch = 256);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game() override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=neural [options]"` tokens, with --type and
  // --name already stripped. Requires --model=<path.onnx>. Throws
  // std::runtime_error on bad input.
  static std::unique_ptr<NeuralAgent> from_spec(const std::vector<std::string>& tokens,
                                                int thread_id, const std::string& name);

  static std::string options_help();

  // The post-move input for candidate `mv`, encoded exactly as make_move()
  // does. Public so the encoding the model actually sees can be checked
  // against an independent replay.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, float* dst) const;

 private:
  static std::unique_ptr<nn::EvalService> make_service(const nn::NeuralNetParams& net_params);

  // The arm the model's ONNX metadata_props declare, validated against its
  // input widths through input_encoder.h's registry (a disagreement throws).
  static InputEncodingSpec derive_spec(const Dictionary& dict, const nn::EvalService& service);

  // Validate parameters and size the input scratch buffer.
  void init();

  std::vector<double> candidate_equities(const MoveRequest& req,
                                         const std::vector<Move>& plays) const;

  int greedy_equity_index(const MoveRequest& req, const std::vector<Move>& plays) const;

  // Fill cand_idx_ with this turn's candidates and return the count.
  int select_candidates(const MoveRequest& req, const std::vector<Move>& plays);

  float objective_value(const nn::Eval& e) const;

  // Fill eval_buf_[0..k), chunked to max_batch_ rows per evaluate() call.
  void evaluate_candidates(const MoveRequest& req, const std::vector<Move>& plays, int k);

  // Index into the first `k` candidates to play.
  int select_index(int k);

  int top_k_;
  Objective objective_;
  double temperature_;
  int max_batch_;
  std::unique_ptr<nn::EvalService> service_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;
  std::mt19937_64 rng_;

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<int> cand_idx_;
  std::vector<float> input_buf_;
  std::vector<nn::Eval> eval_buf_;
  std::vector<double> obj_values_;
  util::SoftmaxSampler sampler_;
};

}  // namespace scribblez
