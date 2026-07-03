#pragma once

#include "scribblez/agent.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/nn/eval_service.h"
#include "scribblez/nn/neural_net.h"
#include "util/math.h"

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

// Post-move value-model agent (M_post). On its turn it picks a candidate set of
// legal plays, applies each to a copy of its tracked GameStateEncoder, encodes
// the resulting post-move position from its own POV, batch-evaluates them with
// the model, and selects among them by the configured objective.
//
// The candidate set is controlled by top_k:
//   top_k == 0 : evaluate EVERY legal play, with no reference to HastyBot static
//                equity -- the move distribution is independent of HastyBot,
//                which yields the most diverse self-play data (but every play
//                hits the GPU, so it is the slowest mode).
//   top_k  > 0 : rank the legal plays by HastyBot static equity and keep the
//                top_k. Cheaper, and a safety valve against the blank explosion
//                (positions with thousands of plays). When a position has fewer
//                than top_k plays, all of them are evaluated.
//
// Selection among the candidates is greedy (the highest-rated post-move state)
// when temperature is 0, or a softmax(objective / temperature) sample when
// positive (used to add exploration when generating self-play training data).
//
// In the endgame (the bag is empty), the agent bypasses the model entirely and
// plays the greedy HastyBot static-equity move. The value model never trains on
// bag-empty positions and evaluates them poorly, whereas static equity is the
// strong heuristic for these perfect-information endgames.
//
// The agent maintains a GameStateEncoder that mirrors the real game via the
// begin_game() / observe_move() hooks (the encoder's placement-plane features
// depend on both players' most-recent moves, which make_move() alone cannot
// see). On a position with no legal plays it passes, matching HastyBot.
class NeuralAgent : public Agent {
 public:
  // Which model head drives move selection among the candidates.
  //   kScoreDiff -- highest expected final score differential (the ScoreDiff
  //                 head's mean on the post-move position).
  //   kWinProb   -- highest P(win) + 0.5*P(draw) from the WLD head. The default
  //                 chosen by from_spec().
  enum class Objective { kScoreDiff, kWinProb };

  // Agent identity plus move-selection configuration, shared by both
  // constructors.
  //   thread_id, name -- the base Agent identity.
  //   top_k       -- candidate set: 0 evaluates every legal play; K > 0 keeps
  //                  the top-K plays by HastyBot static equity.
  //   objective   -- which model head ranks the candidates.
  //   temperature -- 0 == greedy argmax; > 0 == softmax sampling.
  //   seed        -- seeds the sampler (used only when temperature > 0).
  struct Params {
    int thread_id = 0;
    std::string name;
    int top_k = 0;
    Objective objective = Objective::kWinProb;
    double temperature = 0.0;
    uint64_t seed = 0;
  };

  // Production constructor: builds an NNEvaluationService from `net_params`,
  // loading the model at `net_params.onnx_path` into it.
  NeuralAgent(const Params& params, const nn::NeuralNetParams& net_params);

  // Test/injection constructor: takes an already-constructed evaluator (real or
  // a scripted stub). Performs no model loading and touches no GPU. `max_batch`
  // is the per-evaluate row limit the chunked evaluation respects.
  NeuralAgent(const Params& params, std::unique_ptr<nn::EvalService> service, int max_batch = 256);

  Move make_move(const MoveRequest& req) override;
  void begin_game() override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=neural [options]"` tokens (after the factory
  // strips --type and --name). Requires --model=<path.onnx>; optional --top-k=K
  // (0 = all legal plays), --batch-size=N, --precision={FP16,FP32},
  // --cuda-device=D, --objective={scorediff,winprob}, --temperature=T (0 =
  // greedy; > 0 = softmax sampling), --seed=N (sampler seed; defaults to the
  // SeedProducer). Throws std::runtime_error on bad input.
  static std::unique_ptr<NeuralAgent> from_spec(const std::vector<std::string>& tokens,
                                                int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

  // Encode the post-move position for candidate `mv` into `dst` (kInputFloats
  // floats), exactly as make_move() does it: it copies the agent's tracked
  // encoder, applies `mv`, and encodes from `my_seat`'s POV with the rack that
  // remains after playing `mv` (no diagonal flip). `dict` supplies the
  // lexicon-derived cross-check planes. Public so the encoding the model
  // actually sees can be checked against an independent replay.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, const Dictionary& dict,
                        float* dst) const;

 private:
  // Build an NNEvaluationService from `net_params` and load its model into it.
  static std::unique_ptr<nn::EvalService> make_service(const nn::NeuralNetParams& net_params);

  // Validate parameters and size the input scratch buffer. Shared by both ctors.
  void init();

  // HastyBot static equity for every play in `plays`, indexed parallel to it.
  // Shared by candidate selection and the endgame fallback.
  std::vector<double> candidate_equities(const MoveRequest& req,
                                         const std::vector<Move>& plays) const;

  // Index into `plays` of the highest static-equity play. Used in the
  // endgame (empty bag) to fall back to greedy HastyBot equity.
  int greedy_equity_index(const MoveRequest& req, const std::vector<Move>& plays) const;

  // Fill cand_idx_ with the candidate indices into `plays` for this turn and
  // return the count: all plays when top_k_ == 0 or fewer plays than top_k_,
  // else the top_k_ by HastyBot static equity.
  int select_candidates(const MoveRequest& req, const std::vector<Move>& plays);

  // The selection score for an eval under the configured objective.
  float objective_value(const nn::Eval& e) const;

  // Encode and evaluate the first `k` candidates in cand_idx_, filling
  // eval_buf_[0..k). Evaluation is chunked to max_batch_ rows per call.
  void evaluate_candidates(const MoveRequest& req, const std::vector<Move>& plays, int k);

  // Index into the first `k` candidates to play: argmax of the objective when
  // temperature_ is 0, otherwise a softmax(objective / temperature_) sample.
  int select_index(int k);

  int top_k_;  // 0 == evaluate all legal plays
  Objective objective_;
  double temperature_;
  int max_batch_;
  std::unique_ptr<nn::EvalService> service_;
  GameStateEncoder encoder_;
  std::mt19937_64 rng_;  // drives softmax sampling when temperature_ > 0

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<int> cand_idx_;       // candidate indices into legal_plays
  std::vector<float> input_buf_;    // max_batch_ rows x kInputFloats
  std::vector<nn::Eval> eval_buf_;  // per-candidate evals (grows as needed)
  std::vector<double> obj_values_;  // per-candidate objective values for sampling
  util::SoftmaxSampler sampler_;    // draws a candidate when temperature_ > 0
};

}  // namespace scribblez
