#pragma once

// The position-evaluation core shared by the agents that rank candidate moves
// with the position evaluation model (NeuralAgent, NeuralSimAgent). It owns
// the EvalService, derives the input-encoding spec the served model declares,
// mirrors the live game through a GameStateEncoder, and batch-evaluates the
// post-move rows of a turn's candidates -- so every model-driven agent feeds
// the model identical inputs and none can drift from the training encoding.

#include "encoding/game_state_encoder.h"
#include "nn/eval_service.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;
struct MoveRequest;  // agent.h

// Which model head orders candidates: the ScoreDiff head's predicted mean
// final differential, or P(win) + 0.5*P(draw) from the WLD head.
enum class EvalObjective { kScoreDiff, kWinProb };

float objective_value(const nn::Eval& e, EvalObjective objective);

// "scorediff" or "winprob"; anything else throws std::runtime_error naming
// `flag` as the offending option.
EvalObjective parse_eval_objective(const std::string& name, const std::string& flag);

class CandidateEvaluator {
 public:
  // Takes an already-constructed evaluator (real or a scripted stub), loading
  // no model and touching no GPU. `max_batch` bounds one evaluate() call.
  CandidateEvaluator(const Dictionary& dict, std::unique_ptr<nn::EvalService> service,
                     int max_batch);

  // The owning agent forwards its own begin_game() / observe_move() here, so
  // the mirrored encoder sees both seats' moves; its placement-plane features
  // depend on them, which make_move() alone cannot see.
  void begin_game();
  void observe_move(const Move& move);

  // Seat to move in the mirrored game -- the owning agent's own seat when it
  // is deciding a turn.
  int active_player() const { return encoder_.active_player(); }

  // Evaluate candidates[idx[0..k)]'s post-move rows from the mover's POV,
  // chunked to max_batch rows per service call; results land in evals()[0..k)
  // in the same order.
  void evaluate(const MoveRequest& req, const std::vector<Move>& candidates,
                const std::vector<int>& idx, int k);
  const std::vector<nn::Eval>& evals() const { return eval_buf_; }

  // The post-move input for candidate `mv`, encoded exactly as evaluate()
  // does. Public so the encoding the model actually sees can be checked
  // against an independent replay. `opp_leave` is ignored unless the model's
  // input layout carries the opponent-leave block.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, const Rack& opp_leave,
                        float* dst) const;

 private:
  // The arm the model's ONNX metadata_props declare, validated against its
  // input widths through input_encoder.h's registry (a disagreement throws).
  static InputEncodingSpec derive_spec(const Dictionary& dict, const nn::EvalService& service);

  int max_batch_;
  std::unique_ptr<nn::EvalService> service_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<float> input_buf_;
  std::vector<nn::Eval> eval_buf_;
};

}  // namespace scribblez
