#pragma once

// The position-evaluation core shared by the agents that rank candidate moves
// with the position evaluation model (NeuralAgent, NeuralSimAgent). It owns
// the EvalService, derives the input-encoding spec the served model declares,
// mirrors the live game through a GameStateEncoder, and batch-evaluates the
// post-move rows of a turn's candidates -- so every model-driven agent feeds
// the model identical inputs and none can drift from the training encoding.
//
// Deriving that spec is the one part every model-driven agent needs, whichever
// model family it serves, so it lives here as a free function (MsetSimAgent
// calls it too).

#include "encoding/game_state_encoder.h"
#include "nn/eval_service.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;
struct MoveRequest;       // agent.h
struct BeginGameRequest;  // agent.h

// Which model head orders candidates: the ScoreDiff head's predicted mean
// final differential, or P(win) + 0.5*P(draw) from the WLD head.
enum class EvalObjective { kScoreDiff, kWinProb };

// The candidate-ranking value read off one candidate's decoded scoring-head
// rows: the score-diff mean, or expected game points under the WLD head
// (draws counting half).
float objective_value(const float* wld_row, const float* score_diff_row, EvalObjective objective);

// "scorediff" or "winprob"; anything else throws util::CleanException naming
// `flag` as the offending option.
EvalObjective parse_eval_objective(const std::string& name, const std::string& flag);

// The InputEncodingSpec implied by the input-encoding arm a served model
// declares, cross-checked against the input widths that model accepts -- a
// disagreement means the exporter's metadata and the exported graph describe
// different rows, and neither can be trusted to encode one, so it throws with
// `who` naming the caller.
InputEncodingSpec derive_input_spec(const Dictionary& dict, const nn::ServedModelInputs& model,
                                    const std::string& who);

class CandidateEvaluator {
 public:
  // Takes an already-constructed evaluator (real or a scripted stub), loading
  // no model and touching no GPU. `max_batch` bounds one evaluate() call.
  CandidateEvaluator(const Dictionary& dict, std::unique_ptr<nn::PositionEvalService> service,
                     int max_batch);

  // The owning agent forwards its own begin_game() / observe_move() here, so
  // the mirrored encoder sees both seats' moves; its placement-plane features
  // depend on them, which make_move() alone cannot see.
  void begin_game(const BeginGameRequest& req);
  void observe_move(const Move& move);

  // Seat to move in the mirrored game -- the owning agent's own seat when it
  // is deciding a turn.
  int active_player() const { return encoder_.active_player(); }

  // The owned service, for an agent that reuses its model elsewhere -- e.g.
  // as the value-truncated rollout leaf evaluator (SimRunner::Params).
  nn::PositionEvalService& service() { return *service_; }

  // Evaluate candidates[idx[0..k)]'s post-move rows from the mover's POV,
  // chunked to max_batch rows per service call; the decoded scoring-head rows
  // then land at wld_row()/score_diff_row() [0..k) in the same order.
  void evaluate(const MoveRequest& req, const std::vector<Move>& candidates,
                const std::vector<int>& idx, int k);
  const float* wld_row(int i) const { return wld_buf_.data() + i * nn::WldOutput::kRowElems; }
  const float* score_diff_row(int i) const {
    return score_diff_buf_.data() + i * nn::ScoreDiffOutput::kRowElems;
  }

  // The post-move input for candidate `mv`, encoded exactly as evaluate()
  // does. Public so the encoding the model actually sees can be checked
  // against an independent replay. `opp_leave` is ignored unless the model's
  // input layout carries the opponent-leave block.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, const Rack& opp_leave,
                        float* dst) const;

 private:
  int max_batch_;
  std::unique_ptr<nn::PositionEvalService> service_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<float> input_buf_;
  std::vector<float> wld_buf_;         // evaluate()'s decoded WLD rows
  std::vector<float> score_diff_buf_;  // evaluate()'s decoded score-diff rows
};

}  // namespace scribblez
