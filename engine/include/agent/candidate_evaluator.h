#pragma once

// The position-evaluation core shared by NeuralAgent and NeuralSimAgent: it
// mirrors the live game through a GameStateEncoder and batch-evaluates the
// post-move row of each candidate move. Keeping this in one class means every
// agent that ranks moves with the position evaluation model feeds it exactly
// the training encoding.
//
// derive_input_spec() is a free function because every served model needs it,
// including those of other model families (MsetSimAgent, UltimateBotAgent) and
// the rollout leaf evaluator (SimRunner).

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

// One candidate's ranking value under `objective`, read off its decoded
// scoring-head rows.
float objective_value(const float* wld_row, const float* score_diff_row, EvalObjective objective);

// "scorediff" or "winprob"; anything else throws util::CleanException naming
// `flag` as the offending option.
EvalObjective parse_eval_objective(const std::string& name, const std::string& flag);

// The InputEncodingSpec for the input arm a served model declares. Throws,
// naming `who`, if that spec's row widths differ from the widths the model
// accepts: the exporter's metadata and the exported graph then describe
// different rows, and neither can be trusted.
InputEncodingSpec derive_input_spec(const Dictionary& dict, const nn::ServedModelInputs& model,
                                    const std::string& who);

class CandidateEvaluator {
 public:
  // `service` is shared by every game thread's evaluator
  // (nn::PositionEvalService::create() in production, a scripted stub in
  // tests); the constructor loads no model. `max_batch` bounds the rows of one
  // service call.
  CandidateEvaluator(const Dictionary& dict, std::shared_ptr<nn::PositionEvalService> service,
                     int max_batch);

  // The owning agent forwards its begin_game() / observe_move() here. The
  // encoder's features depend on the whole move history, which make_move()
  // alone does not see.
  void begin_game(const BeginGameRequest& req);
  void observe_move(const Move& move);

  // Seat to move in the mirrored game -- the owning agent's own seat when it
  // is deciding a turn.
  int active_player() const { return encoder_.active_player(); }

  // For an agent that reuses the model elsewhere, e.g. as its rollout leaf
  // evaluator.
  nn::PositionEvalService& service() { return *service_; }

  // Evaluate the post-move positions of candidates[idx[0..k)] from the
  // mover's POV. The decoded head rows land at wld_row(i) / score_diff_row(i)
  // for i in [0, k), in the same order.
  void evaluate(const MoveRequest& req, const std::vector<Move>& candidates,
                const std::vector<int>& idx, int k);
  const float* wld_row(int i) const { return wld_buf_.data() + i * nn::WldOutput::kRowElems; }
  const float* score_diff_row(int i) const {
    return score_diff_buf_.data() + i * nn::ScoreDiffOutput::kRowElems;
  }

  // The post-move input row for candidate `mv`, encoded exactly as evaluate()
  // encodes it. Public so a test can check it against an independent replay.
  // `opp_leave` is ignored unless the model's input layout carries the
  // opponent-leave block.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, const Rack& opp_leave,
                        float* dst) const;

 private:
  int max_batch_;
  std::shared_ptr<nn::PositionEvalService> service_;
  InputEncodingSpec spec_;
  GameStateEncoder encoder_;

  // Reused across turns to avoid per-move allocation.
  std::vector<float> input_buf_;
  std::vector<float> wld_buf_;
  std::vector<float> score_diff_buf_;
};

}  // namespace scribblez
