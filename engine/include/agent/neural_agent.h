#pragma once

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/endgame_turn_policy.h"
#include "endgame/endgame_solver.h"
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
// legal plays, hands them to its CandidateEvaluator (which encodes each
// resulting post-move position from the agent's POV and batch-evaluates them
// with the model), and selects among them by the configured objective --
// greedily at temperature 0, else by a softmax(objective / temperature)
// sample.
//
// top_k == 0 evaluates every legal play, keeping the move distribution
// independent of HastyBot and so yielding the most diverse self-play data, at
// the price of putting every play through the GPU. A positive top_k keeps the
// best plays by HastyBot static equity instead -- cheaper, and a safety valve
// against the blank explosion of positions with thousands of plays.
//
// In the endgame the agent bypasses the model entirely -- the value model never
// trains on bag-empty positions and evaluates them poorly -- and hands the turn
// to an EndgameTurnPolicy's exact solve, falling back to the greedy HastyBot
// move on the turns the solver declines (or when a zero budget disables it).
class NeuralAgent : public Agent {
 public:
  // `dict` is required and must outlive the agent; `seed` is read only when
  // temperature is positive. An `endgame` budget of 0 turns endgame solving off,
  // leaving the greedy HastyBot move to play the endgame out.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    int top_k = 0;
    EvalObjective objective = EvalObjective::kWinProb;
    double temperature = 0.0;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  using NetParams = nn::NeuralNetParams<nn::PositionEvaluationSpec>;

  // Takes a shared evaluation service (nn::PositionEvalService::create() in
  // production, a scripted stub in tests). Loads no model and touches no GPU;
  // `max_batch` bounds one evaluate() call.
  NeuralAgent(const Params& params, std::shared_ptr<nn::PositionEvalService> service,
              int max_batch = 256);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=neural [options]"` tokens, with --type and
  // --name already stripped. Requires --model=<path.onnx>. Throws
  // util::CleanException on bad input.
  static std::unique_ptr<NeuralAgent> from_spec(const std::vector<std::string>& tokens,
                                                int thread_id, const std::string& name);

  static std::string options_help();

  // The post-move input for candidate `mv`, encoded exactly as make_move()
  // does. Public so the encoding the model actually sees can be checked
  // against an independent replay. `opp_leave` is ignored unless the model's
  // input layout carries the opponent-leave block.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, const Rack& opp_leave,
                        float* dst) const;

 private:
  // Validate parameters.
  void init();

  std::vector<double> candidate_equities(const MoveRequest& req,
                                         const std::vector<Move>& plays) const;

  int greedy_equity_index(const MoveRequest& req, const std::vector<Move>& plays) const;

  // Fills cand_idx_ with this turn's candidates and returns their count.
  int select_candidates(const MoveRequest& req, const std::vector<Move>& plays);

  // Index, into the first `k` evaluated candidates, of the one to play.
  int select_index(int k);

  // The configured objective read off evaluated candidate `i`'s head rows.
  float objective(int i) const;

  int top_k_;
  EvalObjective objective_;
  double temperature_;
  CandidateEvaluator evaluator_;
  EndgameTurnPolicy endgame_;
  std::mt19937_64 rng_;

  // Scratch reused across turns to avoid per-move allocation.
  std::vector<int> cand_idx_;
  std::vector<double> obj_values_;
  util::SoftmaxSampler sampler_;
};

}  // namespace scribblez
