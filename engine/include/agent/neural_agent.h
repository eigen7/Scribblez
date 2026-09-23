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

// The position evaluation model as a player (--type=neural): evaluates the
// post-move position of each candidate play and picks by the configured
// objective, greedily at temperature 0, else by sampling
// softmax(objective / temperature). Its candidates are plays only: it never
// exchanges, and passes only when it has no play.
//
// top_k == 0 evaluates every legal play, which keeps the move distribution
// independent of HastyBot and the self-play data most diverse, at the cost of
// putting every play through the GPU. A positive top_k keeps the best plays by
// static equity: cheaper, and a guard against blank-heavy racks with
// thousands of plays.
//
// Bag-empty turns bypass the model, which never trains on them: they go to the
// endgame solver, or to the static-equity move when the solver declines.
class NeuralAgent : public Agent {
 public:
  // `dict` is required and must outlive the agent; `seed` is read only when
  // temperature is positive. An `endgame` budget of 0 turns solving off.
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

  // See CandidateEvaluator's constructor for `service` and `max_batch`.
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

  // Forwards to CandidateEvaluator::encode_candidate, for tests.
  void encode_candidate(const Move& mv, const Rack& my_rack, int my_seat, const Rack& opp_leave,
                        float* dst) const;

 private:
  // Throws on out-of-range params.
  void init();

  std::vector<double> candidate_equities(const MoveRequest& req,
                                         const std::vector<Move>& plays) const;

  int greedy_equity_index(const MoveRequest& req, const std::vector<Move>& plays) const;

  // Fills cand_idx_ with the plays worth evaluating and returns their count.
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

  // Reused across turns to avoid per-move allocation.
  std::vector<int> cand_idx_;
  std::vector<double> obj_values_;
  util::SoftmaxSampler sampler_;
};

}  // namespace scribblez
