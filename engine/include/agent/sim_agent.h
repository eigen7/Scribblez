#pragma once

// The Monte-Carlo simming agent (--type=sim): keep the top K legal moves by
// static equity, roll each out under common random numbers, and play whichever
// the rollouts liked best. Bag-empty turns go to the endgame solver.
//
// This is the project's baseline simming opponent. It resembles Macondo's
// BestBot in shape, not detail: Macondo's rollouts stop at a fixed ply and read
// static equity, where ours run to the game's end or to a learned leaf
// evaluation (sim_horizon), and Macondo prunes weak candidates as it sims where
// we sim every candidate equally. BestBot itself is ported in best_bot.h.

#include "agent/agent.h"
#include "agent/endgame_turn_policy.h"
#include "endgame/endgame_solver.h"
#include "nn/eval_service.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

class SimAgent : public Agent {
 public:
  // `dict` is required and must outlive the agent. An `endgame` budget of 0
  // turns solving off, leaving the static-equity move to play the endgame.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    int top_k = 10;  // candidates simmed per turn, by static equity
    // Rollouts per candidate, and their threading. 400 is where the measured
    // strength curve flattens. A rollout to the game's end carries the noise
    // of every ply in it, so below a few hundred the ranking is swamped and
    // the agent plays worse than the static equity it started from: 35%
    // against HastyBot at 50 rollouts, 48% at 200, 57% at 400, 58% at 800.
    SimRunner::Params sim = {400, 1};
    // Value truncation; see SimRunner::Params::horizon_plies. The leaf service
    // handed to the constructor scores the horizon.
    int sim_horizon = 0;
    SimObjective objective = SimObjective::kWinRate;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  // `leaf_service` is the rollout leaf evaluator (real or a scripted stub);
  // give it iff params.sim_horizon is set.
  explicit SimAgent(const Params& params,
                    std::shared_ptr<nn::PositionEvalService> leaf_service = nullptr);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;

  // Build from `--player "--type=sim [options]"` tokens, with --type and --name
  // already stripped. Throws util::CleanException on bad input.
  static std::unique_ptr<SimAgent> from_spec(const std::vector<std::string>& tokens, int thread_id,
                                             const std::string& name);

  static std::string options_help();

  // The seed SimRunner::run is given on the turn after `ply` moves have been
  // observed. Public so a test can reproduce a decision's rollouts exactly.
  uint64_t sim_seed(int ply) const;

 private:
  int top_k_;
  SimObjective objective_;
  uint64_t seed_;
  std::shared_ptr<nn::PositionEvalService> leaf_service_;  // null = terminal sims
  SimRunner runner_;
  EndgameTurnPolicy endgame_;
  int ply_ = 0;  // moves observed this game, by either seat
};

}  // namespace scribblez
