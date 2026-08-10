#pragma once

// The Monte-Carlo simming agent (docs/roadmap.md, track A): filter the legal
// moves by static equity, roll the survivors out under common random numbers,
// and play whichever the rollouts liked best. Once the bag empties the turn
// goes to the exact solver, as it does for every agent that plays the endgame
// properly.
//
// This is the baseline the move set evaluation model has to beat, and the
// harness the sim-quality and scheduling tracks take their match readouts in.
// It is also the closest thing we have to Macondo's BestBot -- simming plus an
// endgame solver, with no rack inference -- which is what makes a win rate
// against it comparable to published results. The resemblance is in the shape,
// not the details: our rollouts run to a natural game end where Macondo's stop
// at a fixed ply and read static equity, and we sim every candidate to the same
// depth where Macondo trims the field as it goes.

#include "agent/agent.h"
#include "agent/endgame_turn_policy.h"
#include "endgame/endgame_solver.h"
#include "selfplay/sim_runner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

class SimAgent : public Agent {
 public:
  // `dict` is required and must outlive the agent. An `endgame` budget of 0
  // turns endgame solving off, leaving the greedy static-equity move to play
  // the endgame out.
  struct Params {
    int thread_id = 0;
    std::string name;
    const Dictionary* dict = nullptr;
    int top_k = 10;  // candidates simmed per turn, by static equity
    // Rollouts per candidate, and their threading. 400 is where the measured
    // strength curve flattens: a rollout run to a natural game end carries the
    // noise of every ply in it, so below a few hundred the ranking is swamped
    // and the agent plays WORSE than the static equity it started from -- 35%
    // against HastyBot at 50 rollouts, 48% at 200, 57% at 400, 58% at 800.
    SimRunner::Params sim = {400, 1};
    SimObjective objective = SimObjective::kWinRate;
    uint64_t seed = 0;
    EndgameSolver::Params endgame = {};  // the solver's own defaults
  };

  explicit SimAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game() override;
  void observe_move(const Move& move) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=sim [options]"` tokens, with --type and --name
  // already stripped. Throws std::runtime_error on bad input.
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
  SimRunner runner_;
  EndgameTurnPolicy endgame_;
  int ply_ = 0;  // moves observed this game, by either seat
};

}  // namespace scribblez
