#pragma once

#include "agent/macondo_bot.h"
#include "endgame/endgame_solver.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A HastyBot that hands the endgame off to an EndgameSolver, so it converts won
// endgames and defends lost ones optimally rather than greedily. Until the bag
// empties it plays exactly like HastyBotAgent.
//
// The solver's answer is trusted only when its first iteration completed. Below
// that -- the solve was declined as too rich for the budget, or cut off
// mid-iteration -- its answer reflects an arbitrary fraction of the root, and
// HastyBot's static-equity move is the stronger policy at that price. The node
// budget therefore tunes how often endgames get a real search, never how noisy
// a search is.
//
// The agent tracks the consecutive-scoreless-turn count the solver needs, which
// no MoveRequest carries, from observe_move: any play resets it, a pass or
// exchange increments it.
//
// The two seats of a game thread share one pooled EndgameSolver. The solver's
// node hash is seat-agnostic, so the second seat's solves reuse the
// transposition entries the first seat just wrote; a thread runs its two agents
// sequentially, so the shared solver never sees concurrent use. Construction
// and begin_game() both clear the table.
class EndgameHastyBotAgent : public HastyBotAgent {
 public:
  // A solver budget of 0 disables endgame solving entirely.
  // solver.spread_matters defaults to false -- the self-play break-out setting,
  // which stops at the class proof and presumes a projection-respecting game
  // loop; pass true for games played to their end, where points still matter.
  struct Params {
    HastyBotAgent::Params hasty;
    EndgameSolver::Params solver;
  };

  explicit EndgameHastyBotAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  void observe_move(const Move& move) override;
  void begin_game() override;

  // Totals over a game, one contribution per solve, for a benchmark to report.
  // Reset by begin_game(). Everything but solve_ns is deterministic.
  //
  // max_solve_nodes lets a budget sweep tell when a smaller-budget run would be
  // bit-identical: if no solve of a game spent more than a smaller budget b',
  // re-running at b' changes nothing (no solve hit the larger cap, and a solve
  // declined for having more root moves than the larger budget is declined at
  // b' too).
  struct SolveTotals {
    uint64_t solves = 0;
    uint64_t nodes = 0;
    uint64_t movegens = 0;
    uint64_t certificate_nodes = 0;
    uint64_t max_solve_nodes = 0;
    uint64_t solve_ns = 0;  // wall time inside solve(), the agent's whole endgame cost
  };
  const SolveTotals& solve_totals() const { return solve_totals_; }

  // The seat-shared solver makes this apply to the seat-mate too.
  void set_incremental_movegen(bool on) { solver_->set_incremental_movegen(on); }

  // Build from `--player "--type=hastybot-endgame [options]"` tokens, with
  // --type and --name already stripped. Accepts every HastyBot option plus the
  // solver Params under an "endgame-" prefix. Throws on bad input.
  static std::unique_ptr<EndgameHastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                         int thread_id, const std::string& name);

  static std::string options_help();

 private:
  EndgameSolver::Params solver_params_;
  int scoreless_turns_ = 0;
  std::shared_ptr<EndgameSolver> solver_;  // shared with the seat-mate
  SolveTotals solve_totals_;
};

}  // namespace scribblez
