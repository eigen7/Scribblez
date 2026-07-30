#pragma once

#include "agent/agent.h"
#include "endgame/endgame_solver.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace scribblez {

// The endgame half of an agent: an agent that picks its moves heuristically
// while the bag holds tiles hands its bag-empty turns here, so it converts won
// endgames and defends lost ones optimally rather than greedily.
//
// The solver's answer is trusted only when its first iteration completed. Below
// that -- the solve was declined as too rich for the budget, or cut off
// mid-iteration -- its answer reflects an arbitrary fraction of the root, and
// the owning agent's own move is the stronger policy at that price. The node
// budget therefore tunes how often endgames get a real search, never how noisy
// a search is.
//
// The consecutive-scoreless-turn count the solver needs rides on observe_move,
// no MoveRequest carrying it: any play resets it, a pass or exchange increments
// it.
//
// The two seats of a game thread share one pooled EndgameSolver. The solver's
// node hash is seat-agnostic, so the second seat's solves reuse the
// transposition entries the first seat just wrote; a thread runs its two agents
// sequentially, so the shared solver never sees concurrent use. Construction
// and begin_game() both clear the table.
class AgentEndgameSolver {
 public:
  // A budget of 0 in `params` disables endgame solving entirely: try_solve()
  // then always declines, leaving the owning agent's own policy to play the
  // endgame out.
  //
  // params.spread_matters defaults to false -- the self-play break-out setting,
  // which stops at the class proof and presumes a projection-respecting game
  // loop; pass true for games played to their end, where points still matter.
  AgentEndgameSolver(int thread_id, const EndgameSolver::Params& params);

  // The decision for a turn the solver owns: its move, annotated with the proof
  // certificate as the decision's projection, so a projection-respecting loop
  // (self-play generation) fast-tracks the game to its proven end instead of
  // prompting for the remaining turns. Nullopt means the owning agent plays its
  // own move -- the bag still holds tiles, solving is disabled, or the solve
  // came back untrusted.
  std::optional<MoveDecision> try_solve(const MoveRequest& req);

  void begin_game();
  void observe_move(const Move& move);

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
    uint64_t solve_ns = 0;  // wall time inside solve(), the whole endgame cost
  };
  const SolveTotals& solve_totals() const { return solve_totals_; }

  // The seat-shared solver makes this apply to the seat-mate too.
  void set_incremental_movegen(bool on) { solver_->set_incremental_movegen(on); }

 private:
  EndgameSolver::Params params_;
  int scoreless_turns_ = 0;
  std::shared_ptr<EndgameSolver> solver_;  // shared with the seat-mate
  SolveTotals solve_totals_;
};

}  // namespace scribblez
