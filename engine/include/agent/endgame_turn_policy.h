#pragma once

#include "agent/agent.h"
#include "endgame/endgame_solver.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace scribblez {

// The endgame half of an agent: an agent that plays heuristically while the
// bag holds tiles hands its bag-empty turns here, so it converts won endgames
// and defends lost ones exactly.
//
// The solver's answer is trusted only when its first iteration completed.
// Below that (the solve was declined because the root has more moves than the
// budget has nodes, or was cut off mid-iteration) the answer reflects an
// arbitrary fraction of the root, and the owning agent's own move is the
// better policy. The node budget therefore tunes how often an endgame gets a
// real search, never how noisy a search is.
//
// The two seats of a game thread share one pooled EndgameSolver, keyed by
// thread id. The solver's hash is seat-agnostic, so one seat's solves reuse
// the transposition entries the other just wrote; a thread runs its two agents
// sequentially, so the solver never sees concurrent use.
class EndgameTurnPolicy {
 public:
  // A budget of 0 in `params` disables solving: try_solve() always declines.
  //
  // params.spread_matters defaults to false, the self-play setting: it stops at
  // the win/draw/loss proof and relies on a projection-respecting game loop to
  // end the game there. Set it for games played to the end, where points still
  // matter.
  EndgameTurnPolicy(int thread_id, const EndgameSolver::Params& params);

  // The solver's move, with its proof certificate (if any) as the decision's
  // projection. Nullopt means the owning agent should play its own move: the
  // bag still holds tiles, solving is disabled, or the solve is untrusted.
  std::optional<MoveDecision> try_solve(const MoveRequest& req);

  // Clears the per-game state and the shared transposition table.
  void begin_game();
  // Tracks the consecutive-scoreless-turn count the solver needs, which no
  // MoveRequest carries.
  void observe_move(const Move& move);

  // Per-game totals over every solve, for a benchmark to report. Reset by
  // begin_game(). Everything but solve_ns is deterministic.
  //
  // max_solve_nodes lets a budget sweep skip redundant runs: if no solve of a
  // game spent more than a smaller budget b', re-running at b' is
  // bit-identical (no solve hit the cap, and a solve declined at the larger
  // budget is declined at b' too).
  struct SolveTotals {
    uint64_t solves = 0;
    uint64_t nodes = 0;
    uint64_t movegens = 0;
    uint64_t certificate_nodes = 0;
    uint64_t max_solve_nodes = 0;
    uint64_t solve_ns = 0;  // wall time inside solve(), the whole endgame cost
  };
  const SolveTotals& solve_totals() const { return solve_totals_; }

  // Also applies to the seat-mate, which shares the solver.
  void set_incremental_movegen(bool on) { solver_->set_incremental_movegen(on); }

 private:
  EndgameSolver::Params params_;
  int scoreless_turns_ = 0;
  std::shared_ptr<EndgameSolver> solver_;  // shared with the seat-mate
  SolveTotals solve_totals_;
};

}  // namespace scribblez
