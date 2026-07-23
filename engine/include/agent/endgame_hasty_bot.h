#pragma once

#include "agent/macondo_bot.h"
#include "endgame/endgame_solver.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A HastyBot that hands the endgame off to an exact/near-exact search. While the
// bag holds tiles it plays exactly like HastyBotAgent (static-equity greedy or
// softmax sampling). Once the bag empties and both racks are fully known, it runs
// an EndgameSolver -- a negamax search with iterative deepening -- on the
// position and plays the solver's best move, so it converts won endgames and
// defends lost ones optimally rather than greedily.
//
// The solver's answer is trusted only when its first iteration completed, i.e.
// every root move was evaluated by a full greedy playout. Below that -- the
// solve was declined as too rich for the budget, or cut off mid-iteration --
// its answer reflects an arbitrary fraction of the root, and the agent plays
// HastyBot's static-equity move instead, the stronger policy at that price. The
// node budget therefore tunes how often endgames get a real search, never how
// noisy a search is.
//
// The solver needs the running consecutive-scoreless-turn count (six such turns
// end the game), which no MoveRequest carries, so the agent tracks it from
// observe_move exactly as the game loop does: any play resets it, a pass or
// exchange increments it.
//
// The two seats of a GameRunner thread share one EndgameSolver, obtained
// from a per-thread pool keyed by thread_id: the solver's node hash is
// seat-agnostic, so the second seat's solves reuse the transposition entries
// the first seat's solves of the same game tree just wrote instead of
// recomputing them. A thread's game loop runs its two agents sequentially,
// so the shared solver never sees concurrent use; distinct threads get
// distinct solvers. Construction and begin_game() both clear the shared
// table, so a fresh agent pair always starts from a cold table.
class EndgameHastyBotAgent : public HastyBotAgent {
 public:
  // HastyBot configuration plus the endgame solver's own Params. A solver
  // budget of 0 disables endgame solving entirely (the agent then plays pure
  // HastyBot all game). solver.spread_matters defaults to false -- the
  // self-play break-out setting, which stops at the class proof and presumes
  // a projection-respecting game loop; pass true for games played to their
  // end, where points still matter.
  struct Params {
    HastyBotAgent::Params hasty;
    EndgameSolver::Params solver;
  };

  explicit EndgameHastyBotAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  void observe_move(const Move& move) override;
  void begin_game() override;

  // Deterministic operation totals accumulated across a game, one contribution
  // per make_move that ran the solver: the number of such solves plus the
  // nodes, logical move generations, and certificate nodes each EndgameResult
  // reported. A benchmark reads these to convert deterministic operation counts
  // into modeled time. Reset by begin_game().
  //
  // max_solve_nodes is the largest single solve's nodes across the game. A
  // budget-sweep benchmark uses it to decide when a smaller-budget run would be
  // bit-identical: if no solve of a game spent more than a smaller budget b',
  // then re-running the game at b' changes nothing (no solve hit the larger
  // cap, and any solve declined for having more root moves than the larger
  // budget is declined at b' too).
  struct SolveTotals {
    uint64_t solves = 0;
    uint64_t nodes = 0;
    uint64_t movegens = 0;
    uint64_t certificate_nodes = 0;
    uint64_t max_solve_nodes = 0;
  };
  const SolveTotals& solve_totals() const { return solve_totals_; }

  // Enable or disable the solver's move-generation memo (off by default),
  // forwarding to the underlying solver. Because the two seats of a thread share
  // one pooled solver, this setting applies to both seat-mates.
  void set_movegen_memo(bool on) { solver_->set_movegen_memo(on); }

  // Enable or disable the solver's incremental move-list maintenance (on by
  // default), forwarding to the underlying solver. As with the memo, the
  // seat-shared solver makes this apply to both seat-mates.
  void set_incremental_movegen(bool on) { solver_->set_incremental_movegen(on); }

  // Build an EndgameHastyBotAgent from `--player "--type=hastybot-endgame
  // [options]"` tokens (after the factory has stripped --type and --name).
  // Accepts every HastyBot option plus the solver Params under an "endgame-"
  // prefix: --endgame-budget=N (0 disables the solver), --endgame-plies=P,
  // --endgame-spread-matters=0|1. Throws on bad input.
  static std::unique_ptr<EndgameHastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                         int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  EndgameSolver::Params solver_params_;
  int scoreless_turns_ = 0;  // consecutive zero-score turns, tracked from observe_move
  std::shared_ptr<EndgameSolver> solver_;  // shared with the seat-mate; see the class comment
  SolveTotals solve_totals_;               // accumulated across the game; reset by begin_game()
};

}  // namespace scribblez
