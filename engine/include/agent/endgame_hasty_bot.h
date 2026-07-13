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
// Each agent instance owns one EndgameSolver. Agents are per-thread and the
// solver is single-threaded, so there is no sharing to guard.
class EndgameHastyBotAgent : public HastyBotAgent {
 public:
  // Default per-solve node budget, tuned with `endgame_bench --mode=games` on
  // NWL23: the largest budget whose endgame-vs-endgame games stay within ~2x
  // the plain HastyBot-vs-HastyBot game time (measured 2.00x, with a
  // +0.40 +/- 0.03 points/game head-to-head edge over greedy HastyBot across
  // 4800 seat-mirrored games). Strength keeps rising superlinearly with the
  // budget -- richer positions pass the decline gate -- at proportionally more
  // game time: 400 gives +0.85 at 4.5x, 1600 gives +5.0 at 21x.
  static constexpr uint64_t kDefaultEndgameNodes = 200;

  // HastyBot configuration plus the two endgame-solver knobs.
  //   endgame_nodes : per-turn solver node budget; 0 disables the solver
  //                   entirely (the agent then plays pure HastyBot all game).
  //   endgame_plies : iterative-deepening depth cap for the solver.
  struct Params {
    HastyBotAgent::Params hasty;
    uint64_t endgame_nodes = kDefaultEndgameNodes;
    int endgame_plies = 25;
  };

  explicit EndgameHastyBotAgent(const Params& params);

  Move make_move(const MoveRequest& req) override;
  void observe_move(const Move& move) override;
  void begin_game() override;

  // Build an EndgameHastyBotAgent from `--player "--type=hastybot-endgame
  // [options]"` tokens (after the factory has stripped --type and --name).
  // Accepts every HastyBot option plus --endgame-nodes=N (0 disables the solver)
  // and --endgame-plies=P. Throws on bad input.
  static std::unique_ptr<EndgameHastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                         int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  uint64_t endgame_nodes_;
  int endgame_plies_;
  int scoreless_turns_ = 0;  // consecutive zero-score turns, tracked from observe_move
  EndgameSolver solver_;
};

}  // namespace scribblez
