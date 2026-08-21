#pragma once

#include "agent/agent.h"
#include "game/rack.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A diagnostic self-play opponent that deterministically forces its
// highest-value retained leave tile onto its best cross-check square, so the
// "opponent-leave-letter x cross-check-plane" conjunction becomes the dominant,
// consistent signal in the training data. It exists to test whether the
// position-eval model can learn that conjunction; it is not a production agent.
//
// The rule keys off the agent's OWN leave -- the tiles it retained after its
// last move -- because that leave is exactly the opponent-leave input the model
// sees, so only a leave-driven rule is learnable. MoveRequest exposes the full
// rack (my_rack) and the opponent's face-up leave (opp_rack) but never the
// agent's own leave, so WeirdBot tracks it: leave = my_rack minus the tiles its
// own chosen move consumed, updated at the end of make_move and reset in
// begin_game.
class WeirdBotAgent : public Agent {
 public:
  WeirdBotAgent(int thread_id, const std::string& name);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=weirdbot [options]"` tokens, with --type and
  // --name already stripped. Takes no options of its own; throws on any input.
  // Ensures the process-wide HastyEquity tables are loaded, as the fallback
  // path (hasty_best_move_wmp) and the survivor ranking both read them.
  static std::unique_ptr<WeirdBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name);

  static std::string options_help();

 private:
  // The move actually chosen this turn (steps 1-5 of the forcing rule), before
  // the leave is updated from it.
  Move choose_move(const MoveRequest& req) const;

  // The agent's own retained tiles after its last move; empty at game start and
  // whenever the last move consumed the whole rack.
  Rack leave_;
};

}  // namespace scribblez
