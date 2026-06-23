#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// Total order on (equity, move) used to pick HastyBot's move: higher equity
// wins; exact-equity ties are broken by a canonical move ordering so the choice
// is deterministic and independent of move-generation order. Returns true iff
// (eq_a, a) is the better hasty choice.
bool hasty_move_better(double eq_a, const Move& a, double eq_b, const Move& b);

// Reference HastyBot selection: generate every legal play and take the
// hasty_move_better argmax. This is the specification the shadow-play search in
// HastyBotAgent::make_move reproduces, used to verify the optimized path.
Move hasty_best_move_reference(const MoveRequest& req);

// An in-process HastyBot player: finds the play with the highest static equity
// (score + leave value + opening/PEG/endgame adjustments) via a shadow-play
// search -- it bounds each anchor's best possible equity, processes anchors
// best-first, and stops once no remaining anchor can beat the move found,
// instead of generating every legal play. Ties broken by hasty_move_better.
// Thread-safe after HastyEquity::init() has been called.
class HastyBotAgent : public Agent {
 public:
  HastyBotAgent(int thread_id, const std::string& name);
  Move make_move(const MoveRequest& req) override;
  bool supports_parallelism() const override { return true; }

  // Build a HastyBotAgent from `--player "--type=hastybot [options]"` tokens
  // (after the factory has stripped --type and --name). Throws on bad input.
  static std::unique_ptr<HastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();
};

}  // namespace scribblez
