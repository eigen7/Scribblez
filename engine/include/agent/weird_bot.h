#pragma once

#include "agent/agent.h"
#include "game/rack.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A diagnostic self-play opponent, not a production agent. It deterministically
// forces its highest-value leave tile onto the best cross-check square for it,
// making the conjunction "opponent-leave letter x cross-check plane" a
// dominant, consistent signal in its games. It exists to test whether the
// position evaluation model can learn that conjunction.
//
// The rule keys off the agent's own leave (the tiles it kept from its last
// move) because that is exactly the opponent-leave input the model sees in a
// face-up-leaves game, so only a leave-driven rule is learnable. MoveRequest
// does not carry the agent's own leave, so the agent tracks it.
class WeirdBotAgent : public Agent {
 public:
  WeirdBotAgent(int thread_id, const std::string& name);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=weirdbot"` tokens, with --type and --name
  // already stripped. It takes no options, so any token throws.
  static std::unique_ptr<WeirdBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name);

  static std::string options_help();

 private:
  // This turn's move under the forcing rule (weird_bot.cpp).
  Move choose_move(const MoveRequest& req) const;

  // The tiles the agent kept from its last move; empty at game start.
  Rack leave_;
};

}  // namespace scribblez
