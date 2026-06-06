#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A HastyBot player: delegates each move to Macondo's "best static play"
// (HastyBot) via a per-thread MacondoOracle keyed off this agent's
// thread_id() (see MacondoOraclePool). The path to the `macondo` binary is
// configured once via MacondoOraclePool::set_params() at process startup
// (driven by play_game's --macondo option).
class HastyBotAgent : public Agent {
 public:
  HastyBotAgent(int thread_id, const std::string& name);
  Move make_move(const MoveRequest& req) override;

  // Build a HastyBotAgent from `--player "--type=hastybot [options]"` tokens
  // (after the factory has stripped --type and --name). Throws on bad input.
  static std::unique_ptr<HastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();
};

}  // namespace scribblez
