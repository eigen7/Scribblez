#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// An in-process HastyBot player: generates all legal plays and picks the one
// with the highest static equity (score + leave value + opening/PEG/endgame
// adjustments) using the process-wide HastyEquity singleton.  Thread-safe
// after HastyEquity::init() has been called.
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
