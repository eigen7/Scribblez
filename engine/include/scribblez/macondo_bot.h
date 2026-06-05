#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A HastyBot player: delegates each move to Macondo's "best static play"
// (HastyBot) via the process-wide Macondo singleton. The path to the
// `macondo` binary is configured once via Macondo::set_params() at process
// startup (driven by play_game's --macondo option), so every HastyBot seat
// shares one subprocess that loads its lexicon/leaves exactly once.
class HastyBotAgent : public Agent {
 public:
  explicit HastyBotAgent(const std::string& name);
  std::string name() const override { return name_; }
  Move make_move(const MoveRequest& req) override;

  // Build a HastyBotAgent from `--player "--type=hastybot [options]"` tokens
  // (after the factory has stripped --type and --name). Throws on bad input.
  static std::unique_ptr<HastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  std::string name_;
};

}  // namespace scribblez
