#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A HastyBot player: delegates each move to Macondo's "best static play"
// (HastyBot). It talks to a single persistent `macondo` shell subprocess that
// is shared by every HastyBot seat and started lazily from `macondo_binary`
// (Macondo loads its lexicon/leaves once, so we never re-spawn it).
class HastyBotAgent : public Agent {
 public:
  HastyBotAgent(std::string macondo_binary, std::string name);
  std::string name() const override { return name_; }
  Move make_move(const MoveRequest& req) override;

  // Build a HastyBotAgent from `--player "--type=hastybot [options]"` tokens
  // (after the factory has stripped --type and --name). Throws on bad input.
  static std::unique_ptr<HastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  std::string name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  std::string macondo_binary_;
  std::string name_;
};

}  // namespace scribblez
