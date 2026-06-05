#pragma once

#include "scribblez/agent.h"

#include <string>

namespace scribblez {

// A HastyBot player: delegates each move to Macondo's "best static play"
// (HastyBot). It talks to a single persistent `macondo` shell subprocess that
// is shared by every HastyBot seat and started lazily from `macondo_binary`
// (Macondo loads its lexicon/leaves once, so we never re-spawn it).
class HastyBotAgent : public Agent {
 public:
  HastyBotAgent(std::string macondo_binary, std::string name);
  std::string name() const override { return name_; }
  Move choose(const AgentContext& ctx, std::mt19937_64& rng) override;

 private:
  std::string macondo_binary_;
  std::string name_;
};

}  // namespace scribblez
