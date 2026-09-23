#pragma once

#include "agent/agent.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

// Plays the highest-scoring play, breaking ties at random. With no legal play
// it exchanges its whole rack if the bag allows, else passes.
class GreedyAgent : public Agent {
 public:
  explicit GreedyAgent(int thread_id, const std::string& name = "Greedy");
  GreedyAgent(int thread_id, const std::string& name, uint64_t seed);

  MoveDecision make_move(const MoveRequest& req) override;

  // Build from `--player "--type=greedy [options]"` tokens, with --type and
  // --name already stripped. Throws util::CleanException on bad input.
  static std::unique_ptr<GreedyAgent> from_spec(const std::vector<std::string>& tokens,
                                                int thread_id, const std::string& name);

  static std::string options_help();

 private:
  std::mt19937_64 rng_;
};

}  // namespace scribblez
