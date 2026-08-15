#pragma once

#include "agent/agent.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

// Picks the highest-scoring PLAY. If none exists, exchanges the entire rack (if
// the bag has >= RACK_SIZE tiles), otherwise passes. Ties broken randomly.
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
