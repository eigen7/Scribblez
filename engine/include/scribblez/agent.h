#pragma once

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <random>
#include <string>
#include <vector>

namespace scribblez {

struct AgentContext {
  const Board& board;
  const Rack& my_rack;
  int my_score;
  int opp_score;
  int bag_size;
  int opp_rack_size;              // tiles on the opponent's rack (hidden contents)
  std::vector<Move> legal_plays;  // PLAY moves only; agent may pass/exchange
};

class Agent {
 public:
  virtual ~Agent() = default;
  virtual std::string name() const = 0;
  virtual Move choose(const AgentContext& ctx, std::mt19937_64& rng) = 0;
};

// Picks the highest-scoring PLAY. If none exists, exchanges the entire rack
// (if the bag has >= RACK_SIZE tiles), otherwise passes. Ties broken randomly.
class GreedyAgent : public Agent {
 public:
  explicit GreedyAgent(std::string name = "Greedy") : name_(std::move(name)) {}
  std::string name() const override { return name_; }
  Move choose(const AgentContext& ctx, std::mt19937_64& rng) override;

 private:
  std::string name_;
};

}  // namespace scribblez
