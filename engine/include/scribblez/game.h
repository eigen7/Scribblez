#pragma once

#include "scribblez/agent.h"
#include "scribblez/bag.h"
#include "scribblez/board.h"
#include "scribblez/dictionary.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

struct TurnRecord {
  int player;  // 0 or 1
  Rack rack_before;
  int bag_size_before;
  Move move;
  int score_delta;  // points scored on this turn (may be 0 or negative)
  std::array<int, 2> cumulative_scores;
  std::vector<Letter> drawn;  // tiles drawn from bag after the move (in draw order)
};

struct GameLog {
  uint64_t seed = 0;
  std::array<std::string, 2> player_names;
  std::vector<TurnRecord> turns;
  std::array<int, 2> final_scores = {0, 0};
  std::string end_reason;  // "out", "stalemate", or "max_turns"
};

class Game {
 public:
  Game(std::unique_ptr<Agent> p0, std::unique_ptr<Agent> p1, const Dictionary& dict, uint64_t seed);

  void play();
  const GameLog& log() const { return log_; }

  // Live game accessors (valid after construction; reflect final state after
  // play() returns). Used by the web front-end to render the end-of-game board.
  const Board& board() const { return board_; }
  int score(int player) const { return scores_[player]; }
  const Rack& rack(int player) const { return racks_[player]; }
  int bag_size() const { return bag_.size(); }

 private:
  std::unique_ptr<Agent> players_[2];
  const Dictionary& dict_;
  uint64_t seed_;
  Bag bag_;
  Board board_;
  Rack racks_[2];
  std::array<int, 2> scores_{0, 0};
  std::mt19937_64 rng_;
  GameLog log_;

  void refill_rack(int p, std::vector<Letter>& drawn);
};

}  // namespace scribblez
