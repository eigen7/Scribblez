#pragma once

#include "scribblez/agent.h"
#include "scribblez/bag.h"
#include "scribblez/board.h"
#include "scribblez/dictionary.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <array>
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
  // Tiles drawn from the bag after the move (in draw order). Trailing entries
  // of `drawn` are empty Tiles. A turn draws at most RACK_SIZE tiles.
  Rack drawn;
};

struct GameLog {
  uint64_t seed = 0;
  std::array<std::string, 2> player_names;
  std::array<Rack, 2> initial_racks;  // tiles dealt to each player at game start
  std::vector<TurnRecord> turns;
  std::array<int, 2> final_scores = {0, 0};
  std::array<Rack, 2> final_racks;  // tiles left on each rack at game end
  std::string end_reason;           // "out", "stalemate", or "max_turns"
};

class Game {
 public:
  // The Game does not own the agents; the caller (typically play_game) keeps
  // them alive across multiple Game instances. This lets the same human or
  // bot persist (including any per-process state like a WebSession or a
  // background Macondo subprocess) over a series of games.
  Game(Agent& p0, Agent& p1, const Dictionary& dict, uint64_t seed);

  void play();
  const GameLog& log() const { return log_; }

  // Live game accessors (valid after construction; reflect final state after
  // play() returns). Used by the web front-end to render the end-of-game board.
  const Board& board() const { return board_; }
  int score(int player) const { return scores_[player]; }
  const Rack& rack(int player) const { return racks_[player]; }
  int bag_size() const { return bag_.size(); }

 private:
  Agent* players_[2];
  const Dictionary& dict_;
  uint64_t seed_;
  Bag bag_;
  Board board_;
  Rack racks_[2];
  std::array<int, 2> scores_{0, 0};
  GameLog log_;

  // Draw from the bag until the player's rack is at RACK_SIZE. If
  // `drawn_out` is non-null, the drawn tiles are added to it.
  void refill_rack(int p, Rack* drawn_out);
};

}  // namespace scribblez
