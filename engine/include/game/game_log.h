#pragma once

#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// One completed game as a sequence of turns: what Game produces, and the single
// currency the tensorization path consumes. Its own header because the data,
// encoding and GCG paths all speak it without ever touching the Game class that
// fills it in.

namespace scribblez {

struct TurnRecord {
  int player;  // 0 or 1
  Rack rack_before;
  int bag_size_before;
  Move move;
  int score_delta;  // may be 0 or negative
  std::array<int, 2> cumulative_scores;
  Rack drawn;  // tiles drawn after the move resolved, in draw order
};

// Non-owning view of one completed game's log; its variable-length backing
// store (a GameLogStorage, or a decoder's scratch buffer) must outlive it. The
// single currency the tensorization path consumes, so self-play and on-disk
// replay funnel through one encoder.
struct GameLog {
  uint64_t seed = 0;
  std::array<const char*, 2> player_names = {nullptr, nullptr};
  std::array<int, 2> initial_scores = {0, 0};  // head-start handicap, if any
  std::array<Rack, 2> initial_racks;           // tiles dealt to each player at game start
  const TurnRecord* records = nullptr;         // backing store owned elsewhere
  int num_records = 0;
  std::array<int, 2> final_scores = {0, 0};
  std::array<Rack, 2> final_racks;   // tiles left on each rack at game end
  const char* end_reason = nullptr;  // "out", "stalemate", or "max_turns"
  // Leading plies played uniformly at random via Game::set_random_opening
  // rather than by the seated agents (0 for a normal game). Positions before
  // the last of these have a random move after them and are excluded from
  // training (see binlog::eligible_span).
  int num_random_opening_plies = 0;
};

// Owning backing store for a game's log. `view()` points into it, and stays
// valid while the storage lives and its `turns` vector is not reallocated.
struct GameLogStorage {
  uint64_t seed = 0;
  std::array<std::string, 2> player_names;
  std::array<int, 2> initial_scores = {0, 0};
  std::array<Rack, 2> initial_racks;
  std::vector<TurnRecord> turns;
  std::array<int, 2> final_scores = {0, 0};
  std::array<Rack, 2> final_racks;
  std::string end_reason;
  int num_random_opening_plies = 0;  // see GameLog::num_random_opening_plies

  GameLog view() const;
};

}  // namespace scribblez
