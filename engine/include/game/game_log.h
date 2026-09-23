#pragma once

#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// One completed game as a sequence of turns: what Game produces, and what the
// data, encoding and GCG code consume without depending on Game itself.

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

// Non-owning view of one game's log; its backing store (a GameLogStorage, or a
// decoder's buffer) must outlive it. Both self-play and on-disk replay produce
// this type, so one encoder serves both.
struct GameLog {
  uint64_t seed = 0;
  std::array<const char*, 2> player_names = {nullptr, nullptr};
  std::array<int, 2> initial_scores = {0, 0};  // head-start handicap, if any
  std::array<Rack, 2> initial_racks;           // tiles dealt to each player at game start
  const TurnRecord* records = nullptr;         // backing store owned elsewhere
  int num_records = 0;
  std::array<int, 2> final_scores = {0, 0};
  std::array<Rack, 2> final_racks;   // tiles left on each rack at game end
  const char* end_reason = nullptr;  // "out", "stalemate", "max_turns", or
                                     // "truncated" (Game::set_max_plies)
  // Leading plies played at random (Game::set_random_opening). Positions
  // followed by a random move are excluded from training
  // (binlog::eligible_span).
  int num_random_opening_plies = 0;
};

// Owning backing store for a game's log. A view() stays valid while the
// storage lives and `turns` is not reallocated.
struct GameLogStorage {
  uint64_t seed = 0;
  std::array<std::string, 2> player_names;
  std::array<int, 2> initial_scores = {0, 0};
  std::array<Rack, 2> initial_racks;
  std::vector<TurnRecord> turns;
  std::array<int, 2> final_scores = {0, 0};
  std::array<Rack, 2> final_racks;
  std::string end_reason;
  int num_random_opening_plies = 0;

  GameLog view() const;
};

}  // namespace scribblez
