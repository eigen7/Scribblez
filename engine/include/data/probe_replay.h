#pragma once

// The game state along a .sprobe position's probes (data/probe_log.h), by
// replay: what SupremeBot M1a's token encoder reads beside each probe's moves
// (docs/plans/supreme_bot_m1a.md, PR 3). The .sprobe keeps only what replay
// cannot recompute; this recomputes the rest from the position's .slog game.
//
// Racks are written as tile codes (game/tile.h: 0-25 letters, 26 the blank),
// ascending and padded with Tile::kEmpty, so a reader needs no Rack layout.
// Score differences are the root mover's score minus the opponent's.

#include "data/probe_log.h"
#include "game/game_log.h"
#include "game/tile.h"

#include <cstdint>
#include <vector>

namespace scribblez {

#pragma pack(push, 1)

// The decision point a position's candidates are made from.
struct ProbeRootState {
  int16_t score_diff;
  uint8_t bag_size;
  uint8_t mover;              // the root mover, 0 or 1
  char rack[RACK_SIZE];       // the root mover's full rack
  char opp_leave[RACK_SIZE];  // the opponent's kept tiles, known under face-up leaves
};
static_assert(sizeof(ProbeRootState) == 18, "ProbeRootState must be 18 bytes");

// The state right after a candidate, before the root mover's refill.
struct ProbeCandidateState {
  char leave[RACK_SIZE];  // the root mover's kept tiles
  int16_t score_diff;
  uint8_t bag_size;  // after the refill, which takes the same count in every probe
};
static_assert(sizeof(ProbeCandidateState) == 10, "ProbeCandidateState must be 10 bytes");

// A probe's two deals before the opponent's reply: the root mover's refill
// and the opponent's rack, each as the tiles dealt and the rack they made.
struct ProbeStartState {
  char mover_drawn[RACK_SIZE];
  char mover_rack[RACK_SIZE];
  char opp_drawn[RACK_SIZE];  // beyond the opponent's known leave
  char opp_rack[RACK_SIZE];
};
static_assert(sizeof(ProbeStartState) == 28, "ProbeStartState must be 28 bytes");

// One probe turn: the move's surroundings and the draw after it.
struct ProbeTurnState {
  char leave[RACK_SIZE];       // the mover's kept tiles
  char drawn[RACK_SIZE];       // the draw after the move
  char rack_after[RACK_SIZE];  // leave plus draw
  uint8_t root_mover;          // 1 when the root mover makes the turn
  uint8_t ply;                 // 1 = the opponent's reply to the candidate
  uint8_t bag_size;            // before the move
  int16_t score_diff;          // before the move
};
static_assert(sizeof(ProbeTurnState) == 26, "ProbeTurnState must be 26 bytes");

#pragma pack(pop)

// Replayed states, in the .sprobe's orders: positions in file order, each
// position's candidates as listed, records candidate-major, turns in record
// order.
struct ProbeReplay {
  std::vector<ProbeRootState> roots;  // one per position
  std::vector<ProbeCandidateState> candidates;
  std::vector<ProbeStartState> starts;  // one per record
  std::vector<ProbeTurnState> turns;
};

// Appends the replay of `pos`, whose position is turn pos.header->turn_index
// of `game` (a make_game_view view of its .slog game). Throws util::Exception
// when a record's racks do not follow from the position, which means the two
// files do not belong together.
void replay_probe_position(const GameLog& game, const ProbeReader::Position& pos, int probes,
                           ProbeReplay* out);

// Replays every position of `probes` against `slog`, the loaded bytes of its
// companion .slog, whose magic and version the caller has checked.
ProbeReplay replay_probe_file(const char* slog, const ProbeReader& probes);

}  // namespace scribblez
