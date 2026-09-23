#pragma once

#include "game/game_log.h"

#include <optional>
#include <ostream>
#include <string>
#include <vector>

// Writes GameLogs as GCG, the de-facto standard Scrabble game-log format (as
// written by Macondo and Quackle): header pragmata, then one event line per
// turn,
//   >nick: rack POS WORD +score cumulative   (play; '.' = a tile already on
//                                              the board, lowercase = a blank)
//   >nick: rack -TILES +0 cumulative         (exchange)
//   >nick: rack - +0 cumulative              (pass)
// then the end-of-game rack adjustment lines.

namespace scribblez {

struct GcgWriteOptions {
  struct PostEventRacks {
    std::optional<std::string> rack1;
    std::optional<std::string> rack2;
  };

  // Emitted as '#lexicon <name>'.
  std::optional<std::string> lexicon_name;

  // Emitted as '#note ...' lines in the header.
  std::vector<std::string> notes;

  // Per turn, whether its line includes the rack. Empty includes them all.
  std::vector<bool> include_rack_before;

  // Per turn, the rack field to write, or nullopt to omit it. If non-empty,
  // replaces both include_rack_before and the TurnRecord's rack.
  std::vector<std::optional<std::string>> rack_before_fields;

  // Per turn, the exchanged-tile field without its leading '-', overriding the
  // move's tiles. For logs of games with hidden racks, which record unknown
  // exchanged tiles as '_' or as a count.
  std::vector<std::optional<std::string>> exchange_fields;

  // Per turn, '#Rack1' / '#Rack2' pragmata to emit after its event line.
  std::vector<PostEventRacks> post_event_racks;

  // '#Rack1' / '#Rack2' pragmata to emit in the header. gcg_reader.h applies
  // these to the game's final position.
  std::optional<std::string> initial_rack1;
  std::optional<std::string> initial_rack2;
};

// One move in GCG event notation, given the board it is played on: "POS WORD"
// for a play (row first for horizontal, "8D"; column first for vertical,
// "D8"), "-TILES" for an exchange, "-" for a pass.
std::string move_notation(const Board& board_before, const Move& m);

// move_notation for human readers: played-through tiles are spelled out in
// parentheses instead of dotted, e.g. "A4 (mO)u(N)T" for GCG's "A4 ..u.T".
std::string spelled_move_notation(const Board& board_before, const Move& m);

std::string game_log_to_gcg(const GameLog& log);
std::string game_log_to_gcg(const GameLog& log, const GcgWriteOptions& options);
void write_game_log_gcg(const GameLog& log, std::ostream& out);
void write_game_log_gcg(const GameLog& log, std::ostream& out, const GcgWriteOptions& options);

}  // namespace scribblez
