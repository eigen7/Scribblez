#pragma once

#include "game/game.h"

#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace scribblez {

// Serialize a GameLog as a GCG file -- the de-facto standard Scrabble game-log
// format (as written by Macondo/Quackle). Header pragmata plus one event line
// per turn:
//   >nick: rack POS WORD +score cumulative      (tile placement; '.' = a tile
//                                                 already on the board, lower-
//                                                 case = a designated blank)
//   >nick: rack -TILES +0 cumulative            (exchange)
//   >nick: rack - +0 cumulative                 (pass)
// followed by end-of-game rack adjustments (END_RACK_PTS / END_RACK_PENALTY).
struct GcgWriteOptions {
  struct PostEventRacks {
    std::optional<std::string> rack1;
    std::optional<std::string> rack2;
  };

  // If set, emits '#lexicon <name>' near the top of the file.
  std::optional<std::string> lexicon_name;

  // Extra '#note ...' lines to include near the top of the file.
  std::vector<std::string> notes;

  // If non-empty, controls whether each turn line includes rack_before.
  // When omitted, every turn includes rack_before (legacy behavior).
  std::vector<bool> include_rack_before;

  // Exact rack field to use on each event line. When present, this overrides
  // include_rack_before and the rack string derived from TurnRecord.
  std::vector<std::optional<std::string>> rack_before_fields;

  // Exact exchanged-tile field to use on exchange event lines. The stored
  // string does not include the leading '-'. Useful for incomplete-rack GCG
  // logs that encode unknown exchanged tiles as '_' or by count.
  std::vector<std::optional<std::string>> exchange_fields;

  // Optional '#rack1' / '#rack2' pragmata to emit immediately after each
  // event line.
  std::vector<PostEventRacks> post_event_racks;

  // Optional initial '#Rack1' / '#Rack2' pragmata to emit after the header.
  std::optional<std::string> initial_rack1;
  std::optional<std::string> initial_rack2;
};

std::string game_log_to_gcg(const GameLog& log);
std::string game_log_to_gcg(const GameLog& log, const GcgWriteOptions& options);
void write_game_log_gcg(const GameLog& log, std::ostream& out);
void write_game_log_gcg(const GameLog& log, std::ostream& out, const GcgWriteOptions& options);

}  // namespace scribblez
