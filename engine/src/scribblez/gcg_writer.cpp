#include "scribblez/gcg_writer.h"

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/tile.h"

#include <array>
#include <sstream>
#include <string>

namespace scribblez {

namespace {

// GCG nicknames are single whitespace-free tokens. Derive one from a display
// name; the caller makes the pair unique.
std::string nickify(const std::string& name) {
  std::string nick;
  for (char c : name) nick.push_back(c == ' ' || c == '\t' ? '_' : c);
  if (nick.empty()) nick = "player";
  return nick;
}

// Coordinate of the main word's first square: "8D" for a horizontal play (row
// number first), "D8" for a vertical one (column letter first). The origin is
// recovered from the board as it stood before the move.
std::string position(const Board& board_before, const Move& m) {
  auto [r, c] = m.word_origin(board_before);
  std::string col(1, static_cast<char>('A' + c));
  std::string row = std::to_string(r + 1);
  return m.horizontal() ? row + col : col + row;
}

// The played word in GCG form: a '.' for every square already occupied before
// this move (played through), and a lowercase letter for a designated blank.
std::string played_word(const Board& board_before, const Move& m) {
  std::string out;
  const int dr = m.horizontal() ? 0 : 1, dc = m.horizontal() ? 1 : 0;
  const int n = m.num_glyphs();
  auto [r, c] = m.word_origin(board_before);
  int gi = 0;
  while (board_before.in_bounds(r, c)) {
    if (!board_before.at(r, c).is_empty()) {
      out.push_back('.');  // played through an existing tile
    } else if (gi < n) {
      Glyph g = m.glyph(gi++);
      char ch = g.letter().to_char();
      out.push_back(g.is_blank() ? static_cast<char>(ch - 'A' + 'a') : ch);
    } else {
      break;
    }
    r += dr;
    c += dc;
  }
  return out;
}

std::string exchanged_tiles(const Move& m) {
  std::string s;
  const int n = m.num_glyphs();
  for (int i = 0; i < n; ++i) s.push_back(m.glyph(i).rack_tile().to_char());
  return s;
}

bool include_rack_field(const GcgWriteOptions& options, size_t turn_idx) {
  if (!options.rack_before_fields.empty()) {
    return turn_idx < options.rack_before_fields.size() &&
           options.rack_before_fields[turn_idx].has_value();
  }
  return options.include_rack_before.empty() ||
         (turn_idx < options.include_rack_before.size() && options.include_rack_before[turn_idx]);
}

std::string rack_field(const GameLog& log, const GcgWriteOptions& options, size_t turn_idx) {
  if (!options.rack_before_fields.empty() && turn_idx < options.rack_before_fields.size() &&
      options.rack_before_fields[turn_idx].has_value()) {
    return *options.rack_before_fields[turn_idx];
  }
  return log.turns[turn_idx].rack_before.to_string();
}

void maybe_emit_post_event_racks(std::ostringstream& o, const GcgWriteOptions& options,
                                 size_t turn_idx) {
  if (turn_idx >= options.post_event_racks.size()) return;
  const auto& racks = options.post_event_racks[turn_idx];
  if (racks.rack1.has_value()) o << "#Rack1 " << *racks.rack1 << "\n";
  if (racks.rack2.has_value()) o << "#Rack2 " << *racks.rack2 << "\n";
}

}  // namespace

std::string game_log_to_gcg(const GameLog& log) { return game_log_to_gcg(log, GcgWriteOptions{}); }

std::string game_log_to_gcg(const GameLog& log, const GcgWriteOptions& options) {
  std::ostringstream o;

  std::array<std::string, 2> nick = {nickify(log.player_names[0]), nickify(log.player_names[1])};
  if (nick[0] == nick[1]) {
    nick[0] += "1";
    nick[1] += "2";
  }

  o << "#character-encoding UTF-8\n";
  if (options.lexicon_name.has_value() && !options.lexicon_name->empty()) {
    o << "#lexicon " << *options.lexicon_name << "\n";
  }
  o << "#player1 " << nick[0] << " " << log.player_names[0] << "\n";
  o << "#player2 " << nick[1] << " " << log.player_names[1] << "\n";
  if (options.initial_rack1.has_value()) o << "#Rack1 " << *options.initial_rack1 << "\n";
  if (options.initial_rack2.has_value()) o << "#Rack2 " << *options.initial_rack2 << "\n";
  for (const std::string& note : options.notes) {
    if (!note.empty()) o << "#note " << note << "\n";
  }

  // Replay the board so each play can be rendered relative to the tiles already
  // down, and track each player's last cumulative score for the end-game lines.
  Board board;
  std::array<int, 2> last_cumulative = {0, 0};
  for (size_t turn_idx = 0; turn_idx < log.turns.size(); ++turn_idx) {
    const TurnRecord& t = log.turns[turn_idx];
    const Move& m = t.move;
    const int cumulative = t.cumulative_scores[t.player];
    last_cumulative[t.player] = cumulative;
    const bool include_rack = include_rack_field(options, turn_idx);
    const std::string rack = include_rack ? rack_field(log, options, turn_idx) : std::string{};

    o << ">" << nick[t.player] << ": ";
    switch (m.type()) {
      case MoveType::PLAY:
        if (include_rack) {
          o << rack << " ";
        }
        o << position(board, m) << " " << played_word(board, m) << " +" << m.score() << " "
          << cumulative << "\n";
        board.apply(m);
        maybe_emit_post_event_racks(o, options, turn_idx);
        break;
      case MoveType::EXCHANGE:
        if (include_rack) {
          o << rack << " ";
        }
        if (turn_idx < options.exchange_fields.size() &&
            options.exchange_fields[turn_idx].has_value()) {
          o << "-" << *options.exchange_fields[turn_idx] << " +0 " << cumulative << "\n";
        } else {
          o << "-" << exchanged_tiles(m) << " +0 " << cumulative << "\n";
        }
        maybe_emit_post_event_racks(o, options, turn_idx);
        break;
      case MoveType::PASS:
        if (include_rack) {
          o << rack << " ";
        }
        o << "- +0 " << cumulative << "\n";
        maybe_emit_post_event_racks(o, options, turn_idx);
        break;
    }
  }

  // End-of-game rack adjustments. A player who went out gains the value of the
  // opponent's leftover tiles (END_RACK_PTS); a player left holding tiles loses
  // their value (END_RACK_PENALTY). Emit the positive adjustment first.
  for (int pass = 0; pass < 2; ++pass) {
    for (int p = 0; p < 2; ++p) {
      const int delta = log.final_scores[p] - last_cumulative[p];
      const bool positive = delta > 0;
      if (delta == 0 || positive != (pass == 0)) continue;
      if (positive) {
        // Tiles scored are the *other* player's leftovers.
        o << ">" << nick[p] << ": (" << log.final_racks[1 - p].to_string() << ") +" << delta << " "
          << log.final_scores[p] << "\n";
      } else {
        const std::string rack = log.final_racks[p].to_string();
        o << ">" << nick[p] << ": " << rack << " (" << rack << ") -" << -delta << " "
          << log.final_scores[p] << "\n";
      }
    }
  }

  return o.str();
}

void write_game_log_gcg(const GameLog& log, std::ostream& out) { out << game_log_to_gcg(log); }

void write_game_log_gcg(const GameLog& log, std::ostream& out, const GcgWriteOptions& options) {
  out << game_log_to_gcg(log, options);
}

}  // namespace scribblez
