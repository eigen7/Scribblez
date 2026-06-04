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
// number first), "D8" for a vertical one (column letter first).
std::string position(const Move& m) {
  std::string col(1, static_cast<char>('A' + m.start_col));
  std::string row = std::to_string(m.start_row + 1);
  return m.horizontal ? row + col : col + row;
}

// The played word in GCG form: a '.' for every square already occupied before
// this move (played through), and a lowercase letter for a designated blank.
std::string played_word(const Board& board_before, const Move& m) {
  std::string out;
  for (size_t i = 0; i < m.main_word.size(); ++i) {
    int r = m.horizontal ? m.start_row : m.start_row + static_cast<int>(i);
    int c = m.horizontal ? m.start_col + static_cast<int>(i) : m.start_col;
    if (!is_empty(board_before.at(r, c))) {
      out.push_back('.');
      continue;
    }
    char ch = m.main_word[i];
    bool blank = false;
    for (const auto& t : m.tiles) {
      if (t.row == r && t.col == c) {
        blank = t.glyph.is_blank();
        break;
      }
    }
    out.push_back(blank ? static_cast<char>(ch - 'A' + 'a') : ch);
  }
  return out;
}

std::string exchanged_tiles(const Move& m) {
  std::string s;
  for (Tile t : m.exchanged) s.push_back(t.to_char());
  return s;
}

}  // namespace

std::string game_log_to_gcg(const GameLog& log) {
  std::ostringstream o;

  std::array<std::string, 2> nick = {nickify(log.player_names[0]), nickify(log.player_names[1])};
  if (nick[0] == nick[1]) {
    nick[0] += "1";
    nick[1] += "2";
  }

  o << "#character-encoding UTF-8\n";
  o << "#player1 " << nick[0] << " " << log.player_names[0] << "\n";
  o << "#player2 " << nick[1] << " " << log.player_names[1] << "\n";

  // Replay the board so each play can be rendered relative to the tiles already
  // down, and track each player's last cumulative score for the end-game lines.
  Board board;
  std::array<int, 2> last_cumulative = {0, 0};
  for (const TurnRecord& t : log.turns) {
    const Move& m = t.move;
    const std::string rack = t.rack_before.to_string();
    const int cumulative = t.cumulative_scores[t.player];
    last_cumulative[t.player] = cumulative;

    o << ">" << nick[t.player] << ": ";
    switch (m.type) {
      case MoveType::PLAY:
        o << rack << " " << position(m) << " " << played_word(board, m) << " +" << m.score << " "
          << cumulative << "\n";
        board.apply(m.tiles);
        break;
      case MoveType::EXCHANGE:
        o << rack << " -" << exchanged_tiles(m) << " +0 " << cumulative << "\n";
        break;
      case MoveType::PASS:
        o << rack << " - +0 " << cumulative << "\n";
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

}  // namespace scribblez
