#include "scribblez/hasty_equity.h"

#include "scribblez/glyph.h"
#include "scribblez/tile.h"

#include <boost/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scribblez {

namespace {

// ---- leave computation --------------------------------------------------

// Subtract the tiles placed by `move` from `rack`, yielding the leave.
TileCounts compute_leave(const Rack& rack, const Move& move) {
  TileCounts leave = rack.counts();
  for (const Glyph& g : move.glyphs) {
    if (g.is_empty()) break;
    leave.remove(g.rack_tile());
  }
  return leave;
}

// ---- leave equity -------------------------------------------------------

double leave_equity(const TileCounts& leave, int bag_size, const LeaveValues& lv) {
  if (bag_size <= 0) return 0.0;
  return static_cast<double>(lv.lookup(leave));
}

// ---- opening adjustment -------------------------------------------------

// Vowel tile indices (A=0, E=4, I=8, O=14, U=20).
bool is_vowel(int letter_index) {
  switch (letter_index) {
    case 0:
    case 4:
    case 8:
    case 14:
    case 20:
      return true;
    default:
      return false;
  }
}

// 2LS column (or row) positions on a standard 15x15 board that adjoin the
// star; identical set for horizontal and vertical first plays.
bool is_penalised_position(int pos) { return pos == 2 || pos == 6 || pos == 8 || pos == 12; }

// -0.7 penalty per vowel that lands on a 2LS square adjacent to the star on
// an empty-board opening play.  Matches Maven / Macondo's heuristic.
double opening_adjustment(const Move& move, const Board& board) {
  if (!board.empty_board()) return 0.0;
  if (move.type != MoveType::PLAY) return 0.0;

  static constexpr double kVowelPenalty = -0.7;
  double penalty = 0.0;
  int start = move.horizontal ? move.start_col : move.start_row;
  int gi = 0;
  for (const Glyph& g : move.glyphs) {
    if (g.is_empty()) break;
    if (g.has_letter() && !g.is_blank() && is_penalised_position(start + gi) &&
        is_vowel(g.letter().index())) {
      penalty += kVowelPenalty;
    }
    ++gi;
  }
  return penalty;
}

// ---- pre-endgame adjustment ---------------------------------------------

double peg_adjustment(const Move& move, int bag_size, const std::vector<double>& peg_table) {
  if (bag_size <= 0 || peg_table.empty()) return 0.0;
  int bag_after = bag_size - move.num_glyphs() + 7;
  if (bag_after < 0 || static_cast<size_t>(bag_after) >= peg_table.size()) return 0.0;
  return peg_table[static_cast<size_t>(bag_after)];
}

// ---- endgame adjustment -------------------------------------------------

// When the bag is empty: penalise non-out plays by leave tile value;
// reward out plays by opponent rack value.  Matches Macondo's endgameAdjustment.
double endgame_adjustment(const TileCounts& leave, const Rack& opp_rack, int bag_size) {
  if (bag_size > 0) return 0.0;
  if (!leave.empty()) return -2.0 * leave.point_value() - 10.0;
  return 2.0 * opp_rack.point_value();
}

// ---- PEG JSON loader ----------------------------------------------------

std::vector<double> load_peg_table(const std::string& path) {
  if (path.empty()) return {};
  std::ifstream in(path);
  if (!in) return {};

  std::ostringstream buf;
  buf << in.rdbuf();

  boost::json::error_code ec;
  auto val = boost::json::parse(buf.str(), ec);
  if (ec || !val.is_array()) return {};

  std::vector<double> table;
  for (const auto& elem : val.as_array()) table.push_back(elem.as_double());
  return table;
}

}  // namespace

// -------------------------------------------------------------------------

HastyEquity& HastyEquity::instance() {
  static HastyEquity inst;
  return inst;
}

void HastyEquity::init(const std::string& klv2_path, const std::string& peg_json_path) {
  auto& inst = instance();
  inst.leave_values_ = LeaveValues::load(klv2_path);
  inst.peg_table_ = load_peg_table(peg_json_path);
  inst.ready_ = true;
}

double HastyEquity::equity(const Move& move, const Board& board, int bag_size, const Rack& my_rack,
                           const Rack& opp_rack) const {
  if (!ready_) throw std::runtime_error("HastyEquity::init() was not called");

  TileCounts leave = compute_leave(my_rack, move);

  return static_cast<double>(move.score) + leave_equity(leave, bag_size, leave_values_) +
         opening_adjustment(move, board) + peg_adjustment(move, bag_size, peg_table_) +
         endgame_adjustment(leave, opp_rack, bag_size);
}

}  // namespace scribblez
