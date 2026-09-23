#include "lexicon/hasty_equity.h"

#include "game/glyph.h"
#include "util/exception.h"

#include <boost/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

constexpr char kStrategyRoot[] = "/workspace/mount/macondo/data/strategy";

// Squares on the center line whose perpendicular neighbors are DLS (rows and
// columns 6 and 8 have DLS at 2, 6, 8, 12), so a vowel there sets up the
// opponent's parallel play across a DLS. The same for either direction.
bool is_penalised_position(int pos) { return pos == 2 || pos == 6 || pos == 8 || pos == 12; }

// Maven's and Macondo's opening heuristic: kVowelPenalty per vowel an opening
// play puts on one of those squares.
double opening_adjustment(const Move& move, const Board& board) {
  if (!board.empty_board()) return 0.0;
  if (move.type() != MoveType::PLAY) return 0.0;

  static constexpr double kVowelPenalty = -0.7;
  double penalty = 0.0;
  uint16_t mask = move.square_mask();
  int gi = 0;
  for (int pos = 0; mask; ++pos, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    Glyph g = move.glyph(gi++);
    if (is_penalised_position(pos) && g.is_vowel()) penalty += kVowelPenalty;
  }
  return penalty;
}

// With the bag empty: going out collects twice the opponent's rack's face
// value; otherwise the mover pays twice its own leave's, plus 10.
double endgame_adjustment(int leave_point_value, bool leave_empty, const Rack& opp_rack,
                          int bag_size) {
  if (bag_size > 0) return 0.0;
  if (!leave_empty) return -2.0 * leave_point_value - 10.0;
  return 2.0 * opp_rack.point_value();
}

// Returns an empty table (disabling the adjustment) if the file is missing or
// isn't a JSON array.
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

void HastyEquity::ensure_initialized(const std::string& lexicon) {
  if (instance().ready_) return;
  init(default_leaves_path(lexicon), default_peg_path());
}

std::string HastyEquity::default_leaves_path(const std::string& lexicon) {
  return std::string(kStrategyRoot) + "/" + lexicon + "/leaves.klv2";
}

std::string HastyEquity::default_peg_path() {
  return std::string(kStrategyRoot) + "/default/preendgame.json";
}

double HastyEquity::equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                           const Rack& my_rack) const {
  if (!ready_) throw util::Exception("HastyEquity::init() was not called");

  // One move at a time isn't hot, so skip building a TurnLeaves.
  Rack leave = my_rack;
  for (int i = 0; i < move.num_glyphs(); ++i) leave.remove(move.glyph(i).rack_tile());

  double lv = (bag_size > 0) ? double(leave_values_.lookup(leave)) : 0.0;
  double eg = endgame_adjustment(leave.point_value(), leave.empty(), opp_rack, bag_size);
  return double(move.score()) + lv + opening_adjustment(move, board) +
         peg_for_tiles(move.num_glyphs(), bag_size) + eg;
}

TurnLeaves HastyEquity::turn_leaves(const Rack& my_rack) const {
  // Guarded like equity(): an unloaded table would silently value every leave
  // at 0, which is a different bot.
  if (!ready_) throw util::Exception("HastyEquity::init() was not called");
  return TurnLeaves(my_rack, leave_values_);
}

double HastyEquity::equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                           TurnLeaves& leaves) const {
  const uint8_t mask = leaves.mask_for(move);
  const double lv = (bag_size > 0) ? leaves.value(mask) : 0.0;
  const double eg = endgame_adjustment(leaves.point_value(mask), mask == 0, opp_rack, bag_size);
  return double(move.score()) + lv + opening_adjustment(move, board) +
         peg_for_tiles(move.num_glyphs(), bag_size) + eg;
}

namespace {

// Enumerates sub-leaves by choosing how many of each distinct tile to keep,
// recording the best value per leave size in `best`.
void enum_sub_leaves(const std::vector<std::pair<Tile, int>>& types, size_t i, Rack& leave,
                     int kept, const LeaveValues& lv, std::array<double, RACK_SIZE + 1>& best) {
  if (i == types.size()) {
    best[kept] = std::max(best[kept], double(lv.lookup(leave)));
    return;
  }
  const Tile t = types[i].first;
  const int cnt = types[i].second;
  for (int k = 0; k <= cnt; ++k) {
    enum_sub_leaves(types, i + 1, leave, kept + k, lv, best);
    if (k < cnt) leave.add(t);
  }
  for (int k = 0; k < cnt; ++k) leave.remove(t);
}

}  // namespace

void HastyEquity::best_leaves_by_size(const Rack& my_rack,
                                      std::array<double, RACK_SIZE + 1>& out) const {
  if (!ready_) throw util::Exception("HastyEquity::init() was not called");
  out.fill(-1e18);
  std::vector<std::pair<Tile, int>> types;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    const int c = my_rack.count(L);
    if (c > 0) types.emplace_back(L, c);
  }
  const int b = my_rack.blanks();
  if (b > 0) types.emplace_back(BLANK, b);
  Rack leave;
  enum_sub_leaves(types, 0, leave, 0, leave_values_, out);
}

double HastyEquity::peg_for_tiles(int tiles_played, int bag_size) const {
  if (bag_size <= 0) return 0.0;
  const int bag_after = bag_size - tiles_played + 7;
  if (bag_after < 0 || size_t(bag_after) >= peg_table_.size()) return 0.0;
  return peg_table_[size_t(bag_after)];
}

std::vector<double> HastyEquity::equities(const std::vector<Move>& moves, const Board& board,
                                          int bag_size, const Rack& opp_rack,
                                          const Rack& my_rack) const {
  if (!ready_) throw util::Exception("HastyEquity::init() was not called");

  std::vector<double> out(moves.size(), 0.0);
  if (moves.empty()) return out;

  TurnLeaves leaves = turn_leaves(my_rack);
  for (int i = 0; i < int(moves.size()); ++i) {
    out[i] = equity(moves[i], board, bag_size, opp_rack, leaves);
  }
  return out;
}

}  // namespace scribblez
