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

// Root of Macondo's per-lexicon strategy data (leaves + pre-endgame tables),
// from which HastyBot's default file locations are derived.
constexpr char kStrategyRoot[] = "/workspace/mount/macondo/data/strategy";

// ---- leave computation --------------------------------------------------

// ---- opening adjustment -------------------------------------------------

// 2LS column (or row) positions on a standard 15x15 board that adjoin the
// star; identical set for horizontal and vertical first plays.
bool is_penalised_position(int pos) { return pos == 2 || pos == 6 || pos == 8 || pos == 12; }

// A kVowelPenalty charge per vowel that lands on a 2LS square adjacent to the
// star on an empty-board opening play. Matches Maven / Macondo's heuristic.
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

// ---- pre-endgame adjustment ---------------------------------------------

double peg_adjustment(const Move& move, int bag_size, const std::vector<double>& peg_table) {
  if (bag_size <= 0 || peg_table.empty()) return 0.0;
  int bag_after = bag_size - move.num_glyphs() + 7;
  if (bag_after < 0 || size_t(bag_after) >= peg_table.size()) return 0.0;
  return peg_table[size_t(bag_after)];
}

// ---- endgame adjustment -------------------------------------------------

double endgame_adjustment(int leave_point_value, bool leave_empty, const Rack& opp_rack,
                          int bag_size) {
  if (bag_size > 0) return 0.0;
  if (!leave_empty) return -2.0 * leave_point_value - 10.0;
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

  // Derive the leave directly from the rack and the move's played tiles; the
  // single-move path is not perf-critical, so it skips the per-turn table.
  Rack leave = my_rack;
  for (int i = 0; i < move.num_glyphs(); ++i) leave.remove(move.glyph(i).rack_tile());

  double lv = (bag_size > 0) ? double(leave_values_.lookup(leave)) : 0.0;
  double eg = endgame_adjustment(leave.point_value(), leave.empty(), opp_rack, bag_size);
  return double(move.score()) + lv + opening_adjustment(move, board) +
         peg_adjustment(move, bag_size, peg_table_) + eg;
}

TurnLeaves HastyEquity::turn_leaves(const Rack& my_rack) const {
  // The greedy HastyBot path prices every leave through this table rather
  // than equity(), so it needs the same guard: an unloaded table silently
  // values every leave at 0, which is a different bot.
  if (!ready_) throw util::Exception("HastyEquity::init() was not called");
  return TurnLeaves(my_rack, leave_values_);
}

double HastyEquity::equity(const Move& move, const Board& board, int bag_size, const Rack& opp_rack,
                           TurnLeaves& leaves) const {
  const uint8_t mask = leaves.mask_for(move);
  const double lv = (bag_size > 0) ? leaves.value(mask) : 0.0;
  const double eg = endgame_adjustment(leaves.point_value(mask), mask == 0, opp_rack, bag_size);
  return double(move.score()) + lv + opening_adjustment(move, board) +
         peg_adjustment(move, bag_size, peg_table_) + eg;
}

namespace {

// Recurse over the rack's distinct tile types, choosing how many of each to
// KEEP, and track the max leave value by leave size in `best[size]`.
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
