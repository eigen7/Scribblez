#include "scribblez/hasty_equity.h"

#include "scribblez/glyph.h"

#include <boost/json.hpp>

#include <array>
#include <bitset>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scribblez {

namespace {

// Root of Macondo's per-lexicon strategy data (leaves + pre-endgame tables),
// from which HastyBot's default file locations are derived.
constexpr char kStrategyRoot[] = "/workspace/mount/macondo/data/strategy";

// ---- leave computation --------------------------------------------------

// Precomputes leave values for every subset of the mover's rack within a
// single turn, indexed by the sorted-rack bitmask carried on each Move.
// Values are filled lazily and cached, so each distinct leave is looked up
// at most once per turn.
class TurnLeaves {
 public:
  TurnLeaves(const Rack& rack, const LeaveValues& lv) : lv_(lv), size_(rack.size()) {
    const auto& t = rack.tiles();
    for (int i = 0; i < size_; ++i) tile_of_bit_[i] = t[i];
  }

  double value(uint8_t mask) {
    ensure(mask);
    return static_cast<double>(value_[mask]);
  }

  int point_value(uint8_t mask) {
    ensure(mask);
    return pv_[mask];
  }

 private:
  void ensure(uint8_t mask) {
    if (computed_[mask]) return;
    Rack leave;
    for (int i = 0; i < size_; ++i)
      if (mask & (1u << i)) leave.add(tile_of_bit_[i]);
    value_[mask] = lv_.lookup(leave);
    pv_[mask] = static_cast<int16_t>(leave.point_value());
    computed_[mask] = true;
  }

  const LeaveValues& lv_;
  int size_;
  std::array<Tile, RACK_SIZE> tile_of_bit_{};
  std::array<float, 128> value_{};
  std::array<int16_t, 128> pv_{};
  std::bitset<128> computed_{};
};

// ---- opening adjustment -------------------------------------------------

// 2LS column (or row) positions on a standard 15x15 board that adjoin the
// star; identical set for horizontal and vertical first plays.
bool is_penalised_position(int pos) { return pos == 2 || pos == 6 || pos == 8 || pos == 12; }

// -0.7 penalty per vowel that lands on a 2LS square adjacent to the star on
// an empty-board opening play.  Matches Maven / Macondo's heuristic.
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
  if (bag_after < 0 || static_cast<size_t>(bag_after) >= peg_table.size()) return 0.0;
  return peg_table[static_cast<size_t>(bag_after)];
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
  if (!ready_) throw std::runtime_error("HastyEquity::init() was not called");

  // Derive the leave directly from the rack and the move's played tiles; the
  // single-move path is not perf-critical, so it skips the per-turn table.
  Rack leave = my_rack;
  for (int i = 0; i < move.num_glyphs(); ++i) leave.remove(move.glyph(i).rack_tile());

  double lv = (bag_size > 0) ? static_cast<double>(leave_values_.lookup(leave)) : 0.0;
  double eg = endgame_adjustment(leave.point_value(), leave.empty(), opp_rack, bag_size);
  return static_cast<double>(move.score()) + lv + opening_adjustment(move, board) +
         peg_adjustment(move, bag_size, peg_table_) + eg;
}

std::vector<double> HastyEquity::equities(const std::vector<Move>& moves, const Board& board,
                                          int bag_size, const Rack& opp_rack,
                                          const Rack& my_rack) const {
  if (!ready_) throw std::runtime_error("HastyEquity::init() was not called");

  std::vector<double> out(moves.size(), 0.0);
  if (moves.empty()) return out;

  TurnLeaves leaves(my_rack, leave_values_);
  for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
    const Move& m = moves[i];
    const uint8_t mask = m.leave_mask();
    const double lv = (bag_size > 0) ? leaves.value(mask) : 0.0;
    const double eg = endgame_adjustment(leaves.point_value(mask), mask == 0, opp_rack, bag_size);
    out[i] = static_cast<double>(m.score()) + lv + opening_adjustment(m, board) +
             peg_adjustment(m, bag_size, peg_table_) + eg;
  }
  return out;
}

}  // namespace scribblez
