#include "scribblez/hasty_equity.h"

#include "scribblez/glyph.h"

#include <boost/json.hpp>

#include <algorithm>
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

struct LeaveRunEntry {
  Rack leave;
  int move_index;
};

struct LeaveRunEntryLess {
  bool operator()(const LeaveRunEntry& a, const LeaveRunEntry& b) const {
    return a.leave < b.leave;
  }
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

double endgame_adjustment_rack(const Rack& leave, const Rack& opp_rack, int bag_size) {
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

double HastyEquity::equity(const Move& move, const Board& board, int bag_size,
                           const Rack& opp_rack) const {
  std::vector<Move> one{move};
  auto vals = equities(one, board, bag_size, opp_rack);
  return vals.empty() ? 0.0 : vals[0];
}

std::vector<double> HastyEquity::equities(const std::vector<Move>& moves, const Board& board,
                                          int bag_size, const Rack& opp_rack) const {
  if (!ready_) throw std::runtime_error("HastyEquity::init() was not called");

  std::vector<double> out(moves.size(), 0.0);
  if (moves.empty()) return out;

  std::vector<LeaveRunEntry> entries;
  entries.reserve(moves.size());
  for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
    entries.push_back(LeaveRunEntry{moves[i].leave(), i});
  }

  std::sort(entries.begin(), entries.end(), LeaveRunEntryLess());

  size_t run_start = 0;
  while (run_start < entries.size()) {
    size_t run_end = run_start + 1;
    while (run_end < entries.size() && entries[run_end].leave == entries[run_start].leave) {
      ++run_end;
    }

    double lv =
      (bag_size > 0) ? static_cast<double>(leave_values_.lookup(entries[run_start].leave)) : 0.0;
    double eg = endgame_adjustment_rack(entries[run_start].leave, opp_rack, bag_size);

    for (size_t k = run_start; k < run_end; ++k) {
      int idx = entries[k].move_index;
      out[idx] = static_cast<double>(moves[idx].score()) + lv +
                 opening_adjustment(moves[idx], board) +
                 peg_adjustment(moves[idx], bag_size, peg_table_) + eg;
    }
    run_start = run_end;
  }
  return out;
}

}  // namespace scribblez
