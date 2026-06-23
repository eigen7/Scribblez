#include "scribblez/macondo_bot.h"

#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/move.h"
#include "scribblez/movegen.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

// Canonical total order on distinct PLAY moves (orientation, anchor, placed
// squares, then placed glyphs) -- used only to break exact-equity ties.
bool move_order_less(const Move& a, const Move& b) {
  if (a.horizontal() != b.horizontal()) return a.horizontal() < b.horizontal();
  if (a.start() != b.start()) return a.start() < b.start();
  if (a.square_mask() != b.square_mask()) return a.square_mask() < b.square_mask();
  const int na = a.num_glyphs(), nb = b.num_glyphs();
  if (na != nb) return na < nb;
  for (int i = 0; i < na; ++i) {
    if (a.glyph(i).code() != b.glyph(i).code()) return a.glyph(i).code() < b.glyph(i).code();
  }
  return false;
}

}  // namespace

bool hasty_move_better(double eq_a, const Move& a, double eq_b, const Move& b) {
  if (eq_a != eq_b) return eq_a > eq_b;
  return move_order_less(a, b);
}

Move hasty_best_move_reference(const MoveRequest& req) {
  const std::vector<Move> plays = generate_legal_plays(req);
  const HastyEquity& eq = HastyEquity::instance();
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);
  bool have = false;
  Move best;
  double best_eq = 0.0;
  for (const Move& m : plays) {
    const double e = eq.equity(m, req.board, req.bag_size, req.opp_rack, leaves);
    if (!have || hasty_move_better(e, m, best_eq, best)) {
      best = m;
      best_eq = e;
      have = true;
    }
  }
  return have ? best : Move::pass();
}

HastyBotAgent::HastyBotAgent(int thread_id, const std::string& name) : Agent(thread_id, name) {}

Move HastyBotAgent::make_move(const MoveRequest& req) {
  const HastyEquity& eq = HastyEquity::instance();
  ShadowMoveGen smg(req.board, req.dict);
  const std::vector<ShadowAnchor> anchors = smg.anchors(req.my_rack);
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);

  const int rack_size = req.my_rack.size();
  std::array<double, RACK_SIZE + 1> leave_by_size;
  eq.best_leaves_by_size(req.my_rack, leave_by_size);
  const bool endgame = req.bag_size <= 0;
  const double endgame_term = endgame ? 2.0 * static_cast<double>(req.opp_rack.point_value()) : 0.0;

  // Tight admissible equity bound for an anchor: for a play of e tiles, equity =
  // score + leave(size rack-e) + opening(<=0) + peg(e), so pairing each e-tile
  // score bound with the best leave of the complementary size bounds equity.
  auto equity_bound = [&](const ShadowAnchor& a) {
    double best = -1e18;
    for (int e = 1; e <= rack_size && e <= kMaxPlayTiles; ++e) {
      const int sb = a.score_bound_by_size[e];
      if (sb < 0) continue;  // no play places e tiles here
      double term = static_cast<double>(sb);
      term +=
        endgame ? endgame_term : (leave_by_size[rack_size - e] + eq.peg_for_tiles(e, req.bag_size));
      best = std::max(best, term);
    }
    return best;
  };

  // Rank anchors by equity bound (descending) so we can stop early.
  std::vector<std::pair<double, int>> order;
  order.reserve(anchors.size());
  for (int i = 0; i < static_cast<int>(anchors.size()); ++i) {
    order.emplace_back(equity_bound(anchors[i]), i);
  }
  std::sort(order.begin(), order.end(),
            [](const auto& x, const auto& y) { return x.first > y.first; });

  bool have = false;
  Move best;
  double best_eq = 0.0;
  std::vector<Move> moves;
  for (const auto& [bound, i] : order) {
    if (have && bound < best_eq) break;  // no remaining anchor can beat the best move
    moves.clear();
    smg.generate_anchor(anchors[i], req.my_rack, moves);
    for (const Move& m : moves) {
      const double e = eq.equity(m, req.board, req.bag_size, req.opp_rack, leaves);
      if (!have || hasty_move_better(e, m, best_eq, best)) {
        best = m;
        best_eq = e;
        have = true;
      }
    }
  }
  return have ? best : Move::pass();
}

std::unique_ptr<HastyBotAgent> HastyBotAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name) {
  namespace po = boost::program_options;

  // No agent-specific options at present. The equity tables (leaves +
  // pre-endgame) are process-wide: a play_game --leaves-file overrides them,
  // but otherwise we lazily load Macondo's defaults for the active lexicon, so
  // running a HastyBot never requires extra command-line flags.
  po::options_description desc("hastybot options");
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=hastybot options: ") + e.what());
  }

  HastyEquity::ensure_initialized(Lexicon::instance().name());
  return std::make_unique<HastyBotAgent>(thread_id, name);
}

std::string HastyBotAgent::options_help() {
  return "  In-process HastyBot: enumerates all legal plays and picks the one\n"
         "  with highest static equity (score + leave value + adjustments).\n"
         "  Options: (none)\n";
}

}  // namespace scribblez
