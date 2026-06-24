#include "scribblez/macondo_bot.h"

#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/move.h"
#include "scribblez/movegen.h"
#include "scribblez/word_map.h"

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

namespace {

// The per-move inputs the equity bounds depend on.
struct BoundInputs {
  int rack_size;
  bool endgame;
  double endgame_term;
  int bag_size;
  std::array<double, RACK_SIZE + 1> leave_by_size;
};

BoundInputs make_bound_inputs(const MoveRequest& req, const HastyEquity& eq) {
  BoundInputs in;
  in.rack_size = req.my_rack.size();
  in.endgame = req.bag_size <= 0;
  in.endgame_term = in.endgame ? 2.0 * static_cast<double>(req.opp_rack.point_value()) : 0.0;
  in.bag_size = req.bag_size;
  eq.best_leaves_by_size(req.my_rack, in.leave_by_size);
  return in;
}

// Admissible equity bound for placing `placed` tiles whose raw score is at most
// `score_bound`: equity = score + leave(rack - placed) + opening(<=0) + peg, so
// adding the best leave of the complementary size (or the endgame term) bounds it.
double equity_bound(const HastyEquity& eq, const BoundInputs& in, int placed, int score_bound) {
  const double leave_term =
    in.endgame ? in.endgame_term
               : in.leave_by_size[in.rack_size - placed] + eq.peg_for_tiles(placed, in.bag_size);
  return static_cast<double>(score_bound) + leave_term;
}

// Best move found so far, with hasty tie-break.
struct BestMove {
  bool have = false;
  Move move;
  double eq = 0.0;
};

// Fold each candidate play's equity into the running best.
void consider_moves(const HastyEquity& eq, const MoveRequest& req, TurnLeaves& leaves,
                    const std::vector<Move>& moves, BestMove& bm) {
  for (const Move& m : moves) {
    const double e = eq.equity(m, req.board, req.bag_size, req.opp_rack, leaves);
    if (!bm.have || hasty_move_better(e, m, bm.eq, bm.move)) {
      bm.move = m;
      bm.eq = e;
      bm.have = true;
    }
  }
}

// Indices into `units` paired with their equity bound, sorted best-first so the
// search can stop as soon as a bound can no longer beat the best move found.
std::vector<std::pair<double, int>> rank_by_bound(const std::vector<double>& bounds) {
  std::vector<std::pair<double, int>> order;
  order.reserve(bounds.size());
  for (int i = 0; i < static_cast<int>(bounds.size()); ++i) order.emplace_back(bounds[i], i);
  std::sort(order.begin(), order.end(),
            [](const auto& x, const auto& y) { return x.first > y.first; });
  return order;
}

// The GADDAG HastyBot path: bound each anchor by its best tile-count, then
// generate anchors best-first with the GADDAG.
Move hasty_best_move_gaddag(const MoveRequest& req) {
  const HastyEquity& eq = HastyEquity::instance();
  ShadowMoveGen smg(req.board, req.dict);
  const std::vector<ShadowAnchor> anchors = smg.anchors(req.my_rack);
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);
  const BoundInputs in = make_bound_inputs(req, eq);

  std::vector<double> bounds(anchors.size());
  for (size_t i = 0; i < anchors.size(); ++i) {
    double best = -1e18;
    for (int e = 1; e <= in.rack_size && e <= kMaxPlayTiles; ++e) {
      if (anchors[i].score_bound_by_size[e] >= 0) {
        best = std::max(best, equity_bound(eq, in, e, anchors[i].score_bound_by_size[e]));
      }
    }
    bounds[i] = best;
  }

  BestMove bm;
  std::vector<Move> moves;
  for (const auto& [bound, i] : rank_by_bound(bounds)) {
    if (bm.have && bound < bm.eq) break;  // no remaining anchor can beat the best move
    moves.clear();
    smg.generate_anchor(anchors[i], req.my_rack, moves);
    consider_moves(eq, req, leaves, moves, bm);
  }
  return bm.have ? bm.move : Move::pass();
}

// The WordMap HastyBot path: bound each individual extent, then generate extents
// best-first with WordMap lookups. The finer (per-extent) granularity lets the
// early-exit prune whole extents, the regime where WordMap beats the GADDAG.
//
// Racks holding a blank fall back to the GADDAG path: the WordMap is blank-free,
// and resolving blanks by enumerating their letters at query time is too slow
// (MAGPIE uses precomputed blank tables instead). Blanks are a minority of racks,
// so self-play still gets the WordMap speedup on the common (blank-free) case.
Move hasty_best_move_wmp_impl(const MoveRequest& req, const WordMap& wm) {
  if (req.my_rack.counts().blanks() > 0) return hasty_best_move_gaddag(req);
  const HastyEquity& eq = HastyEquity::instance();
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);
  const BoundInputs in = make_bound_inputs(req, eq);

  WmpSubracks subracks;
  int rack_tiles = 0;
  wmp_rack_subracks(req.my_rack, subracks, rack_tiles);

  // One pass over the rack's subracks computes three things MAGPIE derives
  // together in its move generator:
  //   - sub_terms[size][j]: each subrack's equity contribution (leave value +
  //     pre-endgame term), for the per-subrack generation cutoff.
  //   - has_word[size]: whether any size-k subrack forms a k-letter word -- the
  //     shadow's nonplaythrough existence prune (handed to extents()).
  //   - np_best_leave_term[size]: the best leave term among WORD-FORMING subracks
  //     of that size (MAGPIE's nonplaythrough_best_leave_values), a tighter leave
  //     bound for nonplaythrough extents than the global best-leave-by-size.
  // The word-existence lookups here are shadow-time checks, not generation probes.
  std::array<std::vector<double>, kMaxPlayTiles + 1> sub_terms;
  std::array<bool, kMaxPlayTiles + 1> has_word{};
  std::array<double, kMaxPlayTiles + 1> np_best_leave_term;
  np_best_leave_term.fill(-1e18);
  const TileCounts& rack_counts = req.my_rack.counts();
  for (int size = 1; size <= rack_tiles && size <= kMaxPlayTiles; ++size) {
    sub_terms[size].reserve(subracks[size].size());
    for (const BitRack& sub : subracks[size]) {
      double term = in.endgame_term;
      if (!in.endgame) {
        Rack leave;
        for (Tile L = Tile::of(0); L < 26; ++L) {
          const int c = rack_counts.count(L) - sub.get(L.index());
          for (int k = 0; k < c; ++k) leave.add(L);
        }
        term = eq.leave_value(leave) + eq.peg_for_tiles(size, in.bag_size);
      }
      sub_terms[size].push_back(term);
      if (size >= 2 && wm.lookup(size, sub).count > 0) {
        has_word[size] = true;
        if (term > np_best_leave_term[size]) np_best_leave_term[size] = term;
      }
    }
  }

  ShadowMoveGen smg(req.board, req.dict);
  const std::vector<ShadowExtent> extents = smg.extents(req.my_rack, &wm, &has_word);

  std::vector<double> bounds(extents.size());
  for (size_t i = 0; i < extents.size(); ++i) {
    const ShadowExtent& e = extents[i];
    // A nonplaythrough extent (word == its placed tiles) is bounded by the best
    // leave among the size-placed subracks that actually form a word; a playthrough
    // extent keeps the looser global best-leave-by-size.
    double leave_term;
    if (e.length == e.placed && has_word[e.placed]) {
      leave_term = np_best_leave_term[e.placed];
    } else if (in.endgame) {
      leave_term = in.endgame_term;
    } else {
      leave_term =
        in.leave_by_size[in.rack_size - e.placed] + eq.peg_for_tiles(e.placed, in.bag_size);
    }
    bounds[i] = static_cast<double>(e.score_bound) + leave_term;
  }

  BestMove bm;
  std::vector<Move> moves;
  for (const auto& [bound, i] : rank_by_bound(bounds)) {
    if (bm.have && bound < bm.eq) break;  // no remaining extent can beat the best move
    const ShadowExtent& e = extents[i];
    moves.clear();
    wmp_generate_extent(req.board, wm, subracks, e, moves, bm.have ? bm.eq : -1e18,
                        sub_terms[e.placed].data());
    consider_moves(eq, req, leaves, moves, bm);
  }
  return bm.have ? bm.move : Move::pass();
}

}  // namespace

Move HastyBotAgent::make_move(const MoveRequest& req) { return hasty_best_move_gaddag(req); }

Move hasty_best_move_wmp(const MoveRequest& req, const WordMap& wm) {
  return hasty_best_move_wmp_impl(req, wm);
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
