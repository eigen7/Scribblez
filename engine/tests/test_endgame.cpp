// The endgame solver and its supporting pieces: Board make/unmake, the
// out-play futility machinery (endgame/outplays.h), incremental move lists
// (endgame/path_move_lists.h), and EndgameSolver itself. Solver correctness is
// checked against a brute-force negamax over small random endgames on the tiny
// dictionary; each search optimization is A/B-tested against the solver with
// it disabled, for identical results and for the nodes it saves.

#include "data/gcg_reader.h"
#include "endgame/endgame_solver.h"
#include "endgame/outplays.h"
#include "endgame/path_move_lists.h"
#include "endgame_positions.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/movegen.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

Rack rack_from(const std::string& s) {
  Rack r;
  for (char c : s) {
    if (c == '?')
      r.add(BLANK);
    else
      r.add(Tile::from_char(c));
  }
  return r;
}

// A canonical key for a play (placed squares and glyphs, plus score), for
// comparing move lists regardless of order.
std::string move_key(const Move& m) {
  struct Placement {
    int r, c, code;
  };
  std::vector<Placement> tiles;
  if (m.type() == MoveType::PLAY) {
    const bool horiz = m.horizontal();
    uint16_t mask = m.square_mask();
    int gi = 0;
    for (int pos = 0; mask; ++pos, mask >>= 1) {
      if ((mask & 1u) == 0) continue;
      const int r = horiz ? m.start() : pos;
      const int c = horiz ? pos : m.start();
      tiles.push_back({r, c, m.glyph(gi++).code()});
    }
  }
  std::sort(tiles.begin(), tiles.end(), [](const Placement& a, const Placement& b) {
    if (a.r != b.r) return a.r < b.r;
    return a.c < b.c;
  });
  std::string k;
  char buf[32];
  for (const auto& t : tiles) {
    std::snprintf(buf, sizeof(buf), "%d,%d,%d;", t.r, t.c, t.code);
    k += buf;
  }
  std::snprintf(buf, sizeof(buf), "|%d", m.score());
  k += buf;
  return k;
}

std::set<std::string> key_set(const std::vector<Move>& ms) {
  std::set<std::string> s;
  for (const auto& m : ms) s.insert(move_key(m));
  return s;
}

bool squares_equal(const Board& a, const Board& b) {
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c)
      if (a.at(r, c).code() != b.at(r, c).code()) return false;
  return true;
}

bool cross_equal(const std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>& a,
                 const std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>& b) {
  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
    if (a[i].mask != b[i].mask || a[i].score != b[i].score ||
        a[i].has_neighbor != b[i].has_neighbor)
      return false;
  }
  return true;
}

// Applies `m` with an undo and unapplies it, then checks that the squares,
// cross-check and anchor tables, and legal plays for `probe_rack` are all
// unchanged.
void expect_apply_unapply_is_identity(Board& b, const Dictionary& d, const Move& m,
                                      const Rack& probe_rack) {
  std::array<Glyph, BOARD_SIZE * BOARD_SIZE> sq;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) sq[r * BOARD_SIZE + c] = b.at(r, c);
  const auto cc0 = b.cross_checks(false);
  const auto cc1 = b.cross_checks(true);
  const auto an0 = b.gaddag_anchors(false);
  const auto an1 = b.gaddag_anchors(true);
  const auto before_keys = key_set(MoveGenerator(b, d).generate(probe_rack));

  BoardUndo undo;
  b.apply(m, &undo);
  b.unapply(undo);

  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c)
      ASSERT_EQ(b.at(r, c).code(), sq[r * BOARD_SIZE + c].code());
  ASSERT_TRUE(cross_equal(b.cross_checks(false), cc0));
  ASSERT_TRUE(cross_equal(b.cross_checks(true), cc1));
  ASSERT_EQ(b.gaddag_anchors(false), an0);
  ASSERT_EQ(b.gaddag_anchors(true), an1);
  ASSERT_EQ(key_set(MoveGenerator(b, d).generate(probe_rack)), before_keys);
}

// A copy of `src`'s squares with invalid caches, the state in which apply()
// updates squares only and leaves the caches for a later rebuild.
Board squares_only_copy(const Board& src) {
  Board out;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) out.set(r, c, src.at(r, c));
  return out;
}

void roundtrip_random_games(const Dictionary& d, unsigned seed, int games, int steps) {
  std::mt19937 rng(seed);
  for (int g = 0; g < games; ++g) {
    Board b;
    for (int s = 0; s < steps; ++s) {
      const Rack r = random_rack(rng);
      const std::vector<Move> moves = MoveGenerator(b, d).generate(r);  // ensures caches
      if (moves.empty()) break;
      std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
      const Move m = moves[pick(rng)];

      // With valid caches.
      expect_apply_unapply_is_identity(b, d, m, r);

      // With invalid caches: the squares are restored, and the caches rebuild
      // to the correct values.
      Board inv = squares_only_copy(b);
      BoardUndo undo;
      inv.apply(m, &undo);
      inv.unapply(undo);
      ASSERT_TRUE(squares_equal(inv, b));
      inv.ensure_movegen_caches(d);
      Board fresh = squares_only_copy(b);
      fresh.ensure_movegen_caches(d);
      ASSERT_TRUE(cross_equal(inv.cross_checks(false), fresh.cross_checks(false)));
      ASSERT_TRUE(cross_equal(inv.cross_checks(true), fresh.cross_checks(true)));

      b.apply(m);
    }
  }
}

// ---------------------------------------------------------------------------
// Brute-force reference: plain negamax with no transposition table, move
// ordering, PVS, or leaf playout. At a depth beyond any possible line, every
// leaf is a real game end, so the result is the exact optimal final spread for
// the side to move. It applies EndgameSolver's end rules: the out bonus, and
// the game ending after two consecutive scoreless turns (so any scoreless turn
// before the solve counts as one).
// ---------------------------------------------------------------------------

struct RefState {
  Board board;
  Rack racks[2];
  int scores[2];
  int scoreless;
  int stm;
};

// The state after the side to move plays `m` (a play or a pass); sets `over`
// when the move ends the game.
RefState ref_apply(const RefState& s, const Move& m, bool& over) {
  const int mover = s.stm, opp = 1 - s.stm;
  RefState ns = s;
  ns.stm = opp;
  over = false;
  if (m.type() == MoveType::PLAY) {
    ns.board.apply(m);
    for (int i = 0; i < m.num_glyphs(); ++i) ns.racks[mover].remove(m.glyph(i).rack_tile());
    ns.scores[mover] += m.score();
    ns.scoreless = 0;
    if (ns.racks[mover].empty()) {
      ns.scores[mover] += 2 * ns.racks[opp].point_value();
      over = true;
    }
  } else {
    ns.scoreless = s.scoreless + 1;
    if (ns.scoreless >= 2) {
      ns.scores[mover] -= ns.racks[mover].point_value();
      ns.scores[opp] -= ns.racks[opp].point_value();
      over = true;
    }
  }
  return ns;
}

int32_t ref_negamax(const RefState& s, const Dictionary& d, int depth) {
  if (depth == 0) return s.scores[s.stm] - s.scores[1 - s.stm];
  std::vector<Move> moves = MoveGenerator(s.board, d).generate(s.racks[s.stm]);
  moves.push_back(Move::pass());
  const int mover = s.stm, opp = 1 - s.stm;
  int32_t best = -2'000'000;
  for (const Move& m : moves) {
    bool over = false;
    const RefState ns = ref_apply(s, m, over);
    const int32_t v = over ? (ns.scores[mover] - ns.scores[opp]) : -ref_negamax(ns, d, depth - 1);
    best = std::max(best, v);
  }
  return best;
}

RefState make_ref_state(const Board& b, const Rack& my, const Rack& opp, int my_score,
                        int opp_score, int scoreless_turns) {
  RefState s;
  s.board = b;
  s.racks[0] = my;
  s.racks[1] = opp;
  s.scores[0] = my_score;
  s.scores[1] = opp_score;
  s.scoreless = scoreless_turns > 0 ? 1 : 0;
  s.stm = 0;
  return s;
}

int32_t ref_solve(const Board& b, const Dictionary& d, const Rack& my, const Rack& opp,
                  int my_score, int opp_score, int scoreless_turns, int depth) {
  return ref_negamax(make_ref_state(b, my, opp, my_score, opp_score, scoreless_turns), d, depth);
}

// The exact value of forcing `first` as the first move, with optimal play
// after it.
int32_t ref_value_after_first(const Board& b, const Dictionary& d, const Rack& my, const Rack& opp,
                              int my_score, int opp_score, int scoreless_turns, const Move& first,
                              int depth) {
  const RefState s = make_ref_state(b, my, opp, my_score, opp_score, scoreless_turns);
  bool over = false;
  const RefState ns = ref_apply(s, first, over);
  if (over) return ns.scores[0] - ns.scores[1];
  return -ref_negamax(ns, d, depth);
}

// The generated move that empties `rack` (goes out) in one, or nullptr.
const Move* find_out_move(const std::vector<Move>& moves, const Rack& rack) {
  for (const Move& m : moves)
    if (m.type() == MoveType::PLAY && m.num_glyphs() == rack.size()) return &m;
  return nullptr;
}

constexpr int kRefDepth = 24;  // longer than any line in these small endgames
constexpr uint64_t kBigBudget = 1ull << 40;

// A play of dummy 'A' tiles at the given positions along one lane, for tests
// that depend only on a move's shape, never on its letters.
Move lane_play(bool horizontal, int line, const std::vector<int>& along, uint16_t score = 0) {
  uint16_t mask = 0;
  std::vector<Glyph> glyphs;
  for (int p : along) {
    mask |= 1u << p;
    glyphs.push_back(Glyph::of(Tile::of(0)));
  }
  return Move::play(horizontal, line, mask, score, glyphs.data(), int(glyphs.size()));
}
Move horiz_play(int row, const std::vector<int>& cols, uint16_t score = 0) {
  return lane_play(true, row, cols, score);
}
Move vert_play(int col, const std::vector<int>& rows, uint16_t score = 0) {
  return lane_play(false, col, rows, score);
}

// The exact value of forcing `m` as the first move in `p`, with the rest scored
// by `control`, a full-window solver the caller trusts. Used to accept a
// divergent best move only when it is a true tie.
int32_t forced_move_value(const Dictionary& d, const EndgamePos& p, const Move& m,
                          EndgameSolver& control) {
  const RefState s = make_ref_state(p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0);
  bool over = false;
  const RefState ns = ref_apply(s, m, over);
  if (over) return ns.scores[0] - ns.scores[1];
  const EndgameResult r = control.solve(
    {&d, ns.board, ns.racks[1], ns.racks[0], ns.scores[1], ns.scores[0], ns.scoreless},
    {kBigBudget, kRefDepth, true});
  return -r.value;
}

// Total nodes one A/B batch searched with out-play futility pruning on and off.
struct PruningNodes {
  uint64_t pruned = 0;
  uint64_t unpruned = 0;
};

// Solves a batch of random endgames with futility pruning on and off, at full
// window and depth so every line resolves. The value must not change. The best
// move may differ, because pruning reorders moves and can settle on another
// equal-valued optimum, but only if the control solver confirms it is a tie.
// Node counts are summed rather than compared per position, since reordering
// can shift one position's count either way.
PruningNodes check_pruning_ab(const Dictionary& d, unsigned seed, int count) {
  std::mt19937 rng(seed);
  PruningNodes nodes;
  int ties = 0;
  for (int i = 0; i < count; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));  // 2..4 tiles
    EndgameSolver on, off;
    off.set_outplay_futility(false);
    const EndgameResult a =
      on.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
               {kBigBudget, kRefDepth, true});
    const EndgameResult b =
      off.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                {kBigBudget, kRefDepth, true});
    EXPECT_EQ(a.value, b.value) << "seed " << seed << " position " << i;
    if (!(a.best == b.best)) {
      EXPECT_EQ(forced_move_value(d, p, a.best, off), b.value)
        << "seed " << seed << " position " << i << ": divergent best move is not a tie";
      ++ties;
    }
    nodes.pruned += a.nodes;
    nodes.unpruned += b.nodes;
  }
  if (ties > 0) std::cout << "  (" << ties << "/" << count << " tie-verified best moves)\n";
  return nodes;
}

// Prints the batch's node saving, so a regression that quietly defeats the
// pruning shows up as the cut shrinking towards zero.
void report_pruning_cut(const char* label, const PruningNodes& nodes) {
  std::printf("  outplay-futility nodes (%s): %llu pruned vs %llu unpruned (%.1f%% cut)\n", label,
              static_cast<unsigned long long>(nodes.pruned),
              static_cast<unsigned long long>(nodes.unpruned),
              100.0 * (1.0 - double(nodes.pruned) / double(nodes.unpruned)));
}

}  // namespace

TEST(EndgameBoardUndo, RoundtripTinyDict) {
  roundtrip_random_games(tiny_dict(), 0xC0FFEEu, /*games=*/40, /*steps=*/12);
}

TEST(EndgameBoardUndo, RoundtripRealLexicon) {
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) {
    GTEST_SKIP() << "no lexicon at " << path;
  }
  Dictionary d = Dictionary::load_kwg(path);
  roundtrip_random_games(d, 0xBEEF01u, /*games=*/8, /*steps=*/14);
}

// With max_plies >= kRefDepth every line reaches a real game end and the leaf
// playout is never consulted, so the solver must match the brute-force
// reference exactly, for either side to move.
TEST(EndgameSolver, DifferentialVsBruteForce) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0x5EED1234u);
  int checked = 0;
  for (int i = 0; i < 120; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2);
    solver.clear();
    const EndgameResult r = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kBigBudget, kRefDepth, true});
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    ASSERT_EQ(r.value, ref) << "position " << i << " (solving side to move)";

    // The same position with the opponent to move.
    solver.clear();
    const EndgameResult r2 =
      solver.solve({&d, p.board, p.opp_rack, p.my_rack, p.opp_score, p.my_score, 0},
                   {kBigBudget, kRefDepth, true});
    const int32_t ref2 =
      ref_solve(p.board, d, p.opp_rack, p.my_rack, p.opp_score, p.my_score, 0, kRefDepth);
    ASSERT_EQ(r2.value, ref2) << "position " << i << " (opponent to move)";
    checked += 2;
  }
  std::cout << "  differential-checked " << checked << " endgames\n";
}

// A position where going out in one, with its bonus of twice the opponent's
// rack, is optimal. Checks the value arithmetic and that the move empties the
// rack.
TEST(EndgameSolver, OutInOneOptimal) {
  Dictionary d = tiny_dict();
  Board b;  // empty board: "GO" opens across the center and empties the rack

  const Rack my = rack_from("GO");
  const Rack opp = rack_from("BD");
  const int my_score = 30, opp_score = 25;

  const std::vector<Move> plays = MoveGenerator(b, d).generate(my);
  const Move* out = find_out_move(plays, my);
  ASSERT_NE(out, nullptr);
  const int32_t expected = (my_score + out->score() + 2 * opp.point_value()) - opp_score;

  EndgameSolver solver;
  const EndgameResult r =
    solver.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, 8, true});
  EXPECT_EQ(r.value, expected);
  EXPECT_EQ(r.value, ref_solve(b, d, my, opp, my_score, opp_score, 0, kRefDepth));
  ASSERT_EQ(r.best.type(), MoveType::PLAY);
  EXPECT_EQ(r.best.num_glyphs(), my.size());
}

// Neither side can play, so two passes end the game and each side loses the
// value of its own tiles.
TEST(EndgameSolver, StalematePassPass) {
  Dictionary d = tiny_dict();
  Board b;
  const Rack my = rack_from("VV");   // V appears in no tiny-dict word
  const Rack opp = rack_from("WW");  // W appears in no tiny-dict word
  const int my_score = 40, opp_score = 12;

  ASSERT_TRUE(MoveGenerator(b, d).generate(my).empty());
  ASSERT_TRUE(MoveGenerator(b, d).generate(opp).empty());

  const int32_t expected = (my_score - my.point_value()) - (opp_score - opp.point_value());
  EndgameSolver solver;
  const EndgameResult r =
    solver.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, 6, true});
  EXPECT_EQ(r.value, expected);
  EXPECT_EQ(r.best.type(), MoveType::PASS);
  EXPECT_EQ(r.value, ref_solve(b, d, my, opp, my_score, opp_score, 0, kRefDepth));
}

// The opponent can never play. The solver's value must match the reference and
// beat passing immediately.
TEST(EndgameSolver, StuckOpponentMultiTurnOut) {
  Dictionary d = tiny_dict();
  Board b;
  const Rack my = rack_from("CATS");
  const Rack opp = rack_from("VW");  // V and W are in no tiny_dict word
  const int my_score = 10, opp_score = 8;

  ASSERT_TRUE(MoveGenerator(b, d).generate(opp).empty());

  EndgameSolver solver;
  const EndgameResult deep =
    solver.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, 12, true});
  const int32_t ref = ref_solve(b, d, my, opp, my_score, opp_score, 0, kRefDepth);
  EXPECT_EQ(deep.value, ref);
  const int32_t pass_now = (my_score - my.point_value()) - (opp_score - opp.point_value());
  EXPECT_GT(deep.value, pass_now);
}

// A depth-1 solve is greedy. Scans for a position where its move is provably
// suboptimal, and checks that the deep solve plays something else.
TEST(EndgameSolver, GreedyHoldbackSuboptimal) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0xA5A5F00Du);
  bool found = false;
  for (int i = 0; i < 400 && !found; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    solver.clear();
    const EndgameResult deep =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {kBigBudget, kRefDepth, true});
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    ASSERT_EQ(deep.value, ref);

    solver.clear();
    const EndgameResult greedy =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {kBigBudget, /*max_plies=*/1, true});
    ASSERT_EQ(greedy.depth_completed, 1);
    const int32_t greedy_true = ref_value_after_first(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                                      p.opp_score, 0, greedy.best, kRefDepth);
    if (greedy_true < ref) {
      EXPECT_NE(greedy.best, deep.best);
      found = true;
    }
  }
  ASSERT_TRUE(found) << "no greedy-suboptimal position found in the scan";
}

// Identical inputs give identical results, after a clear() and on a fresh
// solver.
TEST(EndgameSolver, Determinism) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xD37E211Du);
  for (int i = 0; i < 20; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    EndgameSolver s1;
    const EndgameResult a = s1.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0}, {kBigBudget, 12, true});
    s1.clear();
    const EndgameResult b = s1.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0}, {kBigBudget, 12, true});
    EndgameSolver s2;
    const EndgameResult c = s2.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0}, {kBigBudget, 12, true});
    ASSERT_EQ(a.value, b.value);
    ASSERT_EQ(a.value, c.value);
    ASSERT_EQ(a.depth_completed, b.depth_completed);
    ASSERT_EQ(a.nodes, b.nodes);
    ASSERT_EQ(a.nodes, c.nodes);
    ASSERT_TRUE(a.best == b.best && a.best == c.best);
  }
}

// The node budget caps every pass of a solve together. It can be overshot only
// by one greedy playout, which runs before the next negamax entry notices the
// budget is spent. A legal move comes back even at tiny budgets.
//
// Depth and nodes are not monotone in the budget, so they are not asserted.
// Control flow forks at budget-dependent thresholds (the class pass proves or
// gives up at half the budget, and the fallback searches a different window),
// so a larger budget can legitimately report less depth.
TEST(EndgameSolver, NodeBudget) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xB0DA711Eu);
  EndgameSolver solver;
  // One playout of up to 40 plies (kMaxPlayout), plus the detecting node.
  constexpr uint64_t kSlack = 41;

  for (int i = 0; i < 10; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/4);
    const std::set<std::string> legal = key_set(MoveGenerator(p.board, d).generate(p.my_rack));
    for (bool spread_matters : {false, true}) {
      for (uint64_t budget : {5ull, 50ull, 500ull, 5000ull, 200000ull}) {
        solver.clear();
        const EndgameResult r =
          solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                       {budget, 12, spread_matters});
        EXPECT_LE(r.nodes, budget + kSlack)
          << "budget " << budget << " position " << i << " spread_matters " << spread_matters;
        if (r.best.type() != MoveType::PASS) {
          EXPECT_TRUE(legal.count(move_key(r.best)) > 0)
            << "budget " << budget << " position " << i << " spread_matters " << spread_matters;
        }
      }
    }
  }
}

// A position with more root moves than budget cannot finish its first
// iteration, so solve() declines it without spending nodes and returns a legal
// root move chosen statically.
TEST(EndgameSolver, DeclinesRichPositionsBeyondBudget) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xDEC11983u);
  EndgameSolver solver;
  int checked = 0;
  for (int i = 0; i < 20; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/4);
    const std::vector<Move> plays = MoveGenerator(p.board, d).generate(p.my_rack);
    if (plays.empty()) continue;
    solver.clear();
    // The root moves are the plays plus a pass, so this budget is one short.
    const EndgameResult r = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0}, {plays.size(), 12, true});
    EXPECT_EQ(r.depth_completed, 0) << "position " << i;
    EXPECT_EQ(r.nodes, 0u) << "position " << i;
    if (r.best.type() != MoveType::PASS) {
      EXPECT_TRUE(key_set(plays).count(move_key(r.best)) > 0) << "position " << i;
    }
    ++checked;
  }
  ASSERT_GT(checked, 0);
}

// Transposition-table entries store spread relative to the root, so they stay
// valid across turns. Solving a position, then the child after its best move
// on the warm table, must give the child the negated parent value.
TEST(EndgameSolver, TTReuseAcrossTurns) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0x7700FACEu);
  EndgameSolver solver;
  for (int i = 0; i < 20; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    const EndgameResult parent =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {kBigBudget, kRefDepth, true});

    RefState s = make_ref_state(p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0);
    bool over = false;
    const RefState after = ref_apply(s, parent.best, over);
    if (over) continue;

    const EndgameResult child = solver.solve({&d, after.board, after.racks[1], after.racks[0],
                                              after.scores[1], after.scores[0], after.scoreless},
                                             {kBigBudget, kRefDepth, true});
    EXPECT_EQ(child.value, -parent.value);
  }
}

// With spread_matters off the solve searches the window (kFirstWinAlpha,
// kFirstWinBeta), which resolves only the win/draw/loss class. The class must
// match the brute-force reference, and in a won or drawn position the chosen
// move must keep that class under optimal play. (In a lost position every move
// loses, so there is nothing to check per move.)
TEST(EndgameSolver, FirstWinPreservesDecidedOutcomes) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0xF1257114u);
  int wins = 0, draws = 0, losses = 0, checked = 0;
  for (int i = 0; i < 40; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    const int32_t v_star =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);

    solver.clear();
    const EndgameResult r = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kBigBudget, kRefDepth, false});
    ++checked;

    if (v_star > 0) {
      ASSERT_GE(r.value, EndgameSolver::kFirstWinBeta) << "position " << i;
      const int32_t after = ref_value_after_first(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                                  p.opp_score, 0, r.best, kRefDepth);
      ASSERT_GT(after, 0) << "position " << i;
      ++wins;
    } else if (v_star == 0) {
      ASSERT_EQ(r.value, 0) << "position " << i;
      const int32_t after = ref_value_after_first(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                                  p.opp_score, 0, r.best, kRefDepth);
      ASSERT_GE(after, 0) << "position " << i;
      ++draws;
    } else {
      ASSERT_LE(r.value, EndgameSolver::kFirstWinAlpha) << "position " << i;
      ++losses;
    }
  }
  ASSERT_GT(checked, 10);
  std::cout << "  wld-checked " << checked << " endgames: " << wins << " win, " << draws
            << " draw, " << losses << " loss\n";
}

// The class-only window is narrower than the full window, so over a batch it
// never searches more nodes.
TEST(EndgameSolver, FirstWinSearchesNoMoreNodes) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0x50DE5EEDu);
  uint64_t full_nodes = 0, wld_nodes = 0;
  for (int i = 0; i < 40; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);

    solver.clear();
    const EndgameResult full = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kBigBudget, kRefDepth, true});
    solver.clear();
    const EndgameResult wld = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kBigBudget, kRefDepth, false});
    full_nodes += full.nodes;
    wld_nodes += wld.nodes;
  }
  EXPECT_LE(wld_nodes, full_nodes);
}

// A value reported as proven is exact and must equal the brute-force
// reference. These endgames end well within kRefDepth, so most solves prove,
// and the check is not vacuous.
TEST(EndgameSolver, ProvenMatchesBruteForce) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0x9E3779B9u);
  int proven = 0, checked = 0;
  for (int i = 0; i < 40; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2);
    solver.clear();
    const EndgameResult r = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kBigBudget, kRefDepth, true});
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    if (r.proven) {
      ASSERT_EQ(r.value, ref) << "position " << i;
      ++proven;
    }
    ++checked;
  }
  ASSERT_GE(proven * 2, checked) << "proven fired on only " << proven << "/" << checked;
  std::cout << "  proven " << proven << "/" << checked << " endgames\n";
}

// Iterative deepening stops once the value is proven. That must return exactly
// what capping max_plies at the proof depth returns: the same iterations, so
// the same value, best move and node count. Runs with spread_matters off,
// whose single pass makes "the same iterations" well-defined; with it on, two
// passes deepen on their own schedules.
TEST(EndgameSolver, EarlyExitPreservesResults) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0x1234ABCDu);
  int truncated = 0, checked = 0;
  for (int i = 0; i < 40; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2);

    solver.clear();
    const EndgameResult early =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {kBigBudget, kRefDepth, false});
    ASSERT_GE(early.depth_completed, 1) << "position " << i;
    solver.clear();
    const EndgameResult ctrl =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {kBigBudget, early.depth_completed, false});
    ASSERT_EQ(early.value, ctrl.value) << "position " << i;
    ASSERT_TRUE(early.best == ctrl.best) << "position " << i;
    ASSERT_EQ(early.nodes, ctrl.nodes) << "position " << i;
    if (early.proven && early.depth_completed < kRefDepth) ++truncated;
    ++checked;
  }
  ASSERT_GT(truncated, 0) << "no proven solve truncated below the horizon";
  std::cout << "  early-exit truncated " << truncated << "/" << checked << " solves\n";
}

// The proof early exit fires and saves nodes. With 1- and 2-tile racks the
// game tree ends within a few plies, so solves prove shallow, while a solver
// with the early exit disabled keeps deepening to the ply cap for the same
// value.
TEST(EndgameSolver, ProofShortCircuitSavesNodes) {
  Dictionary d = tiny_dict();
  EndgameSolver on;
  EndgameSolver off;
  off.set_proof_early_exit(false);

  std::mt19937 rng(0x5C1Fu);
  uint64_t nodes_on = 0, nodes_off = 0;
  int proven = 0, checked = 0;
  for (int i = 0; i < 30; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/(i % 2) ? 1 : 2);
    on.clear();
    off.clear();
    const EndgameResult a = on.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0}, {kBigBudget, 25, true});
    const EndgameResult b = off.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0}, {kBigBudget, 25, true});
    ASSERT_EQ(a.value, b.value) << "position " << i;
    if (!(a.best == b.best)) {
      // The early exit changes how many class-pass iterations run, and the root
      // cutoff leaves skipped moves ranked by stale values, so the two solvers
      // may settle on different equal-valued moves. A divergent best must be a
      // true tie. The control is a fresh solver so the measured solvers' tables
      // stay untouched.
      EndgameSolver control;
      EXPECT_EQ(forced_move_value(d, p, a.best, control), b.value)
        << "position " << i << ": divergent best move is not a tie";
    }
    nodes_on += a.nodes;
    nodes_off += b.nodes;
    ++checked;
    if (!a.proven) continue;
    ++proven;
    EXPECT_LT(a.depth_completed, 25) << "position " << i;
    EXPECT_LT(a.nodes, b.nodes) << "position " << i;
    EXPECT_EQ(b.depth_completed, 25) << "position " << i;
  }
  ASSERT_GT(proven, checked / 2) << "shallow endgames should mostly be provable";
  EXPECT_LT(nodes_on * 2, nodes_off);  // at least halves the batch's nodes
  std::cout << "  proof short-circuit: " << proven << "/" << checked << " proven, nodes "
            << nodes_on << " vs " << nodes_off << " without early exit\n";
}

// With spread_matters off, a proven solve's sign matches the reference's, and
// never costs more nodes than the full-window solve.
TEST(EndgameSolver, WldEarlyExitSettlesClass) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0xC0DEC0DEu);
  int proven = 0, checked = 0;
  for (int i = 0; i < 40; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2);
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);

    solver.clear();
    const EndgameResult wld = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*scoreless_turns=*/0},
      {kBigBudget, kRefDepth, false});
    solver.clear();
    const EndgameResult full =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {kBigBudget, kRefDepth, true});
    if (wld.proven) {
      const int wld_sign = (wld.value > 0) - (wld.value < 0);
      const int ref_sign = (ref > 0) - (ref < 0);
      ASSERT_EQ(wld_sign, ref_sign) << "position " << i;
      ++proven;
    }
    ASSERT_LE(wld.nodes, full.nodes) << "position " << i;
    ++checked;
  }
  ASSERT_GT(proven, 0) << "no wld solve proved a class";
  std::cout << "  wld-proven " << proven << "/" << checked << " endgames\n";
}

// Every class-proven solve carries a valid certificate: after the best move,
// each entry is a legal move in its position, the line ends the game under the
// solver's rules, and the final spread has the proven class. Checked with
// spread_matters on and off, at a starved and an unlimited budget.
TEST(EndgameSolver, ContinuationCertificateIsSound) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0xCE47B00Cu);
  int with_certificate = 0, class_proven = 0;
  for (int i = 0; i < 120; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    const uint64_t budget = (i % 2) ? kBigBudget : 400;
    const bool spread_matters = (i % 4) < 2;
    solver.clear();
    const EndgameResult r =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {budget, kRefDepth, spread_matters});
    if (r.proven_class == EndgameResult::kClassUnknown) {
      ASSERT_TRUE(r.continuation.empty()) << "position " << i;
      continue;
    }
    ++class_proven;

    RefState st = make_ref_state(p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0);
    bool over = false;
    st = ref_apply(st, r.best, over);
    // A best move that ends the game needs no certificate.
    if (!over) {
      ASSERT_FALSE(r.continuation.empty())
        << "position " << i << ": class proven but no certificate";
      ++with_certificate;
    }
    for (const Move& m : r.continuation) {
      ASSERT_FALSE(over) << "position " << i << ": continuation past the game end";
      if (m.type() == MoveType::PLAY) {
        const std::vector<Move> plays = MoveGenerator(st.board, d).generate(st.racks[st.stm]);
        ASSERT_NE(std::find(plays.begin(), plays.end(), m), plays.end())
          << "position " << i << ": illegal continuation move";
      }
      st = ref_apply(st, m, over);
    }
    ASSERT_TRUE(over) << "position " << i << ": certified line does not end the game";
    const int32_t final_spread = st.scores[0] - st.scores[1];
    ASSERT_EQ((final_spread > 0) - (final_spread < 0), r.proven_class) << "position " << i;
  }
  ASSERT_GT(class_proven, 0);
  ASSERT_GT(with_certificate, 0);
  std::cout << "  certificates " << with_certificate << "/" << class_proven
            << " class-proven endgames (the rest end with the chosen move)\n";
}

// At an unlimited budget a spread_matters solve proves every position, and its
// proven class matches the reference's sign. The two solvers here are
// configured identically, so the value and move comparisons between them
// check only determinism across fresh solvers.
TEST(EndgameSolver, LexicographicMatchesSpreadAtFullBudget) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0x1E0C0DE5u);
  for (int i = 0; i < 80; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    EndgameSolver lex_solver, spread_solver;
    const EndgameResult lex =
      lex_solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                       {kBigBudget, kRefDepth, true});
    const EndgameResult spread =
      spread_solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                          {kBigBudget, kRefDepth, true});
    ASSERT_EQ(lex.value, spread.value) << "position " << i;
    ASSERT_TRUE(lex.proven) << "position " << i;
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    ASSERT_EQ(lex.proven_class, (ref > 0) - (ref < 0)) << "position " << i;
    if (!(lex.best == spread.best)) {
      EXPECT_EQ(forced_move_value(d, p, lex.best, spread_solver), spread.value)
        << "position " << i << ": divergent best move is not a tie";
    }
  }
}

// The gate for "never sacrifice the class for points". With spread_matters on
// and constrained budgets, whenever a solve proves a class, its move must keep
// that class under optimal play, and the class must match the reference. (In a
// lost position every move loses, so there is nothing to check per move.)
TEST(EndgameSolver, LexicographicPreservesProvenClass) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0xC1A55E5Fu);
  int class_proven = 0, checked = 0;
  for (int i = 0; i < 120; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    const uint64_t budget = 60 + 140 * (i % 5);  // 60..620, starved to roomy
    solver.clear();
    const EndgameResult r = solver.solve(
      {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0}, {budget, kRefDepth, true});
    ++checked;
    if (r.depth_completed == 0 || r.proven_class == EndgameResult::kClassUnknown) continue;
    ++class_proven;
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    ASSERT_EQ(r.proven_class, (ref > 0) - (ref < 0)) << "position " << i;
    if (r.proven_class == -1) continue;
    const int32_t after = ref_value_after_first(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                                p.opp_score, 0, r.best, kRefDepth);
    ASSERT_EQ((after > 0) - (after < 0), r.proven_class)
      << "position " << i << " budget " << budget << ": played move forfeits the proven class";
  }
  ASSERT_GT(class_proven, checked / 4) << "class proofs fired too rarely to gate anything";
  std::cout << "  lexicographic class-proven " << class_proven << "/" << checked << " endgames\n";
}

// With spread_matters on or off, a proven class matches the reference's sign.
TEST(EndgameSolver, ProvenClassMatchesReference) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0xC1A5537Du);
  for (int i = 0; i < 40; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2);
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    const int ref_class = (ref > 0) - (ref < 0);
    for (const bool spread_matters : {false, true}) {
      solver.clear();
      const EndgameResult r =
        solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                     {kBigBudget, kRefDepth, spread_matters});
      if (r.proven) {
        ASSERT_EQ(r.proven_class, ref_class)
          << "position " << i << " spread_matters " << spread_matters;
      }
    }
  }
}

// The soundness gate for opponent out-play futility pruning, and the check on
// what it saves. With pruning on and off, solves must agree on value and on
// the best move up to ties. A wrong halo-survival classification, a stale
// incrementally maintained out-play entry, or a wrong upper bound would prune
// a move that mattered and change one of them.
//
// Pruning must never search more nodes in aggregate, and on the real lexicon
// it must search strictly fewer, so a regression that quietly disables it
// fails here. The strict bar applies only to the real lexicon, where pruning
// saves around a quarter of the nodes; the tiny dictionary's boards are too
// bare for it to bite.
TEST(EndgameSolver, OutplayFutilityPruningIsSound) {
  const Dictionary tiny = tiny_dict();
  const PruningNodes tiny_nodes = check_pruning_ab(tiny, 0x0F0DD5EAu, 200);
  ASSERT_GT(tiny_nodes.unpruned, 0u);
  EXPECT_LE(tiny_nodes.pruned, tiny_nodes.unpruned);
  report_pruning_cut("tiny dict", tiny_nodes);

  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (std::ifstream(path).good()) {
    const Dictionary real = Dictionary::load_kwg(path);
    const PruningNodes real_nodes = check_pruning_ab(real, 0x5EAF00D1u, 60);
    ASSERT_GT(real_nodes.unpruned, 0u);
    EXPECT_LT(real_nodes.pruned, real_nodes.unpruned);
    report_pruning_cut("real lexicon", real_nodes);
  } else {
    std::cout << "  (no lexicon at " << path << "; real-lexicon batch skipped)\n";
  }
}

// collect_rack_outplays keeps only the rack-emptying plays, best score first;
// best_surviving_score is the best one a given move provably leaves intact;
// assign_surviving builds a child set with the same halo filter.
TEST(OutplaySet, CollectQueryFilter) {
  Board b;
  std::vector<Move> plays;
  plays.push_back(vert_play(3, {0, 1}, 30));    // out-play at the top of column 3
  plays.push_back(vert_play(3, {13, 14}, 20));  // out-play at the bottom of column 3
  plays.push_back(vert_play(9, {7}, 50));       // one tile: not rack-emptying
  OutplaySet outs;
  collect_rack_outplays(b, plays, /*rack_size=*/2, outs);
  ASSERT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].move.score(), 30);

  EXPECT_EQ(best_surviving_score(outs, Move::pass()), 30);
  EXPECT_EQ(best_surviving_score(outs, horiz_play(7, {6, 7, 8})), 30);  // far from both
  // (2,3) extends the top out-play's word, so only the bottom one survives.
  EXPECT_EQ(best_surviving_score(outs, vert_play(3, {2})), 20);
  EXPECT_EQ(best_surviving_score(outs, vert_play(3, {2, 12})), kNoOutplaySurvivor);

  OutplaySet child;
  assign_surviving(outs, vert_play(3, {2}), child);
  ASSERT_EQ(child.size(), 1u);
  EXPECT_EQ(child[0].move.score(), 20);

  std::vector<Move> singles;
  singles.push_back(vert_play(9, {7}, 50));
  collect_rack_outplays(b, singles, /*rack_size=*/2, outs);
  EXPECT_TRUE(outs.empty());
}

// LeaveOutplays buckets a node's play list by the tiles each play uses. After
// move m, the child's out-plays are the plays that spend exactly m's leave,
// minus those m disturbs. With an "AA" rack, a one-tile play leaves one A,
// which the other one-tile plays spend.
TEST(LeaveOutplays, BucketsByLeave) {
  Board b;
  const Rack rack = rack_from("AA");
  std::vector<Move> plays;
  plays.push_back(vert_play(3, {0, 1}, 30));  // spends AA: an out-play
  plays.push_back(vert_play(9, {7}, 10));     // spends one A
  plays.push_back(horiz_play(11, {4}, 12));   // spends one A
  LeaveOutplays lo(b, rack, plays);

  // After the play at (7,9), the only surviving one-tile play is the far-away
  // (11,4); a play always disturbs itself.
  OutplaySet out;
  lo.collect_after(plays[1], out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].move.score(), 12);

  lo.collect_after(Move::pass(), out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].move.score(), 30);

  // The two-tile play empties the rack.
  lo.collect_after(plays[0], out);
  EXPECT_TRUE(out.empty());
}

// An out-play's halo is every cell whose occupation by a reply could stop it:
// its own squares, the cross-word neighbours of its placed tiles, and the cell
// just beyond each end of its word.
TEST(OutplayHalo, CoversBlockingCells) {
  Board b;
  const OutplayHalo h = build_outplay_halo(b, horiz_play(7, {7, 8}));
  EXPECT_TRUE(h.contains(7, 7));  // placed
  EXPECT_TRUE(h.contains(7, 8));  // placed
  EXPECT_TRUE(h.contains(6, 7));  // cross-word neighbour above
  EXPECT_TRUE(h.contains(8, 8));  // cross-word neighbour below
  EXPECT_TRUE(h.contains(7, 6));  // before the word
  EXPECT_TRUE(h.contains(7, 9));  // after the word
}

// A tile placed at the end of an existing perpendicular run forms a cross-word
// that includes the whole run, so a reply can change that word only at the
// run's far end. The halo must reach past the run.
TEST(OutplayHalo, CoversPerpendicularRunEnds) {
  Board b;
  b.apply(vert_play(7, {3, 4, 5}));
  // Placing at (6,7) forms the cross-word on rows 3-6 of column 7.
  const OutplayHalo h = build_outplay_halo(b, horiz_play(6, {7}));
  EXPECT_TRUE(h.contains(2, 7));  // past the top of the run
  EXPECT_TRUE(h.contains(7, 7));  // past the bottom
}

// The soundness gate for the root beta cutoff, which stops the root scan as
// soon as a fail-high settles the class. With spread_matters off, a proven
// class must match the reference, and in a won or drawn position the move must
// keep the class. A cutoff that skipped a move that mattered would fail one of
// these.
TEST(EndgameSolver, RootCutoffPreservesVerdicts) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0x2000C0DEu);
  int class_proven = 0, checked = 0;
  for (int i = 0; i < 120; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    const uint64_t budget = (i % 2) ? kBigBudget : 400;
    solver.clear();
    const EndgameResult r =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {budget, kRefDepth, /*spread_matters=*/false});
    ++checked;
    if (r.proven_class == EndgameResult::kClassUnknown) continue;
    ++class_proven;
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    ASSERT_EQ(r.proven_class, (ref > 0) - (ref < 0)) << "position " << i;
    if (r.proven_class == -1) continue;
    const int32_t after = ref_value_after_first(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                                p.opp_score, 0, r.best, kRefDepth);
    ASSERT_EQ((after > 0) - (after < 0), r.proven_class)
      << "position " << i << " budget " << budget << ": played move forfeits the proven class";
  }
  ASSERT_GT(class_proven, checked / 4) << "class proofs fired too rarely to gate anything";
  std::cout << "  root-cutoff class-proven " << class_proven << "/" << checked << " endgames\n";
}

// The root cutoff saves nodes on decisive wins, where a winning root move
// fails high early and the rest of the root is skipped. It only ever skips
// work, so it is never worse per position. It can tie when the fail-high lands
// on the last root move scanned, so the strict saving is asserted on most
// positions and on the batch total, not on each one.
TEST(EndgameSolver, RootCutoffSavesNodes) {
  Dictionary d = tiny_dict();
  EndgameSolver on, off;
  off.set_root_cutoff(false);
  std::mt19937 rng(0x2C07A5EDu);
  uint64_t nodes_on = 0, nodes_off = 0;
  int blowouts = 0, improved = 0;
  for (int i = 0; i < 400 && blowouts < 20; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    if (ref < 30) continue;  // only decisive wins
    const EndgameState state = {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0};
    on.clear();
    off.clear();
    const EndgameResult a = on.solve(state, {kBigBudget, kRefDepth, /*spread_matters=*/false});
    const EndgameResult b = off.solve(state, {kBigBudget, kRefDepth, /*spread_matters=*/false});
    ASSERT_EQ(a.proven_class, 1) << "position " << i;
    ASSERT_EQ(b.proven_class, a.proven_class) << "position " << i;
    nodes_on += a.nodes;
    nodes_off += b.nodes;
    EXPECT_LE(a.nodes, b.nodes) << "position " << i;
    if (a.nodes < b.nodes) ++improved;
    ++blowouts;
  }
  ASSERT_GT(blowouts, 0) << "no won-blowout positions found in the scan";
  EXPECT_GT(improved, blowouts / 2) << "the cutoff should strictly win on most blowouts";
  EXPECT_LT(nodes_on, nodes_off);
  std::cout << "  root-cutoff on " << blowouts << " won blowouts (" << improved
            << " strictly fewer): nodes " << nodes_on << " vs " << nodes_off << " without cutoff\n";
}

// The soundness gate for root-level out-play futility pruning. With
// spread_matters off, at a starved and an unlimited budget, both solvers' proven
// classes must match the reference, and in a won or drawn position the pruned
// solver's move must keep the class. A wrong root bound would either prune a
// class-saving move or turn an unsound bound into a false loss.
//
// A loss is the one verdict root pruning can return without searching, so the
// scan must contain proven losses. At the unlimited budget pruning must not
// cost nodes in aggregate; per position the count can move either way, because
// pruned moves are re-ranked by their bounds.
TEST(EndgameSolver, RootFutilityPruningIsSound) {
  Dictionary d = tiny_dict();
  EndgameSolver on, off;
  off.set_root_futility(false);
  std::mt19937 rng(0x0007F007u);
  uint64_t full_nodes_on = 0, full_nodes_off = 0;
  int losses = 0, checked = 0;
  for (int i = 0; i < 160; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    const uint64_t budget = (i % 2) ? kBigBudget : 400;
    const EndgameState state = {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0};
    on.clear();
    off.clear();
    const EndgameResult a = on.solve(state, {budget, kRefDepth, /*spread_matters=*/false});
    const EndgameResult b = off.solve(state, {budget, kRefDepth, /*spread_matters=*/false});
    ++checked;
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    const int ref_class = (ref > 0) - (ref < 0);
    if (budget == kBigBudget) {
      // Every line resolves within kRefDepth, so both solves must prove.
      ASSERT_TRUE(a.proven && b.proven) << "position " << i;
      full_nodes_on += a.nodes;
      full_nodes_off += b.nodes;
    }
    if (a.proven_class != EndgameResult::kClassUnknown) {
      ASSERT_EQ(a.proven_class, ref_class) << "position " << i;
      if (ref_class == -1) {
        ++losses;
      } else {
        const int32_t after = ref_value_after_first(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                                    p.opp_score, 0, a.best, kRefDepth);
        ASSERT_EQ((after > 0) - (after < 0), ref_class)
          << "position " << i << " budget " << budget << ": played move forfeits the proven class";
      }
    }
    if (b.proven_class != EndgameResult::kClassUnknown) {
      ASSERT_EQ(b.proven_class, ref_class) << "position " << i;
    }
  }
  ASSERT_GT(losses, 0) << "no proven losses in the scan; root pruning was never stressed";
  EXPECT_LE(full_nodes_on, full_nodes_off);
  std::cout << "  root-futility A/B over " << checked << " endgames (" << losses
            << " proven losses): full-budget nodes " << full_nodes_on << " pruned vs "
            << full_nodes_off << " unpruned\n";
}

// When no root move can block the opponent's out-plays or outscore them, the
// loss is proven from the bounds alone. Here the mover can only pass while the
// opponent can go out, so a spread_matters-off solve proves the loss with zero
// nodes searched and still produces a certificate. With root pruning disabled
// the same proof takes search.
TEST(EndgameSolver, RootFutilityProvesLossFromBoundsAlone) {
  Dictionary d = tiny_dict();
  Board b;
  const Rack my = rack_from("VV");  // V is in no tiny_dict word
  const Rack opp = rack_from("GO");
  const int my_score = 20, opp_score = 20;
  ASSERT_TRUE(MoveGenerator(b, d).generate(my).empty());
  ASSERT_NE(find_out_move(MoveGenerator(b, d).generate(opp), opp), nullptr);

  EndgameSolver on;
  const EndgameResult a =
    on.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, kRefDepth, false});
  EXPECT_TRUE(a.proven);
  EXPECT_EQ(a.proven_class, -1);
  EXPECT_EQ(a.nodes, 0u);
  EXPECT_EQ(a.best.type(), MoveType::PASS);
  EXPECT_FALSE(a.continuation.empty());
  EXPECT_LT(ref_solve(b, d, my, opp, my_score, opp_score, 0, kRefDepth), 0);

  EndgameSolver off;
  off.set_root_futility(false);
  const EndgameResult c =
    off.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, kRefDepth, false});
  EXPECT_TRUE(c.proven);
  EXPECT_EQ(c.proven_class, -1);
  EXPECT_GT(c.nodes, 0u);
}

namespace {

// A curated endgame in tests/data/. The solver must prove the expected class
// for the side to move within `max_nodes`, with a certificate. To add a case,
// commit a .gcg with a #RackN pragma for the mover and append a row.
struct GcgEndgameCase {
  const char* file;
  int expected_class;  // +1 win, 0 draw, -1 loss, for the side to move
  uint64_t max_nodes;  // proof must land within this many nodes
};

constexpr GcgEndgameCase kGcgEndgameCases[] = {
  // Alice, down 141 with AABCGNT, cannot block all of Bob's FOE out-plays nor
  // outscore them: a cheap proven loss.
  {"FOE.gcg", -1, 10000},
};

}  // namespace

// Needs the real lexicon; skips without it.
TEST(EndgameGcgCases, ProvenClassAndCost) {
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) GTEST_SKIP() << "no lexicon at " << path;
  Dictionary d = Dictionary::load_kwg(path);

  for (const GcgEndgameCase& c : kGcgEndgameCases) {
    const std::string file = std::string(SCRIBBLEZ_TEST_DATA_DIR) + "/" + c.file;
    std::ifstream in(file);
    ASSERT_TRUE(in.good()) << "cannot read " << file;
    std::stringstream buffer;
    buffer << in.rdbuf();

    ParsedGcgEndgame endgame;
    std::string error;
    ASSERT_TRUE(read_gcg_endgame(buffer.str(), &endgame, &error)) << c.file << ": " << error;

    EndgameSolver solver;
    const EndgameResult r = solver.solve(
      {&d, endgame.board, endgame.racks[endgame.mover], endgame.racks[1 - endgame.mover],
       endgame.scores[endgame.mover], endgame.scores[1 - endgame.mover], 0},
      {/*budget=*/1'000'000, /*plies=*/25, /*spread_matters=*/false});

    EXPECT_EQ(r.proven_class, c.expected_class) << c.file;
    EXPECT_LE(r.nodes, c.max_nodes) << c.file << ": the proof should be cheap";
    EXPECT_FALSE(r.continuation.empty()) << c.file << ": no certificate";
    std::cout << "  " << c.file << ": class " << r.proven_class << ", nodes " << r.nodes
              << ", certificate length " << r.continuation.size() << "\n";
  }
}

// The movegens count is nonzero for a real solve and does not shrink with a
// larger budget: a bigger budget repeats the smaller one's deterministic work
// and then some.
TEST(EndgameSolver, MovegensCounted) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xB16B00B5u);
  const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/4);
  const EndgameState state = {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0};

  EndgameSolver small_solver, big_solver;
  const EndgameResult small = small_solver.solve(state, {/*budget=*/500, kRefDepth, true});
  const EndgameResult big = big_solver.solve(state, {kBigBudget, kRefDepth, true});
  EXPECT_GT(small.movegens, 0u);
  EXPECT_GE(big.movegens, small.movegens);
}

// --- Incremental move-list maintenance (PathMoveLists) ----------------------

TEST(PackedCounts, SubsetRespectsMultiplicityAndBlanks) {
  const PackedCounts eea = pack_rack(rack_from("EEA"));
  EXPECT_TRUE(counts_subset(pack_rack(Rack{}), eea));
  EXPECT_TRUE(counts_subset(pack_rack(rack_from("E")), eea));
  EXPECT_TRUE(counts_subset(pack_rack(rack_from("EE")), eea));
  EXPECT_TRUE(counts_subset(pack_rack(rack_from("EEA")), eea));
  EXPECT_FALSE(counts_subset(pack_rack(rack_from("EEE")), eea));
  EXPECT_FALSE(counts_subset(pack_rack(rack_from("B")), eea));
  // The blank is a tile type of its own, not a wildcard, and lives in the high
  // half together with Q..Z.
  EXPECT_FALSE(counts_subset(pack_rack(rack_from("?")), eea));
  EXPECT_TRUE(counts_subset(pack_rack(rack_from("?")), pack_rack(rack_from("A?"))));
  EXPECT_TRUE(counts_subset(pack_rack(rack_from("ZZ")), pack_rack(rack_from("QZZ"))));
  EXPECT_FALSE(counts_subset(pack_rack(rack_from("ZZZ")), pack_rack(rack_from("QZZ"))));
}

namespace {

uint16_t lane_mask(const std::vector<int>& lanes) {
  uint16_t m = 0;
  for (int lane : lanes) m |= uint16_t(1u << lane);
  return m;
}

}  // namespace

TEST(MoveLaneInfluence, MarksPlacedAndNeighborLanes) {
  const Board b;  // empty
  const LaneTouch t = move_lane_influence(b, horiz_play(7, {6, 7}));
  EXPECT_EQ(t.rows, lane_mask({6, 7, 8}));
  EXPECT_EQ(t.cols, lane_mask({5, 6, 7, 8}));
  const LaneTouch pass = move_lane_influence(b, Move::pass());
  EXPECT_EQ(pass.rows, 0);
  EXPECT_EQ(pass.cols, 0);
}

TEST(MoveLaneInfluence, WalksThroughExistingRuns) {
  // An existing vertical run at (5,7)-(6,7): placing at (7,7) reaches through
  // it to the run's far end (4,7), and rows 5-6 themselves host no reachable
  // placement, so they stay unmarked.
  Board b;
  b.apply(vert_play(7, {5, 6}));
  const LaneTouch t = move_lane_influence(b, horiz_play(7, {7}));
  EXPECT_EQ(t.rows, lane_mask({4, 7, 8}));
  EXPECT_EQ(t.cols, lane_mask({6, 7, 8}));
}

TEST(MoveLaneInfluence, BoardEdgeDropsOffWalks) {
  const Board b;
  const LaneTouch t = move_lane_influence(b, horiz_play(0, {0}));
  EXPECT_EQ(t.rows, lane_mask({0, 1}));
  EXPECT_EQ(t.cols, lane_mask({0, 1}));
}

namespace {

void expect_moves_identical(const std::vector<Move>& inc, const std::vector<Move>& ref,
                            const char* ctx, int ply) {
  ASSERT_EQ(inc.size(), ref.size()) << ctx << " ply " << ply;
  for (size_t i = 0; i < inc.size(); ++i)
    ASSERT_TRUE(inc[i] == ref[i]) << ctx << " ply " << ply << " index " << i;
}

void rack_remove_move(Rack& rack, const Move& m) {
  for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
}

void rack_add_move(Rack& rack, const Move& m) {
  for (int i = 0; i < m.num_glyphs(); ++i) rack.add(m.glyph(i).rack_tile());
}

// Walks random game paths and checks every PathMoveLists list against a
// scratch generation, including after a sibling probe (make a move, read the
// child's list, unmake, continue with another move), which is the pattern a
// search's move scan produces.
void check_path_lists_on_random_paths(const Dictionary& d, unsigned seed, int games,
                                      int max_plies) {
  std::mt19937 rng(seed);
  for (int g = 0; g < games; ++g) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/5);
    Board board = p.board;
    Rack racks[2] = {p.my_rack, p.opp_rack};
    if ((rng() % 2) != 0) racks[rng() % 2].add(BLANK);

    PathMoveLists pl;
    pl.reset(&board, &d, max_plies + 2);
    pl.set_root_list(0, MoveGenerator(board, d).generate(racks[0]));
    pl.set_root_list(1, MoveGenerator(board, d).generate(racks[1]));

    for (int ply = 0; ply < max_plies; ++ply) {
      const int side = ply % 2;
      const std::vector<Move> ref = MoveGenerator(board, d).generate(racks[side]);
      {
        const std::vector<Move>& inc = pl.moves_at(ply, racks[side]);
        expect_moves_identical(inc, ref, "path", ply);
      }
      if (!ref.empty()) {
        const Move probe = ref[rng() % ref.size()];
        pl.on_make(ply, probe);
        BoardUndo undo;
        board.apply(probe, &undo);
        rack_remove_move(racks[side], probe);
        const std::vector<Move>& pinc = pl.moves_at(ply + 1, racks[1 - side]);
        expect_moves_identical(pinc, MoveGenerator(board, d).generate(racks[1 - side]), "probe",
                               ply + 1);
        board.unapply(undo);
        rack_add_move(racks[side], probe);
      }
      const Move chosen =
        (!ref.empty() && (rng() % 8) != 0) ? ref[rng() % ref.size()] : Move::pass();
      pl.on_make(ply, chosen);
      board.apply(chosen);
      rack_remove_move(racks[side], chosen);
      if (racks[side].empty()) break;
    }
  }
}

}  // namespace

TEST(PathMoveLists, MatchesScratchTinyDict) {
  check_path_lists_on_random_paths(tiny_dict(), 0x9A7E11u, /*games=*/40, /*max_plies=*/6);
}

TEST(PathMoveLists, MatchesScratchRealLexicon) {
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) {
    GTEST_SKIP() << "no lexicon at " << path;
  }
  Dictionary d = Dictionary::load_kwg(path);
  check_path_lists_on_random_paths(d, 0x9A7E22u, /*games=*/6, /*max_plies=*/8);
}

namespace {

// Incremental move lists must not change any solver output, with spread_matters
// on and off, at a starved and an unlimited budget.

void check_incremental_ab(const Dictionary& d, unsigned seed, int count) {
  std::mt19937 rng(seed);
  for (int i = 0; i < count; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    const uint64_t budget = (i % 2) ? kBigBudget : 60;
    const bool spread_matters = (i % 4) < 2;
    const EndgameState state = {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0};
    EndgameSolver inc, scratch;
    scratch.set_incremental_movegen(false);
    const EndgameResult a = inc.solve(state, {budget, kRefDepth, spread_matters});
    const EndgameResult b = scratch.solve(state, {budget, kRefDepth, spread_matters});
    EXPECT_EQ(a.value, b.value) << "position " << i;
    EXPECT_TRUE(a.best == b.best) << "position " << i;
    EXPECT_EQ(a.depth_completed, b.depth_completed) << "position " << i;
    EXPECT_EQ(a.nodes, b.nodes) << "position " << i;
    EXPECT_EQ(a.movegens, b.movegens) << "position " << i;
    EXPECT_EQ(a.proven, b.proven) << "position " << i;
    EXPECT_EQ(a.proven_class, b.proven_class) << "position " << i;
    ASSERT_EQ(a.continuation.size(), b.continuation.size()) << "position " << i;
    for (size_t j = 0; j < a.continuation.size(); ++j)
      EXPECT_TRUE(a.continuation[j] == b.continuation[j]) << "position " << i << " move " << j;
  }
}

}  // namespace

TEST(EndgameSolver, IncrementalMovegenBitIdenticalTinyDict) {
  check_incremental_ab(tiny_dict(), 0x1AB2CD3u, /*count=*/60);
}

TEST(EndgameSolver, IncrementalMovegenBitIdenticalRealLexicon) {
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) {
    GTEST_SKIP() << "no lexicon at " << path;
  }
  Dictionary d = Dictionary::load_kwg(path);
  check_incremental_ab(d, 0x1AB2CD4u, /*count=*/16);
}
