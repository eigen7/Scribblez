// GoogleTest suite for the endgame solver and its Board make/unmake support:
//   - Board::apply(move, &undo) / unapply(undo) round-trips, and
//   - EndgameSolver correctness against a brute-force reference plus oracle
//     positions, determinism, node-budget behavior, and cross-turn TT reuse.

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

// A canonical key for a play (placed squares/glyphs + score), so two move lists
// can be compared ignoring enumeration order.
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

// Apply `m` with an undo, unapply it, and assert every observable -- squares,
// both cross-check tables, both anchor tables, and the set of legal plays for
// `probe_rack` -- returns to exactly what it was before the apply.
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

// Copy `src`'s squares into a fresh board via set(), which leaves the caches
// invalid -- the state in which apply() takes its "caches were stale anyway"
// branch (squares change, caches untouched).
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

      // Caches-valid path: full round-trip identity on the live board.
      expect_apply_unapply_is_identity(b, d, m, r);

      // Caches-invalid path: apply/unapply on a squares-only copy restores the
      // squares and leaves the caches rebuildable to the correct values.
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

      b.apply(m);  // advance the game
    }
  }
}

// ---------------------------------------------------------------------------
// Brute-force reference: exact endgame value with no transposition table, move
// ordering, PVS, or leaf playout. Given a depth larger than any possible line
// length, every leaf is a real game end, so this is the exact optimal final
// spread from the side-to-move's perspective. It mirrors EndgameSolver's end
// rules exactly: the out bonus, the internal scoreless cap of 2, and the
// scoreless rebase at entry.
// ---------------------------------------------------------------------------

struct RefState {
  Board board;
  Rack racks[2];
  int scores[2];
  int scoreless;
  int stm;
};

// Apply `m` (a play or a pass) to `s` for the side to move, returning the next
// state and setting `over` when the move ends the game.
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

// Exact value of forcing `first` as the solving side's first move, then optimal
// play by both sides -- used to prove a specific first move is suboptimal.
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

constexpr int kRefDepth = 24;  // > any small-endgame line length
constexpr uint64_t kBigBudget = 1ull << 40;

// A synthetic PLAY placing dummy tiles at the given lane positions, for exercising
// the halo geometry directly (build_outplay_halo reads only the move's shape and
// the board, never the dictionary, so the glyphs' identity is irrelevant).
Move lane_play(bool horizontal, int line, const std::vector<int>& along, uint16_t score = 0) {
  uint16_t mask = 0;
  std::vector<Glyph> glyphs;
  for (int p : along) {
    mask |= 1u << p;
    glyphs.push_back(Glyph::of(Tile::of(0)));  // an 'A'; value is unused here
  }
  return Move::play(horizontal, line, mask, score, glyphs.data(), int(glyphs.size()));
}
Move horiz_play(int row, const std::vector<int>& cols, uint16_t score = 0) {
  return lane_play(true, row, cols, score);
}
Move vert_play(int col, const std::vector<int>& rows, uint16_t score = 0) {
  return lane_play(false, col, rows, score);
}

// The exact value of forcing `m` as the solving side's first move in position
// `p`, scored by `control` (a solver whose configuration the caller trusts):
// the terminal spread when `m` ends the game, else the negation of the control
// solver's full-window value for the resulting child position.
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

// What one A/B batch cost each way: total nodes searched with outplay futility
// pruning on, and over the same positions with it off.
struct PruningNodes {
  uint64_t pruned = 0;
  uint64_t unpruned = 0;
};

// A/B one batch of random endgames with futility pruning on vs off at full
// window and kRefDepth (so every line resolves and no playout leaf is
// consulted): the pruning must not change the value. The best move must either
// be identical or a proven tie: the scheme reorders moves (the ordering cap,
// and the fail-low bounds root re-ordering sorts by), so it can legitimately
// settle on a different optimum among equal-valued moves -- a divergent best
// move is accepted only if the control solver credits it with the same optimal
// value when forced. Node counts are returned per batch rather than compared
// per position: a reordering scheme can shift an individual position's node
// count either way. Failures use EXPECT, so all are reported.
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

// The batch's search-cost win, in the test output where a regression that
// silently defeats the pruning is visible as the cut shrinking towards zero.
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

// The solver's exact value must match the brute-force reference over many
// randomized small endgames, for both sides to move. With max_plies >= kRefDepth
// every line resolves to a real game end, so the leaf playout is never consulted
// and the two searches must agree exactly.
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

    // Other parity: opponent's rack/score becomes the side to move.
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

// A hand-picked position where an out-in-one exists and its 2x-remaining bonus
// makes it the optimal play. Verifies the exact value arithmetic and that the
// chosen move empties the rack.
TEST(EndgameSolver, OutInOneOptimal) {
  Dictionary d = tiny_dict();
  Board b;  // empty board: "GO" opens across the center and empties the rack

  const Rack my = rack_from("GO");
  const Rack opp = rack_from("BD");  // no tiny-dict word: opponent is stuck
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
  EXPECT_EQ(r.best.num_glyphs(), my.size());  // the out play
}

// Neither side can play: two passes end the game as a stalemate, and each side
// subtracts its own remaining tiles.
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

// One side is stuck while the other plays out over more than one turn. The
// solver's exact value matches the reference and beats what any single move can
// secure, since going out requires multiple plays.
TEST(EndgameSolver, StuckOpponentMultiTurnOut) {
  Dictionary d = tiny_dict();
  Board b;
  const Rack my = rack_from("CATS");  // needs multiple turns to shed all tiles
  const Rack opp = rack_from("VW");   // stuck: no legal play, ever
  const int my_score = 10, opp_score = 8;

  ASSERT_TRUE(MoveGenerator(b, d).generate(opp).empty());

  EndgameSolver solver;
  const EndgameResult deep =
    solver.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, 12, true});
  const int32_t ref = ref_solve(b, d, my, opp, my_score, opp_score, 0, kRefDepth);
  EXPECT_EQ(deep.value, ref);
  // No single opening play empties this 4-tile rack (the longest tiny-dict word
  // reachable here is 4 letters only via the board), so the optimal line spans
  // multiple turns; confirm the value strictly beats a here-and-now pass.
  const int32_t pass_now = (my_score - my.point_value()) - (opp_score - opp.point_value());
  EXPECT_GT(deep.value, pass_now);
}

// Depth-1 play is greedy; deeper search can find a higher-value hold-back. Scan
// randomized positions for one where the greedy first move is provably
// suboptimal (optimal play after it yields less than the true optimum) and
// assert the deep solve's exact value matches the reference.
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
      // The greedy first move genuinely forfeits value a deeper hold-back keeps.
      EXPECT_NE(greedy.best, deep.best);
      found = true;
    }
  }
  ASSERT_TRUE(found) << "no greedy-suboptimal position found in the scan";
}

// Identical inputs and budget must produce identical results, both across a
// clear() on the same solver and against a freshly constructed solver.
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

// The node budget is a hard cap across every pass a solve runs (class pass,
// spread pass, verification probes): nodes spent never exceed it by more than
// one greedy playout's plies -- the overshoot before the next negamax entry
// detects exhaustion -- under either objective. A legal move comes back even at
// absurdly small budgets, where depth_completed may be 0.
//
// Depth and nodes are deliberately not asserted monotone in the budget. Both
// objectives fork control flow at budget-dependent thresholds (the class pass
// proves or aborts at its half-budget cap, and what the fallback then reports
// comes from a different window), so both can legitimately report less depth at
// a larger budget.
TEST(EndgameSolver, NodeBudget) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0xB0DA711Eu);
  EndgameSolver solver;
  // One playout can run past the cap before the abort lands (kMaxPlayout = 40
  // plies), plus the detecting node itself.
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
        // The returned move is always legal: a pass or a generated play.
        if (r.best.type() != MoveType::PASS) {
          EXPECT_TRUE(legal.count(move_key(r.best)) > 0)
            << "budget " << budget << " position " << i << " spread_matters " << spread_matters;
        }
      }
    }
  }
}

// A position with more root moves than the node budget provably cannot complete
// its first iteration, so solve() declines it up front: no nodes are spent,
// depth_completed is 0, and the returned move is the statically best-estimated
// root move (still legal).
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
    // Root moves are the plays plus the pass, so a budget of plays.size() falls
    // one node short of the first iteration.
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

// The transposition table is spread-rebased, so entries survive across turns. A
// solve, then applying the PV move plus a reply, then solving again must not
// crash and must return consistent values: the child position's value from the
// opponent's perspective equals the negated parent value when the parent solve
// was exact.
TEST(EndgameSolver, TTReuseAcrossTurns) {
  Dictionary d = tiny_dict();
  std::mt19937 rng(0x7700FACEu);
  EndgameSolver solver;
  for (int i = 0; i < 20; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/3);
    const EndgameResult parent =
      solver.solve({&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0},
                   {kBigBudget, kRefDepth, true});

    // Apply the parent's best move; if it ends the game there is no child solve.
    RefState s = make_ref_state(p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0);
    bool over = false;
    const RefState after = ref_apply(s, parent.best, over);
    if (over) continue;

    // Reuse the warm TT: solve the child from the opponent's perspective.
    const EndgameResult child = solver.solve({&d, after.board, after.racks[1], after.racks[0],
                                              after.scores[1], after.scores[0], after.scoreless},
                                             {kBigBudget, kRefDepth, true});
    // The child's optimal value (opponent to move) is the negation of the value
    // the parent assigned to reaching it, i.e. the parent's optimum.
    EXPECT_EQ(child.value, -parent.value);
  }
}

// first_win pins the root window to (kFirstWinAlpha, kFirstWinBeta), so the
// search only has to resolve the win/draw/loss class instead of the exact
// spread. Over many random endgames, the wld value's class matches the
// brute-force reference's sign, and whenever the position is a proven win or
// draw the chosen move exactly preserves that class under optimal play by both
// sides afterward (a proven loss makes every root move losing, so there is no
// per-move claim to check there).
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

// The first-win window is narrower than the full (-inf, +inf) window, so
// pruning is never worse: the same batch of positions spends no more total
// search effort under first_win than under a full-window solve.
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

// Whenever a full-window solve reports its value as proven, that value is the
// exact game-theoretic spread and must equal the brute-force reference. The
// tiny endgames here terminate well within kRefDepth, so the proof fires on the
// clear majority of them -- the bit is not vacuously false.
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

// The iterative-deepening early exit returns exactly what stopping at the proof
// depth would: capping max_plies at the early-exit run's depth_completed runs
// the identical iterations, so value, best move, and node count all match. And
// because the proof depth is far below the kRefDepth horizon, the full-horizon
// solve spends no more nodes than the shallow depth-capped one -- the proof
// truncated the deepening. Run without spread_matters: its single pass makes
// "the same iterations up to the proof depth" well-defined, while the
// spread_matters driver's two passes each deepen on their own schedule.
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

// The short-circuit demonstrably fires and demonstrably pays: on positions
// whose whole game tree ends within a few plies (1-tile racks: every line is
// an out, a pass-out, or a pass-pass stalemate), the solve is proven at a
// shallow depth, and an identical solve with the short-circuit disabled keeps
// deepening over the same tiny tree -- same value, same move, strictly more
// nodes. Random 2-tile endgames then quantify the aggregate saving.
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
      // The short-circuit changes how many narrow-window class-pass iterations
      // run before the spread pass, and a root beta cutoff leaves the moves it
      // skips stale-ranked, so a different iteration count can reorder the root
      // and settle on a different equal-optimal move. A divergent best is
      // accepted only if a fresh control solver credits it with the same exact
      // value when forced (a true tie); a fresh solver keeps the measured
      // solvers' warm tables untouched.
      EndgameSolver control;
      EXPECT_EQ(forced_move_value(d, p, a.best, control), b.value)
        << "position " << i << ": divergent best move is not a tie";
    }
    nodes_on += a.nodes;
    nodes_off += b.nodes;
    ++checked;
    if (!a.proven) continue;
    ++proven;
    // The proof landed well below the horizon and the disabled control kept
    // deepening past it: the short-circuit saved real nodes on this position.
    EXPECT_LT(a.depth_completed, 25) << "position " << i;
    EXPECT_LT(a.nodes, b.nodes) << "position " << i;
    EXPECT_EQ(b.depth_completed, 25) << "position " << i;
  }
  ASSERT_GT(proven, checked / 2) << "shallow endgames should mostly be provable";
  EXPECT_LT(nodes_on * 2, nodes_off);  // the short-circuit at least halves the batch
  std::cout << "  proof short-circuit: " << proven << "/" << checked << " proven, nodes "
            << nodes_on << " vs " << nodes_off << " without early exit\n";
}

// In the first_win window a proven early exit settles the win/draw/loss class:
// the wld value's sign matches the brute-force reference's, and the narrower
// window plus its earlier proof never search more nodes than the full window.
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

// Every class-proven solve carries a valid proof certificate: starting from
// the returned best move, every entry is legal in its position (regenerated
// and matched exactly), the line ends the game under the solver's rules, and
// the final spread's class is the proven class. Checked across objectives and
// budgets.
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
    // A best move that itself ends the game needs no continuation; otherwise
    // the certificate must be present and finish the game.
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

// At a budget large enough to prove everything, the lexicographic objective is
// exactly the spread objective: the class pass proves the class, the spread
// pass proves the exact optimum, and an exact optimum's sign is its class. The
// two must agree on value, on the proven class (which must match the
// brute-force reference's sign), and on the move up to equal-valued ties.
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

// THE gate for "never sacrifice the class for points": under constrained
// budgets, whenever a lexicographic solve reports a proven class, the move it
// returns preserves that class under optimal play by both sides afterward (a
// proven loss exempts itself: every move loses). The scan also checks the
// proven class against the brute-force reference.
TEST(EndgameSolver, LexicographicPreservesProvenClass) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0xC1A55E5Fu);
  int class_proven = 0, checked = 0;
  for (int i = 0; i < 120; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));
    const uint64_t budget = 60 + 140 * (i % 5);  // 60..620: from starved to roomy
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

// proven_class is reported consistently across objectives: at full budget both
// the spread and first-win objectives prove these tiny endgames, and the class
// each reports matches the brute-force reference's sign.
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

// THE soundness gate for opponent-outplay futility pruning, and the guard on
// what it buys. Over a large random batch (tiny dictionary, and the real
// lexicon when installed) solving with pruning on and off must agree exactly on
// value and best move: a wrong halo-survival classification, a stale
// incrementally-maintained out-play entry, or a wrong U(m) bound would prune a
// mover move that mattered and change one of them. Pruning must also never
// search more nodes than its control, and on the real lexicon it must search
// strictly fewer: the same A/B solves already measure that, so pinning the
// search-cost win costs nothing beyond the comparison, and a regression that
// silently defeats the pruning fails here rather than passing as a merely sound
// no-op. The bar sits on the real-lexicon batch because that is where the win
// is large (a quarter of the nodes) rather than incidental -- the tiny
// dictionary's boards are too bare for the pruning to bite, and a few hundred
// nodes either way there says nothing about the scheme.
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

// collect_rack_outplays keeps only the rack-emptying plays, ranked by score;
// best_surviving_score reports the best one a query move provably leaves
// intact, and assign_surviving derives a child set by the same halo filter.
TEST(OutplaySet, CollectQueryFilter) {
  Board b;
  std::vector<Move> plays;
  plays.push_back(vert_play(3, {0, 1}, 30));    // out-play at the top of column 3
  plays.push_back(vert_play(3, {13, 14}, 20));  // out-play at the bottom of column 3
  plays.push_back(vert_play(9, {7}, 50));       // one tile: not rack-emptying
  OutplaySet outs;
  collect_rack_outplays(b, plays, /*rack_size=*/2, outs);
  ASSERT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].move.score(), 30);  // sorted by descending score

  // A pass places nothing, so every out-play survives and the best score wins.
  EXPECT_EQ(best_surviving_score(outs, Move::pass()), 30);
  // A move elsewhere on the board disturbs neither out-play.
  EXPECT_EQ(best_surviving_score(outs, horiz_play(7, {6, 7, 8})), 30);
  // Occupying the in-line extension cell of the top out-play kills it; the
  // bottom one, untouched, survives.
  EXPECT_EQ(best_surviving_score(outs, vert_play(3, {2})), 20);
  // A move reaching into both halos leaves no survivor.
  EXPECT_EQ(best_surviving_score(outs, vert_play(3, {2, 12})), kNoOutplaySurvivor);

  // The child set after that top-killing move holds exactly the bottom out-play.
  OutplaySet child;
  assign_surviving(outs, vert_play(3, {2}), child);
  ASSERT_EQ(child.size(), 1u);
  EXPECT_EQ(child[0].move.score(), 20);

  // A play list with no rack-emptying play yields an empty set.
  std::vector<Move> singles;
  singles.push_back(vert_play(9, {7}, 50));
  collect_rack_outplays(b, singles, /*rack_size=*/2, outs);
  EXPECT_TRUE(outs.empty());
}

// LeaveOutplays buckets a node's own play list by used-tile multiset: the
// out-plays it hands the child after move m are exactly the plays that spend
// m's leave, minus those m itself disturbs. All plays here spend 'A's from an
// "AA" rack, so a one-tile play's leave (one A) is emptied by the other
// one-tile plays and the whole rack by the two-tile play.
TEST(LeaveOutplays, BucketsByLeave) {
  Board b;
  const Rack rack = rack_from("AA");
  std::vector<Move> plays;
  plays.push_back(vert_play(3, {0, 1}, 30));  // spends AA: an out-play
  plays.push_back(vert_play(9, {7}, 10));     // spends one A
  plays.push_back(horiz_play(11, {4}, 12));   // spends one A
  LeaveOutplays lo(b, rack, plays);

  // After the one-tile play at (7, 9), the leave is a single A: its out-plays
  // are the other one-tile plays that survive it -- (11, 4) is far away, and
  // the play's own placement always touches its own halo.
  OutplaySet out;
  lo.collect_after(plays[1], out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].move.score(), 12);

  // A pass keeps the whole rack, so its bucket is the full-rack out-plays.
  lo.collect_after(Move::pass(), out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].move.score(), 30);

  // The two-tile play empties the rack: no leave, no out-plays.
  lo.collect_after(plays[0], out);
  EXPECT_TRUE(out.empty());
}

// The halo of a candidate out-play covers every cell whose occupation by a reply
// could stop it: its own squares, the cross-word neighbors of its placed tiles,
// and the in-line cell just beyond each end of its word.
TEST(OutplayHalo, CoversBlockingCells) {
  Board b;  // empty: the span never extends through existing tiles here
  const OutplayHalo h = build_outplay_halo(b, horiz_play(7, {7, 8}));
  EXPECT_TRUE(h.contains(7, 7));  // placed square
  EXPECT_TRUE(h.contains(7, 8));  // placed square
  EXPECT_TRUE(h.contains(6, 7));  // cross-word neighbor above a placed tile
  EXPECT_TRUE(h.contains(8, 8));  // cross-word neighbor below a placed tile
  EXPECT_TRUE(h.contains(7, 6));  // in-line cell before the word
  EXPECT_TRUE(h.contains(7, 9));  // in-line cell after the word
}

// When a placed cell sits at the end of an existing perpendicular run, the
// cross-word it forms includes the whole run, so the cells a reply can rewrite
// that cross-word from are the run's far ends -- not the (occupied) immediate
// neighbors. The halo must reach past the run.
TEST(OutplayHalo, CoversPerpendicularRunEnds) {
  Board b;
  b.apply(vert_play(7, {3, 4, 5}));  // an existing run at rows 3-5 of column 7
  // The out-play places at (6, 7), directly beneath the run: its cross-word
  // there is rows 3-6 of column 7.
  const OutplayHalo h = build_outplay_halo(b, horiz_play(6, {7}));
  EXPECT_TRUE(h.contains(2, 7));  // just past the run's far (top) end
  EXPECT_TRUE(h.contains(7, 7));  // just past the near (bottom) end
}

// THE soundness gate for the root beta cutoff: over a large random batch of
// first-win (spread_matters=false) solves, whenever the solve proves a class it
// must equal the brute-force reference's sign, and for a proven win or draw the
// returned move must preserve that class under optimal play by both sides
// afterward (a proven loss makes every move class-equal, so there is no per-move
// claim to check). The cutoff stops a root scan the instant a fail-high settles
// the class, so a cutoff that dropped a move that mattered would surface here as
// a wrong class or a class-forfeiting best move.
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

// The root cutoff demonstrably pays on won blowouts: scan for random endgames
// the reference scores as a decisive win, where a top-ordered winning root move
// fails high early and the cutoff skips the rest of the root. A solver with the
// cutoff disabled scans every root move at every depth instead. The cutoff is
// never worse per position (it only ever skips work), it strictly wins on most
// of the batch, and it cuts the aggregate node count. A blowout can tie when its
// fail-high lands on the last-scanned root move at the proving depth (nothing
// left to skip), which is why the strict claim is the batch total, not each
// position. Both solvers must still prove the same (won) class.
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
    if (ref < 30) continue;  // want a decisive win whose fail-high lands early
    const EndgameState state = {&d, p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0};
    on.clear();
    off.clear();
    const EndgameResult a = on.solve(state, {kBigBudget, kRefDepth, /*spread_matters=*/false});
    const EndgameResult b = off.solve(state, {kBigBudget, kRefDepth, /*spread_matters=*/false});
    ASSERT_EQ(a.proven_class, 1) << "position " << i;  // a blowout is a proven win
    ASSERT_EQ(b.proven_class, a.proven_class) << "position " << i;
    nodes_on += a.nodes;
    nodes_off += b.nodes;
    EXPECT_LE(a.nodes, b.nodes) << "position " << i;  // the cutoff never costs more
    if (a.nodes < b.nodes) ++improved;
    ++blowouts;
  }
  ASSERT_GT(blowouts, 0) << "no won-blowout positions found in the scan";
  EXPECT_GT(improved, blowouts / 2) << "the cutoff should strictly win on most blowouts";
  EXPECT_LT(nodes_on, nodes_off);
  std::cout << "  root-cutoff on " << blowouts << " won blowouts (" << improved
            << " strictly fewer): nodes " << nodes_on << " vs " << nodes_off << " without cutoff\n";
}

// THE soundness gate for root-level outplay futility pruning: over a large
// random batch of first-win solves at budgets from starved to unlimited,
// pruning on and off must prove the same classes, each matching the
// brute-force reference's sign, and for a proven win or draw the pruned
// solver's move must preserve that class under optimal play by both sides
// afterward (a proven loss makes every move class-equal -- and a loss is the
// one verdict root pruning can return without searching a single move, so the
// scan requires losses to occur). A wrong root bound would either prune a
// class-saving move (wrong class or class-forfeiting best) or fold an unsound
// bound into a fail-low (wrong class versus the reference). At the unlimited
// budget both solves must prove; there the pruning must also not cost nodes in
// aggregate (per-position counts can shift either way: pruned moves re-rank by
// their bounds, so later iterations search the root in a different order).
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

// The end of the block-or-outscore chain: when no root move blocks the
// opponent's out-plays or outscores them, the loss is proven from the bounds
// alone, with no search below the root. Here the mover is stuck (no plays, so
// the only root move is a pass) while the opponent holds an out-play; the pass
// leaves that out-play alive and its bound is a loss, so a first-win solve
// proves the loss having spent zero search nodes -- and still produces a valid
// certificate. The control solver with root pruning disabled must reach the
// same proven loss by actually searching.
TEST(EndgameSolver, RootFutilityProvesLossFromBoundsAlone) {
  Dictionary d = tiny_dict();
  Board b;
  const Rack my = rack_from("VV");   // V appears in no tiny-dict word: pass only
  const Rack opp = rack_from("GO");  // "GO" opens across the center and goes out
  const int my_score = 20, opp_score = 20;
  ASSERT_TRUE(MoveGenerator(b, d).generate(my).empty());
  ASSERT_NE(find_out_move(MoveGenerator(b, d).generate(opp), opp), nullptr);

  EndgameSolver on;
  const EndgameResult a =
    on.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, kRefDepth, false});
  EXPECT_TRUE(a.proven);
  EXPECT_EQ(a.proven_class, -1);
  EXPECT_EQ(a.nodes, 0u);  // the bounds settled the root; nothing was searched
  EXPECT_EQ(a.best.type(), MoveType::PASS);
  EXPECT_FALSE(a.continuation.empty());
  EXPECT_LT(ref_solve(b, d, my, opp, my_score, opp_score, 0, kRefDepth), 0);

  EndgameSolver off;
  off.set_root_futility(false);
  const EndgameResult c =
    off.solve({&d, b, my, opp, my_score, opp_score, 0}, {kBigBudget, kRefDepth, false});
  EXPECT_TRUE(c.proven);
  EXPECT_EQ(c.proven_class, -1);
  EXPECT_GT(c.nodes, 0u);  // without root pruning the same proof takes search
}

namespace {

// One curated endgame position, committed as a GCG under tests/data/: the
// solver must prove the expected win/draw/loss class for the side to move
// while expending at most `max_nodes` of a generous budget, and its
// certificate must exist. Add cases by dropping a GCG (with a #RackN pragma
// for the mover) into tests/data/ and appending a row.
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

// Curated GCG endgames resolve to their known class, cheaply. These run the
// real lexicon and skip when it is not installed.
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

// The logical movegens count is nonzero for any real solve and never shrinks
// with a larger budget on a fixed position: a bigger budget searches a superset
// of the smaller budget's deterministic work (each requests a move list at the
// same points), so it performs at least as many generations.
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

// Walk random game paths and assert every PathMoveLists materialization is
// byte-identical to a scratch generation -- including for a sibling probe
// (make a move, materialize the child, unmake, proceed with another move),
// the pattern a search's move scan produces.
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
        // Sibling probe.
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

// Incremental maintenance must be invisible in every solver observable: value,
// best move, node/movegen counts, proof state, and continuation, across
// windows and budget levels.
namespace {

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
