// GoogleTest suite for the endgame solver and its Board make/unmake support:
//   - Board::apply(move, &undo) / unapply(undo) round-trips, and
//   - EndgameSolver correctness against a brute-force reference plus oracle
//     positions, determinism, node-budget behavior, and cross-turn TT reuse.

#include "endgame/endgame_solver.h"
#include "endgame/outplay_threat.h"
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

Dictionary tiny_dict() {
  return Dictionary::build_from_words({"CAT", "CATS", "AT",     "AS",     "BAT", "BATS", "HE",
                                       "TO",  "ON",   "NO",     "IT",     "IS",  "OAT",  "OATS",
                                       "HAT", "HATS", "RAT",    "RATS",   "DOG", "GOD",  "GO",
                                       "OD",  "DO",   "AERIES", "PARTIED"});
}

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

Rack random_rack(std::mt19937& rng) {
  Rack r;
  std::uniform_int_distribution<int> pick(0, 26);  // 26 -> blank
  for (int i = 0; i < RACK_SIZE; ++i) {
    int v = pick(rng);
    r.add(v == 26 ? BLANK : Tile::of(v));
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

// A small endgame position: a board seeded with a few random plays, two short
// racks drawn from a curated tiny-dict letter set (so plays exist but branching
// stays low), and small random scores.
struct EndgamePos {
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score;
  int opp_score;
};

EndgamePos random_endgame(std::mt19937& rng, const Dictionary& d, int rack_tiles) {
  static const char kLetters[] = "ATSOCHEBDGRINO";
  EndgamePos p;
  const int setup = std::uniform_int_distribution<int>(1, 3)(rng);
  for (int k = 0; k < setup; ++k) {
    const Rack seed = random_rack(rng);
    const std::vector<Move> plays = MoveGenerator(p.board, d).generate(seed);
    if (plays.empty()) break;
    p.board.apply(plays[std::uniform_int_distribution<size_t>(0, plays.size() - 1)(rng)]);
  }
  std::uniform_int_distribution<int> letter(0, static_cast<int>(sizeof(kLetters) - 2));
  for (int i = 0; i < rack_tiles; ++i) {
    p.my_rack.add(Tile::from_char(kLetters[letter(rng)]));
    p.opp_rack.add(Tile::from_char(kLetters[letter(rng)]));
  }
  p.my_score = std::uniform_int_distribution<int>(0, 40)(rng);
  p.opp_score = std::uniform_int_distribution<int>(0, 40)(rng);
  return p;
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
Move lane_play(bool horizontal, int line, const std::vector<int>& along) {
  uint16_t mask = 0;
  std::vector<Glyph> glyphs;
  for (int p : along) {
    mask |= static_cast<uint16_t>(1u << p);
    glyphs.push_back(Glyph::of(Tile::of(0)));  // an 'A'; value is unused here
  }
  return Move::play(horizontal, line, mask, 0, glyphs.data(), static_cast<int>(glyphs.size()));
}
Move horiz_play(int row, const std::vector<int>& cols) { return lane_play(true, row, cols); }
Move vert_play(int col, const std::vector<int>& rows) { return lane_play(false, col, rows); }

// A/B one batch of random endgames with threat pruning on vs off at full window
// and kRefDepth (so every line resolves and no playout leaf is consulted): the
// pruning must not change the value or best move, and must not raise node count.
// Accumulates total nodes and counts any mismatch (via EXPECT, so all are
// reported).
void check_pruning_ab(const Dictionary& d, unsigned seed, int count, uint64_t& pruned_nodes,
                      uint64_t& unpruned_nodes) {
  std::mt19937 rng(seed);
  for (int i = 0; i < count; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2 + (i % 3));  // 2..4 tiles
    EndgameSolver on, off;
    off.set_threat_pruning(false);
    const EndgameResult a = on.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0,
                                     kBigBudget, kRefDepth);
    const EndgameResult b = off.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0,
                                      kBigBudget, kRefDepth);
    EXPECT_EQ(a.value, b.value) << "seed " << seed << " position " << i;
    EXPECT_TRUE(a.best == b.best) << "seed " << seed << " position " << i;
    EXPECT_LE(a.nodes, b.nodes) << "seed " << seed << " position " << i;
    pruned_nodes += a.nodes;
    unpruned_nodes += b.nodes;
  }
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
    const EndgameResult r = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score,
                                         /*scoreless_turns=*/0, kBigBudget, kRefDepth);
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    ASSERT_EQ(r.value, ref) << "position " << i << " (solving side to move)";

    // Other parity: opponent's rack/score becomes the side to move.
    solver.clear();
    const EndgameResult r2 = solver.solve(p.board, d, p.opp_rack, p.my_rack, p.opp_score,
                                          p.my_score, 0, kBigBudget, kRefDepth);
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
  const EndgameResult r = solver.solve(b, d, my, opp, my_score, opp_score, 0, kBigBudget, 8);
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
  const EndgameResult r = solver.solve(b, d, my, opp, my_score, opp_score, 0, kBigBudget, 6);
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
  const EndgameResult deep = solver.solve(b, d, my, opp, my_score, opp_score, 0, kBigBudget, 12);
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
    const EndgameResult deep = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                            p.opp_score, 0, kBigBudget, kRefDepth);
    const int32_t ref =
      ref_solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kRefDepth);
    ASSERT_EQ(deep.value, ref);

    solver.clear();
    const EndgameResult greedy = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                              p.opp_score, 0, kBigBudget, /*max_plies=*/1);
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
    const EndgameResult a =
      s1.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kBigBudget, 12);
    s1.clear();
    const EndgameResult b =
      s1.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kBigBudget, 12);
    EndgameSolver s2;
    const EndgameResult c =
      s2.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, kBigBudget, 12);
    ASSERT_EQ(a.value, b.value);
    ASSERT_EQ(a.value, c.value);
    ASSERT_EQ(a.depth_completed, b.depth_completed);
    ASSERT_EQ(a.nodes, b.nodes);
    ASSERT_EQ(a.nodes, c.nodes);
    ASSERT_TRUE(a.best == b.best && a.best == c.best);
  }
}

// The node budget is a hard cap: nodes spent never exceed it by more than one
// greedy playout's plies (the overshoot before the next negamax entry detects
// exhaustion), a legal move comes back even at absurdly small budgets (where
// depth_completed may be 0), and depth/nodes are monotone in the budget.
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
    int prev_depth = 0;
    uint64_t prev_nodes = 0;
    for (uint64_t budget : {5ull, 50ull, 500ull, 5000ull, 200000ull}) {
      solver.clear();
      const EndgameResult r =
        solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, budget, 12);
      EXPECT_LE(r.nodes, budget + kSlack) << "budget " << budget << " position " << i;
      EXPECT_GE(r.depth_completed, prev_depth);  // depth is monotone in budget
      EXPECT_GE(r.nodes, prev_nodes);            // nodes grow with budget
      // The returned move is always legal: a pass or a generated play.
      if (r.best.type() != MoveType::PASS) {
        EXPECT_TRUE(legal.count(move_key(r.best)) > 0) << "budget " << budget << " position " << i;
      }
      prev_depth = r.depth_completed;
      prev_nodes = r.nodes;
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
    const EndgameResult r =
      solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0, plays.size(), 12);
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
    const EndgameResult parent = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                              p.opp_score, 0, kBigBudget, kRefDepth);

    // Apply the parent's best move; if it ends the game there is no child solve.
    RefState s = make_ref_state(p.board, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0);
    bool over = false;
    const RefState after = ref_apply(s, parent.best, over);
    if (over) continue;

    // Reuse the warm TT: solve the child from the opponent's perspective.
    const EndgameResult child =
      solver.solve(after.board, d, after.racks[1], after.racks[0], after.scores[1], after.scores[0],
                   after.scoreless, kBigBudget, kRefDepth);
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
    const EndgameResult r =
      solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score,
                   /*scoreless_turns=*/0, kBigBudget, kRefDepth, /*first_win=*/true);
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
    const EndgameResult full =
      solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score,
                   /*scoreless_turns=*/0, kBigBudget, kRefDepth, /*first_win=*/false);
    solver.clear();
    const EndgameResult wld =
      solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score,
                   /*scoreless_turns=*/0, kBigBudget, kRefDepth, /*first_win=*/true);
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
    const EndgameResult r = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score,
                                         /*scoreless_turns=*/0, kBigBudget, kRefDepth);
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
// truncated the deepening.
TEST(EndgameSolver, EarlyExitPreservesResults) {
  Dictionary d = tiny_dict();
  EndgameSolver solver;
  std::mt19937 rng(0x1234ABCDu);
  int truncated = 0, checked = 0;
  for (int i = 0; i < 40; ++i) {
    const EndgamePos p = random_endgame(rng, d, /*rack_tiles=*/2);

    solver.clear();
    const EndgameResult early = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                             p.opp_score, 0, kBigBudget, kRefDepth);
    ASSERT_GE(early.depth_completed, 1) << "position " << i;
    solver.clear();
    const EndgameResult ctrl = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                            p.opp_score, 0, kBigBudget, early.depth_completed);
    ASSERT_EQ(early.value, ctrl.value) << "position " << i;
    ASSERT_TRUE(early.best == ctrl.best) << "position " << i;
    ASSERT_EQ(early.nodes, ctrl.nodes) << "position " << i;
    if (early.proven && early.depth_completed < kRefDepth) ++truncated;
    ++checked;
  }
  ASSERT_GT(truncated, 0) << "no proven solve truncated below the horizon";
  std::cout << "  early-exit truncated " << truncated << "/" << checked << " solves\n";
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
    const EndgameResult wld =
      solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score,
                   /*scoreless_turns=*/0, kBigBudget, kRefDepth, /*first_win=*/true);
    solver.clear();
    const EndgameResult full = solver.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score,
                                            p.opp_score, 0, kBigBudget, kRefDepth);
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

// THE soundness gate for outplay-threat futility pruning: over a large random
// batch (tiny dictionary, and the real lexicon when installed), solving with
// pruning on and off must agree exactly on value and best move, and pruning must
// never search more nodes. A wrong halo classification or a wrong U(g) bound
// would drop a reply that mattered and change one of these.
TEST(EndgameSolver, ThreatPruningIsSound) {
  uint64_t pruned = 0, unpruned = 0;
  const Dictionary tiny = tiny_dict();
  check_pruning_ab(tiny, 0x7A5C0DE1u, 200, pruned, unpruned);

  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (std::ifstream(path).good()) {
    const Dictionary real = Dictionary::load_kwg(path);
    check_pruning_ab(real, 0x1E51C04Du, 60, pruned, unpruned);
  } else {
    std::cout << "  (no lexicon at " << path << "; real-lexicon batch skipped)\n";
  }
  ASSERT_GT(unpruned, 0u);
  std::cout << "  threat-pruning nodes: " << pruned << " pruned vs " << unpruned << " unpruned\n";
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

// The pair-disjointness test fires (declares unblockable) only when no single
// reply can touch both halos. Each blocking mechanism -- occupying a square,
// poisoning a cross word from an adjacent column, extending a word in line --
// leaves a shared row or column that keeps the pair blockable.
TEST(OutplayHalo, PairUnblockableOnlyWhenDisjoint) {
  Board b;
  // Opposite corners share no row or column: a single reply cannot reach both.
  EXPECT_TRUE(halos_unblockable(build_outplay_halo(b, horiz_play(0, {0, 1})),
                                build_outplay_halo(b, horiz_play(14, {13, 14}))));
  // (a) Same row: a reply there can occupy a square of either out-play.
  EXPECT_FALSE(halos_unblockable(build_outplay_halo(b, horiz_play(7, {2, 3})),
                                 build_outplay_halo(b, horiz_play(7, {10, 11}))));
  // (b) One out-play's cross-word column is the other's word column: a vertical
  // reply there poisons one while building along the other.
  EXPECT_FALSE(halos_unblockable(build_outplay_halo(b, horiz_play(7, {7})),
                                 build_outplay_halo(b, vert_play(7, {2, 3}))));
  // (c) The in-line extension cell of one out-play lies in the other's column.
  EXPECT_FALSE(halos_unblockable(build_outplay_halo(b, horiz_play(7, {7, 8})),
                                 build_outplay_halo(b, vert_play(9, {2, 3}))));
}

// A visible-in-output regression guard: on a fixed batch, pruning must cut the
// total node count. The A/B soundness above already checks per-position value
// and best; this pins the search-cost win so a regression that silently defeats
// the pruning is caught. It prefers the real lexicon, whose richer boards make
// the pruning bite harder, and falls back to the tiny dictionary (a smaller but
// still positive cut) so the guard runs without a lexicon installed.
TEST(EndgameSolver, ThreatPruningCutsNodes) {
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  const bool have_real = std::ifstream(path).good();
  const Dictionary d = have_real ? Dictionary::load_kwg(path) : tiny_dict();
  const int rack_tiles = have_real ? 4 : 3;
  std::mt19937 rng(0xC0FFEE42u);
  uint64_t pruned = 0, unpruned = 0;
  for (int i = 0; i < 60; ++i) {
    const EndgamePos p = random_endgame(rng, d, rack_tiles);
    EndgameSolver on, off;
    off.set_threat_pruning(false);
    const EndgameResult a = on.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0,
                                     kBigBudget, kRefDepth);
    const EndgameResult b = off.solve(p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, 0,
                                      kBigBudget, kRefDepth);
    EXPECT_EQ(a.value, b.value) << "position " << i;  // sound at full resolution
    pruned += a.nodes;
    unpruned += b.nodes;
  }
  std::printf("  threat-pruning node count (%s): %llu pruned vs %llu unpruned (%.1f%% cut)\n",
              have_real ? "real lexicon" : "tiny dict", static_cast<unsigned long long>(pruned),
              static_cast<unsigned long long>(unpruned),
              100.0 * (1.0 - static_cast<double>(pruned) / static_cast<double>(unpruned)));
  EXPECT_LT(pruned, unpruned);
}
