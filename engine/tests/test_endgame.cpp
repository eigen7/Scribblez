// GoogleTest suite for the endgame solver and its Board make/unmake support:
//   - Board::apply(move, &undo) / unapply(undo) round-trips, and
//   - EndgameSolver correctness against a brute-force reference plus oracle
//     positions, determinism, node-budget behavior, and cross-turn TT reuse.

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
