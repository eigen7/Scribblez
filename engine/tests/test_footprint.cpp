#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "training/footprint.h"
#include "training/footprint_mask.h"

#include <gtest/gtest.h>

#include <array>
#include <set>
#include <utility>

namespace scribblez {
namespace {

using CellSet = std::set<std::pair<int, int>>;

Glyph G(int letter_index) { return Glyph::of(Tile::of(letter_index)); }

// A move's placed squares (board frame), transposed when `flip`.
CellSet placed_set(const Move& m, bool flip) {
  CellSet s;
  visit_placed_squares(m,
                       [&](int r, int c) { s.insert(flip ? std::pair{c, r} : std::pair{r, c}); });
  return s;
}

CellSet cells_set(int cls, const Board& b, bool flip) {
  std::array<std::pair<int, int>, kFootprintMaxK> cells{};
  const int n = footprint_cells(cls, b, flip, cells);
  return CellSet(cells.begin(), cells.begin() + n);
}

TEST(Footprint, ClassCountAndCatchAlls) {
  EXPECT_EQ(kSlotsPerCell, 13);
  EXPECT_EQ(kAnchoredFootprints, 2925);
  EXPECT_EQ(kFootprintClasses, 2927);
  EXPECT_EQ(kPassClass, 2925);
  EXPECT_EQ(kExtraClass, 2926);
}

TEST(Footprint, NonPlayIsPassClass) {
  EXPECT_EQ(footprint_class(Move::pass(), false), kPassClass);
  EXPECT_EQ(footprint_class(Move::pass(), true), kPassClass);
}

TEST(Footprint, RoundTripEmptyBoardBothFlips) {
  Board b;
  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t mask = (1u << 5) | (1u << 6) | (1u << 7);  // cols 5,6,7 of row 7
  const Move m = Move::play(true, 7, mask, 0, played, 3);
  for (bool flip : {false, true}) {
    const int cls = footprint_class(m, flip);
    EXPECT_EQ(cells_set(cls, b, flip), placed_set(m, flip)) << "flip=" << flip;
  }
}

TEST(Footprint, SingleTileIsOrientationFree) {
  Glyph played[1] = {G(0)};
  const Move horiz = Move::play(true, 7, (1u << 5), 0, played, 1);  // lone tile at (7,5)
  const Move vert = Move::play(false, 5, (1u << 7), 0, played, 1);  // same square, other axis
  EXPECT_EQ(footprint_class(horiz, false), footprint_class(vert, false));
  EXPECT_EQ(footprint_class(horiz, false), (7 * 15 + 5) * kSlotsPerCell + 0);
}

TEST(Footprint, ThroughTileSkipped) {
  Board b;
  b.set(7, 6, G(4));  // an existing tile mid-span
  Glyph played[2] = {G(0), G(1)};
  const Move m = Move::play(true, 7, (1u << 5) | (1u << 7), 0, played, 2);  // threads col 6
  const CellSet expect = {{7, 5}, {7, 7}};
  EXPECT_EQ(cells_set(footprint_class(m, false), b, false), expect);
}

TEST(Footprint, FlipSwapsOrientation) {
  Glyph played[3] = {G(0), G(1), G(2)};
  const Move m = Move::play(true, 7, (1u << 5) | (1u << 6) | (1u << 7), 0, played, 3);
  // Unflipped: anchor (7,5), horizontal, k=3.
  EXPECT_EQ(footprint_class(m, false), (7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2)));
  // Flipped: anchor transposes to (5,7), orientation becomes vertical.
  EXPECT_EQ(footprint_class(m, true), (5 * 15 + 7) * kSlotsPerCell + (kFootprintMaxK + (3 - 2)));
}

TEST(Footprint, ImpossibleClassOnBoardReturnsZero) {
  Board b;
  b.set(7, 5, G(4));  // occupy the anchor
  std::array<std::pair<int, int>, kFootprintMaxK> cells{};
  const int occupied_anchor = (7 * 15 + 5) * kSlotsPerCell + 0;
  EXPECT_EQ(footprint_cells(occupied_anchor, b, false, cells), 0);
  // A horizontal k=7 anchored too close to the right edge cannot fit.
  const int off_edge = (7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2));
  EXPECT_EQ(footprint_cells(off_edge, b, false, cells), 0);
  // Catch-all classes cover no cells.
  EXPECT_EQ(footprint_cells(kPassClass, b, false, cells), 0);
  EXPECT_EQ(footprint_cells(kExtraClass, b, false, cells), 0);
}

int count_true(const FootprintMask& m) {
  int n = 0;
  for (bool b : m) n += b;
  return n;
}

// On an empty board every square reads as unconstrained, so the opp mask reduces
// to geometry + tile budget -- exercisable without a dictionary. (Cross-check
// gating and the self-mask soundness sweep need a real corpus; that lands with
// the FFI + sweep piece.)
TEST(FootprintMask, EmptyBoardGeometryAndBudget) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*tile_budget=*/7, /*flip=*/false, /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);    // horizontal k=3 fits
  EXPECT_FALSE(m[(7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2))]);  // horizontal k=7 off the edge
  EXPECT_TRUE(m[kPassClass]);                                      // pass always legal
  EXPECT_FALSE(m[kExtraClass]);                                    // dummy for a plays head
  EXPECT_EQ(count_true(m), 2295 + 1);  // every geometrically-fitting footprint + pass
}

TEST(FootprintMask, WinHeadKeepsNotWinSlot) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, 7, false, /*win_head=*/true, m);
  EXPECT_TRUE(m[kExtraClass]);
  EXPECT_EQ(count_true(m), 2295 + 2);  // + pass + not-win
}

TEST(FootprintMask, BudgetCapsK) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*tile_budget=*/2, false, false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (2 - 2))]);   // k=2 within budget
  EXPECT_FALSE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);  // k=3 over budget
}

TEST(FootprintMask, FlipMarksFlippedClassLegal) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, 7, /*flip=*/true, false, m);
  // The flip image of horizontal k=3 at (7,5) is vertical k=3 at (5,7).
  EXPECT_TRUE(m[(5 * 15 + 7) * kSlotsPerCell + (kFootprintMaxK + (3 - 2))]);
}

// The self mask is cross-check-oblivious (pure geometry + BFS), so it is fully
// testable without a dictionary.
TEST(SelfFootprintMask, ReachabilityFromStructure) {
  Board b;
  b.set(7, 7, G(4));  // a lone tile: the only structure to bridge from
  FootprintMask m;
  // opp_budget=1, self_budget=1 -> combined reach 2 tiles.
  self_footprint_mask(b, /*self_budget=*/1, /*opp_budget=*/1, /*flip=*/false, /*win=*/false, m);
  // (7,8) is one empty step from the tile: d=1 <= 2 -> a k=1 footprint is legal.
  EXPECT_TRUE(m[(7 * 15 + 8) * kSlotsPerCell + 0]);
  // (7,10) is three empty steps away: d=3 > 2 -> unreachable.
  EXPECT_FALSE(m[(7 * 15 + 10) * kSlotsPerCell + 0]);
  // A far corner is well beyond reach.
  EXPECT_FALSE(m[(0 * 15 + 0) * kSlotsPerCell + 0]);
  EXPECT_TRUE(m[kPassClass]);
}

TEST(SelfFootprintMask, CombinedBudgetWidensReach) {
  Board b;
  b.set(7, 7, G(4));
  FootprintMask m;
  const int far = (7 * 15 + 10) * kSlotsPerCell + 0;  // d=3 from the tile
  self_footprint_mask(b, 1, 1, false, false, m);      // reach 2 -> out
  EXPECT_FALSE(m[far]);
  self_footprint_mask(b, 2, 1, false, false, m);  // reach 3 -> in
  EXPECT_TRUE(m[far]);
}

TEST(SelfFootprintMask, BudgetCapsK) {
  Board b;
  b.set(7, 4, G(4));  // structure so nearby cells are reachable
  FootprintMask m;
  self_footprint_mask(b, /*self_budget=*/2, /*opp_budget=*/7, false, false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (2 - 2))]);   // k=2 within budget
  EXPECT_FALSE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);  // k=3 over budget
}

TEST(SelfFootprintMask, EmptyBoardTreatsAllReachable) {
  Board b;  // no structure -> game-start guard marks everything reachable
  FootprintMask m;
  self_footprint_mask(b, 7, 7, false, false, m);
  EXPECT_TRUE(m[(7 * 15 + 7) * kSlotsPerCell + 0]);
  EXPECT_EQ(count_true(m), 2295 + 1);  // every fitting footprint reachable + pass
}

}  // namespace
}  // namespace scribblez
