#include "data/gcg_reader.h"
#include "encoding/game_state_encoder.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "training/footprint.h"
#include "training/footprint_collapse.h"
#include "training/footprint_mask.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {
namespace {

using CellSet = std::set<std::pair<int, int>>;

Glyph G(int letter_index) { return Glyph::of(Tile::of(letter_index)); }

// A 27-count availability supply (A..Z then blank at 26) holding one of each
// listed letter; '?' adds a wildcard blank. Unlisted letters stay 0 (out of
// stock). Feeds opp_footprint_mask / collapse_footprint_planes as `supply`.
std::array<uint8_t, 27> supply_of(const std::string& letters) {
  std::array<uint8_t, 27> s{};
  for (char ch : letters) {
    if (ch == '?')
      s[26] = 1;
    else
      s[ch - 'A'] = 1;
  }
  return s;
}

bool opp_admits(const Board& b, const std::array<uint8_t, 27>& supply, int cls) {
  FootprintMask m;
  opp_footprint_mask(b, supply.data(), RACK_SIZE, /*flip=*/false, /*win_head=*/false, m);
  return m[cls];
}

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

// On an empty board every square reads as unconstrained, so with no availability
// supply the opp mask reduces to geometry + tile budget -- exercisable without a
// dictionary. (Cross-check and availability gating need a real board/supply; see
// the availability tests below and the soundness sweep.)
TEST(FootprintMask, EmptyBoardGeometryAndBudget) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*supply=*/nullptr, /*tile_budget=*/7, /*flip=*/false, /*win_head=*/false,
                     m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);    // horizontal k=3 fits
  EXPECT_FALSE(m[(7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2))]);  // horizontal k=7 off the edge
  EXPECT_TRUE(m[kPassClass]);                                      // pass always legal
  EXPECT_FALSE(m[kExtraClass]);                                    // dummy for a plays head
  EXPECT_EQ(count_true(m), 2295 + 1);  // every geometrically-fitting footprint + pass
}

TEST(FootprintMask, WinHeadKeepsNotWinSlot) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*supply=*/nullptr, 7, false, /*win_head=*/true, m);
  EXPECT_TRUE(m[kExtraClass]);
  EXPECT_EQ(count_true(m), 2295 + 2);  // + pass + not-win
}

TEST(FootprintMask, BudgetCapsK) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*supply=*/nullptr, /*tile_budget=*/2, false, false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (2 - 2))]);   // k=2 within budget
  EXPECT_FALSE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);  // k=3 over budget
}

TEST(FootprintMask, FlipMarksFlippedClassLegal) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*supply=*/nullptr, 7, /*flip=*/true, false, m);
  // The flip image of horizontal k=3 at (7,5) is vertical k=3 at (5,7).
  EXPECT_TRUE(m[(5 * 15 + 7) * kSlotsPerCell + (kFootprintMaxK + (3 - 2))]);
}

// The opp mask gates a covered square's hook letters by availability: a footprint
// whose constrained cell has no available legal letter (and no wildcard blank) is
// masked out; supplying any legal hook -- or a blank -- readmits it. Board: 'A'
// above (7,7), so a horizontal tile there hooks the down-word "A_", legal for
// {X, Y} under the dict; (7,8) has no vertical neighbour, so it is unconstrained.
TEST(FootprintMask, AvailabilityGatesHookLetters) {
  Board b;
  b.set(6, 7, G(0));  // 'A'
  const Dictionary d = Dictionary::build_from_words({"AX", "AY"});
  b.ensure_movegen_caches(d);

  // Horizontal 2-tile footprint covering (7,7) [hook {X,Y}] and (7,8) [free].
  Glyph played[2] = {G(23), G(23)};  // placed letters are irrelevant to the mask
  const uint16_t sq = (1u << 7) | (1u << 8);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 2), /*flip=*/false);

  EXPECT_TRUE(opp_admits(b, supply_of("XY"), cls));    // both hooks in stock
  EXPECT_TRUE(opp_admits(b, supply_of("YE"), cls));    // one legal hook (Y) suffices
  EXPECT_TRUE(opp_admits(b, supply_of("XE"), cls));    // the other legal hook (X)
  EXPECT_FALSE(opp_admits(b, supply_of("EIO"), cls));  // no legal hook available -> masked
  EXPECT_TRUE(opp_admits(b, supply_of("?"), cls));     // a blank is a wildcard hook
  EXPECT_TRUE(opp_admits(b, supply_of("EIO?"), cls));  // ... even amid non-hook letters
  EXPECT_TRUE(opp_admits(b, supply_of("XY"), cls));    // sanity: unchanged on re-check

  // A null supply disables availability -- pure board legality readmits it.
  FootprintMask m;
  opp_footprint_mask(b, /*supply=*/nullptr, RACK_SIZE, false, false, m);
  EXPECT_TRUE(m[cls]);
}

// The unconstrained free cell (7,8) still needs SOME tile: an utterly empty
// supply (a blank-less bag with no listed letters) masks even a footprint whose
// only real constraint is "place a tile here".
TEST(FootprintMask, AvailabilityEmptySupplyMasksEverything) {
  Board b;
  b.set(6, 7, G(0));
  const Dictionary d = Dictionary::build_from_words({"AX", "AY"});
  b.ensure_movegen_caches(d);
  Glyph played[2] = {G(23), G(23)};
  const uint16_t sq = (1u << 7) | (1u << 8);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 2), /*flip=*/false);
  EXPECT_FALSE(opp_admits(b, supply_of(""), cls));  // nothing to place at all
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

std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Gate (b): the mask must never exclude a move that actually happens, or the
// masked-softmax cross-entropy would take -log(0) -> NaN on that target. Replay
// each real game; for every played move assert its footprint class is in the opp
// mask on the pre-move board (one ply ahead -- needs the lexicon for cross-checks)
// and in the self mask on the board two plies earlier (the self head's context;
// cross-check-oblivious, so it runs with or without the lexicon).
void sweep_game(const ParsedGcgGame& game, const Dictionary* dict) {
  Board board;
  if (dict) board.ensure_movegen_caches(*dict);
  std::optional<Board> two_plies_ago;  // board before the previous move
  for (const ParsedGcgTurn& turn : game.turns) {
    const Move& m = turn.record.move;
    if (m.type() == MoveType::PLAY) {
      const int cls = footprint_class(m, /*flip=*/false);
      if (dict) {
        // Availability supply the mover of `m` could draw from: everything off
        // the board (the loosest sound pool -- a superset of any one rack, so it
        // can never exclude a move the mover actually makes). This exercises the
        // availability path and, in the fixtures' endgames, its binding regime.
        uint8_t supply[27];
        compute_unseen_pool(supply, board, Rack{});
        FootprintMask opp;
        opp_footprint_mask(board, supply, RACK_SIZE, /*flip=*/false, /*win_head=*/false, opp);
        EXPECT_TRUE(opp[cls]) << "opp mask excluded a real move (class " << cls << ")";
      }
      if (two_plies_ago) {
        FootprintMask self;
        self_footprint_mask(*two_plies_ago, RACK_SIZE, RACK_SIZE, false, false, self);
        EXPECT_TRUE(self[cls]) << "self mask excluded a real move (class " << cls << ")";
      }
    }
    two_plies_ago = board;  // board before THIS move == two plies before the next
    if (m.type() == MoveType::PLAY) {
      board.apply(m);
      if (dict) board.ensure_movegen_caches(*dict);
    }
  }
}

TEST(FootprintMaskSoundness, RealGamesNeverMaskAPlayedMove) {
  std::optional<Dictionary> dict;
  const char* kwg = SCRIBBLEZ_DEFAULT_KWG;
  if (std::ifstream(kwg).good()) dict = Dictionary::load_kwg(kwg);

  const char* fixtures[] = {"boreal.gcg",  "egotize-lane.gcg",   "FOE.gcg",      "ole.gcg",
                            "violets.gcg", "postbingo-gave.gcg", "pos09-gnu.gcg"};
  int swept = 0;
  for (const char* name : fixtures) {
    const std::string text = slurp(std::string(SCRIBBLEZ_TEST_DATA_DIR) + "/" + name);
    if (text.empty()) continue;
    ParsedGcgGame game;
    std::string err;
    if (!read_gcg_text(text, &game, &err)) continue;  // skip an unparseable fixture
    sweep_game(game, dict ? &*dict : nullptr);
    ++swept;
  }
  EXPECT_GT(swept, 0) << "no fixtures swept";
}

// collapse_footprint_planes: overwhelming logit mass on one known footprint
// lands as ~1 on exactly the board cells that footprint covers, and ~0
// elsewhere. Pins the mask -> masked-softmax -> footprint_cells scatter,
// including that the scatter is not row/col transposed.
TEST(FootprintCollapse, MassLandsOnCoveredCells) {
  Board b;  // empty: every square unconstrained, so the opp mask is dict-free
  const Dictionary d = Dictionary::build_from_words({"CAT"});

  // A horizontal 3-tile play at row 7, cols 5,6,7 -- the first placement head's
  // target -- covers (7,5), (7,6), (7,7).
  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t sq = (1u << 5) | (1u << 6) | (1u << 7);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 3), /*flip=*/false);

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next); dwarfs the rest
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_planes(b, d, /*supply=*/nullptr, /*flip=*/false, raw.data(), out.data());

  const float* plane = out.data();  // head 0
  const auto cell = [&](int r, int c) { return plane[r * kFootprintSide + c]; };
  EXPECT_GT(cell(7, 5), 0.99f);
  EXPECT_GT(cell(7, 6), 0.99f);
  EXPECT_GT(cell(7, 7), 0.99f);
  EXPECT_LT(cell(5, 7), 0.01f);  // the transpose of (7,5): a row/col swap would light this
  EXPECT_LT(cell(7, 8), 0.01f);  // just past the covered run
  float total = 0.0f;
  for (int i = 0; i < kFootprintSide * kFootprintSide; ++i) total += plane[i];
  EXPECT_NEAR(total, 3.0f, 0.02f);  // three covered cells, ~all the mass
}

// The collapse actually applies the legality mask, not just softmax+scatter: a
// dominant logit on an ILLEGAL footprint contributes no plane mass. Uses the
// self head, whose mask excludes footprints too far to reach in two plies: a
// lone tile at (0,0) leaves (14,12..14) unreachable (distance 26+ > budget 14).
// Were the mask dropped, that footprint's huge logit would light its cells.
TEST(FootprintCollapse, IllegalFootprintGetsNoMass) {
  Board b;
  b.set(0, 0, G(0));  // the only structure; the far corner is unreachable from it
  const Dictionary d = Dictionary::build_from_words({"CAT"});

  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t sq = (1u << 12) | (1u << 13) | (1u << 14);
  const int illegal = footprint_class(Move::play(true, 14, sq, 0, played, 3), /*flip=*/false);
  ASSERT_LT(illegal, kAnchoredFootprints);

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[1 * kFootprintClasses + illegal] = 20.0f;  // head 1 (self_next); would dominate unmasked
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_planes(b, d, /*supply=*/nullptr, /*flip=*/false, raw.data(), out.data());

  const float* self_plane = out.data() + 1 * kFootprintSide * kFootprintSide;
  EXPECT_LT(self_plane[14 * kFootprintSide + 12], 0.01f);
  EXPECT_LT(self_plane[14 * kFootprintSide + 13], 0.01f);
  EXPECT_LT(self_plane[14 * kFootprintSide + 14], 0.01f);
  // The mass did not vanish -- masked-softmax spread it over the reachable
  // (legal) footprints near the tile, so the plane still sums to ~its tiles.
  float total = 0.0f;
  for (int i = 0; i < kFootprintSide * kFootprintSide; ++i) total += self_plane[i];
  EXPECT_GT(total, 0.5f);
}

// The collapse threads availability into the OPP heads: a dominant logit on an
// opp footprint whose only hook is unavailable contributes NO plane mass -- the
// mask drops it, so its 20-logit probability renormalizes away and (7,7) keeps
// only the faint uniform share of the other footprints that still cover it (a
// lone tile, verticals). With the hook letter supplied, the dominant footprint
// lights (7,7) instead. This is the inference-time "no Y unseen -> the I13 Y-hook
// carries no belief" behaviour in miniature (the real I13 collapses all the way
// because every footprint covering it needs the same unavailable hook).
TEST(FootprintCollapse, OppAvailabilityDropsUnsatisfiableFootprint) {
  Board b;
  b.set(6, 7, G(0));  // 'A' above (7,7): a horizontal hook there needs a "A_" word
  const Dictionary d = Dictionary::build_from_words({"AY"});  // sole hook letter: Y
  b.ensure_movegen_caches(d);

  Glyph played[2] = {G(23), G(23)};
  const uint16_t sq = (1u << 7) | (1u << 8);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 2), /*flip=*/false);

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next); dwarfs the rest
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);

  const std::array<uint8_t, 27> with_y = supply_of("YE");
  collapse_footprint_planes(b, d, with_y.data(), /*flip=*/false, raw.data(), out.data());
  const float lit = out[7 * kFootprintSide + 7];
  EXPECT_GT(lit, 0.9f);  // Y available -> the dominant hook lands on (7,7)

  const std::array<uint8_t, 27> no_y = supply_of("EIO");  // letters, but no Y, no blank
  collapse_footprint_planes(b, d, no_y.data(), /*flip=*/false, raw.data(), out.data());
  const float gated = out[7 * kFootprintSide + 7];
  EXPECT_LT(gated, 0.05f);       // the 20-logit footprint no longer reaches (7,7)
  EXPECT_LT(gated, lit * 0.1f);  // ... a >10x drop vs. when its hook was available
}

}  // namespace
}  // namespace scribblez
