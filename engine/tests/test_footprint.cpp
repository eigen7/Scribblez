// Placement footprints, the class space of the position-evaluation model's
// placement heads: the class encoding (training/footprint.h), the legality
// masks the heads' masked softmax uses (training/footprint_mask.h), and the
// collapse from per-class logits to per-cell planes
// (training/footprint_collapse.h). Class indices are written out as
// (row * 15 + col) * kSlotsPerCell + slot, where slot 0 is a lone tile,
// 1 + (k - 2) a horizontal k-tile footprint, and kFootprintMaxK + (k - 2) a
// vertical one.

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

// A 27-count availability array (A..Z, then blank at 26) with one of each
// listed letter; '?' adds a blank. Unlisted letters are out of stock.
std::array<uint8_t, 27> available_of(const std::string& letters) {
  std::array<uint8_t, 27> s{};
  for (char ch : letters) {
    if (ch == '?')
      s[26] = 1;
    else
      s[ch - 'A'] = 1;
  }
  return s;
}

bool opp_admits(const Board& b, const std::array<uint8_t, 27>& available_counts, int cls) {
  FootprintMask m;
  opp_footprint_mask(b, available_counts.data(), RACK_SIZE, /*win_head=*/false, m);
  return m[cls];
}

CellSet placed_set(const Move& m) {
  CellSet s;
  visit_placed_squares(m, [&](int r, int c) { s.insert({r, c}); });
  return s;
}

CellSet cells_set(int cls, const Board& b) {
  std::array<std::pair<int, int>, kFootprintMaxK> cells{};
  const int n = footprint_cells(cls, b, cells);
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
  EXPECT_EQ(footprint_class(Move::pass()), kPassClass);
  EXPECT_EQ(footprint_class(Move::pass().transpose()), kPassClass);
}

TEST(Footprint, RoundTripEmptyBoardBothFrames) {
  Board b;
  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t mask = (1u << 5) | (1u << 6) | (1u << 7);  // cols 5,6,7 of row 7
  const Move m = Move::play(true, 7, mask, 0, played, 3);
  for (bool transposed : {false, true}) {
    const Board bt = transposed ? b.transpose() : b;
    const Move mt = transposed ? m.transpose() : m;
    EXPECT_EQ(cells_set(footprint_class(mt), bt), placed_set(mt)) << "transposed=" << transposed;
  }
}

TEST(Footprint, SingleTileIsOrientationFree) {
  Glyph played[1] = {G(0)};
  const Move horiz = Move::play(true, 7, (1u << 5), 0, played, 1);  // lone tile at (7,5)
  const Move vert = Move::play(false, 5, (1u << 7), 0, played, 1);  // same square, other axis
  EXPECT_EQ(footprint_class(horiz), footprint_class(vert));
  EXPECT_EQ(footprint_class(horiz), (7 * 15 + 5) * kSlotsPerCell + 0);
}

TEST(Footprint, ThroughTileSkipped) {
  Board b;
  b.set(7, 6, G(4));  // an existing tile mid-span
  Glyph played[2] = {G(0), G(1)};
  const Move m = Move::play(true, 7, (1u << 5) | (1u << 7), 0, played, 2);  // threads col 6
  const CellSet expect = {{7, 5}, {7, 7}};
  EXPECT_EQ(cells_set(footprint_class(m), b), expect);
}

TEST(Footprint, TransposeSwapsOrientation) {
  Glyph played[3] = {G(0), G(1), G(2)};
  const Move m = Move::play(true, 7, (1u << 5) | (1u << 6) | (1u << 7), 0, played, 3);
  // Natural frame: anchor (7,5), horizontal, k=3.
  EXPECT_EQ(footprint_class(m), (7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2)));
  // Transposed: the anchor moves to (5,7) and the orientation becomes vertical.
  EXPECT_EQ(footprint_class(m.transpose()),
            (5 * 15 + 7) * kSlotsPerCell + (kFootprintMaxK + (3 - 2)));
}

TEST(Footprint, ImpossibleClassOnBoardReturnsZero) {
  Board b;
  b.set(7, 5, G(4));  // occupy the anchor
  std::array<std::pair<int, int>, kFootprintMaxK> cells{};
  const int occupied_anchor = (7 * 15 + 5) * kSlotsPerCell + 0;
  EXPECT_EQ(footprint_cells(occupied_anchor, b, cells), 0);
  // A horizontal k=7 anchored too close to the right edge cannot fit.
  const int off_edge = (7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2));
  EXPECT_EQ(footprint_cells(off_edge, b, cells), 0);
  // Catch-all classes cover no cells.
  EXPECT_EQ(footprint_cells(kPassClass, b, cells), 0);
  EXPECT_EQ(footprint_cells(kExtraClass, b, cells), 0);
}

int count_true(const FootprintMask& m) {
  int n = 0;
  for (bool b : m) n += b;
  return n;
}

// On an empty board every square is unconstrained, so without availability
// counts the opp mask reduces to geometry plus the tile budget, and needs no
// dictionary.
TEST(FootprintMask, EmptyBoardGeometryAndBudget) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, /*tile_budget=*/7,
                     /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);    // horizontal k=3 fits
  EXPECT_FALSE(m[(7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2))]);  // horizontal k=7 off the edge
  EXPECT_TRUE(m[kPassClass]);                                      // pass always legal
  EXPECT_FALSE(m[kExtraClass]);                                    // unused by a plays head
  EXPECT_EQ(count_true(m), 2295 + 1);  // every footprint that fits, plus pass
}

TEST(FootprintMask, WinHeadKeepsNotWinSlot) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, 7, /*win_head=*/true, m);
  EXPECT_TRUE(m[kExtraClass]);
  EXPECT_EQ(count_true(m), 2295 + 2);  // plus pass and not-win
}

TEST(FootprintMask, BudgetCapsK) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, /*tile_budget=*/2, /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (2 - 2))]);   // k=2 within budget
  EXPECT_FALSE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);  // k=3 over budget
}

// The class of a footprint's transpose: the anchor cell transposes and a
// multi-tile slot moves between the horizontal and vertical blocks.
int transposed_class(int cls) {
  const int cell = cls / kSlotsPerCell;
  const int slot = cls % kSlotsPerCell;
  int tslot = slot;
  if (slot >= kFootprintMaxK) {
    tslot = slot - (kFootprintMaxK - 1);
  } else if (slot >= 1) {
    tslot = slot + (kFootprintMaxK - 1);
  }
  const int r = cell / kFootprintSide;
  const int c = cell % kFootprintSide;
  return (c * kFootprintSide + r) * kSlotsPerCell + tslot;
}

// The mask of the transposed board is the transposed mask, class for class.
// Also exercises the cross-check caches Board::transpose carries over.
TEST(FootprintMask, TransposedBoardMaskIsTheTransposedMask) {
  Board b;
  b.set(6, 7, G(0));  // 'A' above (7,7): a hook constraint that breaks the symmetry
  b.set(9, 2, G(2));
  const Dictionary d = Dictionary::build_from_words({"AX", "AY", "CAT"});
  b.ensure_movegen_caches(d);
  const std::array<uint8_t, 27> avail = available_of("CATXY");
  FootprintMask m, mt;
  opp_footprint_mask(b, avail.data(), 7, /*win_head=*/false, m);
  opp_footprint_mask(b.transpose(), avail.data(), 7, /*win_head=*/false, mt);
  int legal = 0;
  for (int cls = 0; cls < kAnchoredFootprints; ++cls) {
    legal += m[cls];
    EXPECT_EQ(mt[transposed_class(cls)], m[cls]) << "cls=" << cls;
  }
  EXPECT_GT(legal, 0);
  EXPECT_LT(legal, kAnchoredFootprints);
}

// A footprint is masked when a constrained cell it covers has no legal letter
// in stock and no blank. 'A' above (7,7) means a tile there forms the down-word
// "A_", legal only for X and Y; (7,8) has no vertical neighbour.
TEST(FootprintMask, AvailabilityGatesHookLetters) {
  Board b;
  b.set(6, 7, G(0));  // 'A'
  const Dictionary d = Dictionary::build_from_words({"AX", "AY"});
  b.ensure_movegen_caches(d);

  // Horizontal 2-tile footprint covering (7,7) and (7,8).
  Glyph played[2] = {G(23), G(23)};  // the mask ignores the placed letters
  const uint16_t sq = (1u << 7) | (1u << 8);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 2));

  EXPECT_TRUE(opp_admits(b, available_of("XY"), cls));    // both hooks in stock
  EXPECT_TRUE(opp_admits(b, available_of("YE"), cls));    // one legal hook (Y) suffices
  EXPECT_TRUE(opp_admits(b, available_of("XE"), cls));    // the other legal hook (X)
  EXPECT_FALSE(opp_admits(b, available_of("EIO"), cls));  // no legal hook available -> masked
  EXPECT_TRUE(opp_admits(b, available_of("?"), cls));     // a blank is a wildcard hook
  EXPECT_TRUE(opp_admits(b, available_of("EIO?"), cls));  // ... even amid non-hook letters
  EXPECT_TRUE(opp_admits(b, available_of("XY"), cls));    // no state carried between calls

  // Null availability disables the gate, leaving board legality only.
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, RACK_SIZE, /*win_head=*/false, m);
  EXPECT_TRUE(m[cls]);
}

// Even an unconstrained cell needs some tile, so empty availability masks
// every footprint.
TEST(FootprintMask, AvailabilityEmptyMasksEverything) {
  Board b;
  b.set(6, 7, G(0));
  const Dictionary d = Dictionary::build_from_words({"AX", "AY"});
  b.ensure_movegen_caches(d);
  Glyph played[2] = {G(23), G(23)};
  const uint16_t sq = (1u << 7) | (1u << 8);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 2));
  EXPECT_FALSE(opp_admits(b, available_of(""), cls));  // nothing to place at all
}

// A lone tile below a letter forms the down-word, so it needs an in-stock
// letter that completes that word. An unconstrained across axis does not make
// any letter playable: the tile must satisfy both axes at once. 'A' is above
// (7,7), and "A_" is legal only for X and Y.
TEST(FootprintMask, AvailabilityGatesLoneTileHook) {
  Board b;
  b.set(6, 7, G(0));  // 'A'
  const Dictionary d = Dictionary::build_from_words({"AX", "AY"});
  b.ensure_movegen_caches(d);

  const int lone = (7 * 15 + 7) * kSlotsPerCell + 0;
  EXPECT_TRUE(opp_admits(b, available_of("XY"), lone));    // both down-hooks in stock
  EXPECT_TRUE(opp_admits(b, available_of("YE"), lone));    // one legal down-hook (Y) suffices
  EXPECT_TRUE(opp_admits(b, available_of("?"), lone));     // a blank is a wildcard hook
  EXPECT_FALSE(opp_admits(b, available_of("EIO"), lone));  // no legal down-hook
  EXPECT_FALSE(opp_admits(b, available_of(""), lone));     // nothing to place at all

  // Under board legality alone it is playable, since X and Y exist.
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, RACK_SIZE, /*win_head=*/false, m);
  EXPECT_TRUE(m[lone]);
}

// A lone tile at a cross-point forms both cross-words, so one letter must fit
// both; checking each axis separately is not enough. With 'A' above (7,7)
// (down-word needs X or Y) and 'B' left of it (across-word needs E), no letter
// fits both, so the square is unplayable even with X, Y and E in stock.
TEST(FootprintMask, AvailabilityLoneTileNeedsBothCrossWords) {
  const int lone = (7 * 15 + 7) * kSlotsPerCell + 0;

  Board disjoint;
  disjoint.set(6, 7, G(0));  // 'A' above: down-word "A_" -> {X, Y}
  disjoint.set(7, 6, G(1));  // 'B' left:  across-word "B_" -> {E}
  const Dictionary dd = Dictionary::build_from_words({"AX", "AY", "BE"});
  disjoint.ensure_movegen_caches(dd);
  EXPECT_FALSE(opp_admits(disjoint, available_of("XYE"), lone));  // no letter fits both words
  EXPECT_FALSE(
    opp_admits(disjoint, available_of("?"), lone));  // no jointly-legal square for a blank

  Board overlap;
  overlap.set(6, 7, G(0));                                           // down-word "A_"
  overlap.set(7, 6, G(1));                                           // across-word "B_"
  const Dictionary od = Dictionary::build_from_words({"AE", "BE"});  // both admit only E
  overlap.ensure_movegen_caches(od);
  EXPECT_TRUE(opp_admits(overlap, available_of("E"), lone));    // E fits both words and is in stock
  EXPECT_FALSE(opp_admits(overlap, available_of("XY"), lone));  // only E fits; X/Y satisfy neither
}

// After the opening move, a placement must connect to a tile on the board. An
// empty corner far from any tile has unconstrained cross-checks, so a per-cell
// letter test alone would admit footprints there although no legal move can
// reach them. Connectivity is checked independently of availability.
TEST(FootprintMask, ConnectivityMasksFloatingPlacements) {
  Board b;
  b.set(7, 7, G(0));  // 'A': the sole structure everything must connect to
  const Dictionary d = Dictionary::build_from_words({"AB"});
  b.ensure_movegen_caches(d);

  const int corner_lone = (0 * 15 + 0) * kSlotsPerCell + 0;
  const int floating2 = (0 * 15 + 3) * kSlotsPerCell + (1 + (2 - 2));  // in empty row 0
  const int hook_lone = (7 * 15 + 8) * kSlotsPerCell + 0;              // lone tile right of 'A'
  const int abutting2 = (7 * 15 + 8) * kSlotsPerCell + (1 + (2 - 2));  // horiz k=2 abutting 'A'

  EXPECT_FALSE(opp_admits(b, available_of("AB"), corner_lone));
  EXPECT_FALSE(opp_admits(b, available_of("AB"), floating2));
  // Both cover (7,8), which abuts 'A'. The lone tile hooks "AB"; the pair's
  // cells have free vertical cross-checks.
  EXPECT_TRUE(opp_admits(b, available_of("B"), hook_lone));
  EXPECT_TRUE(opp_admits(b, available_of("AB"), abutting2));

  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, RACK_SIZE, /*win_head=*/false, m);
  EXPECT_FALSE(m[corner_lone]);
  EXPECT_FALSE(m[floating2]);
  EXPECT_TRUE(m[hook_lone]);
  EXPECT_TRUE(m[abutting2]);
}

// The self mask covers the mover's next move, one ply after the opponent's. It
// expands twice: first the squares the opponent's move could cover, then the
// mover's footprints that connect to that enlarged board. Without a dictionary
// every cross-check is unconstrained, so these tests run on geometry alone.
TEST(SelfFootprintMask, ReachabilityFromStructure) {
  Board b;
  b.set(7, 7, G(4));  // the only tile on the board
  FootprintMask m;
  self_footprint_mask(b, /*self_budget=*/1, /*opp_budget=*/1, /*opp_available_counts=*/nullptr,
                      /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 8) * kSlotsPerCell + 0]);  // abuts the tile itself
  // The opponent's one tile reaches at most (7,8), so (7,10) abuts nothing on
  // any board the opponent can leave.
  EXPECT_FALSE(m[(7 * 15 + 10) * kSlotsPerCell + 0]);
  EXPECT_FALSE(m[(0 * 15 + 0) * kSlotsPerCell + 0]);
  EXPECT_TRUE(m[kPassClass]);
}

// A larger self budget widens reach only through footprints that connect. With
// the opponent able to fill (7,8), the mover's pair at (7,9)-(7,10) connects,
// but a lone tile at (7,10) still does not: the cell is coverable, that
// footprint is not.
TEST(SelfFootprintMask, WiderBudgetReachesThroughConnectedFootprints) {
  Board b;
  b.set(7, 7, G(4));
  FootprintMask m;
  const int lone_7_10 = (7 * 15 + 10) * kSlotsPerCell + 0;
  const int pair_7_9 = (7 * 15 + 9) * kSlotsPerCell + (1 + (2 - 2));  // (7,9),(7,10)
  self_footprint_mask(b, /*self_budget=*/1, /*opp_budget=*/1, nullptr, /*win_head=*/false, m);
  EXPECT_FALSE(m[lone_7_10]);
  EXPECT_FALSE(m[pair_7_9]);  // k=2 exceeds self_budget=1
  self_footprint_mask(b, /*self_budget=*/2, /*opp_budget=*/1, nullptr, /*win_head=*/false, m);
  EXPECT_TRUE(m[pair_7_9]);  // (7,9) abuts the opponent's (7,8)
  EXPECT_FALSE(m[lone_7_10]);
}

TEST(SelfFootprintMask, BudgetCapsK) {
  Board b;
  b.set(7, 4, G(4));
  FootprintMask m;
  self_footprint_mask(b, /*self_budget=*/2, /*opp_budget=*/7, nullptr, /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (2 - 2))]);   // k=2 within budget
  EXPECT_FALSE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);  // k=3 over budget
}

TEST(SelfFootprintMask, EmptyBoardTreatsAllReachable) {
  Board b;  // no tiles, so no connectivity requirement (the opening move)
  FootprintMask m;
  self_footprint_mask(b, 7, 7, nullptr, /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 7) * kSlotsPerCell + 0]);
  EXPECT_EQ(count_true(m), 2295 + 1);  // every footprint that fits, plus pass
}

// The opponent's expansion applies cross-checks and the unseen pool; the
// mover's does not. 'A' at (6,7) means (7,7) takes only Y.
TEST(SelfFootprintMask, OppStageGatesReachTheMoverStageDoesNot) {
  Board b;
  b.set(6, 7, G(0));
  const Dictionary d = Dictionary::build_from_words({"AY"});
  b.ensure_movegen_caches(d);
  const int lone_7_7 = (7 * 15 + 7) * kSlotsPerCell + 0;  // the hook square itself
  const int lone_8_7 = (8 * 15 + 7) * kSlotsPerCell + 0;  // below the hook square
  FootprintMask m;

  // With Y in the pool the opponent can fill (7,7), which (8,7) then abuts.
  const std::array<uint8_t, 27> with_y = available_of("YE");
  self_footprint_mask(b, /*self_budget=*/1, /*opp_budget=*/1, with_y.data(), false, m);
  EXPECT_TRUE(m[lone_8_7]);

  const std::array<uint8_t, 27> no_y = available_of("EIO");
  self_footprint_mask(b, 1, 1, no_y.data(), false, m);
  EXPECT_FALSE(m[lone_8_7]);
  // The mover's own tile at (7,7) stays legal: the mover's rack is not drawn
  // from the pool, so it may hold the Y the pool lacks.
  EXPECT_TRUE(m[lone_7_7]);
}

// A ply's reach is its seed (the occupied squares) plus every square its legal
// footprints cover.
TEST(FootprintPly, ReachIsSeedPlusCoveredSquares) {
  Board b;
  b.set(7, 7, G(4));
  const FootprintPly ply = footprint_ply(b, occupied_squares(b), /*budget=*/7,
                                         /*use_cross_checks=*/false, nullptr, /*win_head=*/false);
  EXPECT_TRUE(ply.reach.contains(7, 7));   // the seed
  EXPECT_TRUE(ply.reach.contains(7, 8));   // a lone tile
  EXPECT_TRUE(ply.reach.contains(7, 9));   // a 2-tile word from (7,8)
  EXPECT_TRUE(ply.reach.contains(7, 14));  // a 7-tile word
  EXPECT_FALSE(ply.reach.contains(0, 0));
  EXPECT_TRUE(occupied_squares(Board{}).empty());  // the opener has no seed
  EXPECT_FALSE(occupied_squares(b).empty());
}

std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// The masks must never exclude a move that is actually played: the masked
// softmax cross-entropy would take -log(0) on that target. Replays real games
// and checks every played move against the opp mask on the pre-move board, and
// against the self mask on the board two plies earlier, which is the self
// head's context. The opp check needs the lexicon for cross-checks; the self
// check runs without it, since every cross-check is then unconstrained.
void sweep_game(const ParsedGcgGame& game, const Dictionary* dict) {
  Board board;
  if (dict) board.ensure_movegen_caches(*dict);
  std::optional<Board> two_plies_ago;
  for (const ParsedGcgTurn& turn : game.turns) {
    const Move& m = turn.record.move;
    if (m.type() == MoveType::PLAY) {
      const int cls = footprint_class(m);
      if (dict) {
        // Everything off the board: a superset of the mover's rack, so a sound
        // pool that still exercises availability gating, and binds in the
        // fixtures' endgames.
        uint8_t available_counts[27];
        compute_unseen_pool(available_counts, board, Rack{});
        FootprintMask opp;
        opp_footprint_mask(board, available_counts, RACK_SIZE, /*win_head=*/false, opp);
        EXPECT_TRUE(opp[cls]) << "opp mask excluded a real move (class " << cls << ")";
      }
      if (two_plies_ago) {
        uint8_t opp_available_counts[27];
        compute_unseen_pool(opp_available_counts, *two_plies_ago, Rack{});
        FootprintMask self;
        self_footprint_mask(*two_plies_ago, RACK_SIZE, RACK_SIZE, opp_available_counts,
                            /*win_head=*/false, self);
        EXPECT_TRUE(self[cls]) << "self mask excluded a real move (class " << cls << ")";
      }
    }
    two_plies_ago = board;  // two plies before the next move
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
  for (const char* name : fixtures) {
    const std::string text = slurp(std::string(SCRIBBLEZ_TEST_DATA_DIR) + "/" + name);
    ASSERT_FALSE(text.empty()) << "missing or empty fixture " << name;
    ParsedGcgGame game;
    std::string err;
    ASSERT_TRUE(read_gcg_text(text, &game, &err)) << name << ": " << err;
    sweep_game(game, dict ? &*dict : nullptr);
  }
}

// A dominant logit on one footprint lands ~1 on exactly the cells it covers
// and ~0 elsewhere, which pins the scatter from classes to cells, including
// its orientation.
TEST(FootprintCollapse, MassLandsOnCoveredCells) {
  Board b;  // empty, so every square is unconstrained
  const Dictionary d = Dictionary::build_from_words({"CAT"});

  // Covers (7,5), (7,6), (7,7).
  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t sq = (1u << 5) | (1u << 6) | (1u << 7);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 3));

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next); dwarfs the rest
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_planes(b, d, /*available_counts=*/nullptr, raw.data(), out.data());

  const float* plane = out.data();
  const auto cell = [&](int r, int c) { return plane[r * kFootprintSide + c]; };
  EXPECT_GT(cell(7, 5), 0.99f);
  EXPECT_GT(cell(7, 6), 0.99f);
  EXPECT_GT(cell(7, 7), 0.99f);
  EXPECT_LT(cell(5, 7), 0.01f);  // (7,5) transposed: a row/col swap would light this
  EXPECT_LT(cell(7, 8), 0.01f);  // just past the covered run
  float total = 0.0f;
  for (int i = 0; i < kFootprintSide * kFootprintSide; ++i) total += plane[i];
  EXPECT_NEAR(total, 3.0f, 0.02f);
}

// The collapse applies the legality mask: a dominant logit on an illegal
// footprint contributes no mass. Uses the self head, whose mask excludes
// footprints two plies cannot reach: from a lone tile at (0,0), (14,12..14) is
// 26+ squares away against a combined budget of 14.
TEST(FootprintCollapse, IllegalFootprintGetsNoMass) {
  Board b;
  b.set(0, 0, G(0));  // the only structure; the far corner is unreachable from it
  const Dictionary d = Dictionary::build_from_words({"CAT"});

  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t sq = (1u << 12) | (1u << 13) | (1u << 14);
  const int illegal = footprint_class(Move::play(true, 14, sq, 0, played, 3));
  ASSERT_LT(illegal, kAnchoredFootprints);

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[1 * kFootprintClasses + illegal] = 20.0f;  // head 1 (self_next)
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_planes(b, d, /*available_counts=*/nullptr, raw.data(), out.data());

  const float* self_plane = out.data() + 1 * kFootprintSide * kFootprintSide;
  EXPECT_LT(self_plane[14 * kFootprintSide + 12], 0.01f);
  EXPECT_LT(self_plane[14 * kFootprintSide + 13], 0.01f);
  EXPECT_LT(self_plane[14 * kFootprintSide + 14], 0.01f);
  // The mass went to the legal footprints near the tile instead.
  float total = 0.0f;
  for (int i = 0; i < kFootprintSide * kFootprintSide; ++i) total += self_plane[i];
  EXPECT_GT(total, 0.5f);
}

// The collapse applies availability to the opp heads. A dominant logit on a
// footprint whose only hook letter (Y) is unavailable is masked out, so (7,7)
// keeps only its share of the other legal footprints; with Y available it gets
// ~1. That share is not near zero, because connectivity limits the legal set
// to footprints touching the lone 'A' and (7,7) is one of them. The check is
// that the dominant spike disappears.
TEST(FootprintCollapse, OppAvailabilityDropsUnsatisfiableFootprint) {
  Board b;
  b.set(6, 7, G(0));                                          // 'A' above (7,7)
  const Dictionary d = Dictionary::build_from_words({"AY"});  // (7,7) takes only Y
  b.ensure_movegen_caches(d);

  Glyph played[2] = {G(23), G(23)};
  const uint16_t sq = (1u << 7) | (1u << 8);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 2));

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next); dwarfs the rest
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);

  const std::array<uint8_t, 27> with_y = available_of("YE");
  collapse_footprint_planes(b, d, with_y.data(), raw.data(), out.data());
  const float lit = out[7 * kFootprintSide + 7];
  EXPECT_GT(lit, 0.9f);

  const std::array<uint8_t, 27> no_y = available_of("EIO");  // no Y, no blank
  collapse_footprint_planes(b, d, no_y.data(), raw.data(), out.data());
  const float gated = out[7 * kFootprintSide + 7];
  EXPECT_LT(gated, 0.5f);
  EXPECT_LT(gated, lit * 0.5f);
}

// masked_placement_distributions is the collapse's masked softmax, per class
// rather than scattered to cells: each head sums to ~1 over its legal
// footprints, and an illegal class gets exactly zero.
TEST(FootprintCollapse, MaskedDistributionsAreLegalSoftmaxes) {
  Board b;  // empty, so every square is unconstrained
  const Dictionary d = Dictionary::build_from_words({"CAT"});
  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t sq = (1u << 5) | (1u << 6) | (1u << 7);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 3));

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next)
  std::vector<float> dist(kPlacementHeads * kFootprintClasses, 0.0f);
  masked_placement_distributions(b, d, /*available_counts=*/nullptr, raw.data(), dist.data());

  EXPECT_GT(dist[cls], 0.99f);
  for (int h = 0; h < kPlacementHeads; ++h) {
    float sum = 0.0f;
    for (int c = 0; c < kFootprintClasses; ++c) sum += dist[size_t(h) * kFootprintClasses + c];
    EXPECT_NEAR(sum, 1.0f, 1e-4) << "head " << h;
  }
  const int off_edge = (7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2));  // k=7 off the right edge
  EXPECT_EQ(dist[off_edge], 0.0f);
}

// Every cell collapse_footprint_planes puts mass on is legal according to
// collapse_footprint_legal_cells. Uses MassLandsOnCoveredCells' setup.
TEST(FootprintCollapse, LegalCellsCoverAllProbabilityMass) {
  Board b;  // empty, so every square is unconstrained
  const Dictionary d = Dictionary::build_from_words({"CAT"});

  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t sq = (1u << 5) | (1u << 6) | (1u << 7);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 3));

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next); dwarfs the rest
  std::vector<float> prob(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_planes(b, d, /*available_counts=*/nullptr, raw.data(), prob.data());

  std::vector<float> legal(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_legal_cells(b, d, /*available_counts=*/nullptr, legal.data());

  for (int h = 0; h < kPlacementHeads; ++h) {
    for (int i = 0; i < kFootprintSide * kFootprintSide; ++i) {
      const size_t idx = size_t(h) * kFootprintSide * kFootprintSide + i;
      if (prob[idx] > 1e-6f) {
        EXPECT_GT(legal[idx], 0.5f) << "head " << h << " cell " << i;
      }
    }
  }
  EXPECT_GT(legal[7 * kFootprintSide + 7], 0.5f);  // not vacuous: a covered cell is legal
}

// The legal-cells plane reflects the self head's reach bound: the far corner
// that IllegalFootprintGetsNoMass cannot reach is illegal, a cell next to the
// tile is legal.
TEST(FootprintCollapse, LegalCellsRespectReachBound) {
  Board b;
  b.set(0, 0, G(0));
  const Dictionary d = Dictionary::build_from_words({"CAT"});

  std::vector<float> legal(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_legal_cells(b, d, /*available_counts=*/nullptr, legal.data());

  const float* self_legal =
    legal.data() + 1 * kFootprintSide * kFootprintSide;  // head 1: self_next
  EXPECT_EQ(self_legal[14 * kFootprintSide + 12], 0.0f);
  EXPECT_EQ(self_legal[14 * kFootprintSide + 13], 0.0f);
  EXPECT_EQ(self_legal[14 * kFootprintSide + 14], 0.0f);
  EXPECT_GT(self_legal[0 * kFootprintSide + 1], 0.5f);
}

// On an empty board every square is coverable, and a null pool (board legality
// only) matches a pool holding every tile.
TEST(FootprintReachable, EmptyBoardCoversEverythingAndNullIsFullStock) {
  Board b;
  const Dictionary d = Dictionary::build_from_words({"CAT"});
  b.ensure_movegen_caches(d);

  std::vector<float> null_pool(kFootprintCells, -1.0f);
  footprint_reachable_cells(b, /*available_counts=*/nullptr, kMaskTileBudget, null_pool.data());
  for (int i = 0; i < kFootprintCells; ++i) EXPECT_EQ(null_pool[i], 1.0f) << "cell " << i;

  std::array<uint8_t, 27> all_stock;
  all_stock.fill(9);
  std::vector<float> full_pool(kFootprintCells, -1.0f);
  footprint_reachable_cells(b, all_stock.data(), kMaskTileBudget, full_pool.data());
  EXPECT_EQ(null_pool, full_pool);
}

// A cell is reachable iff some legal footprint for the side to move covers it.
// (7,7) is boxed in by A's on all four sides and only "AYA" is a word, so both
// its cross-checks are {Y}: it is reachable only when Y is in stock.
TEST(FootprintReachable, OccupancyAndAvailabilityGateCells) {
  Board b;
  b.set(6, 7, G(0));
  b.set(8, 7, G(0));
  b.set(7, 6, G(0));
  b.set(7, 8, G(0));
  const Dictionary d = Dictionary::build_from_words({"AYA"});
  b.ensure_movegen_caches(d);

  const auto reach_at = [&](const std::array<uint8_t, 27>& pool, int r, int c) {
    std::vector<float> plane(kFootprintCells, -1.0f);
    footprint_reachable_cells(b, pool.data(), kMaskTileBudget, plane.data());
    return plane[r * kFootprintSide + c];
  };

  const std::array<uint8_t, 27> with_y = available_of("YE");
  EXPECT_EQ(reach_at(with_y, 7, 7), 1.0f);
  EXPECT_EQ(reach_at(with_y, 6, 7), 0.0f);    // occupied
  EXPECT_EQ(reach_at(with_y, 14, 14), 0.0f);  // not connected

  const std::array<uint8_t, 27> no_y = available_of("EIO");  // no Y, no blank
  EXPECT_EQ(reach_at(no_y, 7, 7), 0.0f);
  EXPECT_EQ(reach_at(no_y, 14, 14), 0.0f);
}

// Reachability commutes with transposition, as every input-encoder spatial
// plane must. The caches were built on `b`, so this also exercises
// Board::transpose's cache hand-over.
TEST(FootprintReachable, TransposedBoardIsTheTranspose) {
  Board b;
  b.set(3, 5, G(1));  // asymmetric, so the transpose is non-trivial
  b.set(9, 2, G(2));
  const Dictionary d = Dictionary::build_from_words({"CAT", "AY"});
  b.ensure_movegen_caches(d);
  const std::array<uint8_t, 27> pool = available_of("CATY");

  std::vector<float> normal(kFootprintCells, 0.0f), flipped(kFootprintCells, 0.0f);
  footprint_reachable_cells(b, pool.data(), kMaskTileBudget, normal.data());
  footprint_reachable_cells(b.transpose(), pool.data(), kMaskTileBudget, flipped.data());
  for (int r = 0; r < kFootprintSide; ++r)
    for (int c = 0; c < kFootprintSide; ++c)
      EXPECT_EQ(flipped[r * kFootprintSide + c], normal[c * kFootprintSide + r]) << r << "," << c;
}

}  // namespace
}  // namespace scribblez
