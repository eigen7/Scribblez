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

// A 27-count availability array (A..Z then blank at 26) holding one of each
// listed letter; '?' adds a wildcard blank. Unlisted letters stay 0 (out of
// stock). Feeds opp_footprint_mask / collapse_footprint_planes as available_counts.
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

// On an empty board every square reads as unconstrained, so with no availability
// counts the opp mask reduces to geometry + tile budget -- exercisable without a
// dictionary. (Cross-check and availability gating need a real board/counts; see
// the availability tests below and the soundness sweep.)
TEST(FootprintMask, EmptyBoardGeometryAndBudget) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, /*tile_budget=*/7,
                     /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);    // horizontal k=3 fits
  EXPECT_FALSE(m[(7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2))]);  // horizontal k=7 off the edge
  EXPECT_TRUE(m[kPassClass]);                                      // pass always legal
  EXPECT_FALSE(m[kExtraClass]);                                    // dummy for a plays head
  EXPECT_EQ(count_true(m), 2295 + 1);  // every geometrically-fitting footprint + pass
}

TEST(FootprintMask, WinHeadKeepsNotWinSlot) {
  Board b;
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, 7, /*win_head=*/true, m);
  EXPECT_TRUE(m[kExtraClass]);
  EXPECT_EQ(count_true(m), 2295 + 2);  // + pass + not-win
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

// The mask of the transposed board is the transposed mask, class for class --
// through the cross-check caches Board::transpose hands over.
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
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 2));

  EXPECT_TRUE(opp_admits(b, available_of("XY"), cls));    // both hooks in stock
  EXPECT_TRUE(opp_admits(b, available_of("YE"), cls));    // one legal hook (Y) suffices
  EXPECT_TRUE(opp_admits(b, available_of("XE"), cls));    // the other legal hook (X)
  EXPECT_FALSE(opp_admits(b, available_of("EIO"), cls));  // no legal hook available -> masked
  EXPECT_TRUE(opp_admits(b, available_of("?"), cls));     // a blank is a wildcard hook
  EXPECT_TRUE(opp_admits(b, available_of("EIO?"), cls));  // ... even amid non-hook letters
  EXPECT_TRUE(opp_admits(b, available_of("XY"), cls));    // sanity: unchanged on re-check

  // Null availability disables the gate -- pure board legality readmits it.
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, RACK_SIZE, /*win_head=*/false, m);
  EXPECT_TRUE(m[cls]);
}

// The unconstrained free cell (7,8) still needs SOME tile: an utterly empty
// availability (a blank-less bag with no listed letters) masks even a footprint whose
// only real constraint is "place a tile here".
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

// The lone-tile fix in miniature (the I13 case): a single tile below a vertical
// word forms that word as its DOWN cross-word, so it is playable only with a
// letter that both completes the word and is in stock -- NOT, as the old per-axis
// OR wrongly allowed, any letter merely because the empty across-axis is
// unconstrained. Board: 'A' above (7,7), down-word "A_" legal for {X,Y}.
TEST(FootprintMask, AvailabilityGatesLoneTileHook) {
  Board b;
  b.set(6, 7, G(0));  // 'A'
  const Dictionary d = Dictionary::build_from_words({"AX", "AY"});
  b.ensure_movegen_caches(d);

  const int lone = (7 * 15 + 7) * kSlotsPerCell + 0;       // orientation-free k==1 at (7,7)
  EXPECT_TRUE(opp_admits(b, available_of("XY"), lone));    // both down-hooks in stock
  EXPECT_TRUE(opp_admits(b, available_of("YE"), lone));    // one legal down-hook (Y) suffices
  EXPECT_TRUE(opp_admits(b, available_of("?"), lone));     // a blank is a wildcard hook
  EXPECT_FALSE(opp_admits(b, available_of("EIO"), lone));  // no legal down-hook -> masked (the fix)
  EXPECT_FALSE(opp_admits(b, available_of(""), lone));     // nothing to place at all

  // Board legality only (null availability) still admits it -- some letter (X/Y) exists.
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, RACK_SIZE, /*win_head=*/false, m);
  EXPECT_TRUE(m[lone]);
}

// A lone tile at a cross-point forms BOTH cross-words at once, so the exact test
// intersects the two cross-check letter sets -- a per-axis "some available letter
// fits this axis, AND some available letter fits that axis" is not enough. Board:
// 'A' above (7,7) [down-word {X,Y}] and 'B' left of it [across-word {E}]. No
// single letter is in both sets, so the square is unplayable even with X, Y and E
// all in stock; overlapping cross-words readmit it.
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

// The opp mask requires a placement to CONNECT to the board: a footprint that
// floats free of every tile is masked even though its cells are individually
// playable. This is the reported A1-corner bug in miniature -- an isolated empty
// corner has unconstrained cross-checks, so the per-cell letter test admits it,
// yet no legal move (after the opener) can reach it. Footprints touching the lone
// 'A' at (7,7) are kept; ones adrift in the empty expanse are not. Independent of
// the availability gate, so it also shows under null (board-legality-only) counts.
TEST(FootprintMask, ConnectivityMasksFloatingPlacements) {
  Board b;
  b.set(7, 7, G(0));  // 'A': the sole structure everything must connect to
  const Dictionary d = Dictionary::build_from_words({"AB"});
  b.ensure_movegen_caches(d);

  const int corner_lone = (0 * 15 + 0) * kSlotsPerCell + 0;            // lone tile at (0,0)
  const int floating2 = (0 * 15 + 3) * kSlotsPerCell + (1 + (2 - 2));  // horiz k=2 in empty row 0
  const int hook_lone = (7 * 15 + 8) * kSlotsPerCell + 0;              // lone tile right of 'A'
  const int abutting2 = (7 * 15 + 8) * kSlotsPerCell + (1 + (2 - 2));  // horiz k=2 abutting 'A'

  // Disconnected placements: floated free of the board -> masked (the fix).
  EXPECT_FALSE(opp_admits(b, available_of("AB"), corner_lone));
  EXPECT_FALSE(opp_admits(b, available_of("AB"), floating2));
  // Connected placements: the covered cell at (7,8) abuts 'A'. The lone tile forms
  // the across-word "A_" (legal "AB"), so B both connects and hooks; the k=2 covers
  // (7,8),(7,9) whose vertical cross-checks are free -> both kept.
  EXPECT_TRUE(opp_admits(b, available_of("B"), hook_lone));
  EXPECT_TRUE(opp_admits(b, available_of("AB"), abutting2));

  // Connectivity holds independently of availability -- null (pure board legality)
  // still masks the floating corner and keeps the abutting footprint.
  FootprintMask m;
  opp_footprint_mask(b, /*available_counts=*/nullptr, RACK_SIZE, /*win_head=*/false, m);
  EXPECT_FALSE(m[corner_lone]);
  EXPECT_FALSE(m[floating2]);
  EXPECT_TRUE(m[hook_lone]);
  EXPECT_TRUE(m[abutting2]);
}

// The self mask is cross-check-oblivious (pure geometry + BFS), so it is fully
// testable without a dictionary.
TEST(SelfFootprintMask, ReachabilityFromStructure) {
  Board b;
  b.set(7, 7, G(4));  // a lone tile: the only structure to bridge from
  FootprintMask m;
  // opp_budget=1, self_budget=1 -> combined reach 2 tiles.
  self_footprint_mask(b, /*self_budget=*/1, /*opp_budget=*/1, /*win=*/false, m);
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
  const int far = (7 * 15 + 10) * kSlotsPerCell + 0;    // d=3 from the tile
  self_footprint_mask(b, 1, 1, /*win_head=*/false, m);  // reach 2 -> out
  EXPECT_FALSE(m[far]);
  self_footprint_mask(b, 2, 1, /*win_head=*/false, m);  // reach 3 -> in
  EXPECT_TRUE(m[far]);
}

TEST(SelfFootprintMask, BudgetCapsK) {
  Board b;
  b.set(7, 4, G(4));  // structure so nearby cells are reachable
  FootprintMask m;
  self_footprint_mask(b, /*self_budget=*/2, /*opp_budget=*/7, /*win_head=*/false, m);
  EXPECT_TRUE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (2 - 2))]);   // k=2 within budget
  EXPECT_FALSE(m[(7 * 15 + 5) * kSlotsPerCell + (1 + (3 - 2))]);  // k=3 over budget
}

TEST(SelfFootprintMask, EmptyBoardTreatsAllReachable) {
  Board b;  // no structure -> game-start guard marks everything reachable
  FootprintMask m;
  self_footprint_mask(b, 7, 7, /*win_head=*/false, m);
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
      const int cls = footprint_class(m);
      if (dict) {
        // Availability the mover of `m` could draw from: everything off the board
        // (the loosest sound pool -- a superset of any one rack, so it can never
        // exclude a move the mover actually makes). This exercises the
        // availability path and, in the fixtures' endgames, its binding regime.
        uint8_t available_counts[27];
        compute_unseen_pool(available_counts, board, Rack{});
        FootprintMask opp;
        opp_footprint_mask(board, available_counts, RACK_SIZE, /*win_head=*/false, opp);
        EXPECT_TRUE(opp[cls]) << "opp mask excluded a real move (class " << cls << ")";
      }
      if (two_plies_ago) {
        FootprintMask self;
        self_footprint_mask(*two_plies_ago, RACK_SIZE, RACK_SIZE, /*win_head=*/false, self);
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
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 3));

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next); dwarfs the rest
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_planes(b, d, /*available_counts=*/nullptr, raw.data(), out.data());

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
  const int illegal = footprint_class(Move::play(true, 14, sq, 0, played, 3));
  ASSERT_LT(illegal, kAnchoredFootprints);

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[1 * kFootprintClasses + illegal] = 20.0f;  // head 1 (self_next); would dominate unmasked
  std::vector<float> out(kPlacementHeads * kFootprintSide * kFootprintSide, 0.0f);
  collapse_footprint_planes(b, d, /*available_counts=*/nullptr, raw.data(), out.data());

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
// only the uniform share of the other footprints that still cover it. With the
// hook letter supplied, the dominant footprint lights (7,7) (~1.0) instead. This
// is the inference-time "no Y unseen -> the I13 Y-hook carries no belief"
// behaviour in miniature. The residual share is not vanishing here: connectivity
// confines the legal set to footprints abutting the lone 'A', and (7,7) sits
// against it, so its uniform share is a modest fraction rather than the near-zero
// of a board-wide legal set -- the point is the loss of the DOMINANT spike.
TEST(FootprintCollapse, OppAvailabilityDropsUnsatisfiableFootprint) {
  Board b;
  b.set(6, 7, G(0));  // 'A' above (7,7): a horizontal hook there needs a "A_" word
  const Dictionary d = Dictionary::build_from_words({"AY"});  // sole hook letter: Y
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
  EXPECT_GT(lit, 0.9f);  // Y available -> the dominant hook lands on (7,7)

  const std::array<uint8_t, 27> no_y = available_of("EIO");  // letters, but no Y, no blank
  collapse_footprint_planes(b, d, no_y.data(), raw.data(), out.data());
  const float gated = out[7 * kFootprintSide + 7];
  EXPECT_LT(gated, 0.5f);        // the dominant 20-logit hook is gone -> only the uniform share
  EXPECT_LT(gated, lit * 0.5f);  // ... a clear drop vs. the near-1.0 it held with the hook in stock
}

// masked_placement_distributions returns the same mask + masked-softmax the
// collapse applies, but per class instead of scattered to cells: each head is a
// distribution over its legal footprints (sums to ~1), a dominant logit on a
// legal footprint takes ~all its head's mass, and a structurally illegal class
// stays at zero.
TEST(FootprintCollapse, MaskedDistributionsAreLegalSoftmaxes) {
  Board b;  // empty board: unconstrained, so the opp mask is dict-free
  const Dictionary d = Dictionary::build_from_words({"CAT"});
  Glyph played[3] = {G(0), G(1), G(2)};
  const uint16_t sq = (1u << 5) | (1u << 6) | (1u << 7);
  const int cls = footprint_class(Move::play(true, 7, sq, 0, played, 3));

  std::vector<float> raw(kPlacementHeads * kFootprintClasses, 0.0f);
  raw[0 * kFootprintClasses + cls] = 20.0f;  // head 0 (opp_next) dominant
  std::vector<float> dist(kPlacementHeads * kFootprintClasses, 0.0f);
  masked_placement_distributions(b, d, /*available_counts=*/nullptr, raw.data(), dist.data());

  EXPECT_GT(dist[cls], 0.99f);  // the dominant legal footprint takes ~all head 0's mass
  for (int h = 0; h < kPlacementHeads; ++h) {  // each head is a legal-class distribution
    float sum = 0.0f;
    for (int c = 0; c < kFootprintClasses; ++c) sum += dist[size_t(h) * kFootprintClasses + c];
    EXPECT_NEAR(sum, 1.0f, 1e-4) << "head " << h;
  }
  const int off_edge = (7 * 15 + 12) * kSlotsPerCell + (1 + (7 - 2));  // k=7 off the right edge
  EXPECT_EQ(dist[off_edge], 0.0f);  // structurally illegal -> masked to zero
}

// footprint_reachable_cells: on an empty board every square is coverable (a lone
// tile fits anywhere), a null pool is board-legality-only, and it equals the
// same reduction when the pool holds every tile.
TEST(FootprintReachable, EmptyBoardCoversEverythingAndNullIsFullStock) {
  Board b;  // empty: every square unconstrained
  const Dictionary d = Dictionary::build_from_words({"CAT"});
  b.ensure_movegen_caches(d);

  std::vector<float> null_pool(kFootprintCells, -1.0f);
  footprint_reachable_cells(b, /*available_counts=*/nullptr, kMaskTileBudget, null_pool.data());
  for (int i = 0; i < kFootprintCells; ++i) EXPECT_EQ(null_pool[i], 1.0f) << "cell " << i;

  // A null pool is "everything in stock" -> identical to a full-count array.
  std::array<uint8_t, 27> all_stock;
  all_stock.fill(9);
  std::vector<float> full_pool(kFootprintCells, -1.0f);
  footprint_reachable_cells(b, all_stock.data(), kMaskTileBudget, full_pool.data());
  EXPECT_EQ(null_pool, full_pool);
}

// A cell is reachable iff some legal "moves-next" footprint covers it: occupied
// squares are never covered, and availability gates a fully boxed cell whose
// every covering footprint needs the same hook letter. (7,7) is walled by A's on
// all four sides with only "AYA" legal, so its horizontal and vertical
// cross-checks are both {Y} -- it lights only when Y is in stock.
TEST(FootprintReachable, OccupancyAndAvailabilityGateCells) {
  Board b;
  b.set(6, 7, G(0));  // A above
  b.set(8, 7, G(0));  // A below
  b.set(7, 6, G(0));  // A left
  b.set(7, 8, G(0));  // A right
  const Dictionary d = Dictionary::build_from_words({"AYA"});
  b.ensure_movegen_caches(d);

  const auto reach_at = [&](const std::array<uint8_t, 27>& pool, int r, int c) {
    std::vector<float> plane(kFootprintCells, -1.0f);
    footprint_reachable_cells(b, pool.data(), kMaskTileBudget, plane.data());
    return plane[r * kFootprintSide + c];
  };

  const std::array<uint8_t, 27> with_y = available_of("YE");
  EXPECT_EQ(reach_at(with_y, 7, 7), 1.0f);    // Y in stock -> the boxed cell is reachable
  EXPECT_EQ(reach_at(with_y, 6, 7), 0.0f);    // occupied square is never covered
  EXPECT_EQ(reach_at(with_y, 14, 14), 0.0f);  // the far corner floats free -> unreachable

  const std::array<uint8_t, 27> no_y = available_of("EIO");  // letters, no Y, no blank
  EXPECT_EQ(reach_at(no_y, 7, 7), 0.0f);    // every footprint covering (7,7) needs Y -> gated off
  EXPECT_EQ(reach_at(no_y, 14, 14), 0.0f);  // ... and the disconnected corner stays unreachable
}

// Reachability on the transposed board is the transpose of reachability on the
// board -- the invariant the input encoder's spatial planes all share. The caches
// were built on `b`, so this also exercises Board::transpose's cache hand-over.
TEST(FootprintReachable, TransposedBoardIsTheTranspose) {
  Board b;
  b.set(3, 5, G(1));  // an asymmetric bit of structure so the transpose is non-trivial
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
