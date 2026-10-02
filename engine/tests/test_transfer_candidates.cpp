#include "game/glyph.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "sim/transfer_candidates.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

// A play of `word`'s letters on consecutive squares from (row, col).
Move play(int row, int col, bool horizontal, const std::string& word) {
  std::array<Glyph, RACK_SIZE> glyphs{};
  uint16_t mask = 0;
  const int lane0 = horizontal ? col : row;
  for (size_t i = 0; i < word.size(); ++i) {
    glyphs[i] = Glyph::of(Tile::letter_from_char(word[i]));
    mask |= uint16_t(1u << (lane0 + int(i)));
  }
  return Move::play(horizontal, horizontal ? row : col, mask, /*score=*/0, glyphs.data(),
                    int(word.size()));
}

Move exchange(const std::string& tiles) { return Move::exchange(TileCounts::from_string(tiles)); }

// Two-tile plays with distinct tile pairs, kept off row 7 and column 3, so they
// couple with nothing.
std::vector<Move> fillers(int n) {
  std::vector<Move> out;
  for (char a = 'B'; a <= 'Z' && int(out.size()) < n; ++a) {
    for (char b = char(a + 1); b <= 'Z' && int(out.size()) < n; ++b) {
      const int k = int(out.size());
      out.push_back(play(9 + k % 5, k % 12, /*horizontal=*/true, std::string{a, b}));
    }
  }
  return out;
}

// A ranking that offers exactly one pair of each coupling kind, among fillers
// and exchanges that couple with nothing (theirs are the only tiles with two A's).
std::vector<Move> ranking() {
  std::vector<Move> ranked = fillers(200);
  ranked[0] = play(7, 7, true, "RATE");
  ranked[5] = play(1, 3, false, "TEAR");  // RATE's tiles elsewhere
  ranked[20] = play(7, 7, true, "RAT");   // RATE's lane, one tile less
  ranked[150] = exchange("ART");          // RAT's tiles exchanged
  for (const char* tiles : {"AAB", "AAC", "AAD", "AAE", "AAF"}) ranked.push_back(exchange(tiles));
  return ranked;
}

int count_kind(const std::vector<CoupledPair>& pairs, Coupling kind) {
  return int(std::ranges::count(pairs, kind, &CoupledPair::kind));
}

SimCandidates select(const std::vector<Move>& ranked, uint64_t seed) {
  std::mt19937_64 rng(seed);
  return transfer_selector(TransferRecipe{})({}, SimPosition{}, ranked, Move{}, rng);
}

}  // namespace

TEST(TransferCandidates, PlayAndTheExchangeOfItsTilesShareALeave) {
  EXPECT_EQ(coupling_of(play(7, 7, true, "RATE"), exchange("AERT")), Coupling::kPlayExchange);
  EXPECT_EQ(coupling_of(exchange("AERT"), play(7, 7, true, "RATE")), Coupling::kPlayExchange);
  EXPECT_EQ(coupling_of(play(7, 7, true, "RATE"), exchange("AERS")), Coupling::kNone);
}

TEST(TransferCandidates, SameTilesNeedDifferentSquares) {
  EXPECT_EQ(coupling_of(play(7, 7, true, "RATE"), play(1, 3, false, "TEAR")), Coupling::kSameTiles);
  EXPECT_EQ(coupling_of(play(7, 7, true, "RATE"), play(7, 7, true, "TEAR")), Coupling::kNone);
}

TEST(TransferCandidates, SameLaneOneTileNeedsALaneAndAnOverlap) {
  EXPECT_EQ(coupling_of(play(7, 7, true, "RAT"), play(7, 7, true, "RATE")),
            Coupling::kSameLaneOneTile);
  EXPECT_EQ(coupling_of(play(7, 7, true, "RAT"), play(8, 7, true, "RATE")), Coupling::kNone);
  EXPECT_EQ(coupling_of(play(7, 7, true, "RAT"), play(7, 7, false, "RATE")), Coupling::kNone);
  EXPECT_EQ(coupling_of(play(7, 7, true, "RAT"), play(7, 7, true, "RATES")), Coupling::kNone);
}

TEST(TransferCandidates, SelectionFillsStrataAndTakesOnePairOfEachKind) {
  const std::vector<Move> ranked = ranking();
  const TransferRecipe recipe;
  const SimCandidates c = select(ranked, 7);
  ASSERT_EQ(int(c.moves.size()), recipe.size());
  EXPECT_TRUE(std::ranges::is_sorted(c.equity_ranks));
  for (size_t i = 0; i < c.moves.size(); ++i) EXPECT_EQ(c.moves[i], ranked[c.equity_ranks[i]]);

  std::array<int, 4> strata{};
  for (size_t i = 0; i < c.moves.size(); ++i)
    ++strata[int(stratum_of(c.moves[i], c.equity_ranks[i], recipe))];
  EXPECT_EQ(strata[int(Stratum::kTop)], recipe.top);
  EXPECT_EQ(strata[int(Stratum::kMiddle)], recipe.middle);
  EXPECT_EQ(strata[int(Stratum::kExchange)], recipe.exchanges);
  EXPECT_EQ(strata[int(Stratum::kLow)], recipe.low);

  const std::vector<CoupledPair> pairs = find_couplings(c.moves);
  EXPECT_GE(count_kind(pairs, Coupling::kPlayExchange), 1);
  EXPECT_GE(count_kind(pairs, Coupling::kSameTiles), 1);
  EXPECT_GE(count_kind(pairs, Coupling::kSameLaneOneTile), 1);
}

TEST(TransferCandidates, SelectionIsDeterministicInTheSeed) {
  const std::vector<Move> ranked = ranking();
  EXPECT_EQ(select(ranked, 3).equity_ranks, select(ranked, 3).equity_ranks);
  EXPECT_NE(select(ranked, 3).equity_ranks, select(ranked, 4).equity_ranks);
}

TEST(TransferCandidates, ShortStrataSpillIntoTheRest) {
  const std::vector<Move> ranked = fillers(30);  // no exchanges, no low ranks
  const SimCandidates c = select(ranked, 1);
  EXPECT_EQ(int(c.moves.size()), TransferRecipe{}.size());
}

TEST(TransferCandidates, AnchoredCouplingsAreFoundOnce) {
  const AnchoredCouplings pairs = anchored_couplings(ranking(), TransferRecipe{});
  for (const auto& kind_pairs : pairs) EXPECT_EQ(kind_pairs.size(), 1u);
}
