// training/played_continuation.h on a hand-built log: the played move's
// squares are filled (no label), the later plays' squares take their placers'
// classes and signed scores from the mover's side, the end-of-game targets
// are the mover's, a transposed pre-move board gets the same labels mirrored,
// and the played move is found among candidates in either frame.

#include "encoding/square_fates.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "training/game_end_targets.h"
#include "training/played_continuation.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace scribblez::policy_set {
namespace {

constexpr std::array<Glyph, RACK_SIZE> kGlyphs = {
  Glyph::of(Tile::of(0)), Glyph::of(Tile::of(1)), Glyph::of(Tile::of(2)), Glyph::of(Tile::of(3)),
  Glyph::of(Tile::of(4)), Glyph::of(Tile::of(5)), Glyph::of(Tile::of(6)),
};

Move play(bool horizontal, int lane, int along, int n, int score) {
  const uint16_t mask = uint16_t(((1u << n) - 1) << along);
  return Move::play(horizontal, lane, mask, uint16_t(score), kGlyphs.data(), n);
}

// Player 0 opens with two tiles across row 7, player 1 answers with two down
// column 3, and player 0 bingos along row 14 and goes out, collecting 16.
struct Game {
  std::array<TurnRecord, 3> records{};
  GameLog log{};

  Game() {
    records[0].player = 0;
    records[0].move = play(/*horizontal=*/true, 7, 7, 2, 10);
    records[1].player = 1;
    records[1].move = play(/*horizontal=*/false, 3, 0, 2, 12);
    records[2].player = 0;
    records[2].move = play(/*horizontal=*/true, 14, 0, RACK_SIZE, 70);
    log.records = records.data();
    log.num_records = int(records.size());
    log.final_scores = {96, 12};
  }
};

// Square (r, c) of the natural frame, read as (c, r) in the transposed one.
void expect_square(const SquareFates& f, bool transposed, int r, int c, SquareFate fate,
                   int points) {
  const int sq = transposed ? c * BOARD_SIZE + r : r * BOARD_SIZE + c;
  EXPECT_EQ(f.fate[sq], fate) << r << "," << c;
  EXPECT_EQ(f.points[sq], points) << r << "," << c;
}

void expect_continuation(const Board& board) {
  const Game g;
  const PlayedContinuation c = played_continuation(g.log, /*turn=*/0, board);
  const bool t = board.transposed();
  expect_square(c.fates, t, 7, 7, SquareFate::kFilled, 0);
  expect_square(c.fates, t, 7, 8, SquareFate::kFilled, 0);
  expect_square(c.fates, t, 0, 3, SquareFate::kOpponentOther, -12);
  expect_square(c.fates, t, 1, 3, SquareFate::kOpponentOther, -12);
  for (int col = 0; col < RACK_SIZE; ++col)
    expect_square(c.fates, t, 14, col, SquareFate::kEvaluatorBingo, 70);
  expect_square(c.fates, t, 7, 9, SquareFate::kStaysEmpty, 0);
  EXPECT_EQ(c.going_out, 0.0f);
  EXPECT_EQ(c.end_points, 16.0f);
}

TEST(PlayedContinuation, LabelsTheRestOfTheGameAfterThePlayedMove) { expect_continuation(Board()); }

TEST(PlayedContinuation, ATransposedBoardGetsTheMirroredLabels) {
  expect_continuation(Board().transpose());
}

TEST(PlayedContinuation, ThePlayedMoveIsFoundAmongCandidatesInEitherFrame) {
  const Game g;
  const Move played = g.records[0].move;
  const std::vector<Move> natural = {Move::pass(), played};
  EXPECT_EQ(played_index(g.log, 0, /*transpose=*/false, natural), 1);
  EXPECT_EQ(played_index(g.log, 0, /*transpose=*/true, natural), -1);
  const std::vector<Move> transposed = {played.transpose()};
  EXPECT_EQ(played_index(g.log, 0, /*transpose=*/true, transposed), 0);
  EXPECT_EQ(played_index(g.log, 0, /*transpose=*/false, {Move::pass()}), -1);
}

}  // namespace
}  // namespace scribblez::policy_set
