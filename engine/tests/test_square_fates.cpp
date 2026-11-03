// encoding/square_fates.h on a hand-built game log: each later play's squares
// take its placer's class (a bingo iff it places 7 tiles) and its signed score,
// non-plays are skipped, and a transposed board gets the same fates mirrored.
// The self-play row tests (py/tests/test_leaf_eval_rows.py) cover real games,
// whose short continuations rarely hold a bingo.

#include "encoding/square_fates.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"

#include <gtest/gtest.h>

#include <array>

namespace scribblez {
namespace {

constexpr std::array<Glyph, RACK_SIZE> kGlyphs = {
  Glyph::of(Tile::of(0)), Glyph::of(Tile::of(1)), Glyph::of(Tile::of(2)), Glyph::of(Tile::of(3)),
  Glyph::of(Tile::of(4)), Glyph::of(Tile::of(5)), Glyph::of(Tile::of(6)),
};

// A play of `n` tiles at consecutive squares `along` onward in lane `lane`.
Move play(bool horizontal, int lane, int along, int n, int score) {
  const uint16_t mask = uint16_t(((1u << n) - 1) << along);
  return Move::play(horizontal, lane, mask, uint16_t(score), kGlyphs.data(), n);
}

// Turn 0 is the position's move, by the evaluator; turns 1-5 follow, the
// opponent's on odd turns.
std::array<TurnRecord, 6> continuation() {
  std::array<TurnRecord, 6> records{};
  records[1].move = play(/*horizontal=*/true, 7, 3, RACK_SIZE, 74);
  records[2].move = play(/*horizontal=*/false, 2, 5, 2, 12);
  records[3].move = Move::pass();
  records[4].move = play(/*horizontal=*/false, 12, 0, RACK_SIZE, 80);
  records[5].move = play(/*horizontal=*/true, 10, 0, 2, 9);
  return records;
}

// Square (r, c) of the natural frame, read as (c, r) in the transposed one.
void expect_square(const SquareFates& f, bool transposed, int r, int c, SquareFate fate,
                   int points) {
  const int sq = transposed ? c * BOARD_SIZE + r : r * BOARD_SIZE + c;
  EXPECT_EQ(f.fate[sq], fate) << r << "," << c;
  EXPECT_EQ(f.points[sq], points) << r << "," << c;
}

// The fates of the continuation from `board`, a board with (0, 0) filled.
void expect_fates(const Board& board) {
  const std::array<TurnRecord, 6> records = continuation();
  GameLog g{};
  g.records = records.data();
  g.num_records = int(records.size());
  const SquareFates f = square_fates(g, /*turn=*/0, board);
  const bool t = board.transposed();
  expect_square(f, t, 0, 0, SquareFate::kFilled, 0);
  for (int c = 3; c < 3 + RACK_SIZE; ++c)
    expect_square(f, t, 7, c, SquareFate::kOpponentBingo, -74);
  expect_square(f, t, 5, 2, SquareFate::kEvaluatorOther, 12);
  expect_square(f, t, 6, 2, SquareFate::kEvaluatorOther, 12);
  for (int r = 0; r < RACK_SIZE; ++r) expect_square(f, t, r, 12, SquareFate::kEvaluatorBingo, 80);
  expect_square(f, t, 10, 0, SquareFate::kOpponentOther, -9);
  expect_square(f, t, 10, 1, SquareFate::kOpponentOther, -9);
  expect_square(f, t, 14, 14, SquareFate::kStaysEmpty, 0);
  int labelled = 0;
  for (SquareFate fate : f.fate) labelled += fate != SquareFate::kStaysEmpty;
  EXPECT_EQ(labelled, 1 + RACK_SIZE + 2 + RACK_SIZE + 2);
}

TEST(SquareFates, LabelEachLaterPlaysSquares) {
  Board board;
  board.set(0, 0, kGlyphs[0]);
  expect_fates(board);
}

TEST(SquareFates, FollowTheBoardsFrame) {
  Board board;
  board.set(0, 0, kGlyphs[0]);
  expect_fates(board.transpose());
}

}  // namespace
}  // namespace scribblez
