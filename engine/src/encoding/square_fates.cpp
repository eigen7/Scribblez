#include "encoding/square_fates.h"

#include "util/assert.h"

namespace scribblez {

namespace {

SquareFate placer_fate(bool evaluator, bool bingo) {
  if (evaluator) return bingo ? SquareFate::kEvaluatorBingo : SquareFate::kEvaluatorOther;
  return bingo ? SquareFate::kOpponentBingo : SquareFate::kOpponentOther;
}

}  // namespace

SquareFates square_fates(const GameLog& g, int turn, const Board& board) {
  SquareFates out;
  for (int sq = 0; sq < kBoardCells; ++sq) {
    const bool empty = board.at(sq / BOARD_SIZE, sq % BOARD_SIZE).is_empty();
    out.fate[sq] = empty ? SquareFate::kStaysEmpty : SquareFate::kFilled;
  }
  // Moves alternate, so turn k's mover is the evaluator iff k - turn is even.
  for (int k = turn + 1; k < g.num_records; ++k) {
    const Move& recorded = g.records[k].move;
    if (recorded.type() != MoveType::PLAY) continue;
    const Move m = board.transposed() ? recorded.transpose() : recorded;
    const bool evaluator = (k - turn) % 2 == 0;
    const SquareFate fate = placer_fate(evaluator, m.num_glyphs() == RACK_SIZE);
    const int16_t points = int16_t(evaluator ? m.score() : -int(m.score()));
    visit_placed_squares(m, [&](int r, int c) {
      const int sq = r * BOARD_SIZE + c;
      DEBUG_ASSERT(out.fate[sq] == SquareFate::kStaysEmpty);
      out.fate[sq] = fate;
      out.points[sq] = points;
    });
  }
  return out;
}

}  // namespace scribblez
