#include "training/cross_check_delta.h"

#include "util/assert.h"

namespace scribblez {
namespace move_set {
namespace {

using cross_check_abi::MoveDeltas;

// The cross-check caches index squares in view coordinates (the transposed view
// swaps row and column); an entry's square is in the board's own frame.
uint16_t board_square(const BoardUndo::CrossRec& rec) {
  const int vr = rec.idx / BOARD_SIZE;
  const int vc = rec.idx % BOARD_SIZE;
  return uint16_t(rec.transposed ? vc * BOARD_SIZE + vr : rec.idx);
}

// The changed entries among the cache writes `undo` recorded, read off the
// post-move `board`. A write is skipped when it lands on a square the move
// filled or leaves the mask unchanged. The second test also drops the repeat
// writes to the ends of the move's own word (one per placed tile): only the
// first record holds the true pre-move mask, and the later ones compare equal.
void collect_entries(const Board& board, const BoardUndo& undo, MoveDeltas& out) {
  for (const BoardUndo::CrossRec& rec : undo.crosses) {
    const uint16_t square = board_square(rec);
    if (!board.at(square / BOARD_SIZE, square % BOARD_SIZE).is_empty()) continue;
    const uint32_t new_mask = board.cross_checks(rec.transposed != 0)[rec.idx].mask;
    if (new_mask == rec.old.mask) continue;
    DEBUG_ASSERT(out.count < kMoveMaxCrossDeltas);
    cross_check_abi::insert_sorted(out, {rec.old.mask, new_mask, square, rec.transposed, 0});
  }
}

}  // namespace

MoveDeltas cross_check_deltas(Board& board, const Move& m, BoardUndo& undo) {
  MoveDeltas out;
  board.apply(m, &undo);
  collect_entries(board, undo, out);
  board.unapply(undo);
  return out;
}

}  // namespace move_set
}  // namespace scribblez
