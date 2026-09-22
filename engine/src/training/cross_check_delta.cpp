#include "training/cross_check_delta.h"

#include "util/assert.h"

#include <algorithm>
#include <compare>

namespace scribblez {
namespace move_set {
namespace {

struct Entry {
  uint8_t axis;
  int32_t square;
  uint32_t old_mask;
  uint32_t new_mask;

  auto operator<=>(const Entry&) const = default;
};

// The cross-check caches index in view coordinates, the transposed view
// swapping row and column; an entry's square is in the board's own frame.
int32_t board_square(const BoardUndo::CrossRec& rec) {
  const int vr = rec.idx / BOARD_SIZE;
  const int vc = rec.idx % BOARD_SIZE;
  return rec.transposed ? vc * BOARD_SIZE + vr : rec.idx;
}

// The changed entries among the cache writes `undo` recorded, read off the
// post-move `board`. A write is not an entry when it lands on a square the move
// filled, or leaves the mask as it was -- which also drops the repeat writes to
// the ends of the move's own word, one per placed tile, whose first record
// alone holds the pre-move value.
int collect_entries(const Board& board, const BoardUndo& undo, Entry* out) {
  int n = 0;
  for (const BoardUndo::CrossRec& rec : undo.crosses) {
    const int32_t square = board_square(rec);
    if (!board.at(square / BOARD_SIZE, square % BOARD_SIZE).is_empty()) continue;
    const uint32_t new_mask = board.cross_checks(rec.transposed != 0)[rec.idx].mask;
    if (new_mask == rec.old.mask) continue;
    DEBUG_ASSERT(n < kMoveMaxCrossDeltas);
    out[n++] = {rec.transposed, square, rec.old.mask, new_mask};
  }
  return n;
}

}  // namespace

void encode_cross_check_deltas(Board& board, const Move& m, BoardUndo& undo, uint8_t* axes,
                               int32_t* squares, uint32_t* old_masks, uint32_t* new_masks,
                               uint8_t* delta_mask) {
  std::fill_n(axes, kMoveMaxCrossDeltas, uint8_t(0));
  std::fill_n(squares, kMoveMaxCrossDeltas, 0);
  std::fill_n(old_masks, kMoveMaxCrossDeltas, 0u);
  std::fill_n(new_masks, kMoveMaxCrossDeltas, 0u);
  std::fill_n(delta_mask, kMoveMaxCrossDeltas, uint8_t(0));

  Entry entries[kMoveMaxCrossDeltas];
  board.apply(m, &undo);
  const int n = collect_entries(board, undo, entries);
  board.unapply(undo);

  std::sort(entries, entries + n);
  for (int i = 0; i < n; ++i) {
    axes[i] = entries[i].axis;
    squares[i] = entries[i].square;
    old_masks[i] = entries[i].old_mask;
    new_masks[i] = entries[i].new_mask;
    delta_mask[i] = 1;
  }
}

void encode_cross_check_deltas(const Board& board, const Dictionary& dict, const Move* moves,
                               int64_t n, uint8_t* axes, int32_t* squares, uint32_t* old_masks,
                               uint32_t* new_masks, uint8_t* delta_mask) {
  Board scratch = board;
  scratch.ensure_movegen_caches(dict);
  BoardUndo undo;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t at = i * kMoveMaxCrossDeltas;
    encode_cross_check_deltas(scratch, moves[i], undo, axes + at, squares + at, old_masks + at,
                              new_masks + at, delta_mask + at);
  }
}

}  // namespace move_set
}  // namespace scribblez
