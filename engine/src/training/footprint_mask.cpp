#include "training/footprint_mask.h"

#include <utility>

namespace scribblez {

namespace {

// Can some letter play at board cell (r,c) as part of a word in board-frame
// orientation `horizontal`? The perpendicular cross-check must be non-empty, or
// the square unconstrained. A horizontal word's cross-words run down the columns
// (the non-transposed cache, indexed [r*side+c]); a vertical word's along the
// rows (the transposed cache, indexed [c*side+r]) -- matching the input encoder.
bool cell_admits_letter(const Board& board, bool horizontal, int r, int c) {
  const CrossCheck& cc = horizontal ? board.cross_checks(false)[r * BOARD_SIZE + c]
                                    : board.cross_checks(true)[c * BOARD_SIZE + r];
  return !cc.has_neighbor || cc.mask != 0;
}

}  // namespace

void opp_footprint_mask(const Board& board, int tile_budget, bool flip, bool win_head,
                        FootprintMask& mask) {
  mask.fill(false);
  const int kmax = tile_budget < kFootprintMaxK ? tile_budget : kFootprintMaxK;

  for (int cell = 0; cell < kFootprintCells; ++cell) {
    // Un-flip the anchor into board coordinates once per cell.
    int anchor_r = cell / kFootprintSide;
    int anchor_c = cell % kFootprintSide;
    if (flip) std::swap(anchor_r, anchor_c);
    if (!board.at(anchor_r, anchor_c).is_empty()) continue;  // anchor must be placeable

    for (int slot = 0; slot < kSlotsPerCell; ++slot) {
      bool horizontal;
      int k;
      footprint_slot_decode(slot, horizontal, k);  // orientation in the flip frame
      if (k > kmax) continue;                      // over the tile budget
      const int cls = cell * kSlotsPerCell + slot;

      if (k == 1) {
        // Orientation-free: a lone tile is placeable if it admits a letter along
        // either axis.
        if (cell_admits_letter(board, true, anchor_r, anchor_c) ||
            cell_admits_letter(board, false, anchor_r, anchor_c)) {
          mask[cls] = true;
        }
        continue;
      }

      const bool board_horizontal = flip ? !horizontal : horizontal;
      int count = 0;
      int r = anchor_r;
      int c = anchor_c;
      bool ok = true;
      while (r < kFootprintSide && c < kFootprintSide && count < k) {
        if (board.at(r, c).is_empty()) {
          if (!cell_admits_letter(board, board_horizontal, r, c)) {
            ok = false;
            break;
          }
          ++count;
        }
        if (board_horizontal) {
          ++c;
        } else {
          ++r;
        }
      }
      if (ok && count == k) mask[cls] = true;  // count<k means the edge cut it short
    }
  }

  mask[kPassClass] = true;
  mask[kExtraClass] = win_head;
}

}  // namespace scribblez
