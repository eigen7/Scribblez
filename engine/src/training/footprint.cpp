#include "training/footprint.h"

#include <bit>

namespace scribblez {

namespace {

// The per-cell slot for (orientation, k): slot 0 is the orientation-free k==1
// footprint; slots 1..kFootprintMaxK-1 are horizontal k=2..RACK_SIZE; the next
// block is the vertical k=2..RACK_SIZE.
int slot_for(bool horizontal, int k) {
  if (k <= 1) return 0;
  const int base = horizontal ? 1 : kFootprintMaxK;  // H: 1.., V: kFootprintMaxK..
  return base + (k - 2);
}

}  // namespace

void footprint_slot_decode(int slot, bool& horizontal, int& k) {
  if (slot == 0) {
    horizontal = true;  // orientation-free; a lone tile has no distinct axis
    k = 1;
  } else if (slot < kFootprintMaxK) {
    horizontal = true;
    k = slot + 1;
  } else {
    horizontal = false;
    k = slot - kFootprintMaxK + 2;
  }
}

int footprint_class(const Move& m) {
  if (m.type() != MoveType::PLAY) return kPassClass;
  const int k = m.num_glyphs();                          // == popcount(square_mask)
  const int along0 = std::countr_zero(m.square_mask());  // first placed lane cell
  const bool horizontal = m.horizontal();
  const int r = horizontal ? m.start() : along0;
  const int c = horizontal ? along0 : m.start();
  return (r * kFootprintSide + c) * kSlotsPerCell + slot_for(horizontal, k);
}

int footprint_cells(int cls, const Board& board,
                    std::array<std::pair<int, int>, kFootprintMaxK>& cells) {
  if (cls < 0 || cls >= kAnchoredFootprints) return 0;  // pass / extra
  const int cell = cls / kSlotsPerCell;
  bool horizontal;
  int k;
  footprint_slot_decode(cls % kSlotsPerCell, horizontal, k);

  int r = cell / kFootprintSide;
  int c = cell % kFootprintSide;
  if (!board.at(r, c).is_empty()) return 0;  // anchor must be a placeable square

  int count = 0;
  while (r < kFootprintSide && c < kFootprintSide && count < k) {
    if (board.at(r, c).is_empty()) cells[count++] = {r, c};
    if (horizontal) {
      ++c;
    } else {
      ++r;
    }
  }
  return count == k ? k : 0;  // 0 if the board edge cut the run short
}

}  // namespace scribblez
