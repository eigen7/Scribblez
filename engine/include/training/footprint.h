#pragma once

#include "game/board.h"
#include "game/move.h"

#include <array>
#include <utility>

namespace scribblez {

// A move's placement footprint as a categorical class, the label space of the
// position evaluation model's placement heads (masked softmax cross-entropy
// over footprints; docs/model_architectures.md).
//
// A footprint is (anchor, orientation, k): the anchor is the first newly placed
// square, k the number of tiles placed (1..RACK_SIZE), and the covered cells are
// the first k empty cells from the anchor along the orientation. For a legal
// play those are exactly its newly placed squares, since a play fills every
// empty cell in its span and skips only through-tiles.
//
// Class layout: each anchor cell owns kSlotsPerCell classes. Slot 0 is k == 1,
// orientation-free because a lone tile covers the same cell either way. Then
// come k = 2..RACK_SIZE horizontal, then the same vertical. Two catch-all
// classes follow the per-cell grid: kPassClass (no placement: EXCHANGE, PASS, or
// no move) and kExtraClass (the win heads' "not-win" outcome; unused by the
// plays heads).
//
// A class is expressed in the frame of the move or board it came from (see
// Board::transpose). A diagonal transpose swaps rows with columns AND horizontal
// with vertical, so transposing a class moves its slot between the H and V
// blocks as well as moving its cell.

inline constexpr int kFootprintSide = BOARD_SIZE;                            // 15
inline constexpr int kFootprintMaxK = RACK_SIZE;                             // 7
inline constexpr int kSlotsPerCell = 1 + 2 * (kFootprintMaxK - 1);           // 13
inline constexpr int kFootprintCells = kFootprintSide * kFootprintSide;      // 225
inline constexpr int kAnchoredFootprints = kFootprintCells * kSlotsPerCell;  // 2925
inline constexpr int kPassClass = kAnchoredFootprints;                       // 2925
inline constexpr int kExtraClass = kAnchoredFootprints + 1;                  // 2926
inline constexpr int kFootprintClasses = kAnchoredFootprints + 2;            // 2927

// The number of placement heads, one per placement target in
// training_targets.h. Everything indexed by head (the position model's aux
// outputs, .mset planes, the move-proposal planes and evidence channels) uses
// the targets' declaration order: opp_next, self_next, opp_win, self_win.
inline constexpr int kPlacementHeads = 4;

// The footprint class of a move, in the move's frame. EXCHANGE and PASS map to
// kPassClass.
int footprint_class(const Move& m);

// Decode a per-cell slot in [0, kSlotsPerCell) into its orientation and tile
// count k. Slot 0 (k == 1) reports horizontal by convention.
void footprint_slot_decode(int slot, bool& horizontal, int& k);

// The inverse of footprint_class: the cells a class covers on `board`, the
// pre-move state, written to `cells` as (row, col) pairs, returning their count.
// On m's pre-move board, footprint_class(m) covers exactly m's placed squares.
// Returns 0 for kPassClass and kExtraClass, and for a class impossible on this
// board (occupied anchor, or fewer than k empty cells before the edge); such
// classes never occur as targets and are masked out.
int footprint_cells(int cls, const Board& board,
                    std::array<std::pair<int, int>, kFootprintMaxK>& cells);

}  // namespace scribblez
