#pragma once

#include "game/board.h"
#include "game/move.h"

#include <array>
#include <utility>

namespace scribblez {

// A move's placement FOOTPRINT as a categorical class, for the position-eval
// placement heads (softmax cross-entropy over footprints, replacing per-cell
// BCE). A footprint is (anchor, orientation, k): the anchor is the first newly
// placed square, k the number of tiles placed (1..RACK_SIZE), and the covered
// cells are "the first k empty cells from the anchor along the orientation" --
// which for a legal play are exactly its newly placed squares, since a play
// fills every empty cell in its span and only skips through-tiles.
//
// Class layout: each anchor cell owns kSlotsPerCell classes -- one
// orientation-free slot for k==1 (a lone tile's footprint is a single cell,
// identical either way), then k=2..RACK_SIZE for the horizontal orientation and
// again for the vertical. The per-cell grid is followed by two catch-all
// classes: kPassClass (no placement -- EXCHANGE, PASS, or an absent move) and
// kExtraClass (the win heads' "not-win"; an unused dummy for the plays heads).
//
// Frame: a class is expressed in the (optionally diagonally flipped) frame the
// model sees. A diagonal transpose swaps rows<->cols AND horizontal<->vertical,
// so both calls take the same `flip` the input encoder used for the row.

inline constexpr int kFootprintSide = BOARD_SIZE;                            // 15
inline constexpr int kFootprintMaxK = RACK_SIZE;                             // 7
inline constexpr int kSlotsPerCell = 1 + 2 * (kFootprintMaxK - 1);           // 13
inline constexpr int kFootprintCells = kFootprintSide * kFootprintSide;      // 225
inline constexpr int kAnchoredFootprints = kFootprintCells * kSlotsPerCell;  // 2925
inline constexpr int kPassClass = kAnchoredFootprints;                       // 2925
inline constexpr int kExtraClass = kAnchoredFootprints + 1;                  // 2926
inline constexpr int kFootprintClasses = kAnchoredFootprints + 2;            // 2927

// The footprint class of a played move, in the `flip` frame. A non-PLAY move
// (EXCHANGE / PASS) maps to kPassClass.
int footprint_class(const Move& m, bool flip);

// Decode a per-cell slot [0, kSlotsPerCell) into its orientation (in the same
// frame the slot was encoded -- flipped if the class is a flipped-frame class)
// and tile count k. Slot 0 is the orientation-free k==1 footprint (horizontal
// reported by convention); the caller un-flips the orientation if it needs board
// coordinates.
void footprint_slot_decode(int slot, bool& horizontal, int& k);

// The covered board cells of a footprint class, in the `flip` frame: the first
// k empty cells from the anchor along the orientation on `board` (the state
// BEFORE the move). Writes the (row, col) pairs into `cells` and returns their
// count. Returns 0 for kPassClass / kExtraClass, and for a structurally
// impossible class on this board -- anchor not empty, or fewer than k empty
// cells before the board edge (such classes never occur as targets and are
// masked out). This inverts footprint_class: the covered cells of
// footprint_class(m, flip) on m's pre-move board are exactly m's placed squares.
int footprint_cells(int cls, const Board& board, bool flip,
                    std::array<std::pair<int, int>, kFootprintMaxK>& cells);

}  // namespace scribblez
