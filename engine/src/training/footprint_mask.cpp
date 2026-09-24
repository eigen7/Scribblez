#include "training/footprint_mask.h"

#include "util/math.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace scribblez {

namespace {

// Tile availability as a bitset for a fast per-cell test: bit t is set iff at
// least one of tile t is in stock, indexed like the 27-count array (0..25 =
// A..Z, 26 = blank).
using tile_set_t = uint32_t;
inline constexpr tile_set_t kBlankTile = 1u << 26;
inline constexpr tile_set_t kAllTiles = kAllLettersMask | kBlankTile;

// A null array means everything is in stock, which reduces the availability
// test to board legality.
tile_set_t available_tiles(const uint8_t* counts) {
  if (counts == nullptr) return kAllTiles;
  tile_set_t avail = 0;
  for (int t = 0; t < TILE_KINDS; ++t)
    if (counts[t] > 0) avail |= (1u << t);
  return avail;
}

// Whether some available letter can play at empty cell (r,c) as part of a word
// in board-frame orientation `horizontal`: the perpendicular cross-check must
// allow a letter in `avail` (a blank allows any). The cache indexing matches the
// input encoder: a horizontal word's cross-words run down the columns
// (non-transposed cache, [r*side+c]), a vertical word's along the rows
// (transposed cache, [c*side+r]).
bool cell_admits_letter(const Board& board, tile_set_t avail, bool horizontal, int r, int c) {
  const CrossCheck& cc = horizontal ? board.cross_checks(false)[r * BOARD_SIZE + c]
                                    : board.cross_checks(true)[c * BOARD_SIZE + r];
  if (avail & kBlankTile) return cc.mask != 0;
  return (cc.mask & avail) != 0;
}

// Whether some available letter can play as a lone tile at (r,c). A lone tile
// forms words on both axes at once, so its letter must pass both cross-checks.
bool lone_tile_admits_letter(const Board& board, tile_set_t avail, int r, int c) {
  const CrossCheck& vert = board.cross_checks(false)[r * BOARD_SIZE + c];
  const CrossCheck& horiz = board.cross_checks(true)[c * BOARD_SIZE + r];
  const uint32_t allowed = vert.mask & horiz.mask;
  if (avail & kBlankTile) return allowed != 0;
  return (allowed & avail) != 0;
}

bool touches_seed(const SquareSet& seed, int r, int c) {
  for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
    const int nr = r + dr;
    const int nc = c + dc;
    if (nr < 0 || nr >= kFootprintSide || nc < 0 || nc >= kFootprintSide) continue;
    if (seed.contains(nr, nc)) return true;
  }
  return false;
}

}  // namespace

SquareSet occupied_squares(const Board& board) {
  SquareSet s;
  for (int i = 0; i < kFootprintCells; ++i)
    if (!board.at(i / kFootprintSide, i % kFootprintSide).is_empty()) s.bits.set(i);
  return s;
}

FootprintPly footprint_ply(const Board& board, const SquareSet& seed, int budget,
                           bool use_cross_checks, const uint8_t* available_counts, bool win_head) {
  FootprintPly out;
  out.mask.fill(false);
  out.reach = seed;
  const int kmax = std::min(budget, kFootprintMaxK);
  const tile_set_t avail = use_cross_checks ? available_tiles(available_counts) : kAllTiles;
  const bool adjacency_gate = !seed.empty();
  // TODO(perf): most anchors abut nothing and get masked, yet all
  // kFootprintCells are scanned; growing footprints outward from the seed would
  // visit only the connected few (~222 of 2925 classes after a centred
  // opener). Mask building has not shown up in profiles.
  std::array<int, kFootprintMaxK> covered;
  for (int cell = 0; cell < kFootprintCells; ++cell) {
    const int anchor_r = cell / kFootprintSide;
    const int anchor_c = cell % kFootprintSide;
    if (!board.at(anchor_r, anchor_c).is_empty()) continue;

    for (int slot = 0; slot < kSlotsPerCell; ++slot) {
      bool horizontal;
      int k;
      footprint_slot_decode(slot, horizontal, k);
      if (k > kmax) continue;
      const int cls = cell * kSlotsPerCell + slot;

      if (k == 1) {
        const bool ok =
          (!use_cross_checks || lone_tile_admits_letter(board, avail, anchor_r, anchor_c)) &&
          (!adjacency_gate || touches_seed(seed, anchor_r, anchor_c));
        if (ok) {
          out.mask[cls] = true;
          out.reach.bits.set(cell);
        }
        continue;
      }

      int count = 0;
      int r = anchor_r;
      int c = anchor_c;
      bool ok = true;
      bool connected = !adjacency_gate;
      while (r < kFootprintSide && c < kFootprintSide && count < k) {
        if (board.at(r, c).is_empty()) {
          if (use_cross_checks && !cell_admits_letter(board, avail, horizontal, r, c)) {
            ok = false;
            break;
          }
          if (!connected && touches_seed(seed, r, c)) connected = true;
          covered[count++] = r * kFootprintSide + c;
        }
        if (horizontal) {
          ++c;
        } else {
          ++r;
        }
      }
      // count < k: the board edge cut the run short.
      if (ok && count == k && connected) {
        out.mask[cls] = true;
        for (int i = 0; i < k; ++i) out.reach.bits.set(covered[i]);
      }
    }
  }
  out.mask[kPassClass] = true;
  out.mask[kExtraClass] = win_head;
  return out;
}

void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool win_head, FootprintMask& mask) {
  mask = footprint_ply(board, occupied_squares(board), tile_budget, /*use_cross_checks=*/true,
                       available_counts, win_head)
           .mask;
}

void self_footprint_mask(const Board& board, int self_budget, int opp_budget,
                         const uint8_t* opp_available_counts, bool win_head, FootprintMask& mask) {
  const FootprintPly opp = footprint_ply(board, occupied_squares(board), opp_budget,
                                         /*use_cross_checks=*/true, opp_available_counts,
                                         /*win_head=*/false);
  mask = footprint_ply(board, opp.reach, self_budget, /*use_cross_checks=*/false, nullptr, win_head)
           .mask;
}

void footprint_reachable_cells(const Board& board, const uint8_t* available_counts, int tile_budget,
                               float* out) {
  const SquareSet seed = occupied_squares(board);
  const FootprintPly ply = footprint_ply(board, seed, tile_budget, /*use_cross_checks=*/true,
                                         available_counts, /*win_head=*/false);
  const auto covered = ply.reach.bits & ~seed.bits;
  for (int i = 0; i < kFootprintCells; ++i) out[i] = covered.test(i) ? 1.0f : 0.0f;
}

}  // namespace scribblez
