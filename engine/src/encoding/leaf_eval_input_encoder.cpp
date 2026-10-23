#include "encoding/leaf_eval_input_encoder.h"

#include "game/tile_counts.h"
#include "lexicon/leave_values.h"
#include "util/assert.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace scribblez {

namespace {

using Encoder = LeafEvalInputEncoder;

// One seat's tiles, and whether its fit is a fraction (the pool) or present /
// absent (a leave).
struct Seat {
  TileCounts tiles;
  bool fractional;
};

// How many of a seat's tiles satisfy one square's cross-check, and the highest
// face value among them.
struct SeatFit {
  int fitting = 0;
  int max_value = 0;
};

// Every tile on neither the board nor a leave.
TileCounts leaf_pool(const LeafEvalInput& in) {
  TileCounts pool = in.board->unseen_tiles(in.my_leave);
  for (int i = 0; i < in.opp_leave.size(); ++i) {
    const bool ok = pool.remove(in.opp_leave.tiles()[i]);
    RELEASE_ASSERT(ok, "the opponent's leave is not all unseen");
  }
  return pool;
}

bool occupied(const Board& board, int r, int c) {
  return board.in_bounds(r, c) && !board.at(r, c).is_empty();
}

// Whether (r, c) is empty with a tile across `axis`: above or below it for a
// horizontal play (axis 0), left or right of it for a vertical one.
bool is_hook_square(const Board& board, int axis, int r, int c) {
  if (occupied(board, r, c)) return false;
  const int dr = axis == 0 ? 1 : 0;
  const int dc = 1 - dr;
  return occupied(board, r - dr, c - dc) || occupied(board, r + dr, c + dc);
}

bool fits(Tile t, uint32_t mask) {
  return t.is_blank() ? mask != 0 : ((mask >> t.index()) & 1u) != 0;
}

SeatFit seat_fit(const TileCounts& tiles, uint32_t mask) {
  SeatFit fit;
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    const int n = tiles.count(t);
    if (n == 0 || !fits(t, mask)) continue;
    fit.fitting += n;
    fit.max_value = std::max(fit.max_value, t.value());
  }
  return fit;
}

// Writes one seat's two planes at `cell`: its fit, then the highest fitting
// face value.
void write_seat_fit(const Seat& seat, uint32_t mask, int cell, float* planes) {
  const SeatFit fit = seat_fit(seat.tiles, mask);
  if (fit.fitting == 0) return;
  planes[cell] = seat.fractional ? float(fit.fitting) / float(seat.tiles.size()) : 1.0f;
  planes[kBoardCells + cell] = float(fit.max_value) / Encoder::kFaceValueScale;
}

// The join planes, seat-major, then axis, then (fit, face value).
void encode_join_planes(const Board& board, const CrossCheckMasks& masks,
                        const std::array<Seat, Encoder::kSeats>& seats, float* out) {
  constexpr int kSeatStride = Encoder::kAxes * Encoder::kJoinPlanesPerAxis * kBoardCells;
  constexpr int kAxisStride = Encoder::kJoinPlanesPerAxis * kBoardCells;
  for (int axis = 0; axis < Encoder::kAxes; ++axis) {
    const auto& axis_masks = axis == 0 ? masks.horizontal : masks.vertical;
    for (int r = 0; r < BOARD_SIZE; ++r) {
      for (int c = 0; c < BOARD_SIZE; ++c) {
        if (!is_hook_square(board, axis, r, c)) continue;
        const int cell = r * BOARD_SIZE + c;
        for (int s = 0; s < Encoder::kSeats; ++s) {
          write_seat_fit(seats[s], axis_masks[cell], cell,
                         out + s * kSeatStride + axis * kAxisStride);
        }
      }
    }
  }
}

void encode_scalars(const LeafEvalInput& in, const TileCounts& pool,
                    const LeaveValues& leave_values, float* out) {
  encode_rack_counts(in.my_leave, out + Encoder::kMyLeave0);
  encode_rack_counts(in.opp_leave, out + Encoder::kOppLeave0);
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) out[Encoder::kPool0 + t] = float(pool.count(t));
  encode_score_diff_scalar(in.score_diff, out + Encoder::kScoreDiff);
  out[Encoder::kMyLeaveValue] = leave_values.lookup(in.my_leave) / Encoder::kLeaveValueScale;
  out[Encoder::kOppLeaveValue] = leave_values.lookup(in.opp_leave) / Encoder::kLeaveValueScale;
  out[Encoder::kPoolSize] = float(pool.size()) / Encoder::kCountScale;
}

}  // namespace

void LeafEvalInputEncoder::encode(const LeafEvalInput& in, const Dictionary& dict,
                                  const LeaveValues& leave_values, float* out) {
  std::memset(out, 0, sizeof(float) * size_t(kInputFloats));
  in.board->ensure_movegen_caches(dict);
  const CrossCheckMasks masks = CrossCheckMasks::of(*in.board);
  const TileCounts pool = leaf_pool(in);
  const std::array<Seat, kSeats> seats = {Seat{in.my_leave.counts(), false},
                                          Seat{in.opp_leave.counts(), false}, Seat{pool, true}};

  BoardPlanes::encode(*in.board, out);
  encode_cross_check_planes(masks, out + BoardPlanes::kPlanes * kBoardCells);
  encode_join_planes(*in.board, masks, seats, out + kJoinPlane0 * kBoardCells);
  encode_scalars(in, pool, leave_values, out + kSpatialFloats);
}

}  // namespace scribblez
