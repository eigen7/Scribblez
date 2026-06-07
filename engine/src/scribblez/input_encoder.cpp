#include "scribblez/input_encoder.h"

#include "scribblez/board.h"
#include "scribblez/glyph.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"
#include "scribblez/tile.h"

#include <cstring>

namespace scribblez {
namespace binlog {

namespace {

// Index into a single 15x15 plane: row-major if !flip, transposed if flip.
inline int plane_idx(int r, int c, bool flip) {
  return flip ? (c * kBoardSide + r) : (r * kBoardSide + c);
}

// Plane offsets within the spatial block.
inline constexpr int kBlankPlane = kLetterPlanes;                          // 26
inline constexpr int kPremiumPlane0 = kLetterPlanes + kBlankMarkerPlanes;  // 27
inline constexpr int kLastOppPlane = kPremiumPlane0 + kPremiumPlanes;      // 31

// Letter, blank-marker, and premium planes.
void encode_board_planes(const Glyph* board, bool flip, float* planes_out) {
  for (int r = 0; r < kBoardSide; ++r) {
    for (int c = 0; c < kBoardSide; ++c) {
      const Glyph g = board[r * kBoardSide + c];
      if (g.is_empty()) continue;
      const int letter = g.letter().index();  // 0..25, valid for any non-empty glyph
      planes_out[letter * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
      if (g.is_blank()) {
        planes_out[kBlankPlane * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
      }
    }
  }

  const auto& prem = Board::PREMIUM;
  for (int r = 0; r < kBoardSide; ++r) {
    for (int c = 0; c < kBoardSide; ++c) {
      const Premium p = prem[r * kBoardSide + c];
      if (p == Premium::NONE) continue;
      int offset = -1;
      if (p == Premium::DLS)
        offset = 0;
      else if (p == Premium::TLS)
        offset = 1;
      else if (p == Premium::DWS)
        offset = 2;
      else if (p == Premium::TWS)
        offset = 3;
      if (offset < 0) continue;
      planes_out[(kPremiumPlane0 + offset) * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
    }
  }
}

// Last-opp-placement plane: mark squares the opponent placed tiles on during
// their most recent turn. Exact, using Move::square_mask -- bit i is set iff
// the i-th cell along the move direction was a newly placed tile.
void encode_last_opp_plane(const Move& m, bool flip, float* planes_out) {
  if (m.type != MoveType::PLAY) return;  // EXCHANGE / PASS -> all zeros
  const int dr = m.horizontal ? 0 : 1;
  const int dc = m.horizontal ? 1 : 0;
  int r = m.start_row, c = m.start_col;
  uint16_t mask = m.square_mask;
  for (int i = 0; i < kBoardSide; ++i) {
    if (r < 0 || r >= kBoardSide || c < 0 || c >= kBoardSide) break;
    if (mask == 0) break;
    if (mask & 1u) {
      planes_out[kLastOppPlane * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
    }
    mask = static_cast<uint16_t>(mask >> 1);
    r += dr;
    c += dc;
  }
}

// All 58 scalar features. Raw counts/values throughout; the model handles
// normalization (e.g. via BatchNorm or learned linear layers).
void encode_scalars(const PositionRecord& r, float* out) {
  // Active-rack histogram (raw counts).
  int rack_hist[27] = {0};
  for (Tile t : r.active_rack.tiles()) {
    if (!t.is_empty()) ++rack_hist[t.index()];
  }
  float* p = out;
  for (int i = 0; i < 27; ++i) *p++ = static_cast<float>(rack_hist[i]);

  // Bag composition (raw counts).
  for (int i = 0; i < 27; ++i) *p++ = static_cast<float>(r.bag_counts[i]);

  // Misc scalars (raw).
  *p++ = static_cast<float>(r.score_active - r.score_opp);
  int bag_size = 0;
  for (int i = 0; i < 27; ++i) bag_size += r.bag_counts[i];
  *p++ = static_cast<float>(bag_size);
  *p++ = static_cast<float>(r.active_rack.size());

  // Last opponent move: num_glyphs only. (Move type is recoverable as
  // num_glyphs==0 -> PASS, num_glyphs>0 with empty placement plane -> EXCHANGE,
  // num_glyphs>0 with nonzero placement plane -> PLAY.)
  *p++ = static_cast<float>(r.last_opp_move.num_glyphs());
}

}  // namespace

void encode_input(const PositionRecord& record, bool apply_flip, float* out) {
  std::memset(out, 0, sizeof(float) * static_cast<size_t>(kInputFloats));
  encode_board_planes(record.board, apply_flip, out);
  encode_last_opp_plane(record.last_opp_move, apply_flip, out);
  encode_scalars(record, out + kSpatialFloats);
}

}  // namespace binlog
}  // namespace scribblez
