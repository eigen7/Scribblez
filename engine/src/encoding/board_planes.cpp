#include "encoding/board_planes.h"

#include "encoding/input_encoder.h"
#include "game/glyph.h"
#include "util/assert.h"

namespace scribblez {

namespace {

constexpr int kCells = BOARD_SIZE * BOARD_SIZE;

// Offset from kPremiumPlane0, or -1 for a plain square.
int premium_offset(Premium p) {
  if (p == Premium::DLS) return 0;
  if (p == Premium::TLS) return 1;
  if (p == Premium::DWS) return 2;
  if (p == Premium::TWS) return 3;
  return -1;
}

// A square with no perpendicular neighbor has an all-ones mask, so it is
// written as all 26 letters legal. A 1 then means "legal here" on its own; an
// all-zero encoding would force the network to check the neighbors to read it.
void write_cross_check(uint32_t mask, int cell, float* planes) {
  for (int l = 0; l < 26; ++l) {
    if (mask & (1u << l)) planes[l * kBoardCells + cell] = 1.0f;
  }
}

}  // namespace

void BoardPlanes::encode(const Board& board, float* planes_out) {
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const int idx = r * BOARD_SIZE + c;

      const Glyph g = board.at(r, c);
      if (!g.is_empty()) {
        planes_out[g.letter().index() * kCells + idx] = 1.0f;
        if (g.is_blank()) planes_out[kBlankMarkerPlane * kCells + idx] = 1.0f;
      }

      const int off = premium_offset(board.premium_at(r, c));
      if (off >= 0) planes_out[(kPremiumPlane0 + off) * kCells + idx] = 1.0f;
    }
  }
}

// EXCHANGE and PASS (including the initial last-move placeholder) leave the
// plane all-zero.
int encode_placement_plane(const Move& m, float* out) {
  visit_placed_squares(m, [&](int r, int c) { out[r * kBoardSide + c] = 1.0f; });
  return 1;
}

CrossCheckMasks CrossCheckMasks::of(const Board& board) {
  CrossCheckMasks out;
  // A horizontal word's cross words run down the columns, which is the
  // non-transposed cache; a vertical word's run along the rows.
  const auto& horizontal_play_cross = board.cross_checks(/*transposed=*/false);
  const auto& vertical_play_cross = board.cross_checks(/*transposed=*/true);
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!board.at(r, c).is_empty()) continue;
      out.horizontal[r * BOARD_SIZE + c] = horizontal_play_cross[r * BOARD_SIZE + c].mask;
      out.vertical[r * BOARD_SIZE + c] = vertical_play_cross[c * BOARD_SIZE + r].mask;
    }
  }
  return out;
}

CrossCheckMasks CrossCheckMasks::transpose() const {
  CrossCheckMasks out;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      out.horizontal[c * BOARD_SIZE + r] = vertical[r * BOARD_SIZE + c];
      out.vertical[c * BOARD_SIZE + r] = horizontal[r * BOARD_SIZE + c];
    }
  }
  return out;
}

// Horizontal A..Z, then vertical A..Z. A plane records only the per-square
// cross-check constraint; whether the main word itself is valid depends on
// the whole play and is not a per-square fact.
int encode_cross_check_planes(const CrossCheckMasks& masks, float* out) {
  float* h_planes = out;
  float* v_planes = out + kHorizontalCrossCheckPlanes * kBoardCells;
  for (int cell = 0; cell < kBoardCells; ++cell) {
    write_cross_check(masks.horizontal[cell], cell, h_planes);
    write_cross_check(masks.vertical[cell], cell, v_planes);
  }
  return kCrossCheckPlanes;
}

int encode_cross_check_planes(const Board& board, float* out) {
  return encode_cross_check_planes(CrossCheckMasks::of(board), out);
}

int encode_rack_counts(const Rack& rack, float* out) {
  for (Tile t : rack.tiles()) {
    if (!t.is_empty()) out[t.index()] += 1.0f;
  }
  return kRackCountFloats;
}

int encode_unseen_pool_thermometer(const uint8_t unseen[TILE_KINDS], float* out) {
  int offset = 0;
  for (int i = 0; i < TILE_KINDS; ++i) {
    for (int j = 0; j < unseen[i]; ++j) out[offset + j] = 1.0f;
    offset += TILE_COUNTS[i];
  }
  DEBUG_ASSERT(offset == kUnseenPoolThermoFloats);
  return kUnseenPoolThermoFloats;
}

// TODO(score-diff resolution near the endgame): as the bag empties, the
// win/loss boundary in score difference becomes sharp; a two-point swing can
// flip the likely outcome, where mid-game it barely matters. A single linear
// scalar can express that, but the network must spend capacity learning the
// steep, phase-dependent transition exactly where endgame training positions
// are sparsest. To measure it, slice held-out win calibration by (tiles
// remaining, score difference) and look for win probability flattened across
// the flip point in the near-empty-bag, small-difference cells. If it shows,
// prefer a few nonlinear features (RBF bumps or bins, denser near 0) over a
// full thermometer, and change the move set model's score feature
// (move_set_encoder.cpp) to match. The decisive endgame itself belongs to the
// endgame solver (docs/roadmap.md item 7), not to finer value-net input.
int encode_score_diff_scalar(int score_diff, float* out) {
  out[0] = float(score_diff) / kScoreDiffInputScale;
  return kScoreDiffInputFloats;
}

}  // namespace scribblez
