#include "training/move_set_encoder.h"

#include "encoding/input_encoder.h"  // kScoreDiffInputScale

#include <algorithm>

namespace scribblez {
namespace move_set {

void encode_move(const Move& m, int pre_move_score_diff, int32_t* letters, uint8_t* blanks,
                 int32_t* squares, uint8_t* tile_mask, float* scalars) {
  std::fill_n(letters, kMoveMaxPlaced, 0);
  std::fill_n(blanks, kMoveMaxPlaced, uint8_t(0));
  std::fill_n(squares, kMoveMaxPlaced, 0);
  std::fill_n(tile_mask, kMoveMaxPlaced, uint8_t(0));

  if (m.type() == MoveType::PLAY) {
    // visit_placed_squares yields the newly placed squares in lane order, the
    // same order Move stores its glyphs, so a single counter indexes both. A
    // placed tile always carries a letter (natural or a designated blank).
    int placed = 0;
    visit_placed_squares(m, [&](int r, int c) {
      const Glyph g = m.glyph(placed);
      letters[placed] = g.letter().index() + 1;  // 1..26; 0 is the empty slot
      blanks[placed] = g.is_blank() ? 1 : 0;
      squares[placed] = r * BOARD_SIZE + c;
      tile_mask[placed] = 1;
      ++placed;
    });
  } else {
    // An EXCHANGE has no placed squares (visit_placed_squares yields nothing);
    // its glyphs are the surrendered tiles, encoded so same-size exchanges
    // differ by WHICH tiles leave. An undesignated blank has no letter --
    // letters stays 0 (the pad value; the blank flag alone represents it).
    // A PASS has no glyphs and stays all-zero.
    for (int i = 0; i < m.num_glyphs(); ++i) {
      const Glyph g = m.glyph(i);
      letters[i] = g.has_letter() ? g.letter().index() + 1 : 0;
      blanks[i] = g.is_blank() ? 1 : 0;
      tile_mask[i] = 1;
    }
  }

  // Resultant post-move differential (mover POV): the mover's score advantage
  // plus this move's score, on the board trunk's score-diff scale.
  const int resultant_diff = pre_move_score_diff + int(m.score());
  scalars[0] = float(resultant_diff) / kScoreDiffInputScale;
  scalars[1] = float(m.num_glyphs()) / float(kMoveMaxPlaced);
  scalars[2] = m.type() == MoveType::PLAY ? 1.0f : 0.0f;
}

void encode_moves(const Move* moves, int64_t n, const int32_t* pre_move_score_diffs,
                  int32_t* letters, uint8_t* blanks, int32_t* squares, uint8_t* tile_mask,
                  float* scalars) {
  for (int64_t i = 0; i < n; ++i) {
    encode_move(moves[i], pre_move_score_diffs[i], letters + i * kMoveMaxPlaced,
                blanks + i * kMoveMaxPlaced, squares + i * kMoveMaxPlaced,
                tile_mask + i * kMoveMaxPlaced, scalars + i * kMoveScalars);
  }
}

void MoveFeatureArrays::encode(const Move* moves, int n, int pre_move_score_diff) {
  count = n;
  letters.resize(size_t(n) * kMoveMaxPlaced);
  blanks.resize(size_t(n) * kMoveMaxPlaced);
  squares.resize(size_t(n) * kMoveMaxPlaced);
  tile_mask.resize(size_t(n) * kMoveMaxPlaced);
  scalars.resize(size_t(n) * kMoveScalars);
  for (int i = 0; i < n; ++i) {
    encode_move(moves[i], pre_move_score_diff, letters.data() + i * kMoveMaxPlaced,
                blanks.data() + i * kMoveMaxPlaced, squares.data() + i * kMoveMaxPlaced,
                tile_mask.data() + i * kMoveMaxPlaced, scalars.data() + i * kMoveScalars);
  }
}

}  // namespace move_set
}  // namespace scribblez
