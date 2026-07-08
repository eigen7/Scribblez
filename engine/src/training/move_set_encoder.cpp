#include "training/move_set_encoder.h"

#include "encoding/input_encoder.h"  // kScoreDiffInputScale

#include <algorithm>

namespace scribblez {
namespace move_set {

void encode_move(const Move& m, int pre_move_score_diff, int32_t* letters, uint8_t* blanks,
                 int32_t* squares, uint8_t* tile_mask, float* scalars) {
  std::fill_n(letters, kMoveMaxPlaced, 0);
  std::fill_n(blanks, kMoveMaxPlaced, static_cast<uint8_t>(0));
  std::fill_n(squares, kMoveMaxPlaced, 0);
  std::fill_n(tile_mask, kMoveMaxPlaced, static_cast<uint8_t>(0));

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

  // Resultant post-move differential (mover POV): the mover's score advantage
  // plus this move's score, on the board trunk's score-diff scale.
  const int resultant_diff = pre_move_score_diff + static_cast<int>(m.score());
  scalars[0] = static_cast<float>(resultant_diff) / kScoreDiffInputScale;
  scalars[1] = static_cast<float>(m.num_glyphs()) / static_cast<float>(kMoveMaxPlaced);
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

}  // namespace move_set
}  // namespace scribblez
