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
    // visit_placed_squares yields squares in lane order, the order Move stores
    // its glyphs in, so one counter indexes both. A placed tile always has a
    // letter, a blank's being its designation.
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
    // An EXCHANGE's glyphs are its surrendered tiles; an undesignated blank
    // among them keeps letter 0. A PASS has no glyphs.
    for (int i = 0; i < m.num_glyphs(); ++i) {
      const Glyph g = m.glyph(i);
      letters[i] = g.has_letter() ? g.letter().index() + 1 : 0;
      blanks[i] = g.is_blank() ? 1 : 0;
      tile_mask[i] = 1;
    }
  }

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

void encode_move_cross_checks(Board& board, const Move& m, BoardUndo& undo, int32_t* cells,
                              uint8_t* letters) {
  uint8_t axes[kMoveCrossSlots];
  int32_t squares[kMoveCrossSlots];
  uint32_t old_masks[kMoveCrossSlots];
  uint32_t new_masks[kMoveCrossSlots];
  uint8_t real[kMoveCrossSlots];
  encode_cross_check_deltas(board, m, undo, axes, squares, old_masks, new_masks, real);
  std::fill_n(cells, kMoveCrossSlots, 0);
  std::fill_n(letters, kMoveCrossLetters, uint8_t(0));
  // Real entries lead (encode_cross_check_deltas).
  for (int i = 0; i < kMoveCrossSlots && real[i]; ++i) {
    cells[i] = 1 + axes[i] * kMoveCells + squares[i];
    for (int l = 0; l < 26; ++l) letters[i * 26 + l] = (new_masks[i] >> l) & 1u;
  }
}

void encode_moves_cross_checks(const Board& board, const Dictionary& dict, const Move* moves,
                               int64_t n, int32_t* cells, uint8_t* letters) {
  Board scratch = board;
  scratch.ensure_movegen_caches(dict);
  BoardUndo undo;
  for (int64_t i = 0; i < n; ++i) {
    encode_move_cross_checks(scratch, moves[i], undo, cells + i * kMoveCrossSlots,
                             letters + i * kMoveCrossLetters);
  }
}

void MoveFeatureArrays::resize(int n) {
  count = n;
  letters.assign(size_t(n) * kMoveMaxPlaced, 0);
  blanks.assign(size_t(n) * kMoveMaxPlaced, 0);
  squares.assign(size_t(n) * kMoveMaxPlaced, 0);
  tile_mask.assign(size_t(n) * kMoveMaxPlaced, 0);
  scalars.assign(size_t(n) * kMoveScalars, 0.0f);
  cross_cells.assign(size_t(n) * kMoveCrossSlots, 0);
  cross_letters.assign(size_t(n) * kMoveCrossLetters, 0);
}

void MoveFeatureArrays::encode(const Board& board, const Dictionary& dict, const Move* moves, int n,
                               int pre_move_score_diff) {
  resize(n);
  for (int i = 0; i < n; ++i) {
    encode_move(moves[i], pre_move_score_diff, letters.data() + i * kMoveMaxPlaced,
                blanks.data() + i * kMoveMaxPlaced, squares.data() + i * kMoveMaxPlaced,
                tile_mask.data() + i * kMoveMaxPlaced, scalars.data() + i * kMoveScalars);
  }
  encode_moves_cross_checks(board, dict, moves, n, cross_cells.data(), cross_letters.data());
}

}  // namespace move_set
}  // namespace scribblez
