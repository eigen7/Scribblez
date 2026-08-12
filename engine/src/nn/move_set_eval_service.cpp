#include "nn/move_set_eval_service.h"

#include "encoding/input_encoder.h"
#include "nn/eval_decode.h"
#include "training/training_targets.h"

#include <algorithm>
#include <cstring>

namespace scribblez {
namespace nn {

namespace {

// Copy `rows` rows of `width` elements from a candidate set's buffer into the
// engine's staging buffer, starting at move `start`.
template <typename T>
void stage_rows(const std::vector<T>& src, int start, int rows, int width, T* dst) {
  std::memcpy(dst, src.data() + static_cast<size_t>(start) * width,
              sizeof(T) * static_cast<size_t>(rows) * width);
}

}  // namespace

TrtMoveSetEvalService::TrtMoveSetEvalService(const MoveSetNetParams& params) : net_(params) {}

void TrtMoveSetEvalService::load() { net_.load(); }

void TrtMoveSetEvalService::evaluate(const float* board_row,
                                     const move_set::MoveFeatureArrays& moves, Eval* out) {
  // Split the encoder row into the engine's two board buffers, at the model's
  // own widths. It is staged once and re-sent with every chunk, the engine
  // holding one position per call by construction.
  const size_t spatial_floats = static_cast<size_t>(net_.spatial_planes()) * kBoardCells;
  std::memcpy(net_.input_spatial_host(), board_row, sizeof(float) * spatial_floats);
  std::memcpy(net_.input_scalar_host(), board_row + spatial_floats,
              sizeof(float) * net_.scalar_floats());

  const int max_moves = net_.max_moves();
  for (int start = 0; start < moves.count; start += max_moves) {
    evaluate_chunk(moves, start, std::min(max_moves, moves.count - start), out + start);
  }
}

void TrtMoveSetEvalService::evaluate_chunk(const move_set::MoveFeatureArrays& moves, int start,
                                           int chunk, Eval* out) {
  constexpr int kTiles = move_set::kMoveMaxPlaced;
  stage_rows(moves.letters, start, chunk, kTiles, net_.move_letters_host());
  stage_rows(moves.blanks, start, chunk, kTiles, net_.move_blanks_host());
  stage_rows(moves.squares, start, chunk, kTiles, net_.move_squares_host());
  stage_rows(moves.tile_mask, start, chunk, kTiles, net_.move_tile_mask_host());
  stage_rows(moves.scalars, start, chunk, move_set::kMoveScalars, net_.move_scalars_host());

  net_.predict(chunk);

  const float* wld = net_.wld_host();
  const float* sd = net_.score_diff_host();
  for (int r = 0; r < chunk; ++r) {
    out[r] = decode_eval(wld + static_cast<size_t>(r) * kWldFloats,
                         sd + static_cast<size_t>(r) * kScoreDiffOutputFloats);
  }
}

std::unique_ptr<MoveSetEvalService> make_loaded_move_set_service(const MoveSetNetParams& params) {
  auto svc = std::make_unique<TrtMoveSetEvalService>(params);
  svc->load();
  return svc;
}

}  // namespace nn
}  // namespace scribblez
