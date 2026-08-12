// Standalone sanity-check for the move set evaluation inference path,
// independent of any game logic -- nn_infer_smoke's counterpart for the other
// model family. Loads an exported ONNX model (building or loading a cached
// engine), scores a synthetic candidate set through MoveSetEvalService, and
// prints each candidate's decoded Eval.
//
// Usage:
//   mset_infer_smoke <model.onnx> [num_moves] [FP16|FP32]
//
// A successful run confirms the whole path -- ONNX parse and metadata checks,
// engine build + plan cache, the dtype-aware bindings, chunking, and the Eval
// decode -- works end to end on a real checkpoint, and running it at both
// precisions is the FP32-vs-FP16 spot check. The move features are synthetic,
// not encoder output: what a Move encodes to has its own golden test, and this
// tool is about the engine path.

#include "encoding/input_encoder.h"
#include "nn/move_set_eval_service.h"
#include "nn/trt_util.h"
#include "training/move_set_encoder.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

// A candidate set spanning the shapes the model sees: plays of 1..7 tiles
// spread across the board, and -- every fifth candidate -- an exchange, which
// carries tiles but no squares.
scribblez::move_set::MoveFeatureArrays synthetic_candidates(int num_moves) {
  using namespace scribblez::move_set;
  MoveFeatureArrays moves;
  moves.count = num_moves;
  moves.letters.assign(static_cast<size_t>(num_moves) * kMoveMaxPlaced, 0);
  moves.blanks.assign(static_cast<size_t>(num_moves) * kMoveMaxPlaced, 0);
  moves.squares.assign(static_cast<size_t>(num_moves) * kMoveMaxPlaced, 0);
  moves.tile_mask.assign(static_cast<size_t>(num_moves) * kMoveMaxPlaced, 0);
  moves.scalars.assign(static_cast<size_t>(num_moves) * kMoveScalars, 0.0f);

  for (int m = 0; m < num_moves; ++m) {
    const bool is_play = m % 5 != 0;
    const int tiles = m % kMoveMaxPlaced + 1;
    for (int t = 0; t < tiles; ++t) {
      const size_t slot = static_cast<size_t>(m) * kMoveMaxPlaced + t;
      moves.letters[slot] = (m + t) % 26 + 1;
      moves.tile_mask[slot] = 1;
      if (is_play) moves.squares[slot] = (m * kMoveMaxPlaced + t) % kMoveCells;
    }
    float* scalars = moves.scalars.data() + static_cast<size_t>(m) * kMoveScalars;
    scalars[0] = static_cast<float>(m - num_moves / 2) / 100.0f;
    scalars[1] = static_cast<float>(tiles) / kMoveMaxPlaced;
    scalars[2] = is_play ? 1.0f : 0.0f;
  }
  return moves;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [num_moves] [FP16|FP32]\n";
    return 1;
  }

  const std::string model = argv[1];
  const int num_moves = std::max(argc > 2 ? std::atoi(argv[2]) : 8, 1);
  const std::string precision = argc > 3 ? argv[3] : "FP16";

  try {
    scribblez::nn::MoveSetNetParams params;
    params.onnx_path = model;
    params.precision = scribblez::nn::parse_precision(precision);

    scribblez::nn::TrtMoveSetEvalService service(params);
    service.load();

    // An all-zero board row at the model's own width: the candidates are what
    // this tool varies.
    const size_t row_floats =
      static_cast<size_t>(service.spatial_planes()) * scribblez::kBoardCells +
      service.scalar_floats();
    const std::vector<float> board(row_floats, 0.0f);

    const scribblez::move_set::MoveFeatureArrays moves = synthetic_candidates(num_moves);
    std::vector<scribblez::nn::Eval> evals(num_moves);
    service.evaluate(board.data(), moves, evals.data());

    for (int m = 0; m < num_moves; ++m) {
      const scribblez::nn::Eval& e = evals[m];
      std::cout << "move " << m << ": P(win)=" << e.p_win << " P(draw)=" << e.p_draw
                << " P(loss)=" << e.p_loss << " win_prob=" << e.win_prob
                << " score_diff_mean=" << e.score_diff_mean << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "mset_infer_smoke error: " << ex.what() << "\n";
    return 1;
  }
}
