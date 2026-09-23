// mset_infer_smoke: nn_infer_smoke's counterpart for the move set evaluation
// model. It loads an ONNX export, scores a synthetic candidate set against an
// all-zero board through TrtEvalService, and prints each candidate's decoded
// output. A clean run covers ONNX parse and metadata checks, engine build and
// plan cache, the bindings, chunking, and decoding on a real checkpoint.
// Running it at two precisions is a quick precision spot check.
//
//   mset_infer_smoke model.onnx [num_moves=8] [FP16|BF16|FP32, default BF16]
//
// The move features are synthetic rather than encoder output: the encoder has
// its own tests, and this tool is about the inference path.

#include "encoding/input_encoder.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "synthetic_candidates.h"
#include "training/move_set_encoder.h"
#include "util/misc.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [num_moves] [FP16|BF16|FP32]\n";
    return 1;
  }

  const std::string model = argv[1];
  const int num_moves = std::max(argc > 2 ? std::atoi(argv[2]) : 8, 1);
  const std::string precision = argc > 3 ? argv[3] : "BF16";

  try {
    using Spec = scribblez::nn::MoveSetEvaluationSpec;
    scribblez::nn::NeuralNetParams<Spec> params;
    params.onnx_path = model;
    params.precision = scribblez::nn::parse_precision(precision);

    scribblez::nn::TrtEvalService<Spec> service(params);
    service.load();

    // An all-zero board row at the model's own width; only the candidates vary.
    const size_t row_floats =
      size_t(service.spatial_planes()) * scribblez::kBoardCells + service.scalar_floats();
    const std::vector<float> board(row_floats, 0.0f);

    const scribblez::move_set::MoveFeatureArrays moves =
      scribblez::move_set::synthetic_candidates(num_moves);
    std::vector<float> wld(size_t(num_moves) * scribblez::nn::WldOutput::kRowElems);
    std::vector<float> sd(size_t(num_moves) * scribblez::nn::ScoreDiffOutput::kRowElems);
    float* const head_out[] = {wld.data(), sd.data()};
    service.evaluate({board.data(), &moves}, head_out);

    for (int m = 0; m < num_moves; ++m) {
      const float* w = wld.data() + size_t(m) * scribblez::nn::WldOutput::kRowElems;
      const float* s = sd.data() + size_t(m) * scribblez::nn::ScoreDiffOutput::kRowElems;
      std::cout << "move " << m << ": P(win)=" << w[0] << " P(draw)=" << w[1] << " P(loss)=" << w[2]
                << " win_prob=" << w[0] + 0.5f * w[1] << " score_diff_mean=" << s[0] << "\n";
    }
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
