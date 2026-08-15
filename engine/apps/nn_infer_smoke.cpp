// Standalone sanity-check for the TensorRT inference path, independent of any
// game logic. Loads an exported ONNX model (building or loading a cached
// engine), runs a batch of all-zero inputs through the TrtEvalService, and
// prints the per-row WLD probabilities and ScoreDiff mean.
//
// Usage:
//   nn_infer_smoke <model.onnx> [num_rows] [FP16|FP32]
//
// A successful run confirms the whole engine path -- ONNX parse, engine build +
// plan cache, host<->device copies, and output decoding -- works end to end.

#include "encoding/input_encoder.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "util/misc.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [num_rows] [FP16|FP32]\n";
    return 1;
  }

  const std::string model = argv[1];
  const int rows = argc > 2 ? std::atoi(argv[2]) : 4;
  const std::string precision = argc > 3 ? argv[3] : "FP16";

  try {
    using Spec = scribblez::nn::PositionEvaluationSpec;
    scribblez::nn::NeuralNetParams<Spec> params;
    params.onnx_path = model;
    params.max_rows = std::max(rows, 1);
    params.precision = scribblez::nn::parse_precision(precision);

    scribblez::nn::TrtEvalService<Spec> service(params);
    service.load();

    // All-zero inputs (the canonical game-start-ish encoding) at the model's
    // own row width; we only care that the model produces finite, well-formed
    // outputs.
    const size_t row_floats =
      static_cast<size_t>(service.spatial_planes()) * scribblez::kBoardCells +
      service.scalar_floats();
    std::vector<float> inputs(static_cast<size_t>(rows) * row_floats, 0.0f);
    std::vector<float> wld(static_cast<size_t>(rows) * scribblez::nn::WldOutput::kRowElems);
    std::vector<float> sd(static_cast<size_t>(rows) * scribblez::nn::ScoreDiffOutput::kRowElems);
    float* const head_out[] = {wld.data(), sd.data()};
    service.evaluate({inputs.data(), rows}, head_out);

    for (int r = 0; r < rows; ++r) {
      const float* w = wld.data() + static_cast<size_t>(r) * scribblez::nn::WldOutput::kRowElems;
      const float* s =
        sd.data() + static_cast<size_t>(r) * scribblez::nn::ScoreDiffOutput::kRowElems;
      std::cout << "row " << r << ": P(win)=" << w[0] << " P(draw)=" << w[1] << " P(loss)=" << w[2]
                << " win_prob=" << w[0] + 0.5f * w[1] << " score_diff_mean=" << s[0] << "\n";
    }
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
