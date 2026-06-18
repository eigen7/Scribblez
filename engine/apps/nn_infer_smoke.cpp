// Standalone sanity-check for the TensorRT inference path, independent of any
// game logic. Loads an exported ONNX model (building or loading a cached
// engine), runs a batch of all-zero inputs through the NNEvaluationService, and
// prints the per-row WLD probabilities and ScoreDiff mean.
//
// Usage:
//   nn_infer_smoke <model.onnx> [num_rows] [FP16|FP32]
//
// A successful run confirms the whole engine path -- ONNX parse, engine build +
// plan cache, host<->device copies, and output decoding -- works end to end.

#include "scribblez/input_encoder.h"
#include "scribblez/nn/nn_evaluation_service.h"
#include "scribblez/nn/trt_util.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
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
    scribblez::nn::NeuralNetParams params;
    params.max_batch_size = std::max(rows, 1);
    params.precision = scribblez::nn::parse_precision(precision);

    scribblez::nn::NNEvaluationService service(params);
    service.load(model);

    // All-zero inputs (the canonical game-start-ish encoding); we only care that
    // the model produces finite, well-formed outputs.
    std::vector<float> inputs(static_cast<size_t>(rows) * scribblez::binlog::kInputFloats, 0.0f);
    std::vector<scribblez::nn::Eval> evals = service.evaluate(inputs.data(), rows);

    for (int r = 0; r < rows; ++r) {
      const scribblez::nn::Eval& e = evals[r];
      std::cout << "row " << r << ": P(win)=" << e.p_win << " P(draw)=" << e.p_draw
                << " P(loss)=" << e.p_loss << " win_prob=" << e.win_prob
                << " score_diff_mean=" << e.score_diff_mean << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "nn_infer_smoke error: " << ex.what() << "\n";
    return 1;
  }
}
