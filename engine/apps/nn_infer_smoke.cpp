// nn_infer_smoke: checks that a position evaluation model export runs under
// TensorRT, with no game logic involved. It loads the ONNX file (building or
// reusing the cached engine plan), evaluates a batch of all-zero input rows,
// and prints each row's WLD probabilities and score-diff mean. Finite,
// well-formed output means ONNX parse, engine build, device copies and output
// decoding all work.
//
//   nn_infer_smoke model.onnx [num_rows=4] [FP16|BF16|FP32, default FP16]

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
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [num_rows] [FP16|BF16|FP32]\n";
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

    // Zero rows at the model's own input width: the values are meaningless, the
    // point is that the outputs come back finite and well-formed.
    const size_t row_floats =
      size_t(service.spatial_planes()) * scribblez::kBoardCells + service.scalar_floats();
    std::vector<float> inputs(size_t(rows) * row_floats, 0.0f);
    std::vector<float> wld(size_t(rows) * scribblez::nn::WldOutput::kRowElems);
    std::vector<float> sd(size_t(rows) * scribblez::nn::ScoreDiffOutput::kRowElems);
    float* const head_out[] = {wld.data(), sd.data()};
    service.evaluate({inputs.data(), rows}, head_out);

    for (int r = 0; r < rows; ++r) {
      const float* w = wld.data() + size_t(r) * scribblez::nn::WldOutput::kRowElems;
      const float* s = sd.data() + size_t(r) * scribblez::nn::ScoreDiffOutput::kRowElems;
      std::cout << "row " << r << ": P(win)=" << w[0] << " P(draw)=" << w[1] << " P(loss)=" << w[2]
                << " win_prob=" << w[0] + 0.5f * w[1] << " score_diff_mean=" << s[0] << "\n";
    }
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
