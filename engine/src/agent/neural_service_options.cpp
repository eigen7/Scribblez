#include "agent/neural_service_options.h"

#include "nn/trt_util.h"
#include "util/exception.h"

#include <algorithm>

namespace scribblez {

namespace po = boost::program_options;

// Boost.program_options renders defaults set via default_value() as "(=...)" in
// the help text, so the option descriptions deliberately omit them.
void NeuralServiceOptions::add_options(po::options_description& desc) {
  desc.add_options()  //
    ("model,m", po::value<std::string>(&model), "exported ONNX model (required)")(
      "batch-size,b", po::value<int>(&batch_size)->default_value(batch_size),
      "max candidate rows scored in one GPU call")(
      "cuda-device,d", po::value<int>(&cuda_device)->default_value(cuda_device), "GPU index")(
      "precision,p", po::value<std::string>(&precision)->default_value(precision),
      "TensorRT precision: FP16|FP32");
}

namespace {

// The model path is the one option with no usable default.
const std::string& require_model(const std::string& model) {
  if (model.empty()) throw util::CleanException("this player type requires --model=<path.onnx>");
  return model;
}

}  // namespace

template <typename Spec>
nn::NeuralNetParams<Spec> NeuralServiceOptions::net_params(int min_rows) const {
  nn::NeuralNetParams<Spec> params;
  params.onnx_path = require_model(model);
  params.cuda_device_id = cuda_device;
  params.max_rows = std::max({batch_size, min_rows, 1});
  params.precision = nn::parse_precision(precision);
  return params;
}

template nn::NeuralNetParams<nn::PositionEvaluationSpec>
NeuralServiceOptions::net_params<nn::PositionEvaluationSpec>(int min_rows) const;
template nn::NeuralNetParams<nn::MoveSetEvaluationSpec>
NeuralServiceOptions::net_params<nn::MoveSetEvaluationSpec>(int min_rows) const;

}  // namespace scribblez
