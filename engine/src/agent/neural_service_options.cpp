#include "agent/neural_service_options.h"

#include "nn/trt_util.h"

#include <algorithm>
#include <stdexcept>

namespace scribblez {

namespace po = boost::program_options;

// Boost.program_options renders defaults set via default_value() as "(=...)" in
// the help text, so the option descriptions deliberately omit them.
void NeuralServiceOptions::add_options(po::options_description& desc) {
  desc.add_options()  //
    ("model,m", po::value<std::string>(&model), "exported ONNX model (required)")(
      "batch-size,b", po::value<int>(&batch_size)->default_value(batch_size), "max GPU batch")(
      "cuda-device,d", po::value<int>(&cuda_device)->default_value(cuda_device), "GPU index")(
      "precision,p", po::value<std::string>(&precision)->default_value(precision),
      "TensorRT precision: FP16|FP32");
}

nn::NeuralNetParams NeuralServiceOptions::net_params(int min_batch) const {
  if (model.empty()) throw std::runtime_error("this player type requires --model=<path.onnx>");
  nn::NeuralNetParams params;
  params.onnx_path = model;
  params.cuda_device_id = cuda_device;
  params.max_batch_size = std::max({batch_size, min_batch, 1});
  params.precision = nn::parse_precision(precision);
  return params;
}

}  // namespace scribblez
