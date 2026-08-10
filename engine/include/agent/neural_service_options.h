#pragma once

// The --player options that select and size the served position evaluation
// model, shared by every agent type that drives one (--type=neural,
// --type=neural-sim) so their model-facing CLIs cannot drift.

#include "nn/neural_net.h"

#include <boost/program_options.hpp>

#include <string>

namespace scribblez {

struct NeuralServiceOptions {
  std::string model;
  int batch_size = 256;
  int cuda_device = 0;
  std::string precision = "FP16";

  // Register --model/--batch-size/--cuda-device/--precision, bound to this
  // struct's fields. Call before parsing the agent's option tokens.
  void add_options(boost::program_options::options_description& desc);

  // The validated NeuralNetParams these options describe. `min_batch` lets an
  // agent raise the engine batch to its own per-turn candidate cap, so one
  // chunk can carry the whole set. Throws std::runtime_error on bad input.
  nn::NeuralNetParams net_params(int min_batch) const;
};

}  // namespace scribblez
