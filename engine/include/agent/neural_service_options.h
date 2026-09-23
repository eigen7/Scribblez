#pragma once

// The --player options that name a served evaluation model and set its device,
// precision and batch size, shared by --type=neural, neural-sim and mset-sim so
// their model options cannot drift. (UltimateBot serves a pair of graphs and
// has its own.)

#include "nn/neural_net.h"

#include <boost/program_options.hpp>

#include <string>

namespace scribblez {

struct NeuralServiceOptions {
  std::string model;
  int batch_size = 256;
  int cuda_device = 0;
  // BF16 by default: its FP32-range exponent cannot overflow on any
  // checkpoint's activations, and its lost mantissa costs far less than the
  // model's own error. FP16 is for models known to fit its range.
  std::string precision = "BF16";

  // Register the options, bound to this struct's fields. An agent with a
  // different sensible batch size assigns batch_size first; the help then
  // shows it as the default.
  void add_options(boost::program_options::options_description& desc);

  // The validated params for Spec's model family. `min_rows` raises the
  // engine's per-call row ceiling, so an agent's whole per-turn candidate set
  // fits one call. Throws util::CleanException on bad input.
  template <typename Spec>
  nn::NeuralNetParams<Spec> net_params(int min_rows) const;
};

}  // namespace scribblez
