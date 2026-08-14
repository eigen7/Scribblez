#pragma once

// The --player options that select and size a served evaluation model, shared
// by every agent type that drives one (--type=neural, --type=neural-sim,
// --type=mset-sim) so their model-facing CLIs cannot drift. The two model
// families differ in what a scored row IS -- a candidate's post-move position,
// or a candidate of one position's move set -- but not in how a model is
// named, placed on a device, or sized per GPU call.

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
  // struct's fields. Call before parsing the agent's option tokens; an agent
  // whose sensible per-call ceiling differs assigns batch_size first, which is
  // then what the help renders as the default.
  void add_options(boost::program_options::options_description& desc);

  // The validated params these options describe, for Spec's model family.
  // `min_rows` lets an agent raise the engine's per-call ceiling to its own
  // per-turn candidate cap, so one chunk can carry the whole set. Throws
  // std::runtime_error on bad input.
  template <typename Spec>
  nn::NeuralNetParams<Spec> net_params(int min_rows) const;
};

}  // namespace scribblez
