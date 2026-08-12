#pragma once

// The --player options that select and size a served evaluation model, shared
// by every agent type that drives one (--type=neural, --type=neural-sim,
// --type=mset-sim) so their model-facing CLIs cannot drift. The two model
// families differ in what a scored row IS -- a candidate's post-move position,
// or a candidate of one position's move set -- but not in how a model is
// named, placed on a device, or sized per GPU call.

#include "nn/move_set_net.h"
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

  // The validated NeuralNetParams these options describe. `min_batch` lets an
  // agent raise the engine batch to its own per-turn candidate cap, so one
  // chunk can carry the whole set. Throws std::runtime_error on bad input.
  nn::NeuralNetParams net_params(int min_batch) const;

  // The same for a move set evaluation model, where the per-call ceiling is
  // the candidates of one turn. Throws std::runtime_error on bad input.
  nn::MoveSetNetParams move_set_net_params(int min_moves) const;
};

}  // namespace scribblez
