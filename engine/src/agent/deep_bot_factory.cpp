// DeepBot's command-line construction and backend: the only part that reaches
// CUDA and TensorRT (the GPU rollouts, the leaf network and the leave inference
// model), so deep_bot.cpp's unit tests can inject the CPU reference instead.

#include "agent/deep_bot.h"
#include "belief/served_leave_model.h"
#include "gpu/rollout_backend.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"

#include <boost/program_options.hpp>

#include <memory>
#include <utility>

namespace scribblez {

namespace {

DeepBot::Backend make_backend(const DeepBot::Params& p) {
  RolloutBackend rollouts;
  rollouts.backend = p.leaf_model == RolloutBackend::kSyntheticLeaf ? "cpu" : "gpu";
  rollouts.leaf_model = p.leaf_model;
  rollouts.leaf_precision = p.leaf_precision;
  rollouts.validate();
  DeepBot::Backend b;
  b.values = rollouts.shared(load_dictionary_or_throw());
  if (p.leave_inference.empty()) return b;
  belief::AgentLeaveModel leave =
    belief::agent_leave_model(p.leave_inference, HastyEquity::instance().leave_values());
  b.table = std::move(leave.table);
  b.predict = std::move(leave.predict);
  return b;
}

}  // namespace

DeepBot::DeepBot(const Params& params) : DeepBot(params, make_backend(params)) {}

std::unique_ptr<DeepBot> DeepBot::from_spec(const std::vector<std::string>& tokens, int thread_id,
                                            const std::string& name) {
  boost::program_options::options_description no_extra;
  return std::make_unique<DeepBot>(parse_params(tokens, thread_id, name, no_extra, kType));
}

}  // namespace scribblez
