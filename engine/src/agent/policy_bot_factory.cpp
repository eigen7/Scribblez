// PolicyBot's command-line construction and backend: the only part that
// reaches CUDA and TensorRT (the policy-set model and the leave inference
// model), so policy_bot.cpp's unit tests can inject stubs instead.

#include "agent/agent_options.h"
#include "agent/policy_bot.h"
#include "belief/served_leave_model.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/onnx_metadata.h"
#include "util/exception.h"
#include "util/io.h"

#include <boost/program_options.hpp>

#include <memory>
#include <string>
#include <utility>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// The metadata entries policy_set/onnx_export.py stamps for PolicyBot.
constexpr const char* kBlankPruneKey = "blank_prune_points";
constexpr const char* kLeaveTagKey = "leave_inference_tag";
constexpr const char* kLeaveGenerationKey = "leave_inference_generation";

po::options_description policybot_options(PolicyBot::Params& p) {
  po::options_description desc("policybot options");
  // One call scores a turn's whole pool.
  p.service.batch_size = nn::PolicySetSpec::kDefaultMaxRows;
  p.service.add_options(desc);
  desc.add_options()  //
    ("leave-inference", po::value<std::string>(&p.leave_inference),
     "the leave inference model (ONNX) the model's pools drew their beliefs from, the "
     "generation its export names; required iff it names one");
  return desc;
}

// The leave inference generation `meta` says the model's pools drew their
// beliefs from, as an identity to compare; an empty tag for uniform beliefs.
nn::ModelIdentity trained_leave_model(const nn::OnnxMetadata& meta) {
  const auto tag = meta.entries.find(kLeaveTagKey);
  return {.hash = {},
          .tag = tag == meta.entries.end() ? std::string() : tag->second,
          .generation = meta.int_entry(kLeaveGenerationKey, -1)};
}

// The leave model `p` names, checked against the one the model trained on.
belief::AgentLeaveModel leave_model(const PolicyBot::Params& p, const nn::ModelIdentity& trained) {
  if (trained.tag.empty()) {
    if (!p.leave_inference.empty()) {
      throw util::CleanException("policybot: {} trained on uniform beliefs; drop --leave-inference",
                                 p.service.model);
    }
    return {};
  }
  if (p.leave_inference.empty()) {
    throw util::CleanException(
      "policybot: {} trained on the beliefs of leave_inference tag {} generation {}; pass "
      "--leave-inference=<that export>",
      p.service.model, trained.tag, trained.generation);
  }
  belief::AgentLeaveModel leave =
    belief::agent_leave_model(p.leave_inference, HastyEquity::instance().leave_values());
  if (leave.identity.tag != trained.tag || leave.identity.generation != trained.generation) {
    throw util::CleanException(
      "policybot: {} is leave_inference tag {} generation {}, but {} trained on generation {} "
      "of tag {}",
      p.leave_inference, leave.identity.tag, leave.identity.generation, p.service.model,
      trained.generation, trained.tag);
  }
  return leave;
}

PolicyBot::Backend make_backend(const PolicyBot::Params& p) {
  if (p.service.model.empty()) throw util::CleanException("policybot: --model is required");
  // Metadata first, so a mismatch fails before the TensorRT engine build.
  const nn::OnnxMetadata meta = nn::parse_onnx_metadata(util::read_file_bytes(p.service.model));
  PolicyBot::Backend b;
  b.blank_prune_points = meta.int_entry(kBlankPruneKey, -1);
  if (b.blank_prune_points < 0) {
    throw util::CleanException("policybot: {} records no {}; is it a policy_set export?",
                               p.service.model, kBlankPruneKey);
  }
  belief::AgentLeaveModel leave = leave_model(p, trained_leave_model(meta));
  b.predict = std::move(leave.predict);
  b.table = std::move(leave.table);
  b.policy = nn::PolicySetEvalService::create(p.service.net_params<nn::PolicySetSpec>(0));
  return b;
}

}  // namespace

PolicyBot::PolicyBot(const Params& params) : PolicyBot(params, make_backend(params)) {}

PolicyBot::Params PolicyBot::parse_params(const std::vector<std::string>& tokens, int thread_id,
                                          const std::string& name,
                                          boost::program_options::options_description& extra,
                                          const char* type_label) {
  Params p{.thread_id = thread_id, .name = name};
  po::options_description desc = policybot_options(p);
  desc.add(extra);
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type={} options: {}", type_label, e.what());
  }
  // The equity tables must not load concurrently with queries (hasty_equity.h).
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  p.dict = &Lexicon::instance().dict();
  return p;
}

std::unique_ptr<PolicyBot> PolicyBot::from_spec(const std::vector<std::string>& tokens,
                                                int thread_id, const std::string& name) {
  po::options_description no_extra;
  return std::make_unique<PolicyBot>(parse_params(tokens, thread_id, name, no_extra, kType));
}

std::string PolicyBot::options_help() {
  Params defaults;  // binding targets; only the defaults are read
  return agent_options_help(
    "  PolicyBot: plays the policy-set model's argmax over every legal move, no search. Its\n"
    "  belief about the opponent's rack comes from the leave inference generation the\n"
    "  model's pools drew theirs from. Hidden leaves only; policybot-endgame solves the\n"
    "  endgame.\n",
    policybot_options(defaults));
}

}  // namespace scribblez
