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

PolicyBot::TrainedOn trained_on(const nn::OnnxMetadata& meta) {
  const auto tag = meta.entries.find(kLeaveTagKey);
  return {.blank_prune_points = meta.int_entry(kBlankPruneKey, -1),
          .leave_model = {.hash = {},
                          .tag = tag == meta.entries.end() ? std::string() : tag->second,
                          .generation = meta.int_entry(kLeaveGenerationKey, -1)}};
}

// The --leave-inference export's identity, read from its file without
// building its engine.
nn::ModelIdentity leave_identity(const PolicyBot::Params& p) {
  return nn::read_model_identity(p.leave_inference, "leave_inference_format_version");
}

PolicyBot::Backend make_backend(const PolicyBot::Params& p) {
  if (p.service.model.empty()) throw util::CleanException("policybot: --model is required");
  const PolicyBot::TrainedOn trained =
    trained_on(nn::parse_onnx_metadata(util::read_file_bytes(p.service.model)));
  const nn::ModelIdentity leave =
    p.leave_inference.empty() ? nn::ModelIdentity{} : leave_identity(p);
  PolicyBot::check_trained_on(p, trained, p.leave_inference.empty() ? nullptr : &leave);
  PolicyBot::Backend b;
  b.blank_prune_points = trained.blank_prune_points;
  if (!p.leave_inference.empty()) {
    belief::AgentLeaveModel model =
      belief::agent_leave_model(p.leave_inference, HastyEquity::instance().leave_values());
    b.predict = std::move(model.predict);
    b.table = std::move(model.table);
  }
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
