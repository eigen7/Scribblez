// LeafBot's command-line construction and backend: the only part that reaches
// CUDA and TensorRT (the leaf network and the leave inference model), so
// leaf_bot.cpp's unit tests can inject the CPU reference instead.

#include "agent/agent_options.h"
#include "agent/leaf_bot.h"
#include "belief/served_leave_model.h"
#include "gpu/rollout_backend.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <memory>
#include <string>
#include <utility>

namespace scribblez {

namespace {

namespace po = boost::program_options;

po::options_description leafbot_options(LeafBot::Params& p) {
  po::options_description desc("leafbot options");
  desc.add_options()  //
    ("leaf-model", po::value<std::string>(&p.leaf_model),
     "the leaf network (a deep_bot_leaf_eval ONNX export); 'synthetic' for the CPU reference's "
     "logistic leaf (tests)")  //
    ("leaf-precision", po::value<std::string>(&p.leaf_precision)->default_value(p.leaf_precision),
     "TensorRT precision for the leaf network: fp32, fp16 or bf16")  //
    ("leaf-rows", po::value<int>(&p.leaf_rows)->default_value(p.leaf_rows),
     "rows per leaf network call; the games' leaves are gathered into calls this large")  //
    ("leave-inference", po::value<std::string>(&p.leave_inference),
     "the leave inference model (ONNX) the opponent's leaves are drawn from")  //
    ("candidates", po::value<int>(&p.candidates)->default_value(p.candidates),
     "HastyBot's best moves by static equity that the leaf values")  //
    ("opp-leaves", po::value<int>(&p.opp_leaves)->default_value(p.opp_leaves),
     "the opponent's leaves each candidate is valued over")  //
    ("report-values", po::bool_switch(&p.report_values),
     "also value and report the candidates that keep the bag where another would empty it "
     "(an analysis; play is HastyBot's move there either way)")  //
    ("seed", po::value<uint64_t>(&p.seed), "the leaves' quantile jitter (default: SeedProducer)");
  return desc;
}

void validate(const LeafBot::Params& p) {
  if (p.leaf_model.empty()) throw util::CleanException("leafbot: --leaf-model is required");
  if (p.leave_inference.empty())
    throw util::CleanException("leafbot: --leave-inference is required");
  if (p.candidates < 1) throw util::CleanException("leafbot: --candidates must be positive");
  if (p.opp_leaves < 1) throw util::CleanException("leafbot: --opp-leaves must be positive");
}

LeafBot::Backend make_backend(const LeafBot::Params& p) {
  RolloutBackend rollouts;
  rollouts.backend = p.leaf_model == RolloutBackend::kSyntheticLeaf ? "cpu" : "gpu";
  rollouts.leaf_model = p.leaf_model;
  rollouts.leaf_precision = p.leaf_precision;
  rollouts.leaf_rows = p.leaf_rows;
  rollouts.validate();
  belief::AgentLeaveModel leave =
    belief::agent_leave_model(p.leave_inference, HastyEquity::instance().leave_values());
  return {.values = rollouts.shared(load_dictionary_or_throw()),
          .predict = std::move(leave.predict),
          .table = std::move(leave.table)};
}

}  // namespace

LeafBot::LeafBot(const Params& params) : LeafBot(params, make_backend(params)) {}

LeafBot::Params LeafBot::parse_params(const std::vector<std::string>& tokens, int thread_id,
                                      const std::string& name,
                                      boost::program_options::options_description& extra,
                                      const char* type_label) {
  Params p{.thread_id = thread_id, .name = name};
  po::options_description desc = leafbot_options(p);
  desc.add(extra);
  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type={} options: {}", type_label, e.what());
  }
  if (!have_seed) p.seed = SeedProducer::instance().next();
  validate(p);
  // The equity tables must not load concurrently with queries (hasty_equity.h).
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  return p;
}

std::unique_ptr<LeafBot> LeafBot::from_spec(const std::vector<std::string>& tokens, int thread_id,
                                            const std::string& name) {
  po::options_description no_extra;
  return std::make_unique<LeafBot>(parse_params(tokens, thread_id, name, no_extra, kType));
}

std::string LeafBot::options_help() {
  Params defaults;  // binding targets; only the defaults are read
  return agent_options_help(
    "  LeafBot: the leaf network as a player, no search. Values HastyBot's best moves by\n"
    "  static equity at the position after each, over the opponent's leaves drawn from the\n"
    "  leave inference model's belief, and plays the best. Plays HastyBot's move where a\n"
    "  candidate would empty the bag. Hidden leaves only; leafbot-endgame solves the\n"
    "  endgame.\n",
    leafbot_options(defaults));
}

}  // namespace scribblez
