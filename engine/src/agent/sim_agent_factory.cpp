// Command-line construction of SimAgent, kept separate from the agent's
// selection logic (sim_agent.cpp) for the same reason as the neural agents'
// factories: --leaf-model stands up the concrete TensorRT-backed leaf
// service, so this is the only SimAgent translation unit with a CUDA/TensorRT
// dependency, and the agent's unit tests (which inject a stub, or run
// terminal) never link it.

#include "agent/agent_options.h"
#include "agent/sim_agent.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "util/exception.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <memory>
#include <string>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// The `--player "--type=sim ..."` options, bound to one struct so the parsed
// set and the documented set cannot drift.
struct SimOptions {
  int top_k = 10;
  int rollouts = 400;
  int sim_threads = 1;
  int sim_horizon = 0;
  std::string leaf_model;
  std::string objective = "winrate";
  uint64_t seed = 0;
  EndgameSolver::Params endgame;
};

po::options_description make_options_description(SimOptions& o) {
  po::options_description desc("sim options");
  desc.add_options()                                                 //
    ("top-k", po::value<int>(&o.top_k)->default_value(o.top_k),      //
     "candidates simmed per turn, taken by HastyBot static equity")  //
    ("rollouts", po::value<int>(&o.rollouts)->default_value(o.rollouts),
     "rollouts per candidate; every candidate shares the same rollout seeds, so "
     "rack and draw luck cancels when they are compared")  //
    ("sim-threads", po::value<int>(&o.sim_threads)->default_value(o.sim_threads),
     "threads within one turn's simulation; leave at 1 when the game loop is "
     "already running games in parallel")  //
    ("sim-horizon", po::value<int>(&o.sim_horizon)->default_value(o.sim_horizon),
     "value truncation: rollouts stop after this many plies and --leaf-model scores the "
     "horizon; 0 rolls out to a natural game end")  //
    ("leaf-model", po::value<std::string>(&o.leaf_model),
     "position evaluation model (.onnx) scoring rollout horizons; required with, and only "
     "with, --sim-horizon")  //
    ("objective", po::value<std::string>(&o.objective)->default_value(o.objective),
     "what the rollouts are scored on: 'winrate' or 'spread'")  //
    ("seed", po::value<uint64_t>(&o.seed),
     "PRNG seed for the rollouts (default: derived from SeedProducer)");
  o.endgame.add_options(desc, "endgame-");
  return desc;
}

}  // namespace

std::unique_ptr<SimAgent> SimAgent::from_spec(const std::vector<std::string>& tokens, int thread_id,
                                              const std::string& name) {
  SimOptions opts;
  po::options_description desc = make_options_description(opts);

  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=sim options: {}", e.what());
  }
  if ((opts.sim_horizon > 0) != !opts.leaf_model.empty()) {
    throw util::CleanException("sim agent: --sim-horizon and --leaf-model come together");
  }

  HastyEquity::ensure_initialized(Lexicon::instance().name());

  Params params;
  params.thread_id = thread_id;
  params.name = name;
  params.dict = &Lexicon::instance().dict();
  params.top_k = opts.top_k;
  params.sim.rollouts = opts.rollouts;
  params.sim.threads = opts.sim_threads;
  params.sim_horizon = opts.sim_horizon;
  params.objective = parse_sim_objective(opts.objective, "--objective");
  params.seed = have_seed ? opts.seed : SeedProducer::instance().next();
  params.endgame = opts.endgame;

  std::unique_ptr<nn::PositionEvalService> leaf;
  if (!opts.leaf_model.empty()) {
    nn::NeuralNetParams<nn::PositionEvaluationSpec> leaf_params;
    leaf_params.onnx_path = opts.leaf_model;
    // FP32 unconditionally: current checkpoints overflow FP16 inside the
    // engine on legitimate extreme-advantage states rollouts routinely
    // reach, yielding NaN in every head before any decode this code
    // controls -- there is no host-side fix. SimRunner hard-errors on a
    // NaN readout either way.
    leaf_params.precision = nn::Precision::kFP32;
    leaf = nn::make_loaded_service(leaf_params);
  }
  return std::make_unique<SimAgent>(params, std::move(leaf));
}

std::string SimAgent::options_help() {
  SimOptions defaults;  // scratch binding targets; only the defaults are read
  return agent_options_help(
    "  Monte-Carlo simming bot: it keeps the best --top-k moves by HastyBot static\n"
    "  equity, plays --rollouts games out from each under common random numbers,\n"
    "  and plays whichever scored best. Once the bag empties the turn goes to the\n"
    "  exact endgame solver.\n",
    make_options_description(defaults));
}

}  // namespace scribblez
