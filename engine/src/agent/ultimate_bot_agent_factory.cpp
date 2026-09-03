// Command-line construction of UltimateBotAgent, kept separate from the agent's
// decision logic (ultimate_bot_agent.cpp) for the same reason as
// mset_sim_agent_factory.cpp: this is the only UltimateBotAgent translation
// unit that references the concrete TensorRT-backed move proposal nets and
// session, so the core agent TU -- and the agent's unit tests, which inject a
// stub through the other constructor -- carry no CUDA/TensorRT dependency.

#include "agent/agent_options.h"
#include "agent/move_proposal_nets.h"
#include "agent/move_proposal_session.h"
#include "agent/ultimate_bot_agent.h"
#include "endgame/endgame_solver.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "util/exception.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// Parsed `--type=ultimatebot` option values, with their defaults. A single
// options_description is built over these fields (make_options_description)
// and reused for both parsing (from_spec) and help rendering (options_help),
// so the two can never drift. The model options are this agent's own rather
// than NeuralServiceOptions': it serves a PAIR of graphs, not one model.
struct UltimateBotOptions {
  std::string cache_model;
  std::string step_model;
  int batch_size = nn::MoveProposalCacheSpec::kDefaultMaxRows;
  int cuda_device = 0;
  // FP32: the evidence path's serving precision (docs/roadmap.md item 3); the
  // fusion graph is not yet FP16-gated.
  std::string precision = "FP32";
  int max_sims = 10;
  float gain_threshold = 0.0f;
  int rollouts = 400;
  int sim_threads = 1;
  int sim_horizon = 0;
  std::string leaf_model;
  uint64_t seed = 0;
  EndgameSolver::Params endgame;
};

// Boost.program_options renders defaults set via default_value() as "(=...)" in
// the help text, so the option descriptions deliberately omit them.
po::options_description make_options_description(UltimateBotOptions& o) {
  po::options_description desc("UltimateBot (--type=ultimatebot) options");
  desc.add_options()  //
    ("cache-model", po::value<std::string>(&o.cache_model),
     "the move proposal model's exported move_proposal_cache graph (.onnx, required)")  //
    ("step-model", po::value<std::string>(&o.step_model),
     "its move_proposal_step graph (.onnx, required); exported with --cache-model from one "
     "checkpoint")  //
    ("batch-size,b", po::value<int>(&o.batch_size)->default_value(o.batch_size),
     "max candidate rows the cache graph scores in one GPU call")  //
    ("cuda-device,d", po::value<int>(&o.cuda_device)->default_value(o.cuda_device),
     "GPU index")  //
    ("precision,p", po::value<std::string>(&o.precision)->default_value(o.precision),
     "TensorRT precision: FP32|BF16|FP16")  //
    ("max-sims", po::value<int>(&o.max_sims)->default_value(o.max_sims),
     "sims per turn at most, the anchor included; 1 plays the anchor unsimmed")  //
    ("gain-threshold", po::value<float>(&o.gain_threshold)->default_value(o.gain_threshold),
     "stop simming once no unsimmed candidate's predicted gain (win probability) reaches "
     "this; 0 never stops early")  //
    ("rollouts", po::value<int>(&o.rollouts)->default_value(o.rollouts),
     "rollouts per simmed candidate; every sim of a turn shares the same rollout seeds, so "
     "rack and draw luck cancels when they are compared")  //
    ("sim-threads", po::value<int>(&o.sim_threads)->default_value(o.sim_threads),
     "threads within one sim; leave at 1 when the game loop is already running games in "
     "parallel")  //
    ("sim-horizon", po::value<int>(&o.sim_horizon)->default_value(o.sim_horizon),
     "value truncation: rollouts stop after this many plies and --leaf-model scores the "
     "horizon; 0 rolls out to a natural game end")  //
    ("leaf-model", po::value<std::string>(&o.leaf_model),
     "position evaluation model (.onnx) scoring rollout horizons; required with, and only "
     "with, --sim-horizon")  //
    ("seed,s", po::value<uint64_t>(&o.seed),
     "PRNG seed for the rollouts (default: derived from SeedProducer)");
  o.endgame.add_options(desc, "endgame-");
  return desc;
}

// The loaded, shared engine pair the options name. Both paths are required.
std::shared_ptr<agent::MoveProposalNets> load_nets(const UltimateBotOptions& o) {
  if (o.cache_model.empty() || o.step_model.empty()) {
    throw util::CleanException(
      "--type=ultimatebot requires both --cache-model=<cache.onnx> and --step-model=<step.onnx>");
  }
  agent::MoveProposalNets::Params np;
  np.cache_onnx_path = o.cache_model;
  np.step_onnx_path = o.step_model;
  np.cuda_device_id = o.cuda_device;
  np.max_rows = std::max(o.batch_size, 1);
  np.precision = nn::parse_precision(o.precision);
  return agent::MoveProposalNets::create(np);
}

}  // namespace

std::unique_ptr<UltimateBotAgent> UltimateBotAgent::from_spec(
  const std::vector<std::string>& tokens, int thread_id, const std::string& name) {
  UltimateBotOptions opts;
  po::options_description desc = make_options_description(opts);

  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=ultimatebot options: {}", e.what());
  }

  HastyEquity::ensure_initialized(Lexicon::instance().name());

  Params params;
  params.thread_id = thread_id;
  params.name = name;
  params.dict = &Lexicon::instance().dict();
  params.max_sims = opts.max_sims;
  params.gain_threshold = opts.gain_threshold;
  params.sim.rollouts = opts.rollouts;
  params.sim.threads = opts.sim_threads;
  params.sim_horizon = opts.sim_horizon;
  params.seed = have_seed ? opts.seed : SeedProducer::instance().next();
  params.endgame = opts.endgame;
  // Fail on a bad scalar option now, before the engine builds.
  validate(params);
  SimRunner::validate_horizon("ultimatebot", opts.sim_horizon, !opts.leaf_model.empty());

  // The leaf shares the nets' device; the run's agents share the leaf.
  std::shared_ptr<nn::PositionEvalService> leaf =
    nn::load_leaf_position_service(opts.leaf_model, opts.cuda_device);
  return std::make_unique<UltimateBotAgent>(
    params, std::make_unique<agent::MoveProposalSession>(load_nets(opts)), std::move(leaf));
}

std::string UltimateBotAgent::options_help() {
  UltimateBotOptions defaults;  // scratch binding targets; only the defaults are read
  return agent_options_help(
    "  UltimateBot: the move proposal model scores every legal move of the turn in\n"
    "  one pass, the highest-scoring move is simmed first, and then -- conditioned\n"
    "  on every sim so far -- the model re-scores the set and its proves-best pick\n"
    "  is simmed next, until --max-sims is spent or no candidate's predicted gain\n"
    "  clears --gain-threshold; the rollouts' favourite plays. Once the bag empties\n"
    "  the turn goes to the exact endgame solver.\n",
    make_options_description(defaults));
}

}  // namespace scribblez
