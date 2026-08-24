// Command-line construction of NeuralSimAgent, kept separate from the agent's
// selection logic (neural_sim_agent.cpp) for the same reason as
// neural_agent_factory.cpp: this is the only NeuralSimAgent translation unit
// that references the concrete nn::TrtEvalService, so the core agent TU
// -- and the agent's unit tests, which inject a stub through the other
// constructor -- carry no CUDA/TensorRT dependency.

#include "agent/neural_service_options.h"
#include "agent/neural_sim_agent.h"
#include "endgame/endgame_solver.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "util/exception.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// Parsed `--type=neural-sim` option values, with their defaults. A single
// options_description is built over these fields (make_options_description)
// and reused for both parsing (from_spec) and help rendering (options_help),
// so the two can never drift.
struct NeuralSimOptions {
  NeuralServiceOptions service;
  int shortlist = 50;
  int sim_top_k = 10;
  std::string rank_objective = "winprob";
  std::string sim_objective = "winrate";
  double drop_best_prob = 0.0;
  int rollouts = 400;
  int sim_threads = 1;
  int sim_horizon = 0;
  uint64_t seed = 0;
  EndgameSolver::Params endgame;
};

po::options_description make_options_description(NeuralSimOptions& o) {
  po::options_description desc("Neural-sim agent (--type=neural-sim) options");
  o.service.add_options(desc);
  desc.add_options()  //
    ("shortlist", po::value<int>(&o.shortlist)->default_value(o.shortlist),
     "static-equity shortlist the model evaluates exhaustively; 0 = every legal move")  //
    ("sim-top-k", po::value<int>(&o.sim_top_k)->default_value(o.sim_top_k),
     "candidates simmed per turn, best by model rank")  //
    ("rank-objective", po::value<std::string>(&o.rank_objective)->default_value(o.rank_objective),
     "model head that ranks candidates: 'winprob' or 'scorediff'")  //
    ("sim-objective", po::value<std::string>(&o.sim_objective)->default_value(o.sim_objective),
     "what the rollouts are scored on: 'winrate' or 'spread'")  //
    ("drop-best-prob", po::value<double>(&o.drop_best_prob)->default_value(o.drop_best_prob),
     "per-turn probability of excluding the model's top-ranked candidate from the sim "
     "set -- the A4 sensitivity sweep's controlled recall miss")  //
    ("rollouts", po::value<int>(&o.rollouts)->default_value(o.rollouts),
     "rollouts per simmed candidate; every candidate shares the same rollout seeds, so "
     "rack and draw luck cancels when they are compared")  //
    ("sim-threads", po::value<int>(&o.sim_threads)->default_value(o.sim_threads),
     "threads within one turn's simulation; leave at 1 when the game loop is "
     "already running games in parallel")  //
    ("sim-horizon", po::value<int>(&o.sim_horizon)->default_value(o.sim_horizon),
     "value truncation: rollouts stop after this many plies and this agent's own model scores "
     "the horizon; 0 rolls out to a natural game end. The one net serves ranking and leaves "
     "alike, so pair this with --precision=FP32: FP16 overflows to NaN on the "
     "extreme-advantage states rollouts routinely reach (a hard error)")  //
    ("seed,s", po::value<uint64_t>(&o.seed),
     "PRNG seed for the rollouts and drop decisions (default: derived from SeedProducer)");
  o.endgame.add_options(desc, "endgame-");
  return desc;
}

}  // namespace

NeuralSimAgent::NeuralSimAgent(const Params& params, const NetParams& net_params)
    : NeuralSimAgent(params, nn::make_loaded_service(net_params), net_params.max_rows) {}

std::unique_ptr<NeuralSimAgent> NeuralSimAgent::from_spec(const std::vector<std::string>& tokens,
                                                          int thread_id, const std::string& name) {
  NeuralSimOptions opts;
  po::options_description desc = make_options_description(opts);

  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=neural-sim options: {}", e.what());
  }

  HastyEquity::ensure_initialized(Lexicon::instance().name());

  Params params;
  params.thread_id = thread_id;
  params.name = name;
  params.dict = &Lexicon::instance().dict();
  params.shortlist = opts.shortlist;
  params.sim_top_k = opts.sim_top_k;
  params.rank_objective = parse_eval_objective(opts.rank_objective, "--rank-objective");
  params.sim_objective = parse_sim_objective(opts.sim_objective, "--sim-objective");
  params.drop_best_prob = opts.drop_best_prob;
  params.sim.rollouts = opts.rollouts;
  params.sim.threads = opts.sim_threads;
  params.sim_horizon = opts.sim_horizon;
  params.seed = have_seed ? opts.seed : SeedProducer::instance().next();
  params.endgame = opts.endgame;
  // Fail on a bad scalar option now, before net_params() and the constructor
  // spend seconds loading the model and building the TensorRT engine.
  validate(params);

  // Sizing the engine batch to at least the shortlist just lets the whole
  // shortlist be scored in a single chunk; the agent chunks to the engine
  // batch either way. shortlist == 0 (all moves) is chunked to batch_size.
  const NeuralSimAgent::NetParams net_params =
    opts.service.net_params<nn::PositionEvaluationSpec>(opts.shortlist);
  return std::make_unique<NeuralSimAgent>(params, net_params);
}

std::string NeuralSimAgent::options_help() {
  NeuralSimOptions opts;
  std::ostringstream os;
  os << "  The position-evaluation-top-K agent: the position evaluation model\n"
        "  exactly evaluates a generous static-equity shortlist, the model's top\n"
        "  --sim-top-k candidates are rolled out under common random numbers, and\n"
        "  the rollouts' favourite is played. Once the bag empties the turn goes\n"
        "  to the exact endgame solver.\n"
     << make_options_description(opts);
  return os.str();
}

}  // namespace scribblez
