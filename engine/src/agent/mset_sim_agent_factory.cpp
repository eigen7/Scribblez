// Command-line construction of MsetSimAgent, kept separate from the agent's
// selection logic (mset_sim_agent.cpp) for the same reason as
// neural_sim_agent_factory.cpp: this is the only MsetSimAgent translation unit
// that references the concrete TensorRT-backed move set service, so the core
// agent TU -- and the agent's unit tests, which inject a stub through the other
// constructor -- carry no CUDA/TensorRT dependency.

#include "agent/mset_sim_agent.h"
#include "agent/neural_service_options.h"
#include "endgame/endgame_solver.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// Parsed `--type=mset-sim` option values, with their defaults. A single
// options_description is built over these fields (make_options_description)
// and reused for both parsing (from_spec) and help rendering (options_help),
// so the two can never drift.
struct MsetSimOptions {
  NeuralServiceOptions service;
  int shortlist = 0;
  int sim_top_k = 10;
  std::string rank_objective = "winprob";
  std::string sim_objective = "winrate";
  int rollouts = 400;
  int sim_threads = 1;
  uint64_t seed = 0;
  EndgameSolver::Params endgame;
};

po::options_description make_options_description(MsetSimOptions& o) {
  po::options_description desc("Move-set-evaluation agent (--type=mset-sim) options");
  // One GPU call scores a whole turn's candidate set, so the shared per-call
  // ceiling is sized as the move set spec sizes it (model_specs.h documents
  // why generously): a set past the ceiling costs another board pass.
  o.service.batch_size = nn::MoveSetEvaluationSpec::kDefaultMaxRows;
  o.service.add_options(desc);
  desc.add_options()  //
    ("shortlist", po::value<int>(&o.shortlist)->default_value(o.shortlist),
     "static-equity shortlist the model scores; 0 = every legal move, the default -- "
     "the whole point of this model is that pre-filtering is unnecessary")  //
    ("sim-top-k", po::value<int>(&o.sim_top_k)->default_value(o.sim_top_k),
     "candidates simmed per turn, best by model rank")  //
    ("rank-objective", po::value<std::string>(&o.rank_objective)->default_value(o.rank_objective),
     "model head that ranks candidates: 'winprob' or 'scorediff'")  //
    ("sim-objective", po::value<std::string>(&o.sim_objective)->default_value(o.sim_objective),
     "what the rollouts are scored on: 'winrate' or 'spread'")  //
    ("rollouts", po::value<int>(&o.rollouts)->default_value(o.rollouts),
     "rollouts per simmed candidate; every candidate shares the same rollout seeds, so "
     "rack and draw luck cancels when they are compared")  //
    ("sim-threads", po::value<int>(&o.sim_threads)->default_value(o.sim_threads),
     "threads within one turn's simulation; leave at 1 when the game loop is "
     "already running games in parallel")  //
    ("seed,s", po::value<uint64_t>(&o.seed),
     "PRNG seed for the rollouts (default: derived from SeedProducer)");
  o.endgame.add_options(desc, "endgame-");
  return desc;
}

}  // namespace

MsetSimAgent::MsetSimAgent(const Params& params,
                           const nn::NeuralNetParams<nn::MoveSetEvaluationSpec>& net_params)
    : MsetSimAgent(params, nn::make_loaded_service(net_params)) {}

std::unique_ptr<MsetSimAgent> MsetSimAgent::from_spec(const std::vector<std::string>& tokens,
                                                      int thread_id, const std::string& name) {
  MsetSimOptions opts;
  po::options_description desc = make_options_description(opts);

  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=mset-sim options: ") + e.what());
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
  params.sim.rollouts = opts.rollouts;
  params.sim.threads = opts.sim_threads;
  params.seed = have_seed ? opts.seed : SeedProducer::instance().next();
  params.endgame = opts.endgame;
  // Fail on a bad scalar option now, before net_params() and the
  // constructor spend seconds loading the model and building the TensorRT
  // engine.
  validate(params);

  // Raising the per-pass ceiling to the shortlist just lets the whole shortlist
  // be scored in one pass; the service chunks to the ceiling either way.
  // shortlist == 0 (all moves) is chunked to batch_size.
  return std::make_unique<MsetSimAgent>(
    params, opts.service.net_params<nn::MoveSetEvaluationSpec>(opts.shortlist));
}

std::string MsetSimAgent::options_help() {
  MsetSimOptions opts;
  std::ostringstream os;
  os << "  The move-set-evaluation agent: the move set evaluation model scores\n"
        "  every legal move of the turn in a single pass, the model's top\n"
        "  --sim-top-k candidates are rolled out under common random numbers,\n"
        "  and the rollouts' favourite is played. Once the bag empties the turn\n"
        "  goes to the exact endgame solver.\n"
     << make_options_description(opts);
  return os.str();
}

}  // namespace scribblez
