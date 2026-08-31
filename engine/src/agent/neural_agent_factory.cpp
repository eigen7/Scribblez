// Command-line construction of NeuralAgent, kept separate from the agent's
// selection logic (neural_agent.cpp). from_spec resolves the run-shared model
// through nn::ServiceCache and hands the agent a borrowed service, so the core
// agent TU -- and the agent's unit tests, which inject a stub through the other
// constructor -- carry no CUDA/TensorRT dependency (the cache's construction of
// the concrete nn::TrtEvalService lives in service_cache.cpp). Parsing the
// `--type=neural` option string additionally pulls in Boost.program_options,
// the process-wide Lexicon, and the shared NeuralServiceOptions block
// (neural_service_options.cpp, which resolves --precision through
// nn::parse_precision), all kept out of the core agent TU.

#include "agent/neural_agent.h"
#include "agent/neural_service_options.h"
#include "endgame/endgame_solver.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/service_cache.h"
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

// Parsed `--type=neural` option values, with their defaults. A single
// options_description is built over these fields (make_options_description) and
// reused for both parsing (from_spec) and help rendering (options_help), so the
// two can never drift.
struct NeuralOptions {
  NeuralServiceOptions service;
  int top_k = 10;
  std::string objective = "winprob";
  double temperature = 0.0;
  uint64_t seed = 0;
  EndgameSolver::Params endgame;
};

// Boost.program_options renders defaults set via default_value() as "(=...)" in
// the help text, so the option descriptions deliberately omit them.
po::options_description make_options_description(NeuralOptions& opts) {
  po::options_description desc("Neural agent (--type=neural) options");
  opts.service.add_options(desc);
  desc.add_options()(
    "top-k,k", po::value<int>(&opts.top_k)->default_value(opts.top_k),
    "candidate plays to evaluate: K>0 = top-K by static equity; 0 = ALL legal plays (most "
    "diverse, but slowest -- every play hits the GPU)")(
    "objective,o", po::value<std::string>(&opts.objective)->default_value(opts.objective),
    "selection head: winprob = highest P(win)+0.5*P(draw); scorediff = highest expected final "
    "score differential")("temperature,t",
                          po::value<double>(&opts.temperature)->default_value(opts.temperature),
                          "softmax move-sampling temperature (0 = greedy argmax)")(
    "seed,s", po::value<uint64_t>(&opts.seed), "sampling PRNG seed (default: SeedProducer)");
  opts.endgame.add_options(desc, "endgame-");
  return desc;
}

}  // namespace

std::unique_ptr<NeuralAgent> NeuralAgent::from_spec(const std::vector<std::string>& tokens,
                                                    int thread_id, const std::string& name,
                                                    nn::ServiceCache& services) {
  NeuralOptions opts;
  po::options_description desc = make_options_description(opts);

  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=neural options: {}", e.what());
  }

  if (opts.top_k < 0) throw util::CleanException("--top-k must be >= 0 (0 = all legal plays)");

  // Sizing the engine batch to at least top_k just lets the whole top-K set be
  // scored in a single chunk; the agent chunks to the engine batch either way.
  // top_k == 0 (all plays) is chunked to batch_size.
  const NeuralAgent::NetParams net_params =
    opts.service.net_params<nn::PositionEvaluationSpec>(opts.top_k);

  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const uint64_t resolved_seed = have_seed ? opts.seed : SeedProducer::instance().next();
  // One loaded model per (net_params) shared across this run's threads; the
  // agent borrows it. net_params.max_rows bounds one evaluate() call, matching
  // the batch the shared engine was built for.
  nn::PositionEvalService* service = services.position_service(net_params);
  return std::make_unique<NeuralAgent>(
    NeuralAgent::Params{.thread_id = thread_id,
                        .name = name,
                        .dict = &Lexicon::instance().dict(),
                        .top_k = opts.top_k,
                        .objective = parse_eval_objective(opts.objective, "--objective"),
                        .temperature = opts.temperature,
                        .seed = resolved_seed,
                        .endgame = opts.endgame},
    service, net_params.max_rows);
}

std::string NeuralAgent::options_help() {
  NeuralOptions opts;
  std::ostringstream os;
  os << "  HastyBot move-gen + position evaluation model: applies candidate plays\n"
        "  and plays the one whose post-move state the model's objective head\n"
        "  rates highest. Once the bag empties the model steps aside and an\n"
        "  iterative-deepening negamax solver plays the endgame exactly.\n"
     << make_options_description(opts);
  return os.str();
}

}  // namespace scribblez
