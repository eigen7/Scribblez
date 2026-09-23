// NeuralAgent's command-line construction, kept out of neural_agent.cpp so the
// unit tests that compile the agent with a stub service need none of the
// option-parsing, Lexicon, or model-loading dependencies.

#include "agent/neural_agent.h"
#include "agent/neural_service_options.h"
#include "endgame/endgame_solver.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/eval_service.h"
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

// Parsed `--type=neural` options with their defaults. from_spec and
// options_help build the same options_description over them, so the parsed
// and documented options cannot drift.
struct NeuralOptions {
  NeuralServiceOptions service;
  int top_k = 10;
  std::string objective = "winprob";
  double temperature = 0.0;
  uint64_t seed = 0;
  EndgameSolver::Params endgame;
};

// Help strings omit defaults: program_options renders them as "(=...)".
po::options_description make_options_description(NeuralOptions& opts) {
  po::options_description desc("Neural agent (--type=neural) options");
  opts.service.add_options(desc);
  desc.add_options()(
    "top-k,k", po::value<int>(&opts.top_k)->default_value(opts.top_k),
    "candidate moves (plays and exchanges) to evaluate: K>0 = top-K by static equity; 0 = ALL "
    "legal moves (most diverse, but slowest -- every move hits the GPU)")(
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
                                                    int thread_id, const std::string& name) {
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

  if (opts.top_k < 0) throw util::CleanException("--top-k must be >= 0 (0 = all legal moves)");

  // Raise the engine batch to top_k so the top-K set is scored in one chunk.
  // With top_k == 0 (all moves) the evaluator chunks to batch_size.
  const NeuralAgent::NetParams net_params =
    opts.service.net_params<nn::PositionEvaluationSpec>(opts.top_k);

  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const uint64_t resolved_seed = have_seed ? opts.seed : SeedProducer::instance().next();
  // One loaded model per distinct net_params, shared by the run's threads.
  std::shared_ptr<nn::PositionEvalService> service = nn::PositionEvalService::create(net_params);
  return std::make_unique<NeuralAgent>(
    NeuralAgent::Params{.thread_id = thread_id,
                        .name = name,
                        .dict = &Lexicon::instance().dict(),
                        .top_k = opts.top_k,
                        .objective = parse_eval_objective(opts.objective, "--objective"),
                        .temperature = opts.temperature,
                        .seed = resolved_seed,
                        .endgame = opts.endgame},
    std::move(service), net_params.max_rows);
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
