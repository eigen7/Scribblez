// Command-line construction of NeuralAgent, kept separate from the agent's
// selection logic (neural_agent.cpp). This is also the only translation unit
// that references the concrete nn::NNEvaluationService: the production
// constructor and make_service() live here, so the core agent TU -- and the
// agent's unit tests, which inject a stub through the other constructor --
// carry no CUDA/TensorRT dependency. Parsing the `--type=neural` option string
// additionally pulls in Boost.program_options, the process-wide Lexicon, and
// the TensorRT precision parser, all confined to this file.

#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/neural_agent.h"
#include "scribblez/nn/neural_net.h"
#include "scribblez/nn/nn_evaluation_service.h"
#include "scribblez/nn/trt_util.h"
#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// Parsed `--type=neural` option values, with their defaults. A single
// options_description is built over these fields (make_options_description) and
// reused for both parsing (from_spec) and help rendering (options_help), so the
// two can never drift.
struct NeuralOptions {
  std::string model;
  int top_k = 10;
  int batch_size = 256;
  int cuda_device = 0;
  std::string precision = "FP16";
  std::string objective = "winprob";
  double temperature = 0.0;
  uint64_t seed = 0;
};

// Boost.program_options renders defaults set via default_value() as "(=...)" in
// the help text, so the option descriptions deliberately omit them.
po::options_description make_options_description(NeuralOptions& opts) {
  po::options_description desc("Neural agent (--type=neural) options");
  desc.add_options()("model,m", po::value<std::string>(&opts.model),
                     "exported ONNX model (required)")(
    "top-k,k", po::value<int>(&opts.top_k)->default_value(opts.top_k),
    "candidate plays to evaluate: K>0 = top-K by static equity; 0 = ALL legal plays (most "
    "diverse, but slowest -- every play hits the GPU)")(
    "objective,o", po::value<std::string>(&opts.objective)->default_value(opts.objective),
    "selection head: winprob = highest P(win)+0.5*P(draw); scorediff = highest expected final "
    "score differential")("batch-size,b",
                          po::value<int>(&opts.batch_size)->default_value(opts.batch_size),
                          "max GPU batch")(
    "cuda-device,d", po::value<int>(&opts.cuda_device)->default_value(opts.cuda_device),
    "GPU index")("precision,p",
                 po::value<std::string>(&opts.precision)->default_value(opts.precision),
                 "TensorRT precision: FP16|FP32")(
    "temperature,t", po::value<double>(&opts.temperature)->default_value(opts.temperature),
    "softmax move-sampling temperature (0 = greedy argmax)")(
    "seed,s", po::value<uint64_t>(&opts.seed), "sampling PRNG seed (default: SeedProducer)");
  return desc;
}

NeuralAgent::Objective parse_objective(const std::string& objective) {
  if (objective == "scorediff") return NeuralAgent::Objective::kScoreDiff;
  if (objective == "winprob") return NeuralAgent::Objective::kWinProb;
  throw std::runtime_error("--objective must be 'scorediff' or 'winprob' (got '" + objective +
                           "')");
}

}  // namespace

NeuralAgent::NeuralAgent(const Params& params, const nn::NeuralNetParams& net_params)
    : Agent(params.thread_id, params.name),
      top_k_(params.top_k),
      objective_(params.objective),
      temperature_(params.temperature),
      max_batch_(net_params.max_batch_size),
      service_(make_service(net_params)),
      rng_(params.seed) {
  init();
}

std::unique_ptr<nn::EvalService> NeuralAgent::make_service(const nn::NeuralNetParams& net_params) {
  auto svc = std::make_unique<nn::NNEvaluationService>(net_params);
  svc->load();
  return svc;
}

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
    throw std::runtime_error(std::string("bad --type=neural options: ") + e.what());
  }

  if (opts.model.empty()) throw std::runtime_error("--type=neural requires --model=<path.onnx>");
  if (opts.top_k < 0) throw std::runtime_error("--top-k must be >= 0 (0 = all legal plays)");

  const Objective obj = parse_objective(opts.objective);

  nn::NeuralNetParams net_params;
  net_params.onnx_path = opts.model;
  net_params.cuda_device_id = opts.cuda_device;
  // The agent always chunks candidates to max_batch_size, so any value is
  // correct; sizing the engine batch to at least top_k just lets the whole
  // top-K set be scored in a single chunk. top_k == 0 (all plays) is chunked to
  // batch_size.
  net_params.max_batch_size = std::max({opts.batch_size, opts.top_k, 1});
  net_params.precision = nn::parse_precision(opts.precision);

  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const uint64_t resolved_seed = have_seed ? opts.seed : SeedProducer::instance().next();
  return std::make_unique<NeuralAgent>(NeuralAgent::Params{.thread_id = thread_id,
                                                           .name = name,
                                                           .top_k = opts.top_k,
                                                           .objective = obj,
                                                           .temperature = opts.temperature,
                                                           .seed = resolved_seed},
                                       net_params);
}

std::string NeuralAgent::options_help() {
  NeuralOptions opts;
  std::ostringstream os;
  os << "  HastyBot move-gen + M_post value model: applies candidate plays and\n"
        "  plays the one whose post-move state the model's objective head rates\n"
        "  highest.\n"
     << make_options_description(opts);
  return os.str();
}

}  // namespace scribblez
