// Command-line construction of NeuralAgent, kept separate from the agent's
// selection logic (neural_agent.cpp). Parsing the `--type=neural` option string
// pulls in Boost.program_options, the process-wide Lexicon, and the TensorRT
// precision parser; isolating it here means the core agent translation unit --
// and the agent's unit tests -- carry none of those dependencies.

#include "scribblez/neural_agent.h"

#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/nn/neural_net.h"
#include "scribblez/nn/trt_util.h"
#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace scribblez {

std::unique_ptr<NeuralAgent> NeuralAgent::from_spec(const std::vector<std::string>& tokens,
                                                    int thread_id, const std::string& name) {
  namespace po = boost::program_options;

  std::string model;
  int top_k = 10;
  int batch_size = 256;
  int cuda_device = 0;
  std::string precision = "FP16";
  std::string objective = "scorediff";
  double temperature = 0.0;
  uint64_t seed = 0;
  bool have_seed = false;

  po::options_description desc("neural options");
  desc.add_options()                                                                          //
    ("model", po::value<std::string>(&model), "path to the exported ONNX model (required)")    //
    ("top-k", po::value<int>(&top_k)->default_value(top_k),
     "candidate moves to evaluate: K>0 = top-K by static equity, 0 = all plays")              //
    ("batch-size", po::value<int>(&batch_size)->default_value(batch_size), "max GPU batch")    //
    ("cuda-device", po::value<int>(&cuda_device)->default_value(cuda_device), "GPU index")      //
    ("precision", po::value<std::string>(&precision)->default_value(precision), "FP16|FP32")    //
    ("temperature", po::value<double>(&temperature)->default_value(temperature),
     "softmax sampling temperature (0 = greedy argmax)")                                        //
    ("seed", po::value<uint64_t>(&seed), "sampling PRNG seed (default: SeedProducer)")          //
    ("objective", po::value<std::string>(&objective)->default_value(objective),
     "selection objective: scorediff|winprob");

  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=neural options: ") + e.what());
  }

  if (model.empty()) throw std::runtime_error("--type=neural requires --model=<path.onnx>");
  if (top_k < 0) throw std::runtime_error("--top-k must be >= 0 (0 = all legal plays)");

  Objective obj;
  if (objective == "scorediff") {
    obj = Objective::kScoreDiff;
  } else if (objective == "winprob") {
    obj = Objective::kWinProb;
  } else {
    throw std::runtime_error("--objective must be 'scorediff' or 'winprob' (got '" + objective +
                             "')");
  }

  nn::NeuralNetParams params;
  params.cuda_device_id = cuda_device;
  // top_k > 0 is scored in one evaluate() call, so the engine must admit a batch
  // at least that large; top_k == 0 (all plays) is chunked to batch_size.
  params.max_batch_size = std::max({batch_size, top_k, 1});
  params.precision = nn::parse_precision(precision);

  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const uint64_t resolved_seed = have_seed ? seed : SeedProducer::instance().next();
  return std::make_unique<NeuralAgent>(thread_id, name, model, top_k, obj, params, temperature,
                                       resolved_seed);
}

std::string NeuralAgent::options_help() {
  return "  HastyBot move-gen + M_post value model: applies candidate plays and\n"
         "  plays the one whose post-move state the model's objective head rates\n"
         "  highest.\n"
         "  Options:\n"
         "    --model=<path.onnx>      exported model (required)\n"
         "    --top-k=K                candidate plays to evaluate: K>0 = top-K by\n"
         "                             static equity; 0 = ALL legal plays (most\n"
         "                             diverse, but slowest -- every play hits the GPU)\n"
         "    --objective=scorediff|winprob  selection head (default scorediff:\n"
         "                             highest expected final score differential)\n"
         "    --batch-size=N           max GPU batch (default 256)\n"
         "    --cuda-device=D          GPU index (default 0)\n"
         "    --precision=FP16|FP32    TensorRT precision (default FP16)\n"
         "    --temperature=T          softmax move sampling (default 0 = greedy)\n"
         "    --seed=N                 sampling PRNG seed (default: SeedProducer)\n";
}

}  // namespace scribblez
