#include "scribblez/agent.h"

#include "scribblez/agent_options.h"
#include "scribblez/movegen.h"
#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <stdexcept>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// The command-line options for `--type=greedy`, binding `--seed` to `seed`. Both
// from_spec() (to parse) and options_help() (to document) build this, so the two
// share one source of truth for the option list.
po::options_description greedy_options(uint64_t& seed) {
  po::options_description desc;
  desc.add_options()                      //
    ("seed", po::value<uint64_t>(&seed),  //
     "PRNG seed for tie-breaking (default: derived from SeedProducer)");
  return desc;
}

}  // namespace

std::vector<Move> generate_legal_plays(const MoveRequest& req) {
  return MoveGenerator(req.board, req.dict).generate(req.my_rack);
}

GreedyAgent::GreedyAgent(int thread_id, const std::string& name)
    : Agent(thread_id, name), rng_(SeedProducer::instance().next()) {}

GreedyAgent::GreedyAgent(int thread_id, const std::string& name, uint64_t seed)
    : Agent(thread_id, name), rng_(seed) {}

Move GreedyAgent::make_move(const MoveRequest& req) {
  const std::vector<Move> plays = generate_legal_plays(req);
  if (!plays.empty()) {
    int best = plays.front().score();
    for (const auto& m : plays) best = std::max(best, static_cast<int>(m.score()));
    std::vector<const Move*> top;
    for (const auto& m : plays)
      if (m.score() == best) top.push_back(&m);
    std::uniform_int_distribution<size_t> d(0, top.size() - 1);
    return *top[d(rng_)];
  }
  // No legal plays.
  if (req.bag_size >= RACK_SIZE) {
    return Move::exchange(req.my_rack.counts());
  }
  return Move::pass();
}

std::unique_ptr<GreedyAgent> GreedyAgent::from_spec(const std::vector<std::string>& tokens,
                                                    int thread_id, const std::string& name) {
  uint64_t seed = 0;
  bool have_seed = false;

  po::options_description desc = greedy_options(seed);

  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=greedy options: ") + e.what());
  }

  if (have_seed) return std::make_unique<GreedyAgent>(thread_id, name, seed);
  return std::make_unique<GreedyAgent>(thread_id, name);
}

std::string GreedyAgent::options_help() {
  uint64_t seed = 0;  // unused; greedy_options() only needs a binding target
  return agent_options_help("  Picks the highest-scoring play; ties broken randomly.\n",
                            greedy_options(seed));
}

}  // namespace scribblez
