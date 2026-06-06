#include "scribblez/agent.h"

#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace scribblez {

GreedyAgent::GreedyAgent(int thread_id, const std::string& name)
    : Agent(thread_id, name), rng_(SeedProducer::instance().next()) {}

GreedyAgent::GreedyAgent(int thread_id, const std::string& name, uint64_t seed)
    : Agent(thread_id, name), rng_(seed) {}

Move GreedyAgent::make_move(const MoveRequest& req) {
  if (!req.legal_plays.empty()) {
    int best = req.legal_plays.front().score;
    for (const auto& m : req.legal_plays) best = std::max(best, m.score);
    std::vector<const Move*> top;
    for (const auto& m : req.legal_plays)
      if (m.score == best) top.push_back(&m);
    std::uniform_int_distribution<size_t> d(0, top.size() - 1);
    return *top[d(rng_)];
  }
  // No legal plays.
  if (req.bag_size >= RACK_SIZE) {
    Move m;
    m.type = MoveType::EXCHANGE;
    int gi = 0;
    for (Tile L = Tile::of(0); L <= BLANK; ++L) {
      for (int i = 0; i < req.my_rack.count(L); ++i) m.glyphs[gi++] = Glyph::exchanging(L);
    }
    return m;
  }
  Move m;
  m.type = MoveType::PASS;
  return m;
}

std::unique_ptr<GreedyAgent> GreedyAgent::from_spec(const std::vector<std::string>& tokens,
                                                    int thread_id, const std::string& name) {
  namespace po = boost::program_options;
  uint64_t seed = 0;
  bool have_seed = false;

  po::options_description desc("greedy options");
  desc.add_options()                                  //
    ("seed", po::value<uint64_t>(&seed),              //
     "PRNG seed for tie-breaking (default: derived "  //
     "from SeedProducer)");

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
  return "  Picks the highest-scoring play; ties broken randomly.\n"
         "  Options:\n"
         "    --seed=N    PRNG seed for tie-breaking (default: derived from --seed)\n";
}

}  // namespace scribblez
