#include "agent/best_bot.h"

#include "agent/agent_options.h"
#include "agent/hasty_bot.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "sim/macondo_simmer.h"
#include "sim/sim_runner.h"
#include "sim/win_pct_table.h"
#include "util/exception.h"
#include "util/math.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace scribblez {

namespace {

// Macondo's phase boundaries, in tiles unseen by the mover. With 8 unseen the
// bag holds one tile (Macondo's pre-endgame), with 7 or fewer it is empty.
constexpr int kStaticMaxUnseen = 8;
constexpr int kLateMaxUnseen = 14;
constexpr int kCandidates = 40;
constexpr int kLateCandidates = 80;

// Shared by parse_params() and options_help(), so the parsed and documented
// options cannot drift.
boost::program_options::options_description bestbot_options(BestBot::Params& p) {
  namespace po = boost::program_options;
  po::options_description desc("bestbot options");
  desc.add_options()                                                                     //
    ("min-sim-plies", po::value<int>(&p.min_sim_plies)->default_value(p.min_sim_plies),  //
     "rollout plies while more than 14 tiles are unseen (Macondo uses at least 2)")      //
    ("sim-threads", po::value<int>(&p.sim_threads)->default_value(p.sim_threads),
     "threads within one move's sim; leave at 1 when the game loop already runs games in "
     "parallel. Results do not depend on it")  //
    ("max-iterations", po::value<int>(&p.max_iterations)->default_value(p.max_iterations),
     "cap on sim iterations per move; 0 leaves it to Macondo's stopping rule")  //
    ("seed", po::value<uint64_t>(&p.seed), "sim PRNG seed (default: SeedProducer)");
  return desc;
}

}  // namespace

BestBot::BestBot(const Params& params)
    : Agent(params.thread_id, params.name),
      min_sim_plies_(params.min_sim_plies),
      sim_threads_(params.sim_threads),
      max_iterations_(params.max_iterations),
      seed_(params.seed) {
  if (min_sim_plies_ < 1) throw util::CleanException("bestbot: --min-sim-plies must be >= 1");
  if (sim_threads_ < 1) throw util::CleanException("bestbot: --sim-threads must be >= 1");
  if (max_iterations_ < 0) throw util::CleanException("bestbot: --max-iterations must be >= 0");
}

uint64_t BestBot::sim_seed(int ply) const {
  return util::splitmix64(seed_ ^ util::splitmix64(uint64_t(ply)));
}

void BestBot::begin_game(const BeginGameRequest& /*req*/) {
  ply_ = 0;
  scoreless_turns_ = 0;
}

void BestBot::observe_move(const Move& move) {
  ++ply_;
  scoreless_turns_ = move.type() == MoveType::PLAY ? 0 : scoreless_turns_ + 1;
}

MoveDecision BestBot::make_move(const MoveRequest& req) {
  const int unseen = req.board.unseen_count(req.my_rack.size());
  if (unseen <= kStaticMaxUnseen) return hasty_best_move_wmp(req);

  const bool late = unseen <= kLateMaxUnseen;
  const std::vector<Move> candidates = equity_top_k(req, late ? kLateCandidates : kCandidates);
  if (candidates.size() == 1) return candidates.front();

  const MacondoSimmer::Params params{
    .plies = late ? unseen : std::max(min_sim_plies_, 2),
    .threads = sim_threads_,
    .max_iterations = max_iterations_,
  };
  const MacondoSimmer::Result result = MacondoSimmer::simulate(
    req.dict, sim_position_from(req), candidates, scoreless_turns_, params, sim_seed(ply_));
  return result.plays.front().move;
}

BestBot::Params BestBot::parse_params(const std::vector<std::string>& tokens, int thread_id,
                                      const std::string& name,
                                      boost::program_options::options_description& extra,
                                      const char* type_label) {
  namespace po = boost::program_options;
  Params p{.thread_id = thread_id, .name = name};
  po::options_description desc = bestbot_options(p);
  desc.add(extra);
  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type={} options: {}", type_label, e.what());
  }
  if (!have_seed) p.seed = SeedProducer::instance().next();

  // Load the process-wide tables now, in single-threaded setup: the equity
  // tables must not load concurrently with queries (hasty_equity.h), and a
  // missing win-percentage file should fail here rather than mid-game.
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  WinPctTable::macondo_default();
  return p;
}

std::unique_ptr<BestBot> BestBot::from_spec(const std::vector<std::string>& tokens, int thread_id,
                                            const std::string& name) {
  boost::program_options::options_description no_extra;
  return std::make_unique<BestBot>(parse_params(tokens, thread_id, name, no_extra, kType));
}

std::string BestBot::options_help() {
  Params defaults;  // binding targets; only the defaults are read
  return agent_options_help(
    "  A port of Macondo's BestBot (without its endgame engines): it sims HastyBot's\n"
    "  top moves with fixed-ply rollouts scored by Macondo's win-percentage table,\n"
    "  pruning trailing moves until one remains or the iteration cap is hit. With 8 or\n"
    "  fewer tiles unseen it plays HastyBot's move.\n",
    bestbot_options(defaults));
}

}  // namespace scribblez
