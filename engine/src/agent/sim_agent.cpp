#include "agent/sim_agent.h"

#include "agent/agent_options.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "selfplay/seed_producer.h"
#include "util/math.h"

#include <boost/program_options.hpp>

#include <optional>
#include <stdexcept>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// The `--player "--type=sim ..."` options, bound to one struct so the parsed
// set and the documented set cannot drift.
struct SimOptions {
  int top_k = 10;
  int rollouts = 400;
  int sim_threads = 1;
  std::string objective = "winrate";
  uint64_t seed = 0;
  EndgameSolver::Params endgame;
};

po::options_description make_options_description(SimOptions& o) {
  po::options_description desc("sim options");
  desc.add_options()                                                 //
    ("top-k", po::value<int>(&o.top_k)->default_value(o.top_k),      //
     "candidates simmed per turn, taken by HastyBot static equity")  //
    ("rollouts", po::value<int>(&o.rollouts)->default_value(o.rollouts),
     "rollouts per candidate; every candidate shares the same rollout seeds, so "
     "rack and draw luck cancels when they are compared")  //
    ("sim-threads", po::value<int>(&o.sim_threads)->default_value(o.sim_threads),
     "threads within one turn's simulation; leave at 1 when the game loop is "
     "already running games in parallel")  //
    ("objective", po::value<std::string>(&o.objective)->default_value(o.objective),
     "what the rollouts are scored on: 'winrate' or 'spread'")  //
    ("seed", po::value<uint64_t>(&o.seed),
     "PRNG seed for the rollouts (default: derived from SeedProducer)");
  o.endgame.add_options(desc, "endgame-");
  return desc;
}

SimAgent::Objective parse_objective(const std::string& name) {
  if (name == "winrate") return SimAgent::Objective::kWinRate;
  if (name == "spread") return SimAgent::Objective::kSpread;
  throw std::runtime_error("--objective must be 'winrate' or 'spread', got '" + name + "'");
}

// Checked in the initializer list, where the SimRunner member dereferences it
// before any constructor body could look.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw std::runtime_error("sim agent: a dictionary is required");
  return *dict;
}

double objective_value(const SimObservation& o, SimAgent::Objective objective) {
  if (o.n == 0) return 0.0;
  const double n = o.n;
  if (objective == SimAgent::Objective::kWinRate) return (o.wins + 0.5 * o.draws) / n;
  return static_cast<double>(o.delta_sum) / n;
}

}  // namespace

SimAgent::SimAgent(const Params& params)
    : Agent(params.thread_id, params.name),
      top_k_(params.top_k),
      objective_(params.objective),
      seed_(params.seed),
      runner_(require_dict(params.dict), params.sim),
      endgame_(params.thread_id, params.endgame) {
  if (top_k_ < 1) throw std::runtime_error("sim agent: --top-k must be >= 1");
}

uint64_t SimAgent::sim_seed(int ply) const {
  return util::splitmix64(seed_ ^ util::splitmix64(static_cast<uint64_t>(ply)));
}

void SimAgent::begin_game() {
  endgame_.begin_game();
  ply_ = 0;
}

void SimAgent::observe_move(const Move& move) {
  endgame_.observe_move(move);
  ++ply_;
}

int SimAgent::best_index(const std::vector<SimObservation>& observations) const {
  int best = 0;
  for (size_t i = 1; i < observations.size(); ++i) {
    if (objective_value(observations[i], objective_) >
        objective_value(observations[best], objective_)) {
      best = static_cast<int>(i);
    }
  }
  return best;
}

MoveDecision SimAgent::make_move(const MoveRequest& req) {
  // The endgame belongs to the exact solver, which needs no candidates of ours.
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> candidates = equity_top_k(req, top_k_);
  // Rollouts need a bag to draw the opponent's replenishments from, so a
  // bag-empty turn the solver declined falls back to the static-equity move --
  // which is what equity_top_k already ranked first.
  if (req.bag_size == 0 || candidates.size() == 1) return candidates.front();

  SimPosition pos;
  pos.board = req.board;
  // The rollouts run from the mover's point of view, so seating them as player
  // 0 costs nothing and spares the agent having to know its own seat.
  pos.mover = 0;
  pos.scores = {req.my_score, req.opp_score};
  pos.rack = req.my_rack;
  // Whatever we legitimately know of the opponent's rack (see MoveRequest):
  // under face-up leaves their retained tiles, which then seed every rollout
  // instead of being drawn from the pool.
  pos.opp_leave = req.opp_rack;

  const std::vector<SimObservation> observations = runner_.run(pos, candidates, sim_seed(ply_));
  return candidates[static_cast<size_t>(best_index(observations))];
}

std::unique_ptr<SimAgent> SimAgent::from_spec(const std::vector<std::string>& tokens, int thread_id,
                                              const std::string& name) {
  SimOptions opts;
  po::options_description desc = make_options_description(opts);

  bool have_seed = false;
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=sim options: ") + e.what());
  }

  HastyEquity::ensure_initialized(Lexicon::instance().name());

  Params params;
  params.thread_id = thread_id;
  params.name = name;
  params.dict = &Lexicon::instance().dict();
  params.top_k = opts.top_k;
  params.sim.rollouts = opts.rollouts;
  params.sim.threads = opts.sim_threads;
  params.objective = parse_objective(opts.objective);
  params.seed = have_seed ? opts.seed : SeedProducer::instance().next();
  params.endgame = opts.endgame;
  return std::make_unique<SimAgent>(params);
}

std::string SimAgent::options_help() {
  SimOptions defaults;  // scratch binding targets; only the defaults are read
  return agent_options_help(
    "  Monte-Carlo simming bot: it keeps the best --top-k moves by HastyBot static\n"
    "  equity, plays --rollouts games out from each under common random numbers,\n"
    "  and plays whichever scored best. Once the bag empties the turn goes to the\n"
    "  exact endgame solver.\n",
    make_options_description(defaults));
}

}  // namespace scribblez
