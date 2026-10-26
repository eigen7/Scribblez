#include "agent/deep_bot.h"

#include "agent/agent_options.h"
#include "agent/hasty_bot.h"
#include "belief/leave_query.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "sim/pool_jobs.h"
#include "sim/pool_labeler.h"
#include "sim/sim_runner.h"
#include "util/exception.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <optional>
#include <utility>

namespace scribblez {

namespace {

boost::program_options::options_description deepbot_options(DeepBot::Params& p) {
  namespace po = boost::program_options;
  po::options_description desc("deepbot options");
  desc.add_options()  //
    ("leaf-model", po::value<std::string>(&p.leaf_model),
     "the leaf network (a deep_bot_leaf_eval ONNX export; required), or 'synthetic' for the "
     "CPU reference with a logistic leaf")  //
    ("leaf-precision", po::value<std::string>(&p.leaf_precision)->default_value(p.leaf_precision),
     "TensorRT precision for the leaf network: fp32, fp16 or bf16")  //
    ("leave-inference", po::value<std::string>(&p.leave_inference),
     "the leave inference model (ONNX) Bob's leaves are drawn from; without it his whole rack "
     "is drawn uniformly")  //
    ("candidates", po::value<int>(&p.candidates)->default_value(p.candidates),
     "candidates by static equity")  //
    ("late-candidates", po::value<int>(&p.late_candidates)->default_value(p.late_candidates),
     "candidates with --late-bag or fewer tiles in the bag")  //
    ("late-bag", po::value<int>(&p.late_bag)->default_value(p.late_bag),
     "the bag size from which --late-candidates applies")  //
    ("rollouts", po::value<int>(&p.rollouts)->default_value(p.rollouts),
     "rollouts per candidate, each with its own rack for Bob")  //
    ("race", po::value<bool>(&p.race)->default_value(p.race),
     "stop a candidate once it sits three paired standard errors below the leader")  //
    ("threads", po::value<int>(&p.threads)->default_value(p.threads),
     "CPU threads for the belief samplers and the cpu backend; 0 uses all")  //
    ("seed", po::value<uint64_t>(&p.seed), "sim PRNG seed (default: SeedProducer)");
  return desc;
}

// The labels of one job, run alone through `labeler`.
PoolPosition label(PoolLabeler& labeler, PoolJob job) {
  std::optional<PoolJob> pending(std::move(job));
  std::optional<PoolPosition> out;
  labeler.run([&] { return std::exchange(pending, std::nullopt); },
              [&](PoolPosition&& p) { out = std::move(p); });
  return std::move(*out);
}

void validate(const DeepBot::Params& p) {
  if (p.leaf_model.empty()) throw util::CleanException("deepbot: --leaf-model is required");
  if (p.candidates < 1 || p.late_candidates < 1)
    throw util::CleanException("deepbot: --candidates and --late-candidates must be >= 1");
  if (p.rollouts < 1) throw util::CleanException("deepbot: --rollouts must be >= 1");
  if (p.threads < 0) throw util::CleanException("deepbot: --threads must be >= 0");
}

}  // namespace

DeepBot::DeepBot(const Params& params, Backend backend)
    : Agent(params.thread_id, params.name), params_(params), backend_(std::move(backend)) {
  validate(params_);
  if (params_.threads == 0) params_.threads = util::default_thread_count();
}

void DeepBot::begin_game(const BeginGameRequest& /*req*/) {
  recorder_.begin_game();
  ply_ = 0;
}

void DeepBot::observe_move(const Move& move) {
  recorder_.observe(move);
  ++ply_;
}

uint64_t DeepBot::turn_seed() const {
  return util::splitmix64(params_.seed ^ util::splitmix64(uint64_t(ply_)));
}

std::vector<Move> DeepBot::candidates(const MoveRequest& req) const {
  const bool late = req.bag_size <= params_.late_bag;
  return equity_top_k(req, late ? params_.late_candidates : params_.candidates);
}

std::shared_ptr<const belief::LeaveSampler> DeepBot::root_belief(
  const belief::LeaveQuery& q) const {
  if (!backend_.predict) return nullptr;
  return belief::beliefs_of({&q, 1}, backend_.predict, *backend_.table, params_.threads)[0];
}

MoveDecision DeepBot::make_move(const MoveRequest& req) {
  // With the bag empty the opponent's rack is known and no belief is needed.
  // EndgameAgent solves those turns, so this is its fallback, and every later
  // turn is one too: no query will read the racks not recorded since.
  if (req.bag_size == 0) return hasty_best_move_wmp(req);
  // Every other turn records the rack, which later queries' masks read.
  const belief::LeaveQuery q = recorder_.own_query(req);
  const std::vector<Move> moves = candidates(req);
  if (moves.size() == 1) return moves.front();
  const std::shared_ptr<const belief::LeaveSampler> belief = root_belief(q);
  PoolJob job = position_job(sim_position_from(req), moves, turn_seed(), belief.get(), req.dict,
                             params_.rollouts);
  PoolLabeler labeler(*backend_.values, {.rollouts = params_.rollouts, .race = params_.race});
  const std::vector<PoolLabel> labels = label(labeler, std::move(job)).labels;
  const auto best = std::max_element(
    labels.begin(), labels.end(), [](const PoolLabel& a, const PoolLabel& b) { return a.q < b.q; });
  return moves[size_t(best - labels.begin())];
}

DeepBot::Params DeepBot::parse_params(const std::vector<std::string>& tokens, int thread_id,
                                      const std::string& name,
                                      boost::program_options::options_description& extra,
                                      const char* type_label) {
  namespace po = boost::program_options;
  Params p{.thread_id = thread_id, .name = name};
  po::options_description desc = deepbot_options(p);
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
  validate(p);
  // The equity tables must not load concurrently with queries (hasty_equity.h).
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  return p;
}

std::string DeepBot::options_help() {
  Params defaults;  // binding targets; only the defaults are read
  return agent_options_help(
    "  DeepBot with the hasty reply model: sims its static-equity top moves on the GPU,\n"
    "  each rollout seating a leave for the opponent from the leave inference model's\n"
    "  belief, both replies by HastyBot, and the leaf network at the end; it plays the\n"
    "  candidate most likely best. Hidden leaves only; deepbot-endgame solves the endgame.\n",
    deepbot_options(defaults));
}

}  // namespace scribblez
