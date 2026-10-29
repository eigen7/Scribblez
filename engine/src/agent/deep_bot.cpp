#include "agent/deep_bot.h"

#include "agent/agent_options.h"
#include "agent/alice_subtree.h"
#include "agent/hasty_bot.h"
#include "agent/subtree_dump.h"
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
    ("reply", po::value<std::string>(&p.reply)->default_value(p.reply),
     "how Alice models Bob's reply: subtree (Bob's own sims on his information), hasty "
     "(HastyBot's move), or alice-subtree (HastyBot's move, then Alice's own sims choose her "
     "second move)")  //
    ("rollouts", po::value<int>(&p.rollouts)->default_value(p.rollouts),
     "hasty: rollouts per candidate, each with its own rack for Bob")  //
    ("race", po::value<bool>(&p.race)->default_value(p.race),
     "hasty: stop a candidate once it sits three paired standard errors below the leader")  //
    ("bob-racks", po::value<int>(&p.bob_racks)->default_value(p.bob_racks),
     "subtrees: racks Alice samples for Bob, each with its own subtree")  //
    ("subtree-candidates",
     po::value<int>(&p.subtree_candidates)->default_value(p.subtree_candidates),
     "subtrees: how many candidates, the best by a first pass of hasty sims, get the subtree; "
     "0 = all")  //
    ("sims-per-rack", po::value<int>(&p.sims_per_rack)->default_value(p.sims_per_rack),
     "subtrees: Alice's sims after the subtree's move at each rack, each drawing the "
     "replenishment")  //
    ("alice-racks", po::value<int>(&p.subtree.opp_racks)->default_value(p.subtree.opp_racks),
     "subtree: leaves Bob samples for Alice, one sim of each of his candidates per leave")  //
    ("bob-candidates", po::value<int>(&p.subtree.candidates)->default_value(p.subtree.candidates),
     "subtree: Bob's candidates by static equity")  //
    ("bob-late-candidates",
     po::value<int>(&p.subtree.late_candidates)->default_value(p.subtree.late_candidates),
     "subtree: Bob's candidates with --bob-late-bag or fewer tiles in the bag")  //
    ("bob-late-bag", po::value<int>(&p.subtree.late_bag)->default_value(p.subtree.late_bag),
     "subtree: the bag size Bob sees from which --bob-late-candidates applies")  //
    ("bob-race", po::value<bool>(&p.subtree.race)->default_value(p.subtree.race),
     "subtree: stop Bob's clearly beaten candidates between rounds")  //
    ("second-racks", po::value<int>(&p.second.opp_racks)->default_value(p.second.opp_racks),
     "alice-subtree: leaves Alice samples for Bob at her second turn, one sim of each of her "
     "candidates per leave")  //
    ("second-candidates", po::value<int>(&p.second.candidates)->default_value(p.second.candidates),
     "alice-subtree: Alice's candidates at her second turn by static equity (1: HastyBot's "
     "move, four hasty plies)")  //
    ("second-late-candidates",
     po::value<int>(&p.second.late_candidates)->default_value(p.second.late_candidates),
     "alice-subtree: her second turn's candidates with --second-late-bag or fewer tiles in the "
     "bag")  //
    ("second-late-bag", po::value<int>(&p.second.late_bag)->default_value(p.second.late_bag),
     "alice-subtree: the bag size she sees from which --second-late-candidates applies")  //
    ("second-race", po::value<bool>(&p.second.race)->default_value(p.second.race),
     "alice-subtree: stop her second turn's clearly beaten candidates between rounds")  //
    ("subtree-dump", po::value<std::string>(&p.subtree_dump),
     "subtree: append a JSON line per turn comparing Bob's subtree replies with HastyBot's "
     "(runs the hasty sims and the subtrees a second time)")  //
    ("seed", po::value<uint64_t>(&p.seed), "sim PRNG seed (default: SeedProducer)");
  return desc;
}

// A second run of the subtrees, on fresh samples and draws.
constexpr uint64_t kRerunSeed = 0x2E2D2E2D2E2D2E2DULL;

// A candidate after which Alice's draw empties the bag.
bool empties_bag(const Move& m, int bag) {
  return m.type() == MoveType::PLAY && m.num_glyphs() >= bag;
}

// `values` with each candidate that empties the bag valued by the first
// pass, whose hasty sims play the game to its end.
std::vector<double> with_first_pass(std::vector<double> values, const std::vector<Move>& moves,
                                    const std::vector<PoolLabel>& hasty, int bag) {
  for (size_t c = 0; c < moves.size(); ++c)
    if (empties_bag(moves[c], bag)) values[c] = hasty[c].win_prob;
  return values;
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
  if (p.reply != "subtree" && p.reply != "hasty" && p.reply != "alice-subtree")
    throw util::CleanException("deepbot: --reply must be subtree, hasty or alice-subtree");
  if (p.bob_racks < 1 || p.sims_per_rack < 1 || p.subtree.opp_racks < 1 || p.second.opp_racks < 1)
    throw util::CleanException(
      "deepbot: --bob-racks, --sims-per-rack, --alice-racks and --second-racks must be >= 1");
  if (!p.subtree_dump.empty() && p.reply != "subtree")
    throw util::CleanException("deepbot: --subtree-dump needs --reply subtree");
  if (p.subtree_candidates < 0)
    throw util::CleanException("deepbot: --subtree-candidates must be >= 0");
  if (p.subtree.candidates < 1 || p.subtree.late_candidates < 1)
    throw util::CleanException("deepbot: --bob-candidates and --bob-late-candidates must be >= 1");
  if (p.second.candidates < 1 || p.second.late_candidates < 1)
    throw util::CleanException(
      "deepbot: --second-candidates and --second-late-candidates must be >= 1");
}

}  // namespace

DeepBot::DeepBot(const Params& params, Backend backend)
    : Agent(params.thread_id, params.name), params_(params), backend_(std::move(backend)) {
  validate(params_);
}

void DeepBot::begin_game(const BeginGameRequest& /*req*/) {
  recorder_.begin_game();
  ply_ = 0;
  ++games_;
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
  return belief::beliefs_of({&q, 1}, backend_.predict, *backend_.table)[0];
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
  if (params_.reply != "hasty") return subtree_move(req, moves, belief.get());
  const std::vector<PoolLabel> labels =
    hasty_labels(req, moves, belief.get(), params_.rollouts, params_.race);
  const auto best = std::max_element(
    labels.begin(), labels.end(), [](const PoolLabel& a, const PoolLabel& b) { return a.q < b.q; });
  return moves[size_t(best - labels.begin())];
}

std::vector<PoolLabel> DeepBot::hasty_labels(const MoveRequest& req, const std::vector<Move>& moves,
                                             const belief::LeaveSampler* belief, int rollouts,
                                             bool race) const {
  PoolJob job =
    position_job(sim_position_from(req), moves, turn_seed(), belief, req.dict, rollouts);
  PoolLabeler labeler(*backend_.values, {.rollouts = rollouts, .race = race});
  return label(labeler, std::move(job)).labels;
}

Move DeepBot::subtree_move(const MoveRequest& req, const std::vector<Move>& moves,
                           const belief::LeaveSampler* belief) {
  // Alice's rollout i seats rack i for Bob in every sim of hers, hasty or not.
  const PoolJob root =
    position_job(sim_position_from(req), moves, turn_seed(), belief, req.dict, params_.bob_racks);
  const std::vector<Rack> racks = bob_racks(root.group, root.starts[0], params_.bob_racks);
  const std::vector<belief::ChooserMove> earlier = recorder_.own_moves(req.dict);
  const belief::ReplyPosition position{.board = req.board,
                                       .chooser_rack = req.my_rack,
                                       .chooser_score = req.my_score,
                                       .opp_score = req.opp_score,
                                       .bag = req.bag_size,
                                       .earlier = earlier};
  const AliceTurn turn{position, moves, root.starts, racks, root.group};
  // The first pass: every candidate's hasty sims, on the same racks.
  const std::vector<PoolLabel> hasty =
    hasty_labels(req, moves, belief, params_.bob_racks, /*race=*/false);
  const std::vector<size_t> searched = survivors(moves, hasty, req.bag_size);
  const std::vector<double> values = params_.reply == "subtree"
                                       ? bob_subtree_values(req, turn, hasty, searched)
                                       : second_turn_values(req, turn, searched);
  return moves[best_candidate(with_first_pass(values, moves, hasty, req.bag_size))];
}

std::vector<size_t> DeepBot::survivors(const std::vector<Move>& moves,
                                       const std::vector<PoolLabel>& hasty, int bag) const {
  std::vector<size_t> out;
  for (size_t c = 0; c < moves.size(); ++c)
    if (!empties_bag(moves[c], bag)) out.push_back(c);
  const size_t keep =
    params_.subtree_candidates == 0 ? out.size() : size_t(params_.subtree_candidates);
  if (out.size() <= keep) return out;
  std::stable_sort(out.begin(), out.end(),
                   [&](size_t a, size_t b) { return hasty[a].win_prob > hasty[b].win_prob; });
  out.resize(keep);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<double> DeepBot::bob_subtree_values(const MoveRequest& req, const AliceTurn& turn,
                                                const std::vector<PoolLabel>& hasty,
                                                const std::vector<size_t>& searched) {
  const auto replies = bob_replies(turn, searched, belief_model(), *backend_.values,
                                   params_.subtree, req.dict, turn_seed());
  const std::vector<double> values = values_after_replies(
    turn, searched, replies, *backend_.values, params_.sims_per_rack, req.dict, turn_seed());
  if (!params_.subtree_dump.empty()) {
    dump_turn(req, turn, hasty, searched, replies,
              with_first_pass(values, turn.candidates, hasty, req.bag_size));
  }
  return values;
}

std::vector<double> DeepBot::second_turn_values(const MoveRequest& req, const AliceTurn& turn,
                                                const std::vector<size_t>& searched) {
  // Bob's history only feeds Alice's belief, which a model draws.
  const std::vector<belief::ChooserMove> bob_earlier =
    backend_.predict ? recorder_.opponent_moves(req.dict) : std::vector<belief::ChooserMove>{};
  return alice_subtree_values(turn, searched, bob_earlier, belief_model(), *backend_.values,
                              params_.second, params_.sims_per_rack, req.dict, turn_seed());
}

void DeepBot::dump_turn(const MoveRequest& req, const AliceTurn& turn,
                        const std::vector<PoolLabel>& hasty, const std::vector<size_t>& searched,
                        const std::vector<std::vector<SubtreeMove>>& replies,
                        const std::vector<double>& values) const {
  const auto rerun = bob_replies(turn, searched, belief_model(), *backend_.values, params_.subtree,
                                 req.dict, turn_seed() ^ kRerunSeed);
  SubtreeTurn t{.agent = name(),
                .thread = params_.thread_id,
                .game = games_,
                .ply = ply_,
                .bag = req.bag_size,
                .turn = &turn,
                .searched = std::vector<bool>(turn.candidates.size(), false),
                .subtree_values = values,
                .hasty_values = {},
                .replies = &replies,
                .rerun = &rerun};
  for (size_t c : searched) t.searched[c] = true;
  for (const PoolLabel& l : hasty) t.hasty_values.push_back(l.win_prob);
  append_dump_line(params_.subtree_dump, subtree_turn_json(t));
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
    "  DeepBot: sims its static-equity top moves on the GPU, each rack for the opponent\n"
    "  seating a leave from the leave inference model's belief, the leaf network at the\n"
    "  end. --reply subtree has the opponent choose his reply by his own sims on his own\n"
    "  information; --reply hasty has him play HastyBot's move; --reply alice-subtree\n"
    "  has him play HastyBot's move and her own sims choose her second move. Hidden\n"
    "  leaves only; deepbot-endgame solves the endgame.\n",
    deepbot_options(defaults));
}

}  // namespace scribblez
