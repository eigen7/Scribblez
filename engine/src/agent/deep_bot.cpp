#include "agent/deep_bot.h"

#include "agent/agent_options.h"
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
     "how Alice models Bob's reply: subtree (Bob's own sims on his information) or hasty "
     "(HastyBot's move)")  //
    ("rollouts", po::value<int>(&p.rollouts)->default_value(p.rollouts),
     "hasty: rollouts per candidate, each with its own rack for Bob")  //
    ("race", po::value<bool>(&p.race)->default_value(p.race),
     "hasty: stop a candidate once it sits three paired standard errors below the leader")  //
    ("bob-racks", po::value<int>(&p.bob_racks)->default_value(p.bob_racks),
     "subtree: racks Alice samples for Bob, each with its own subtree")  //
    ("sims-per-rack", po::value<int>(&p.sims_per_rack)->default_value(p.sims_per_rack),
     "subtree: Alice's sims after Bob's reply at each rack, each drawing her replenishment")  //
    ("alice-racks", po::value<int>(&p.subtree.alice_racks)->default_value(p.subtree.alice_racks),
     "subtree: leaves Bob samples for Alice, one sim of each of his candidates per leave")  //
    ("bob-candidates", po::value<int>(&p.subtree.candidates)->default_value(p.subtree.candidates),
     "subtree: Bob's candidates by static equity")  //
    ("bob-late-candidates",
     po::value<int>(&p.subtree.late_candidates)->default_value(p.subtree.late_candidates),
     "subtree: Bob's candidates with --late-bag or fewer tiles in the bag")  //
    ("bob-race", po::value<bool>(&p.subtree.race)->default_value(p.subtree.race),
     "subtree: stop Bob's clearly beaten candidates between rounds")  //
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

size_t argmax(const std::vector<double>& v) {
  return size_t(std::max_element(v.begin(), v.end()) - v.begin());
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
  if (p.reply != "subtree" && p.reply != "hasty")
    throw util::CleanException("deepbot: --reply must be subtree or hasty");
  if (p.bob_racks < 1 || p.sims_per_rack < 1 || p.subtree.alice_racks < 1)
    throw util::CleanException(
      "deepbot: --bob-racks, --sims-per-rack and --alice-racks must be >= 1");
  if (p.subtree.candidates < 1 || p.subtree.late_candidates < 1)
    throw util::CleanException("deepbot: --bob-candidates and --bob-late-candidates must be >= 1");
}

}  // namespace

DeepBot::DeepBot(const Params& params, Backend backend)
    : Agent(params.thread_id, params.name), params_(params), backend_(std::move(backend)) {
  validate(params_);
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
  if (params_.reply == "subtree") return subtree_move(req, moves, belief.get());
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
  const AliceTurn turn{position, moves, root.starts, racks};
  std::vector<size_t> searched;
  for (size_t c = 0; c < moves.size(); ++c)
    if (!empties_bag(moves[c], req.bag_size)) searched.push_back(c);
  const auto replies = bob_replies(turn, searched, bob_belief_model(), *backend_.values,
                                   params_.subtree, req.dict, turn_seed());
  const std::vector<double> values = subtree_values(req, turn, belief, searched, replies);
  if (!params_.subtree_dump.empty()) dump_turn(req, turn, belief, searched, replies, values);
  return moves[argmax(values)];
}

std::vector<double> DeepBot::subtree_values(
  const MoveRequest& req, const AliceTurn& turn, const belief::LeaveSampler* belief,
  const std::vector<size_t>& searched, const std::vector<std::vector<BobReply>>& replies) const {
  std::vector<double> values = values_after_replies(turn, searched, replies, *backend_.values,
                                                    params_.sims_per_rack, req.dict, turn_seed());
  std::vector<size_t> emptying;
  std::vector<Move> to_end;
  for (size_t c = 0; c < turn.candidates.size(); ++c) {
    if (!empties_bag(turn.candidates[c], req.bag_size)) continue;
    emptying.push_back(c);
    to_end.push_back(turn.candidates[c]);
  }
  if (emptying.empty()) return values;
  const std::vector<PoolLabel> labels =
    hasty_labels(req, to_end, belief, params_.bob_racks, /*race=*/false);
  for (size_t k = 0; k < emptying.size(); ++k) values[emptying[k]] = labels[k].win_prob;
  return values;
}

void DeepBot::dump_turn(const MoveRequest& req, const AliceTurn& turn,
                        const belief::LeaveSampler* belief, const std::vector<size_t>& searched,
                        const std::vector<std::vector<BobReply>>& replies,
                        const std::vector<double>& values) const {
  const std::vector<PoolLabel> hasty =
    hasty_labels(req, turn.candidates, belief, params_.bob_racks, /*race=*/false);
  const auto rerun = bob_replies(turn, searched, bob_belief_model(), *backend_.values,
                                 params_.subtree, req.dict, turn_seed() ^ kRerunSeed);
  SubtreeTurn t{.ply = ply_,
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
    "  information; --reply hasty has him play HastyBot's move. Hidden leaves only;\n"
    "  deepbot-endgame solves the endgame.\n",
    deepbot_options(defaults));
}

}  // namespace scribblez
