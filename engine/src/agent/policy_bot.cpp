#include "agent/policy_bot.h"

#include "agent/hasty_bot.h"
#include "sim/sim_runner.h"
#include "training/policy_set_inputs.h"
#include "util/exception.h"

#include <limits>
#include <utility>

// The production constructor and the option parsing live in
// policy_bot_factory.cpp.

namespace scribblez {

namespace {

// Checked in the initializer list, where members dereference the dictionary
// and the model before any constructor body could check them.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw util::Exception("policybot: a dictionary is required");
  return *dict;
}

// Throws unless `leave` (nullptr without --leave-inference) is the leave model
// `trained` names.
void check_leave_model(const PolicyBot::Params& p, const nn::ModelIdentity& trained,
                       const nn::ModelIdentity* leave) {
  if (trained.tag.empty()) {
    if (leave != nullptr) {
      throw util::CleanException("policybot: {} trained on uniform beliefs; drop --leave-inference",
                                 p.service.model);
    }
    return;
  }
  if (leave == nullptr) {
    throw util::CleanException(
      "policybot: {} trained on the beliefs of leave_inference tag {} generation {}; pass "
      "--leave-inference=<that export>",
      p.service.model, trained.tag, trained.generation);
  }
  if (leave->tag != trained.tag || leave->generation != trained.generation) {
    throw util::CleanException(
      "policybot: {} is leave_inference tag {} generation {}, but {} trained on generation {} "
      "of tag {}",
      p.leave_inference, leave->tag, leave->generation, p.service.model, trained.generation,
      trained.tag);
  }
}

}  // namespace

void PolicyBot::check_trained_on(const Params& p, const TrainedOn& trained,
                                 const nn::ModelIdentity* leave) {
  if (trained.blank_prune_points < 0) {
    throw util::CleanException("policybot: {} records no blank prune; is it a policy_set export?",
                               p.service.model);
  }
  check_leave_model(p, trained.leave_model, leave);
}

PolicyBot::PolicyBot(const Params& params, Backend backend)
    : Agent(params.thread_id, params.name),
      backend_(std::move(backend)),
      scorer_(backend_.policy, backend_.cross_checks, require_dict(params.dict), "policybot"),
      encoder_(scorer_.spec()) {}

void PolicyBot::begin_game(const BeginGameRequest& req) {
  encoder_ = GameStateEncoder(scorer_.spec(), req.initial_scores);
  recorder_.begin_game();
}

void PolicyBot::observe_move(const Move& move) {
  encoder_.apply_move(move);
  recorder_.observe(move);
}

MoveDecision PolicyBot::make_move(const MoveRequest& req) {
  // The pools never hold a bag-empty turn. EndgameAgent solves those turns,
  // so this is its fallback, and every later turn is one too: no query will
  // read the racks not recorded since.
  if (req.bag_size == 0) return hasty_best_move_wmp(req);
  // Every other turn records the rack, which later queries' masks read.
  const belief::LeaveQuery q = recorder_.own_query(req);
  const std::vector<Move> moves = candidates(req);
  if (moves.size() == 1) return moves.front();
  // The encoder has observed every prior move, so its active player is this
  // agent's seat.
  return moves[best_logit(
    scorer_.score(encoder_, req.my_rack, req.bag_size, opp_rack_belief(req, q), moves))];
}

std::vector<Move> PolicyBot::candidates(const MoveRequest& req) const {
  if (!req.opp_rack.empty())
    throw util::Exception("policybot: hidden leaves only, but the opponent's rack is face up");
  return policy_set::pool_candidates(equity_top_k(req, std::numeric_limits<int>::max()),
                                     backend_.blank_prune_points)
    .moves;
}

belief::OppRackSummary PolicyBot::opp_rack_belief(const MoveRequest& req,
                                                  const belief::LeaveQuery& q) const {
  const std::shared_ptr<const belief::LeaveSampler> belief =
    backend_.predict ? belief::beliefs_of({&q, 1}, backend_.predict, *backend_.table)[0] : nullptr;
  return policy_set::opp_rack_summary(req.board, req.my_rack, belief.get());
}

}  // namespace scribblez
