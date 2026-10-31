#include "agent/policy_bot.h"

#include "agent/candidate_evaluator.h"
#include "agent/hasty_bot.h"
#include "sim/sim_runner.h"
#include "util/exception.h"

#include <algorithm>
#include <cmath>
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

InputEncodingSpec hidden_leaves_spec(const Dictionary& dict, const PolicyBot::Backend& backend) {
  if (!backend.policy) throw util::Exception("policybot: a policy-set model is required");
  InputEncodingSpec spec = derive_input_spec(dict, *backend.policy, "policybot");
  if (spec.opp_leave_input) {
    throw util::CleanException("policybot: the model reads face-up leaves; its pools are hidden");
  }
  return spec;
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
      dict_(require_dict(params.dict)),
      spec_(hidden_leaves_spec(dict_, backend_)),
      encoder_(spec_) {
  board_row_.resize(size_t(input_floats(spec_)));
}

void PolicyBot::begin_game(const BeginGameRequest& req) {
  encoder_ = GameStateEncoder(spec_, req.initial_scores);
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
  encode(req, moves, opp_rack_belief(req, q));
  return moves[best_candidate()];
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

void PolicyBot::encode(const MoveRequest& req, const std::vector<Move>& moves,
                       const belief::OppRackSummary& opp_rack) {
  // The cross-check input planes read the board's move-generation caches;
  // building them here is a no-op once they are valid.
  encoder_.board().ensure_movegen_caches(dict_);
  // The encoder has observed every prior move, so its active player is this
  // agent's seat.
  encoder_.encode_input(encoder_.active_player(), req.my_rack, req.opp_rack, board_row_.data());
  opp_rack_ = policy_set::encode_opp_rack(opp_rack);
  candidates_.encode(encoder_, dict_, req.my_rack, req.bag_size, moves);
}

size_t PolicyBot::best_candidate() {
  logits_.resize(size_t(candidates_.count()));
  float* const head_out[] = {logits_.data()};
  backend_.policy->evaluate({board_row_.data(), opp_rack_.data(), &candidates_}, head_out);
  if (!std::ranges::all_of(logits_, [](float x) { return std::isfinite(x); }))
    throw util::Exception("policybot: the model gave a non-finite logit");
  return size_t(std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
}

}  // namespace scribblez
