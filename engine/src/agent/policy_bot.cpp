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

bool all_finite(const std::vector<float>& xs) {
  for (float x : xs)
    if (!std::isfinite(x)) return false;
  return true;
}

InputEncodingSpec hidden_leaves_spec(const Dictionary& dict, const PolicyBot::Backend& backend) {
  if (!backend.policy) throw util::Exception("policybot: a policy-set model is required");
  InputEncodingSpec spec = derive_input_spec(dict, *backend.policy, "policybot");
  if (spec.opp_leave_input) {
    throw util::CleanException("policybot: the model reads face-up leaves; its pools are hidden");
  }
  return spec;
}

}  // namespace

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
  if (!all_finite(logits_)) throw util::Exception("policybot: the model gave a non-finite logit");
  return size_t(std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
}

}  // namespace scribblez
