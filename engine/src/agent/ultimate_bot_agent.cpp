#include "agent/ultimate_bot_agent.h"

#include "encoding/input_encoder.h"
#include "nn/model_specs.h"
#include "training/evidence_trajectory_select.h"
#include "util/exception.h"
#include "util/math.h"

#include <limits>
#include <optional>

// The production path -- the only members that reference the concrete
// TensorRT-backed session and nets (and thus pull in CUDA / TensorRT) -- lives
// in ultimate_bot_agent_factory.cpp, so this translation unit, and the agent's
// unit tests that compile it, carry no GPU dependency.

namespace scribblez {

namespace {

// The dictionary reference the members that need it read before any
// constructor body could check it.
const Dictionary& require_dict(const Dictionary* dict) {
  if (dict == nullptr) throw util::Exception("ultimatebot: a dictionary is required");
  return *dict;
}

}  // namespace

UltimateBotAgent::UltimateBotAgent(const Params& params,
                                   std::unique_ptr<agent::MoveProposalService> service,
                                   std::shared_ptr<nn::PositionEvalService> leaf_service)
    : Agent(params.thread_id, params.name),
      max_sims_(params.max_sims),
      policy_(params.gain_threshold),
      seed_(params.seed),
      service_(std::move(service)),
      spec_(derive_input_spec(require_dict(params.dict), *service_, "ultimatebot")),
      encoder_(spec_),
      leaf_service_(std::move(leaf_service)),
      runner_(*params.dict,
              make_runner_params(params.sim, params.sim_horizon, leaf_service_.get())),
      endgame_(params.thread_id, params.endgame) {
  validate(params);
  // validate() bounded the budget by the padded width every step graph shares;
  // the service's own width can only be narrower (a scripted stub's, or a
  // trained-width stamp to come).
  if (params.max_sims > service_->max_evidence()) {
    throw util::CleanException("ultimatebot: --max-sims must be <= the model's evidence width {}",
                               service_->max_evidence());
  }
  board_row_.resize(size_t(input_floats(spec_)));
}

void UltimateBotAgent::validate(const Params& params) {
  if (params.max_sims < 1) throw util::CleanException("ultimatebot: --max-sims must be >= 1");
  // The step graph pads its evidence to a fixed width; a budget past it could
  // never be staged.
  if (params.max_sims > nn::kMaxEvidence) {
    throw util::CleanException("ultimatebot: --max-sims must be <= the evidence width {}",
                               nn::kMaxEvidence);
  }
  if (params.gain_threshold < 0.0f) {
    throw util::CleanException("ultimatebot: --gain-threshold must be >= 0");
  }
  SimRunner::validate(params.sim);
  // The horizon lower bound, checked early (the factory calls validate()
  // before loading the model). The flag pairing against --leaf-model is the
  // factory's, which alone knows whether a leaf path was given.
  SimRunner::validate_min_horizon("ultimatebot", params.sim_horizon);
}

uint64_t UltimateBotAgent::sim_seed(int ply) const {
  return util::splitmix64(seed_ ^ util::splitmix64(uint64_t(ply)));
}

void UltimateBotAgent::begin_game(const BeginGameRequest& req) {
  encoder_ = GameStateEncoder(spec_, req.initial_scores);
  endgame_.begin_game();
  ply_ = 0;
}

void UltimateBotAgent::observe_move(const Move& move) {
  encoder_.apply_move(move);
  endgame_.observe_move(move);
  ++ply_;
}

void UltimateBotAgent::encode_board_row(const MoveRequest& req, float* dst) const {
  // The cross-check input planes read the board's move-generation caches;
  // building them here (a no-op once valid) keeps them lexicon-accurate.
  encoder_.board().ensure_movegen_caches(*spec_.dict);
  // The encoder's active player is this agent's own seat: it has observed every
  // prior move, and this is its turn.
  const int me = encoder_.active_player();
  if (spec_.opp_leave_input) {
    encoder_.encode_input(me, req.my_rack, req.opp_rack, dst);
  } else {
    encoder_.encode_input(me, req.my_rack, dst);
  }
}

MoveDecision UltimateBotAgent::make_move(const MoveRequest& req) {
  // The endgame belongs to the exact solver, which needs no candidates of ours.
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;

  const std::vector<Move> candidates = equity_top_k(req, std::numeric_limits<int>::max());
  // A bag-empty turn the solver declined: the value model is out of its
  // training regime and rollouts have no bag to draw the opponent's
  // replenishments from, so play the static-equity move -- which is what
  // equity_top_k already ranked first.
  if (req.bag_size == 0 || candidates.size() == 1) return candidates.front();
  // A budget of one is the anchor alone, and a lone sim decides nothing: play
  // it without rollouts or a model pass (the greedy agent's move).
  if (max_sims_ == 1) return candidates[evidence::anchor_index(candidates)];

  encode_candidates(req, candidates);

  const SimPosition pos = sim_position_from(req);
  agent::SimRunnerCandidateSimmer simmer(runner_, pos, sim_seed(ply_));
  const agent::EvidenceSet evidence =
    agent::run_evidence_loop(candidates, *service_, simmer, policy_, max_sims_);
  // Win rate is the objective the gain head is trained in, so the pick and the
  // stopping rule agree; this agent has no spread objective. Ties in the
  // observations go to the earlier sim -- the anchor first, then the earlier
  // pick, this agent's own ordering.
  return evidence
    .moves[size_t(best_observation_index(evidence.observations, SimObjective::kWinRate))];
}

void UltimateBotAgent::encode_candidates(const MoveRequest& req,
                                         const std::vector<Move>& candidates) {
  encode_board_row(req, board_row_.data());
  // The differential the moves resolve is read off the same mirrored encoder
  // that wrote the board row's score-diff feature, so a candidate's resultant
  // differential is exactly that feature plus the move's score -- the plain sum
  // the two representations were designed to share (input_encoder.h).
  const int me = encoder_.active_player();
  move_features_.encode(candidates.data(), int(candidates.size()),
                        encoder_.score(me) - encoder_.score(1 - me));
  service_->encode(board_row_.data(), move_features_);
}

}  // namespace scribblez
