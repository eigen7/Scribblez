#include "agent/candidate_evaluator.h"

#include "agent/agent.h"
#include "encoding/input_encoder.h"
#include "util/exception.h"

#include <algorithm>
#include <string>

namespace scribblez {

float objective_value(const float* wld_row, const float* score_diff_row, EvalObjective objective) {
  if (objective == EvalObjective::kScoreDiff) return score_diff_row[0];
  return wld_row[0] + 0.5f * wld_row[1];  // P(win) + 0.5 * P(draw)
}

EvalObjective parse_eval_objective(const std::string& name, const std::string& flag) {
  if (name == "scorediff") return EvalObjective::kScoreDiff;
  if (name == "winprob") return EvalObjective::kWinProb;
  throw util::CleanException("{} must be 'scorediff' or 'winprob' (got '{}')", flag, name);
}

InputEncodingSpec derive_input_spec(const Dictionary& dict, const nn::ServedModelInputs& model,
                                    const std::string& who) {
  const InputEncodingSpec spec{&dict, model.opp_leave_input()};
  if (model.spatial_planes() != spatial_planes() || model.scalar_floats() != scalar_floats(spec)) {
    throw util::Exception(
      "{}: the model declares an input arm (opp_leave={}) whose layout does not match the input "
      "widths it accepts",
      who, model.opp_leave_input());
  }
  return spec;
}

CandidateEvaluator::CandidateEvaluator(const Dictionary& dict,
                                       std::shared_ptr<nn::PositionEvalService> service,
                                       int max_batch)
    : max_batch_(max_batch),
      service_(std::move(service)),
      spec_(derive_input_spec(dict, *service_, "candidate evaluator")),
      encoder_(spec_) {
  if (max_batch_ < 1) throw util::Exception("candidate evaluator: max batch must be >= 1");
  input_buf_.resize(size_t(max_batch_) * input_floats(spec_));
}

void CandidateEvaluator::begin_game(const BeginGameRequest& req) {
  encoder_ = GameStateEncoder(spec_, req.initial_scores);
}

void CandidateEvaluator::observe_move(const Move& move) { encoder_.apply_move(move); }

void CandidateEvaluator::encode_candidate(const Move& mv, const Rack& my_rack, int my_seat,
                                          const Rack& opp_leave, float* dst) const {
  // The cross-check input planes read the board's move-generation caches;
  // building them here (a no-op once valid) keeps them lexicon-accurate on the
  // copy, which then updates them incrementally when the candidate is applied.
  encoder_.board().ensure_movegen_caches(*spec_.dict);
  encode_post_move_row(encoder_, my_seat, my_rack, mv, opp_leave, dst);
}

void CandidateEvaluator::evaluate(const MoveRequest& req, const std::vector<Move>& candidates,
                                  const std::vector<int>& idx, int k) {
  wld_buf_.resize(size_t(k) * nn::WldOutput::kRowElems);
  score_diff_buf_.resize(size_t(k) * nn::ScoreDiffOutput::kRowElems);

  // The encoder's active player is the owning agent's seat (it has observed
  // every prior move). Each candidate is scored from a post-move copy of the
  // encoder, in chunks no larger than the model's batch limit.
  const int my_seat = encoder_.active_player();
  int done = 0;
  while (done < k) {
    const int chunk = std::min(max_batch_, k - done);
    for (int j = 0; j < chunk; ++j) {
      const Move& mv = candidates[size_t(idx[done + j])];
      encode_candidate(mv, req.my_rack, my_seat, req.opp_rack,
                       input_buf_.data() + size_t(j) * input_floats(spec_));
    }
    float* const head_out[] = {
      wld_buf_.data() + size_t(done) * nn::WldOutput::kRowElems,
      score_diff_buf_.data() + size_t(done) * nn::ScoreDiffOutput::kRowElems};
    service_->evaluate({input_buf_.data(), chunk}, head_out);
    done += chunk;
  }
}

}  // namespace scribblez
