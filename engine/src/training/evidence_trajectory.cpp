#include "training/evidence_trajectory.h"

#include "agent/agent.h"
#include "util/exception.h"
#include "util/misc.h"

#include <limits>

namespace scribblez::evidence {

SimRunner::Params sim_params(const TrajectoryOptions& opt, nn::PositionEvalService* leaf) {
  SimRunner::Params p;
  p.rollouts = opt.rollouts;
  p.threads = 1;
  p.horizon_plies = opt.horizon;
  p.leaf_service = leaf;
  return p;
}

void validate(const TrajectoryOptions& opt) {
  SimRunner::validate_min_horizon("evidence trajectory", opt.horizon);
  TrajectoryOptions terminal = opt;  // the leaf pairing is the caller's to check
  terminal.horizon = 0;
  SimRunner::validate(sim_params(terminal, nullptr));
  if (opt.proposals_min < 0) throw util::CleanException("--proposals-min must be >= 0");
  if (opt.proposals_max < opt.proposals_min) {
    throw util::CleanException("--proposals-max must be >= --proposals-min");
  }
  if (opt.temperature <= 0.0) throw util::CleanException("--temperature must be > 0");
}

// --- StudentScorer ---

void StudentScorer::score(const float* board_row, const move_set::MoveFeatureArrays* moves,
                          float* wld_out, float* sd_out) {
  Request req{board_row, moves, wld_out, sd_out};
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.push_back(&req);
  queue_cv_.notify_one();
  done_cv_.wait(lock, [&] { return req.done; });
}

void StudentScorer::run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    queue_cv_.wait(lock, [&] { return !queue_.empty() || stopping_; });
    if (queue_.empty()) return;
    Request* req = queue_.front();
    queue_.pop_front();
    lock.unlock();
    float* const head_out[] = {req->wld_out, req->sd_out};
    service_->evaluate({req->board_row, req->moves}, head_out);
    lock.lock();
    req->done = true;
    done_cv_.notify_all();
  }
}

void StudentScorer::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopping_ = true;
  queue_cv_.notify_all();
}

// --- TrajectoryRunner ---

TrajectoryRunner::TrajectoryRunner(const Dictionary& dict, const InputEncodingSpec& spec,
                                   const TrajectoryOptions& opt, StudentScorer* scorer,
                                   nn::PositionEvalService* leaf)
    : dict_(dict),
      spec_(spec),
      opt_(opt),
      scorer_(scorer),
      runner_(dict, sim_params(opt, leaf)),
      board_row_(size_t(input_floats(spec))) {}

const std::vector<float>& TrajectoryRunner::win_equities(const DecisionPoint& dp,
                                                         const Rack& visible_opp,
                                                         const std::vector<Move>& ranked) {
  const GameStateEncoder& enc = *dp.enc;
  const int mover = dp.pos.mover;
  const int n = ranked.size();
  // The cross-check input planes read the board's move-generation caches;
  // building them here (a no-op once valid) keeps them lexicon-accurate.
  enc.board().ensure_movegen_caches(*spec_.dict);
  if (spec_.opp_leave_input) {
    enc.encode_input(mover, dp.pos.rack, visible_opp, /*apply_flip=*/false, board_row_.data());
  } else {
    enc.encode_input(mover, dp.pos.rack, /*apply_flip=*/false, board_row_.data());
  }
  const int score_diff = enc.score(mover) - enc.score(1 - mover);
  move_features_.encode(ranked.data(), n, score_diff);
  wld_buf_.resize(size_t(n) * nn::WldOutput::kRowElems);
  sd_buf_.resize(size_t(n) * nn::ScoreDiffOutput::kRowElems);
  scorer_->score(board_row_.data(), &move_features_, wld_buf_.data(), sd_buf_.data());
  win_equity_.resize(size_t(n));
  for (int c = 0; c < n; ++c) {
    const float* wld = wld_buf_.data() + size_t(c) * nn::WldOutput::kRowElems;
    win_equity_[size_t(c)] = wld[0] + 0.5f * wld[1];
  }
  return win_equity_;
}

TrajectoryResult TrajectoryRunner::run(const DecisionPoint& dp, uint64_t base_seed) {
  const SimPosition& pos = dp.pos;
  const Rack hidden_opp;
  const Rack& visible_opp = spec_.opp_leave_input ? pos.opp_leave : hidden_opp;
  MoveRequest ranking_req{
    pos.board,  dict_, pos.rack, visible_opp, pos.scores[pos.mover], pos.scores[1 - pos.mover],
    dp.bag_size};
  const std::vector<Move> ranked = equity_top_k(ranking_req, std::numeric_limits<int>::max());
  const std::vector<float>& win_equity = win_equities(dp, visible_opp, ranked);

  TrajectoryResult res;
  res.num_legal_moves = ranked.size();
  std::mt19937_64 rng(util::splitmix64(base_seed ^ 0x7A6A11EC70ull));
  const std::vector<size_t> chosen =
    select_trajectory(ranked, win_equity, opt_, rng, sampler_, &res.uniform_tail);
  res.candidates.reserve(chosen.size());
  for (size_t idx : chosen) res.candidates.push_back(ranked[idx]);
  res.observations = runner_.run(pos, res.candidates, base_seed);
  return res;
}

}  // namespace scribblez::evidence
