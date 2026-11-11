#include "sim/pool_labeler.h"

#include "sim/pool_labels.h"
#include "sim/sim_maps.h"
#include "util/exception.h"

#include <algorithm>
#include <numeric>
#include <utility>

namespace scribblez {

namespace {

Rollout to_rollout(const RolloutOutcome& o) {
  Rollout r;
  r.p_win = o.wld[0];
  r.p_draw = o.wld[1];
  r.p_loss = o.wld[2];
  r.delta = o.result.diff;
  r.delta_sq = double(o.result.diff) * o.result.diff;
  return r;
}

void check_rollouts(const PoolJob& job, const PoolLabeler::Params& params) {
  if (job.rollouts.empty()) return;
  if (params.race) throw util::Exception("PoolLabeler: per-start rollouts need an unraced labeler");
  if (job.rollouts.size() != job.starts.size())
    throw util::Exception("PoolLabeler: {} rollout counts for {} starts", job.rollouts.size(),
                          job.starts.size());
  for (const int n : job.rollouts) {
    if (n < 1 || n > params.rollouts)
      throw util::Exception("PoolLabeler: a start's {} rollouts are outside [1, {}]", n,
                            params.rollouts);
  }
}

}  // namespace

PoolLabeler::PoolLabeler(RolloutValues& values, const Params& params)
    : values_(values), params_(params) {
  if (params_.race) checkpoints_ = race_checkpoints(params_.rollouts);
  if (checkpoints_.empty() || checkpoints_.back() != params_.rollouts)
    checkpoints_.push_back(params_.rollouts);
}

void PoolLabeler::run(const std::function<std::optional<PoolJob>()>& next,
                      const std::function<void(PoolPosition&&)>& done) {
  for (;;) {
    top_up(next);
    if (flight_.empty()) break;
    launch();
    settle();
    emit(done);
  }
}

void PoolLabeler::top_up(const std::function<std::optional<PoolJob>()>& next) {
  const auto pending_tasks = [&] {
    size_t n = 0;
    for (const Flight& f : flight_) n += f.pending.size();
    return n;
  };
  while (!exhausted_ && int(flight_.size()) < params_.max_in_flight &&
         pending_tasks() < size_t(params_.launch_tasks)) {
    std::optional<PoolJob> job = next();
    if (!job) {
      exhausted_ = true;
      break;
    }
    check_rollouts(*job, params_);
    Flight f;
    f.order = pulled_++;
    f.job = std::move(*job);
    f.rollouts.resize(f.job.record.moves.size());
    if (params_.sum_maps) f.maps.assign(f.rollouts.size(), rollout_abi::SimMaps{});
    f.alive.resize(f.rollouts.size());
    std::iota(f.alive.begin(), f.alive.end(), size_t(0));
    for (const Move& m : f.job.record.moves) f.keep.push_back(m.type() == MoveType::PASS);
    for (size_t c = 0; c < f.rollouts.size(); ++c) stats_.full_tasks += budget(f, c);
    begin_round(f);
    flight_.push_back(std::move(f));
    finished_.emplace_back();
  }
}

int PoolLabeler::budget(const Flight& f, size_t c) const {
  return f.job.rollouts.empty() ? params_.rollouts : f.job.rollouts[c];
}

void PoolLabeler::begin_round(Flight& f) const {
  for (const size_t c : f.alive) {
    const int upto = std::min(checkpoints_[f.round], budget(f, c));
    f.rollouts[c].resize(size_t(upto));
    for (int i = f.done; i < upto; ++i) f.pending.push_back({uint32_t(c), uint32_t(i)});
  }
}

void PoolLabeler::launch() {
  RolloutBatch batch;
  batch.sum_maps = params_.sum_maps;
  // Where each launched task's outcome goes: the flight and its task; and
  // each batch start's maps: the flight and its candidate.
  std::vector<std::pair<size_t, rollout_abi::RolloutTask>> where;
  std::vector<std::pair<size_t, uint32_t>> start_where;
  for (size_t k = 0; k < flight_.size() && int(batch.tasks.size()) < params_.launch_tasks; ++k) {
    Flight& f = flight_[k];
    if (f.pending.empty()) continue;
    const uint32_t group = uint32_t(batch.groups.size());
    batch.groups.push_back(f.job.group);
    // The batch's start for each of this flight's candidates, added on first use.
    std::vector<int> start_of(f.job.starts.size(), -1);
    while (!f.pending.empty() && int(batch.tasks.size()) < params_.launch_tasks) {
      const rollout_abi::RolloutTask t = f.pending.front();
      f.pending.pop_front();
      int& s = start_of[t.position];
      if (s < 0) {
        s = int(batch.starts.size());
        batch.starts.push_back(f.job.starts[t.position]);
        batch.starts.back().group = group;
        start_where.emplace_back(k, t.position);
      }
      batch.tasks.push_back({uint32_t(s), t.rollout});
      where.emplace_back(k, t);
      ++f.outstanding;
    }
  }
  const BatchValues values = values_.run(batch);
  const std::vector<RolloutOutcome>& outcomes = values.outcomes;
  for (size_t j = 0; j < outcomes.size(); ++j) {
    Flight& f = flight_[where[j].first];
    const rollout_abi::RolloutTask& t = where[j].second;
    f.rollouts[t.position][t.rollout] = to_rollout(outcomes[j]);
    --f.outstanding;
  }
  for (size_t s = 0; s < values.maps.size(); ++s) {
    const auto [k, c] = start_where[s];
    add_sim_maps(flight_[k].maps[c], values.maps[s]);
  }
  ++stats_.launches;
  stats_.tasks += (long long)outcomes.size();
}

void PoolLabeler::settle() {
  for (Flight& f : flight_) {
    if (f.outstanding > 0 || !f.pending.empty()) continue;
    f.done = checkpoints_[f.round];
    if (f.done < params_.rollouts) {
      stop_the_beaten(f.rollouts, params_.race_sigmas, f.keep, &f.alive);
      ++f.round;
      begin_round(f);
      continue;
    }
    f.job.record.labels = pool_labels(f.rollouts);
    f.job.record.rollout_values = rollout_values(f.rollouts);
    f.job.record.maps = std::move(f.maps);
    finished_[f.order - emitted_] = std::move(f.job.record);
    f.finished = true;
  }
  std::erase_if(flight_, [](const Flight& f) { return f.finished; });
}

void PoolLabeler::emit(const std::function<void(PoolPosition&&)>& done) {
  while (!finished_.empty() && finished_.front()) {
    done(std::move(*finished_.front()));
    finished_.pop_front();
    ++emitted_;
  }
}

}  // namespace scribblez
