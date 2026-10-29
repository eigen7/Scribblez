#pragma once

// The policy-set model's pool labels from rollout batches
// (docs/plans/deep_bot_v0.md, "Data: labeled pools"): every candidate of a
// position simmed by a RolloutValues backend, raced in rounds, and reduced to
// the pool record's labels (sim/pool_labels.h).
//
// A round is a slice of rollout indices for every candidate still in the
// race; the race rule runs between rounds, at the CPU simmer's checkpoints
// (sim/candidate_race.h). The pass is never stopped: its win probability is
// the baseline of the pool's stakes (the best move's over the pass's, which
// the policy-set trainer weights its loss by), so it runs every rollout. Many positions are in
// flight at once, and each launch takes the pending rounds of as many as fill it, so the device
// stays busy while each race narrows. A task's outcome depends only on its start and rollout index,
// so the labels do not depend on how rounds share launches.

#include "data/pool_log.h"
#include "sim/candidate_race.h"
#include "sim/hasty_rollout_task.h"
#include "sim/rollout_values.h"

#include <cstddef>
#include <deque>
#include <functional>
#include <optional>
#include <vector>

namespace scribblez {

// One position to label. `record` arrives with everything but its labels and
// rollout values; `starts` is parallel to record.moves, each start's group 0
// being `group`.
struct PoolJob {
  PoolPosition record;
  RolloutGroup group;
  std::vector<RolloutStart> starts;
};

class PoolLabeler {
 public:
  struct Params {
    int rollouts = 100;  // per candidate, the budget
    bool race = true;
    double race_sigmas = 3.0;
    // Tasks a launch fills toward; positions join the flight until their
    // pending rounds reach it.
    int launch_tasks = 32768;
    int max_in_flight = 256;  // positions
  };

  // What run() did, for the tool's report.
  struct Stats {
    long long launches = 0;
    long long tasks = 0;
    long long full_tasks = 0;  // what an unraced sim would have run
  };

  PoolLabeler(RolloutValues& values, const Params& params);

  // Labels the jobs `next` hands out until it returns nullopt, calling `done`
  // with each finished record in the order the jobs came.
  void run(const std::function<std::optional<PoolJob>()>& next,
           const std::function<void(PoolPosition&&)>& done);

  const Stats& stats() const { return stats_; }

 private:
  // A position in flight: its job, its candidates' rollouts so far, and the
  // round running.
  struct Flight {
    size_t order = 0;  // the job's place in pull order
    PoolJob job;
    CandidateRollouts rollouts;
    std::vector<size_t> alive;
    std::vector<char> keep;  // parallel to rollouts: candidates the race never stops
    size_t round = 0;        // index into checkpoints_ of the round running
    int done = 0;            // rollouts every alive candidate has
    std::deque<rollout_abi::RolloutTask> pending;  // position = the candidate
    int outstanding = 0;                           // launched, not yet back
    bool finished = false;
  };

  // Pulls jobs until the pending tasks fill a launch, or no job is left.
  void top_up(const std::function<std::optional<PoolJob>()>& next);
  // Queues the next round of `f`'s alive candidates.
  void begin_round(Flight& f) const;
  // One launch over the flight's pending tasks.
  void launch();
  // Races, starts the next round or finishes each position whose round is in.
  void settle();
  void emit(const std::function<void(PoolPosition&&)>& done);

  RolloutValues& values_;
  Params params_;
  std::vector<int> checkpoints_;  // ends of the rounds, the last one `rollouts`
  std::vector<Flight> flight_;
  std::deque<std::optional<PoolPosition>> finished_;  // by job order, from emitted_
  size_t pulled_ = 0;
  size_t emitted_ = 0;
  bool exhausted_ = false;
  Stats stats_;
};

}  // namespace scribblez
