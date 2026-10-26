#pragma once

// The opponent's leave beliefs at positions of one .slog, from the leave
// inference model (docs/plans/deep_bot_v0.md, "Workload 2"): what the survey
// tool's --leave-inference sims under. Each position's model input is the
// training row itself, decoded by BlockDecoder's kLeaveInference task from the
// .slog and its .forgone sidecar, so the model sees exactly the rows it was
// trained on. Of the row's target, only what the observer can see is read
// (the leave's length and pool, the cause, the bag); the true leave never is.

#include "belief/leave_params.h"
#include "belief/leave_query.h"
#include "belief/leave_sampler.h"
#include "data/forgone_play_log.h"
#include "data/slog_sampling.h"
#include "game/tile_counts.h"
#include "lexicon/leave_value_arrays.h"
#include "training/leave_inference_task.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

namespace belief {

class SlogLeaveBeliefs {
 public:
  // Reads the .forgone sidecar beside `slog_path`, throwing util::Exception
  // as ForgoneSidecar does. `slog` (the loaded .slog), `dict` and `table`
  // must outlive this object.
  SlogLeaveBeliefs(const std::vector<char>& slog, const std::string& slog_path,
                   const Dictionary& dict, const LeaveValueArrays& table, LeaveParamsFn predict);

  // The decoded row at `at`, a training-eligible turn.
  LeaveQuery query(const binlog::GamePositionIndex& at) const;

  // The belief at `at`, or null where the leave is not modelled. Thread-safe.
  std::shared_ptr<const LeaveSampler> belief(const binlog::GamePositionIndex& at) const;

 private:
  const std::vector<char>& slog_;
  std::string slog_path_;
  const Dictionary& dict_;
  const LeaveValueArrays& table_;
  LeaveParamsFn predict_;
  ForgoneSidecar forgone_;
};

}  // namespace belief
}  // namespace scribblez
