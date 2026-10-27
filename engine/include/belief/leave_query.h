#pragma once

// One query of the leave inference model: an input row and what the belief it
// yields is drawn over. A .slog position's comes from its training row
// (belief/slog_leave_beliefs.h); the opponent's view of a chooser's candidate
// is built live (belief/reply_query.h).

#include "belief/leave_params.h"
#include "belief/leave_sampler.h"
#include "game/tile_counts.h"
#include "lexicon/leave_value_arrays.h"
#include "training/leave_inference_task.h"

#include <memory>
#include <span>
#include <vector>

namespace scribblez::belief {

struct LeaveQuery {
  std::vector<float> input;  // the model's input row
  int length = 0;            // of the leave
  TileCounts leave_pool;     // what the leave is drawn from
  LeaveRowCause cause = LeaveRowCause::kValid;
  int bag = 0;  // tiles in the bag at the observer's turn

  // Whether the model has anything to say. Not at turn 0, after a pass or for
  // an empty leave (the rows training masks), and not with the bag empty,
  // where the opponent holds every unseen tile whatever they kept.
  bool modelled() const { return cause == LeaveRowCause::kValid && bag > 0; }
};

// The model's parameters for each query, from one `predict` call over the
// modelled queries' rows together; an unmodelled query's entry is unused.
std::vector<LeaveParams> predict_beliefs(std::span<const LeaveQuery> queries,
                                         const LeaveParamsFn& predict);

// The query's belief under `params`, null where it is not modelled. `table`
// must outlive the sampler.
std::shared_ptr<const LeaveSampler> belief_of(const LeaveQuery& q, const LeaveParams& params,
                                              const LeaveValueArrays& table);

// Each query's belief: predict_beliefs, then the samplers' table passes in
// parallel (util/work_pool.h).
std::vector<std::shared_ptr<const LeaveSampler>> beliefs_of(std::span<const LeaveQuery> queries,
                                                            const LeaveParamsFn& predict,
                                                            const LeaveValueArrays& table);

}  // namespace scribblez::belief
