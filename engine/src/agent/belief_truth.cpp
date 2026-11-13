#include "agent/belief_truth.h"

#include "belief/leave_params.h"
#include "belief/leave_query.h"
#include "belief/leave_sampler.h"
#include "util/assert.h"
#include "util/work_pool.h"

#include <memory>
#include <vector>

namespace scribblez {

namespace {

// One rack's log p(the kept leave) under the model's belief and under none.
struct RackLogProbs {
  bool modelled = false;
  double informed = 0;
  double prior = 0;
};

RackLogProbs rack_log_probs(const BobBeliefs& beliefs, size_t rack, const Rack& kept,
                            const LeaveValueArrays& table) {
  const std::shared_ptr<const belief::LeaveSampler> informed = beliefs.sampler(rack, table);
  if (!informed) return {};
  const std::shared_ptr<const belief::LeaveSampler> prior =
    belief::belief_of(beliefs.queries[rack], belief::LeaveParams{}, table);
  const size_t row = informed->row_of(kept);
  RELEASE_ASSERT(row < informed->rows(), "the kept leave is not a leave of the query's length");
  return {.modelled = true, .informed = informed->log_prob(row), .prior = prior->log_prob(row)};
}

}  // namespace

BeliefTruth belief_truth(const AliceTurn& turn, size_t c, const BeliefModel& model,
                         const Dictionary& dict) {
  const BobBeliefs beliefs =
    bob_beliefs(turn.position, turn.candidates[c], turn.bob_racks, model, dict);
  if (beliefs.queries.empty()) return {};
  std::vector<RackLogProbs> by_rack(turn.bob_racks.size());
  util::parallel_for(by_rack.size(), [&](size_t i) {
    by_rack[i] = rack_log_probs(beliefs, i, turn.applied[c].leave, *model.table);
  });
  BeliefTruth out;
  for (const RackLogProbs& r : by_rack) {
    if (!r.modelled) continue;
    ++out.racks;
    out.log_prob += r.informed;
    out.prior_log_prob += r.prior;
  }
  if (out.racks > 0) {
    out.log_prob /= out.racks;
    out.prior_log_prob /= out.racks;
  }
  return out;
}

}  // namespace scribblez
