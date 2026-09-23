#pragma once

// The TensorRT-backed MoveProposalService: one consumer's view of the shared
// MoveProposalNets. Each game thread's agent owns a session, which holds the
// cache for the position it last encoded and decodes the nets' raw outputs.
//
// Not thread-safe. Its memory is the retained cache, dominated by the
// predicted planes (~47 KB per candidate).

#include "agent/move_proposal_nets.h"
#include "agent/move_proposal_service.h"

#include <memory>

namespace scribblez {
namespace agent {

class MoveProposalSession : public MoveProposalService {
 public:
  // Over an already-loaded, shared pair (MoveProposalNets::create()).
  explicit MoveProposalSession(std::shared_ptr<MoveProposalNets> nets);

  bool opp_leave_input() const override { return nets_->opp_leave_input(); }
  int spatial_planes() const override { return nets_->spatial_planes(); }
  int scalar_floats() const override { return nets_->scalar_floats(); }

  const MoveProposalPredictions& encode(const float* board_row,
                                        const move_set::MoveFeatureArrays& moves) override;
  const MoveProposalPredictions& condition(const EvidenceSet& evidence) override;

  // For tests and tools that inspect what the evidence is gathered from.
  const MoveProposalCache& cache() const { return cache_; }
  const MoveProposalNets& nets() const { return *nets_; }

 private:
  std::shared_ptr<MoveProposalNets> nets_;
  MoveProposalCache cache_;
  MoveProposalPredictions plain_;
  MoveProposalPredictions conditioned_;
};

}  // namespace agent
}  // namespace scribblez
