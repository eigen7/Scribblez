#pragma once

// One consumer's view of the shared move proposal engines: the TensorRT-backed
// MoveProposalService. A session holds ONE position at a time -- the cache
// graph's retained outputs for the candidate set it last encoded -- and turns
// the nets' raw outputs into the decoded predictions the loop reads (softmax
// on the WLD logits; score_diff and gain pass through). Every game thread's
// agent owns a session; all of them drive one MoveProposalNets, which
// serializes the GPU work and shares nothing per position.
//
// Not thread-safe: a session is one consumer's, driven from one thread at a
// time. Its cost per position is the retained cache, dominated by the single
// copy of the M x (4 * kSlotsPerCell) x 225 predicted planes the evidence
// staging gathers from (~47 KB per candidate).

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

  // The retained position (raw), for a test or tool checking what the step
  // graph's evidence is gathered from.
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
