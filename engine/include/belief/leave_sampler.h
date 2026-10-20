#pragma once

// Sampling the opponent's leave from the leave inference model's belief
// (docs/plans/deep_bot.md, Algorithm 1), on the CPU. One query is one
// position's LeaveParams over the leaves of one length: a pass over
// LeaveValueArrays gives every row's log weight, the CDF in table order, and
// log Z. The trainer's loss (py/scribblez/leave_inference/loss.py) runs the
// same pass over the same rows, which the parity test checks.
//
// Paired sampling: a caller that passes the same quantile u to two samplers
// over similar beliefs gets similar leaves, since both CDFs walk the rows in
// one order. Sims that share u across candidates therefore compare the
// candidates on matched opponent racks.
//
// The sampler draws only the leave. The opponent's refill, a uniform draw
// from the unseen pool less the leave, is the sim's: SimRunner already seats
// a known leave and draws the rest from the bag.

#include "belief/leave_params.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "lexicon/leave_value_arrays.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace scribblez::belief {

class LeaveSampler {
 public:
  // The belief `params` over the length-`length` leaves drawable from
  // `leave_pool` (encoding/leave_inference_input_encoder.h's leave_pool).
  // Reads `table` in place, so it must outlive the sampler. Throws
  // util::Exception for a length outside 1..kLeaveArraysMaxLength or a pool
  // holding no such leave.
  LeaveSampler(const LeaveParams& params, int length, const TileCounts& leave_pool,
               const LeaveValueArrays& table);

  int length() const { return length_; }
  double log_z() const { return log_z_; }

  // The row at quantile `u` in [0, 1) of the CDF. Never a row the pool cannot
  // supply.
  size_t sample(double u) const;

  // log p of the leave at `row`; -infinity for a row the pool cannot supply.
  double log_prob(size_t row) const;

  // The leave at `row`.
  Rack leave(size_t row) const;

 private:
  // The unnormalized log weight of `row`.
  double log_weight(size_t row) const;
  void accumulate();

  int length_;
  // The table's length-`length_` rows.
  std::array<std::span<const uint8_t>, kLeaveArraysMaxLength> codes_;  // [slot]
  std::span<const float> values_;
  float a_;
  float b_;
  // Per tile code: that copy's step of the pool's log prior, plus theta on a
  // letter's first copy. -infinity for a copy the pool cannot supply.
  std::array<double, LeaveValueArrays::kCodes> scores_;
  // Inclusive running sums of exp(log weight - max_log_weight_), by row.
  std::vector<double> cdf_;
  double max_log_weight_ = 0;
  double log_z_ = 0;
};

}  // namespace scribblez::belief
