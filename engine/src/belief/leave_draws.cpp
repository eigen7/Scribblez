#include "belief/leave_draws.h"

#include "game/bag.h"
#include "util/assert.h"

namespace scribblez::belief {

namespace {

// `count` racks of `length` tiles, each drawn from `pool` at random.
std::vector<Rack> random_racks(const TileCounts& pool, int length, int count, uint64_t seed) {
  Bag bag(seed, pool);
  std::vector<Rack> racks(static_cast<size_t>(count));
  for (Rack& rack : racks) {
    for (int i = 0; i < length; ++i) rack.add(bag.draw());
    for (int i = 0; i < rack.size(); ++i) bag.put_back(rack.tiles()[size_t(i)]);
  }
  return racks;
}

}  // namespace

std::vector<Rack> opp_leave_draws(const LeaveQuery& q, const LeaveSampler* belief, int count,
                                  uint64_t seed) {
  if (q.length == 0) return std::vector<Rack>(size_t(count));
  if (q.length > kLeaveArraysMaxLength) return random_racks(q.leave_pool, q.length, count, seed);
  RELEASE_ASSERT(belief != nullptr);
  const double jitter = double(seed >> 11) * 0x1.0p-53;
  std::vector<Rack> draws;
  draws.reserve(size_t(count));
  for (int i = 0; i < count; ++i)
    draws.push_back(belief->leave(belief->sample((double(i) + jitter) / double(count))));
  return draws;
}

}  // namespace scribblez::belief
