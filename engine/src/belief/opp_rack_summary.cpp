#include "belief/opp_rack_summary.h"

#include "game/rack.h"
#include "util/assert.h"

#include <algorithm>

namespace scribblez::belief {

namespace {

// The chance that `draws` tiles drawn without replacement from `pool` tiles,
// `copies` of them of one kind, miss that kind: C(pool - copies, draws) /
// C(pool, draws).
double miss_chance(int pool, int copies, int draws) {
  double p = 1;
  for (int i = 0; i < draws; ++i) p *= double(pool - copies - i) / double(pool - i);
  return std::max(0.0, p);
}

}  // namespace

OppRackSummary summarize_opp_rack(const TileCounts& unseen, const LeaveSampler* leave) {
  RELEASE_ASSERT(unseen.size() > RACK_SIZE, "a summary needs a non-empty bag");
  OppRackSummary s;
  const int kept = leave ? leave->length() : 0;
  const LeaveMoments m = leave ? leave->moments() : LeaveMoments{};
  // A leave lacking a kind leaves all its unseen copies to the refill, so the
  // rack lacks the kind iff both the leave and the refill do.
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    const double leave_lacks = 1 - m.presence[t.index()];
    const double refill_lacks =
      miss_chance(unseen.size() - kept, unseen.count(t), RACK_SIZE - kept);
    s.presence[t.index()] = float(1 - leave_lacks * refill_lacks);
  }
  s.modelled = leave != nullptr;
  s.leave_value_mean = float(m.value_mean);
  s.leave_value_sd = float(m.value_sd);
  return s;
}

}  // namespace scribblez::belief
