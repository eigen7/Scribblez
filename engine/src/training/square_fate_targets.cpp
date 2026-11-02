#include "training/square_fate_targets.h"

#include "encoding/square_fates.h"
#include "util/assert.h"

namespace scribblez {

void OwnershipTarget::encode(const EncodeContext& v, float* out) {
  RELEASE_ASSERT(v.square_fates != nullptr);
  for (int sq = 0; sq < kBoardCells; ++sq) out[sq] = float(int(v.square_fates->fate[sq]));
}

void SquarePointsTarget::encode(const EncodeContext& v, float* out) {
  RELEASE_ASSERT(v.square_fates != nullptr);
  for (int sq = 0; sq < kBoardCells; ++sq) out[sq] = float(v.square_fates->points[sq]);
}

}  // namespace scribblez
