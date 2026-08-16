#include "data/slog_sampling.h"

#include "util/math.h"

namespace scribblez {
namespace binlog {

inline uint64_t position_seed(uint64_t run_seed, uint32_t game_idx, uint32_t turn_idx) {
  return util::splitmix64(run_seed ^ util::splitmix64((uint64_t(game_idx) << 20) | turn_idx));
}

}  // namespace binlog
}  // namespace scribblez
