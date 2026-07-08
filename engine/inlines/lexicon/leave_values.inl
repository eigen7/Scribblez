#include "lexicon/leave_values.h"

namespace scribblez {

inline uint32_t LeaveValues::klv_step(uint32_t arc_list, uint8_t code, uint32_t* index) const {
  if (arc_list == 0) return 0;
  for (uint32_t i = arc_list;; ++i) {
    const uint32_t e = nodes_[i];
    const uint32_t t = e >> kTileShift;
    if (t == code) return i;
    if (t > code) return 0;
    *index += subtree_words_[i];
    if (e & kIsEndBit) return 0;
  }
}

}  // namespace scribblez
