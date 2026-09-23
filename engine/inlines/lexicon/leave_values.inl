#include "lexicon/leave_values.h"

namespace scribblez {

inline uint32_t LeaveValues::klv_step(uint32_t arc_list, uint8_t code, uint32_t* index) const {
  if (arc_list == 0) return 0;
  for (uint32_t i = arc_list;; ++i) {
    const uint32_t e = nodes_[i];
    const uint32_t t = Dictionary::arc_tile(e);
    if (t == code) return i;
    if (t > code) return 0;  // siblings are sorted by tile
    *index += subtree_words_[i];
    if (e & Dictionary::IS_END_BIT) return 0;
  }
}

}  // namespace scribblez
