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

// Visiting siblings in order and an arc's own word before its children
// reproduces word-golib's GetWordIndexOf numbering, which is how values_ is
// ordered.
template <typename F>
void LeaveValues::walk(uint32_t node, Rack& acc, uint32_t& index, F& f) const {
  while (node != 0) {
    const uint32_t entry = nodes_[node];
    const Tile tile = klv_tile(Dictionary::arc_tile(entry));
    acc.add(tile);
    if (entry & Dictionary::ACCEPTS_BIT) f(static_cast<const Rack&>(acc), klv_value_at(index++));
    const uint32_t child = entry & Dictionary::ARC_MASK;
    if (child != 0) walk(child, acc, index, f);
    acc.remove(tile);
    if (entry & Dictionary::IS_END_BIT) break;
    ++node;
  }
}

template <typename F>
void LeaveValues::for_each(F&& f) const {
  Rack acc;
  uint32_t index = 0;
  walk(root_arc_list_, acc, index, f);
}

}  // namespace scribblez
