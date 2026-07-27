#pragma once

#include "game/rack.h"
#include "game/tile.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace scribblez {

// Leave values loaded from a Kurnia Leave Value (.klv2) file. Read-only after
// construction and safe to query concurrently.
//
// Loading walks the embedded KWG once to enumerate every leave into a hash map
// keyed on the canonical Rack, making lookup() a single probe. The KWG itself
// is retained for the cursor API below, which follows one tile per arc instead
// of hashing a whole leave -- what lets a subrack enumeration price every leave
// in a single DFS rather than one hash per subrack.
class LeaveValues {
 public:
  // Throws std::runtime_error on I/O failure.
  static LeaveValues load(const std::string& path);

  // 0.0 for an empty leave or one absent from the table.
  float lookup(const Rack& leave) const;

  // ---- Incremental cursor over the leave KWG ----------------------------
  // The KWG is a minimized DAWG, so a leave's value is keyed by its word index
  // (its pre-order position over the tree expansion) rather than by the node it
  // ends on, suffix nodes being shared. The cursor follows a leave's tiles in
  // ascending KLV code order, accumulating that index: klv_step matches one tile
  // and adds the word counts of the earlier siblings it skipped; after matching,
  // advance past the arc's own word (if accepting) and descend with klv_next.
  // The leave's value is klv_value_at(index) iff the arc it ends on accepts.

  uint32_t klv_root() const { return root_arc_list_; }

  // A..Z -> 1..26; the blank is code 0.
  static uint8_t klv_code(Tile letter) { return static_cast<uint8_t>(letter.index() + 1); }

  // Returns the matched arc, or 0 if the tile is absent -- siblings are sorted,
  // so a larger tile means absent.
  uint32_t klv_step(uint32_t arc_list, uint8_t code, uint32_t* index) const;

  bool klv_accepts(uint32_t arc) const { return (nodes_[arc] & kAcceptsBit) != 0; }
  uint32_t klv_next(uint32_t arc) const { return nodes_[arc] & kArcMask; }
  float klv_value_at(uint32_t index) const {
    return index < values_.size() ? values_[index] : 0.0f;
  }

 private:
  static constexpr uint32_t kArcMask = 0x003fffffu;
  static constexpr uint32_t kIsEndBit = 0x00400000u;
  static constexpr uint32_t kAcceptsBit = 0x00800000u;
  static constexpr uint32_t kTileShift = 24u;

  std::unordered_map<Rack, float> values_by_leave_;  // leave -> value
  std::vector<uint32_t> nodes_;                      // KWG arc nodes
  std::vector<float> values_;                        // leave values, indexed by word index
  std::vector<uint32_t> subtree_words_;              // per-arc word count of its subtree
  uint32_t root_arc_list_ = 0;
};

}  // namespace scribblez

#include "inlines/lexicon/leave_values.inl"
