#pragma once

#include "scribblez/rack.h"
#include "scribblez/tile.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace scribblez {

// Stores leave values loaded from a Kurnia Leave Value (.klv2) file.
//
// At load time the embedded KWG is walked once to enumerate every leave (in
// word-index order) into a hash map keyed on the canonical Rack representation
// of the leave. Thereafter lookup() is a single O(1) hash probe.
//
// The KWG itself (arc nodes plus a per-node value array) is retained so callers
// can also walk leaves incrementally with the cursor API below -- following one
// tile per arc instead of hashing a whole leave. This is what lets a subrack
// enumeration price every leave in a single DFS rather than one hash per subrack.
//
// After construction the object is read-only and safe to query from multiple
// threads concurrently.
class LeaveValues {
 public:
  // Load from a .klv2 file. Throws std::runtime_error on I/O failure.
  static LeaveValues load(const std::string& path);

  // Leave equity for a given rack leave (the tiles remaining after a play).
  // Returns 0.0 for an empty leave or a leave not in the table.
  float lookup(const Rack& leave) const;

  // ---- Incremental cursor over the leave KWG ----------------------------
  // The KWG is a minimized DAWG, so a leave's value is keyed by its word index
  // (its pre-order position over the tree expansion), not by the node it ends on
  // (suffix nodes are shared). The cursor follows a leave's tiles in ascending
  // KLV code order (blank = code 0, A..Z = codes 1..26), accumulating that index:
  // klv_step matches one tile and adds the word counts of the earlier siblings it
  // skipped; after matching, advance past the arc's own word (if accepting) and
  // descend with klv_next. The leave's value is klv_value_at(index) iff the arc it
  // ends on is klv_accepts.

  // The arc-list index of the root's children (where the first tile is matched).
  uint32_t klv_root() const { return root_arc_list_; }

  // KLV code for a board letter (A..Z -> 1..26); the blank is code 0.
  static uint8_t klv_code(Tile letter) { return static_cast<uint8_t>(letter.index() + 1); }

  // Match one tile `code` in sibling list `arc_list`, adding the skipped earlier
  // siblings' subtree word counts to *index. Returns the matched arc, or 0 if the
  // tile is absent (siblings are sorted, so a larger tile means absent).
  uint32_t klv_step(uint32_t arc_list, uint8_t code, uint32_t* index) const {
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
