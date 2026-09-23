#pragma once

#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace scribblez {

// Leave values from a Kurnia Leave Value (.klv2) file. Read-only after load()
// and safe to query concurrently.
//
// Two access paths. lookup() is one hash probe on a map built at load time.
// The cursor API walks the embedded KWG one tile at a time, so a caller that
// enumerates sub-racks depth-first prices each one incrementally instead of
// hashing it from scratch.
class LeaveValues {
 public:
  // Throws util::Exception on I/O failure.
  static LeaveValues load(const std::string& path);

  // 0.0 for an empty leave or one absent from the table.
  float lookup(const Rack& leave) const;

  // ---- Incremental cursor ----
  // The KWG is minimized, so suffix nodes are shared and a node can't carry a
  // value. Instead a leave's value is indexed by its word index: its position
  // in a pre-order enumeration of all leaves. The cursor accumulates that
  // index while following the leave's tiles in ascending klv_code order:
  //   1. klv_step() matches one tile, adding the word counts of the siblings
  //      it skips.
  //   2. If the matched arc accepts, the leave spelled so far is a table
  //      entry and its value is klv_value_at(index).
  //   3. To continue, add 1 if the arc accepts, then descend via klv_next().

  uint32_t klv_root() const { return root_arc_list_; }

  // A..Z -> 1..26; the blank is code 0.
  static uint8_t klv_code(Tile letter) { return letter.index() + 1; }

  // The matched arc, or 0 if `code` has no arc in `arc_list`.
  uint32_t klv_step(uint32_t arc_list, uint8_t code, uint32_t* index) const;

  bool klv_accepts(uint32_t arc) const { return (nodes_[arc] & Dictionary::ACCEPTS_BIT) != 0; }
  uint32_t klv_next(uint32_t arc) const { return nodes_[arc] & Dictionary::ARC_MASK; }
  float klv_value_at(uint32_t index) const {
    return index < values_.size() ? values_[index] : 0.0f;
  }

 private:
  std::unordered_map<Rack, float> values_by_leave_;  // leave -> value
  std::vector<uint32_t> nodes_;                      // KWG arc nodes
  std::vector<float> values_;                        // leave values, indexed by word index
  std::vector<uint32_t> subtree_words_;              // per arc: words at or below it
  uint32_t root_arc_list_ = 0;
};

}  // namespace scribblez

#include "inlines/lexicon/leave_values.inl"
