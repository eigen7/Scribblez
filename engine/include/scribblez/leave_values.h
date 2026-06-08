#pragma once

#include "scribblez/tile_counts.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace scribblez {

// Stores leave values loaded from a Kurnia Leave Value (.klv2) file.
// The file contains a KWG (used only for leave enumeration during load) and
// a parallel float32 array indexed by word-order position in that KWG.
//
// After construction the object is read-only and safe to query from multiple
// threads concurrently.
class LeaveValues {
 public:
  // Load from a .klv2 file. Throws std::runtime_error on I/O failure.
  static LeaveValues load(const std::string& path);

  // Leave equity for a given rack leave (the tiles remaining after a play).
  // Returns 0.0 for an empty leave or a leave not in the table.
  float lookup(const TileCounts& leave) const;

 private:
  // KWG bit-field constants (same layout as Dictionary / word-golib KWG).
  static constexpr uint32_t kArcMask = 0x003fffffu;
  static constexpr uint32_t kIsEndBit = 0x00400000u;
  static constexpr uint32_t kAcceptsBit = 0x00800000u;
  static constexpr uint32_t kTileShift = 24u;

  std::vector<uint32_t> nodes_;       // KWG arc nodes from the KLV file
  std::vector<int32_t> word_counts_;  // subtree word-count table, built during load
  std::vector<float> values_;         // leave values indexed by word-order position

  // Build word_counts_ from nodes_ (must be called after nodes_ is populated).
  void build_word_counts();

  // Recursive helper used by build_word_counts().
  int32_t count_words_at(uint32_t node);

  // Walk the KWG to find the word-index of a leave (sorted tile sequence).
  // Returns -1 if the leave is not present.
  int32_t word_index_of(const TileCounts& leave) const;
};

}  // namespace scribblez
