#pragma once

#include "scribblez/rack.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace scribblez {

// Stores leave values loaded from a Kurnia Leave Value (.klv2) file.
//
// At load time the embedded KWG is walked once to enumerate every leave (in
// word-index order) into a hash map keyed on the canonical Rack representation
// of the leave. Thereafter lookup() is a single O(1) hash probe.
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

 private:
  std::unordered_map<Rack, float> values_by_leave_;  // leave -> value
};

}  // namespace scribblez
