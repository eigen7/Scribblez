#pragma once
#include <array>
#include <string>
#include <vector>

#include "scribblez/tile.h"

namespace scribblez {

// A Rack is a multiset of tiles, each tile being a letter A..Z or a blank.
class Rack {
 public:
  Rack() { counts_.fill(0); }

  void add(Letter t) { ++counts_[t]; }
  bool remove(Letter t) {
    if (counts_[t] == 0) return false;
    --counts_[t];
    return true;
  }
  int count(Letter t) const { return counts_[t]; }
  int blanks() const { return counts_[BLANK]; }

  int size() const {
    int s = 0;
    for (int c : counts_) s += c;
    return s;
  }
  bool empty() const { return size() == 0; }

  // String form: letters in alphabetical order followed by '?' for blanks.
  std::string to_string() const;

  // Sum of tile values currently in the rack (blanks count as 0).
  int point_value() const;

  const std::array<int, 27>& counts() const { return counts_; }

 private:
  std::array<int, 27> counts_{};
};

}  // namespace scribblez
