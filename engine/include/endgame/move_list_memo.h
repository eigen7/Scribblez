#pragma once

#include "game/move.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace scribblez {

class Board;
class Dictionary;
class Rack;

// Move generation, optionally cached by board and rack. Caching trades memory
// for generation work and changes no list, so a benchmark can enable it to
// speed up a measurement run without perturbing what is measured. Off by
// default.
//
// A returned reference is invalidated by later generate() calls; a caller that
// must hold a list across further generation keeps its own copy.
class MoveListMemo {
 public:
  void set_enabled(bool on) { enabled_ = on; }

  // The legal plays for `rack` on `board`, generated on a cache miss.
  // `board_hash` identifies `board` and is the caller's to maintain.
  const std::vector<Move>& generate(const Board& board, const Dictionary& dict, const Rack& rack,
                                    uint64_t board_hash);

 private:
  // Entries depend on neither scores nor search state, so they stay valid for
  // the memo's whole lifetime; the map is instead dropped whole once it exceeds
  // this many entries.
  static constexpr size_t kCap = static_cast<size_t>(1) << 20;

  bool enabled_ = false;
  std::unordered_map<uint64_t, std::vector<Move>> cache_;
  std::vector<Move> scratch_;  // the return buffer while caching is off
};

}  // namespace scribblez
