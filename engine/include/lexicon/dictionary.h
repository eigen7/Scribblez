#pragma once

#include "game/tile.h"
#include "lexicon/word_map.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace scribblez {

// A Kurnia Word Graph (KWG) -- the wolges/Macondo DAWG file format. A
// little-endian array of uint32, one per arc node, laid out as (from wolges):
//   bits  0..21 : arc_index (child arc list, 0 = no children)
//   bit  22     : is_end (last sibling in this arc list)
//   bit  23     : accepts (traversing this arc completes a word)
//   bits 24..31 : tile (1-indexed: A=1, ..., Z=26)
//
// Node 0's arc_index is the DAWG root; node 1's is the GADDAG root.
//
// The DAWG side (a forward trie) is used for whole-word lookup and for the
// move generator's cross-checks. The GADDAG side is used as the main spine of
// move generation (Gordon's algorithm); see movegen.cpp.
//
// GADDAG tile convention: the separator token is tile value 0 (SEPARATOR);
// letters are 1-indexed (A=1..Z=26). A word c1..cn is encoded as
// rev(c1..ci) + SEP + c(i+1..n) for i < n, plus the fully reversed word
// cn..c1 (no trailing separator) for i = n.
class Dictionary {
 public:
  struct Step {
    uint32_t next = 0;     // child arc list (0 if no children below)
    bool valid = false;    // true iff the transition exists from `node`
    bool accepts = false;  // true iff this transition completes a word
  };

  // Throws util::Exception on I/O failure.
  static Dictionary load_kwg(const std::string& path);

  // Lays out a (non-minimized) trie in the KWG node layout, so the same
  // accessors work. Words under 2 letters or holding non-A..Z characters are
  // silently skipped.
  static Dictionary build_from_words(const std::vector<std::string>& words);

  // Arc list indices of the two roots.
  uint32_t root() const { return root_; }
  uint32_t gaddag_root() const { return gaddag_root_; }

  // Transition from `node` by `letter` (0..25), or {0, false, false} if it is
  // not a sibling there.
  Step step(uint32_t node, Tile letter) const;

  // The low-level primitive behind step(), taking a raw KWG tile value.
  Step step_tile(uint32_t node, uint8_t tile_value) const;

  static constexpr uint8_t SEPARATOR = 0;

  bool contains(const std::string& word) const;

  size_t num_nodes() const { return nodes_.size(); }

  // KWG bit layout constants (also used by the in-memory builder).
  static constexpr uint32_t ARC_MASK = 0x003fffffu;
  static constexpr uint32_t IS_END_BIT = 0x00400000u;
  static constexpr uint32_t ACCEPTS_BIT = 0x00800000u;

  // Raw access to a node's child arc list, which starts at the node's index and
  // runs through the arc carrying IS_END_BIT -- cheaper than 26 step() scans
  // when every available transition is wanted.
  uint32_t arc(uint32_t i) const { return nodes_[i]; }
  static uint8_t arc_tile(uint32_t a) { return a >> 24; }

  // Built on first call and cached for the dictionary's lifetime. Thread-safe:
  // concurrent first calls build exactly once, and the result may be shared.
  const WordMap& word_map() const;

 private:
  // Held behind a pointer so Dictionary stays movable (std::once_flag is
  // neither movable nor copyable).
  struct WordMapCache {
    std::once_flag once;
    std::unique_ptr<WordMap> map;
  };

  std::vector<uint32_t> nodes_;
  uint32_t root_ = 0;
  uint32_t gaddag_root_ = 0;
  mutable std::unique_ptr<WordMapCache> word_map_cache_ = std::make_unique<WordMapCache>();

  static uint8_t tile_of(uint32_t n) { return n >> 24; }
};

}  // namespace scribblez
