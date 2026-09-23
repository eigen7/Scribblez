#pragma once

#include "game/tile.h"
#include "lexicon/word_map.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace scribblez {

// A lexicon in Kurnia Word Graph (KWG) form, the wolges/Macondo file format:
// a little-endian uint32 array, one entry per arc:
//   bits  0..21 : arc_index (first arc of the child list; 0 = no children)
//   bit  22     : is_end (last sibling in this arc list)
//   bit  23     : accepts (taking this arc completes a word)
//   bits 24..31 : tile (0 = GADDAG separator, A=1 .. Z=26)
//
// Entry 0's arc_index is the DAWG root; entry 1's is the GADDAG root.
//
// The DAWG (a forward trie) serves whole-word lookup and cross-checks. The
// GADDAG drives Gordon-style move generation where the word map can't (see
// movegen.cpp). It encodes each word c1..cn as rev(c1..ci) + SEPARATOR +
// c(i+1)..cn for every i < n, plus the fully reversed word cn..c1 with no
// separator.
class Dictionary {
 public:
  struct Step {
    uint32_t next = 0;     // child arc list (0 if no children below)
    bool valid = false;    // true iff the transition exists from `node`
    bool accepts = false;  // true iff this transition completes a word
  };

  // Throws util::Exception on I/O failure.
  static Dictionary load_kwg(const std::string& path);

  // An unminimized trie in the KWG layout, for tests and small word lists.
  // Words under 2 letters or with characters outside A..Z are skipped.
  static Dictionary build_from_words(const std::vector<std::string>& words);

  // Arc list indices of the two roots.
  uint32_t root() const { return root_; }
  uint32_t gaddag_root() const { return gaddag_root_; }

  // Transition from arc list `node` by `letter`, or an invalid Step if there is
  // none. step_tile() takes a raw KWG tile value instead, e.g. SEPARATOR.
  Step step(uint32_t node, Tile letter) const;
  Step step_tile(uint32_t node, uint8_t tile_value) const;

  static constexpr uint8_t SEPARATOR = 0;

  bool contains(const std::string& word) const;

  size_t num_nodes() const { return nodes_.size(); }

  static constexpr uint32_t ARC_MASK = 0x003fffffu;
  static constexpr uint32_t IS_END_BIT = 0x00400000u;
  static constexpr uint32_t ACCEPTS_BIT = 0x00800000u;

  // Raw arc access, for callers that want every transition out of an arc list
  // (which runs from its first index through the arc with IS_END_BIT set):
  // one scan instead of a step() per letter.
  uint32_t arc(uint32_t i) const { return nodes_[i]; }
  static uint8_t arc_tile(uint32_t a) { return a >> 24; }

  // Built on first call and cached for the dictionary's lifetime. Thread-safe;
  // concurrent first calls build it once.
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
