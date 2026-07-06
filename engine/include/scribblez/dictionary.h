#pragma once

#include "scribblez/tile.h"
#include "scribblez/word_map.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace scribblez {

// A Kurnia Word Graph (KWG) -- the wolges/Macondo DAWG file format.
//
// The on-disk file is a little-endian array of uint32, one per arc node.
// Bit layout (from wolges):
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

  // Load a .kwg file from disk. Throws std::runtime_error on I/O failure.
  static Dictionary load_kwg(const std::string& path);

  // Build an in-memory dictionary from a list of words. Lays out a (non-
  // minimized) trie using the KWG node layout, so the same accessors work.
  // Words shorter than 2 letters or containing non-A..Z characters are
  // silently skipped.
  static Dictionary build_from_words(const std::vector<std::string>& words);

  // The arc list index for the DAWG (forward-trie) root.
  uint32_t root() const { return root_; }

  // The arc list index for the GADDAG root.
  uint32_t gaddag_root() const { return gaddag_root_; }

  // Transition from `node` by `letter` (0..25). If `letter` is not a valid
  // sibling at `node`, returns {0, false, false}.
  Step step(uint32_t node, Tile letter) const;

  // Transition from `node` by a raw KWG tile value (0 = GADDAG separator,
  // 1..26 = letters A..Z). The low-level primitive behind step().
  Step step_tile(uint32_t node, uint8_t tile_value) const;

  // The GADDAG separator tile value.
  static constexpr uint8_t SEPARATOR = 0;

  // Whole-word lookup. Convenience wrapper around step().
  bool contains(const std::string& word) const;

  size_t num_nodes() const { return nodes_.size(); }

  // KWG bit layout constants (also used by the in-memory builder).
  static constexpr uint32_t ARC_MASK = 0x003fffffu;
  static constexpr uint32_t IS_END_BIT = 0x00400000u;
  static constexpr uint32_t ACCEPTS_BIT = 0x00800000u;

  // Iterate a node's child arcs directly (faster than 26 step() scans when you
  // want all available transitions): the arc list starts at the node's index
  // and runs until the arc carrying IS_END_BIT:
  //   for (uint32_t i = node; node; ++i) {
  //     uint32_t a = arc(i); uint8_t t = arc_tile(a); ...; if (a & IS_END_BIT) break;
  //   }
  uint32_t arc(uint32_t i) const { return nodes_[i]; }
  static uint8_t arc_tile(uint32_t a) { return static_cast<uint8_t>(a >> 24); }

  // The anagram index (WordMap) derived from this dictionary, built on first
  // call and cached for the dictionary's lifetime. Thread-safe: concurrent
  // first calls build exactly once, and the returned reference may be shared
  // across threads (lookup is const).
  const WordMap& word_map() const;

 private:
  // Lazily-built word_map() cache, held behind a pointer so Dictionary stays
  // movable (std::once_flag is neither movable nor copyable).
  struct WordMapCache {
    std::once_flag once;
    std::unique_ptr<WordMap> map;
  };

  std::vector<uint32_t> nodes_;
  uint32_t root_ = 0;
  uint32_t gaddag_root_ = 0;
  mutable std::unique_ptr<WordMapCache> word_map_cache_ = std::make_unique<WordMapCache>();

  static uint8_t tile_of(uint32_t n) { return static_cast<uint8_t>(n >> 24); }
};

}  // namespace scribblez
