#pragma once

// Anagram index ("word map", after MAGPIE's WMP): per word length, a hash map
// from a letter multiset to the dictionary words with exactly those letters.
// Move generation uses it instead of a GADDAG walk: a rack subset's anagrams
// come back as one contiguous, cache-friendly block. Keys hold no blanks; the
// move generator substitutes concrete letters before looking up.

#include "game/tile.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

// A letter multiset packed 4 bits per letter (A..Z in nibbles 0..25), the word
// map's key: a rack subset plus the playthrough tiles, added together. No
// multiset exceeds one word (<= 15 letters), so no nibble overflows and two
// 64-bit adds union two multisets.
struct BitRack {
  uint64_t lo = 0;  // letters 0..15
  uint64_t hi = 0;  // letters 16..25

  bool operator==(const BitRack& o) const { return lo == o.lo && hi == o.hi; }
  BitRack operator+(const BitRack& o) const { return BitRack{lo + o.lo, hi + o.hi}; }
  bool empty() const { return lo == 0 && hi == 0; }
  int get(int letter) const { return (half(letter) >> shift(letter)) & 0xF; }

  void add_letter(int letter, int n = 1);
  // Multiplicative (Fibonacci) hash mixing both halves.
  uint64_t hash() const;

 private:
  const uint64_t& half(int letter) const { return letter < 16 ? lo : hi; }
  uint64_t& half(int letter) { return letter < 16 ? lo : hi; }
  static int shift(int letter) { return 4 * (letter < 16 ? letter : letter - 16); }
};

class WordMap {
 public:
  static WordMap build(const Dictionary& dict);

  // The words of `length` whose letter multiset equals `key`, as a contiguous
  // block: word i at begin[i*length .. i*length+length). `begin` is null when
  // there are none.
  struct WordList {
    const Tile* begin = nullptr;
    int count = 0;
  };
  WordList lookup(int length, const BitRack& key) const;

 private:
  static constexpr int kMaxLen = 15;

  // Linear-probing slot; count == 0 marks it empty.
  struct Slot {
    BitRack key;
    uint32_t word_start = 0;
    uint32_t count = 0;
  };
  struct PerLength {
    std::vector<Tile> words;  // flat; word i occupies [i*len, (i+1)*len)
    std::vector<Slot> slots;  // open-addressing table, power-of-two size
    uint64_t mask = 0;        // slots.size() - 1
  };
  std::array<PerLength, kMaxLen + 1> by_len_;  // indexed by word length (2..15)
};

}  // namespace scribblez

#include "inlines/lexicon/word_map.inl"
