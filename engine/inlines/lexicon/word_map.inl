#include "lexicon/word_map.h"

namespace scribblez {

inline void BitRack::add_letter(int letter, int n) {
  half(letter) += static_cast<uint64_t>(n) << shift(letter);
}

inline uint64_t BitRack::hash() const {
  uint64_t h = lo * 0x9E3779B97F4A7C15ULL + hi * 0xC2B2AE3D27D4EB4FULL;
  h ^= h >> 31;
  return h;
}

inline WordMap::WordList WordMap::lookup(int length, const BitRack& key) const {
  if (length < 2 || length > kMaxLen) return {};
  const PerLength& pl = by_len_[length];
  if (pl.slots.empty()) return {};
  uint64_t i = key.hash() & pl.mask;
  while (pl.slots[i].count != 0) {
    const Slot& s = pl.slots[i];
    if (s.key == key) {
      return WordList{&pl.words[static_cast<size_t>(s.word_start) * length],
                      static_cast<int>(s.count)};
    }
    i = (i + 1) & pl.mask;
  }
  return {};
}

}  // namespace scribblez
