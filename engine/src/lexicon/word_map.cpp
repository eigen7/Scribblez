#include "lexicon/word_map.h"

#include "lexicon/dictionary.h"
#include "util/math.h"

#include <array>
#include <unordered_map>
#include <vector>

namespace scribblez {

namespace {

constexpr int kMaxLen = 15;

struct BitRackHash {
  std::size_t operator()(const BitRack& k) const { return k.hash(); }
};

// Words grouped by (length, multiset) during the build.
using Grouped =
  std::array<std::unordered_map<BitRack, std::vector<std::vector<Tile>>, BitRackHash>, kMaxLen + 1>;

// Letters accumulated along the current DFS path, capped at kMaxLen.
using WordBuf = std::array<Tile, kMaxLen>;

// Depth-first walk of the DAWG, collecting every accepted word. `len` is how
// many of `word`'s leading entries are the current path.
void dfs(const Dictionary& dict, uint32_t node, WordBuf& word, int len, BitRack key,
         Grouped& grouped) {
  if (node == 0) return;
  for (uint32_t i = node;; ++i) {
    const uint32_t a = dict.arc(i);
    const uint8_t tv = Dictionary::arc_tile(a);
    if (tv >= 1 && tv <= 26) {
      const Tile L = Tile::of(tv - 1);
      word[len] = L;
      key.add_letter(L.index());
      const int new_len = len + 1;
      if ((a & Dictionary::ACCEPTS_BIT) && new_len >= 2 && new_len <= kMaxLen) {
        grouped[new_len][key].emplace_back(word.begin(), word.begin() + new_len);
      }
      if (new_len < kMaxLen) dfs(dict, a & Dictionary::ARC_MASK, word, new_len, key, grouped);
      key.add_letter(L.index(), -1);
    }
    if (a & Dictionary::IS_END_BIT) break;
  }
}

}  // namespace

WordMap WordMap::build(const Dictionary& dict) {
  Grouped grouped;
  WordBuf word;
  dfs(dict, dict.root(), word, 0, BitRack{}, grouped);

  WordMap wm;
  for (int len = 2; len <= kMaxLen; ++len) {
    PerLength& pl = wm.by_len_[len];
    const auto& keys = grouped[len];
    if (keys.empty()) continue;

    // ~50% load factor keeps open-addressing probe chains short.
    const uint64_t table_size = util::round_up_pow2(uint64_t(keys.size()) * 2);
    pl.slots.assign(table_size, Slot{});
    pl.mask = table_size - 1;

    for (const auto& [key, words] : keys) {
      const uint32_t start = uint32_t(pl.words.size() / len);
      for (const auto& w : words) {
        for (const Tile t : w) pl.words.push_back(t);
      }
      uint64_t i = key.hash() & pl.mask;
      while (pl.slots[i].count != 0) i = (i + 1) & pl.mask;
      pl.slots[i] = Slot{key, start, uint32_t(words.size())};
    }
  }
  return wm;
}

}  // namespace scribblez
