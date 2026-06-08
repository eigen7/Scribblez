#include "scribblez/leave_values.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace scribblez {

namespace {

// Read a little-endian uint32 array of `count` elements from `in`.
std::vector<uint32_t> read_u32_array(std::ifstream& in, uint32_t count) {
  std::vector<uint32_t> data(count);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count) * 4);
  if (!in) throw std::runtime_error("LeaveValues: truncated read");
  return data;
}

// Read a little-endian float32 array of `count` elements from `in`.
std::vector<float> read_f32_array(std::ifstream& in, uint32_t count) {
  std::vector<float> data(count);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count) * 4);
  if (!in) throw std::runtime_error("LeaveValues: truncated read");
  return data;
}

uint32_t read_u32(std::ifstream& in) {
  uint32_t v = 0;
  in.read(reinterpret_cast<char*>(&v), 4);
  if (!in) throw std::runtime_error("LeaveValues: truncated header read");
  return v;
}

}  // namespace

// -------------------------------------------------------------------------

LeaveValues LeaveValues::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("LeaveValues: cannot open " + path);

  // KLV2 binary layout (little-endian):
  //   uint32  kwg_node_count
  //   uint32[kwg_node_count]  KWG arc nodes
  //   uint32  num_leaves
  //   float32[num_leaves]     leave values
  uint32_t kwg_size = read_u32(in);
  auto nodes = read_u32_array(in, kwg_size);
  uint32_t num_leaves = read_u32(in);
  auto values = read_f32_array(in, num_leaves);

  LeaveValues lv;
  lv.nodes_ = std::move(nodes);
  lv.values_ = std::move(values);
  lv.build_word_counts();
  return lv;
}

// -------------------------------------------------------------------------
// word_counts_ construction (mirrors word-golib CountWords/countWordsAt)

void LeaveValues::build_word_counts() {
  word_counts_.assign(nodes_.size(), 0);
  // Process nodes in reverse index order so children are always computed
  // before parents (same traversal order as the Go implementation).
  for (int p = static_cast<int>(nodes_.size()) - 1; p >= 0; --p)
    count_words_at(static_cast<uint32_t>(p));
}

int32_t LeaveValues::count_words_at(uint32_t p) {
  if (p >= static_cast<uint32_t>(word_counts_.size())) return 0;
  if (word_counts_[p] != 0) return word_counts_[p];

  // Sentinel: already in progress (shouldn't happen in a valid KWG, but match
  // Go's panic-guard by treating -1 as cached).
  word_counts_[p] = -1;

  int32_t a = (nodes_[p] & kAcceptsBit) ? 1 : 0;
  uint32_t child = nodes_[p] & kArcMask;
  int32_t b = (child != 0) ? count_words_at(child) : 0;
  int32_t c = (nodes_[p] & kIsEndBit) ? 0 : count_words_at(p + 1);

  word_counts_[p] = a + b + c;
  return word_counts_[p];
}

// -------------------------------------------------------------------------
// Lookup

int32_t LeaveValues::word_index_of(const TileCounts& leave) const {
  // Build sorted tile sequence (1-indexed KLV tile codes: A=1..Z=26, blank=27).
  // Maximum leave length is 6 tiles; a fixed small buffer avoids allocation.
  uint8_t seq[6];
  int len = 0;
  for (int t = 0; t < 26 && len < 6; ++t) {
    for (int i = 0; i < leave.count(Tile::of(t)) && len < 6; ++i)
      seq[len++] = static_cast<uint8_t>(t + 1);  // A=1..Z=26
  }
  for (int i = 0; i < leave.blanks() && len < 6; ++i)
    seq[len++] = 27u;  // blank = 27 in KLV tile encoding

  if (len == 0) return -1;

  // Walk the KWG from arc 0 (DAWG root, which is the leave KWG root).
  // Mirrors word-golib GetWordIndexOf.
  uint32_t node = nodes_[0] & kArcMask;  // ArcIndex(0) = children of node 0
  int32_t idx = 0;
  int lidx = 0;

  while (node != 0) {
    idx += word_counts_[node];
    while ((nodes_[node] >> kTileShift) != seq[lidx]) {
      if (nodes_[node] & kIsEndBit) return -1;
      ++node;
    }
    idx -= word_counts_[node];
    ++lidx;
    if (lidx > len - 1) {
      return (nodes_[node] & kAcceptsBit) ? idx : -1;
    }
    if (nodes_[node] & kAcceptsBit) ++idx;
    node = nodes_[node] & kArcMask;
  }
  return -1;
}

float LeaveValues::lookup(const TileCounts& leave) const {
  int32_t idx = word_index_of(leave);
  if (idx < 0 || static_cast<size_t>(idx) >= values_.size()) return 0.0f;
  return values_[static_cast<size_t>(idx)];
}

}  // namespace scribblez
