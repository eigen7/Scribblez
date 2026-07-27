#include "lexicon/leave_values.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

// KWG bit-field constants (same layout as Dictionary / word-golib KWG).
constexpr uint32_t kArcMask = 0x003fffffu;
constexpr uint32_t kIsEndBit = 0x00400000u;
constexpr uint32_t kAcceptsBit = 0x00800000u;
constexpr uint32_t kTileShift = 24u;

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

// Map a KLV tile code to an engine Tile. Macondo's leave KWG uses the blank as
// machine letter 0 (so it sorts ahead of the letters), with A..Z as 1..26 --
// distinct from the word KWG's 1-indexed A..Z with no blank. Mapping the blank
// to BLANK is what lets blank-bearing leaves (e.g. the lone "?") be keyed and
// looked up correctly.
Tile tile_from_klv_code(uint8_t code) {
  return code == 0u ? BLANK : Tile::of(static_cast<int>(code) - 1);
}

// Enumerate every leave reachable from `node`'s arc list in word-index order,
// inserting leave -> value into `out`. `acc` carries the prefix tiles and
// `counter` tracks the running word index into `values`.
//
// The traversal visits siblings left-to-right and, at each node, records the
// accepting word before descending into children -- exactly the order in which
// word-golib's GetWordIndexOf assigns indices, so values[counter] aligns.
void enumerate_leaves(const std::vector<uint32_t>& nodes, const std::vector<float>& values,
                      uint32_t node, Rack& acc, size_t& counter,
                      std::unordered_map<Rack, float>& out) {
  while (node != 0) {
    const uint32_t entry = nodes[node];
    const Tile tile = tile_from_klv_code(static_cast<uint8_t>(entry >> kTileShift));
    acc.add(tile);
    if (entry & kAcceptsBit) {
      if (counter < values.size()) out.emplace(acc, values[counter]);
      ++counter;
    }
    const uint32_t child = entry & kArcMask;
    if (child != 0) enumerate_leaves(nodes, values, child, acc, counter, out);
    acc.remove(tile);
    if (entry & kIsEndBit) break;
    ++node;
  }
}

// Fill subtree_words[i] for every arc reachable from sibling list `list` with the
// number of words (accepting nodes) in that arc's pre-order subtree. Memoized via
// `done` so shared DAWG nodes are computed once.
void compute_subtree_words(const std::vector<uint32_t>& nodes, uint32_t list,
                           std::vector<uint32_t>& subtree_words, std::vector<bool>& done) {
  for (uint32_t i = list;; ++i) {
    if (!done[i]) {
      const uint32_t entry = nodes[i];
      const uint32_t child = entry & kArcMask;
      if (child != 0) compute_subtree_words(nodes, child, subtree_words, done);
      uint32_t sw = (entry & kAcceptsBit) ? 1u : 0u;
      if (child != 0)
        for (uint32_t c = child;; ++c) {
          sw += subtree_words[c];
          if (nodes[c] & kIsEndBit) break;
        }
      subtree_words[i] = sw;
      done[i] = true;
    }
    if (nodes[i] & kIsEndBit) break;
  }
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
  lv.values_by_leave_.reserve(num_leaves);
  lv.root_arc_list_ = nodes.empty() ? 0u : (nodes[0] & kArcMask);
  // Enumerate every leave once, starting from the children of the root node.
  Rack acc;
  size_t counter = 0;
  if (!nodes.empty())
    enumerate_leaves(nodes, values, nodes[0] & kArcMask, acc, counter, lv.values_by_leave_);
  // Per-arc subtree word counts power the incremental cursor's index accumulation.
  lv.subtree_words_.assign(nodes.size(), 0u);
  if (!nodes.empty()) {
    std::vector<bool> done(nodes.size(), false);
    compute_subtree_words(nodes, nodes[0] & kArcMask, lv.subtree_words_, done);
  }
  lv.nodes_ = std::move(nodes);
  lv.values_ = std::move(values);
  return lv;
}

// -------------------------------------------------------------------------
// Lookup

float LeaveValues::lookup(const Rack& leave) const {
  auto it = values_by_leave_.find(leave);
  return it == values_by_leave_.end() ? 0.0f : it->second;
}

}  // namespace scribblez
