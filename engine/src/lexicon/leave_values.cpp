#include "lexicon/leave_values.h"

#include "util/exception.h"
#include "util/hash.h"

#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

// Readers for the file's little-endian arrays; assume a little-endian host.
std::vector<uint32_t> read_u32_array(std::ifstream& in, uint32_t count) {
  std::vector<uint32_t> data(count);
  in.read(reinterpret_cast<char*>(data.data()), std::streamsize(count) * 4);
  if (!in) throw util::Exception("LeaveValues: truncated read");
  return data;
}

std::vector<float> read_f32_array(std::ifstream& in, uint32_t count) {
  std::vector<float> data(count);
  in.read(reinterpret_cast<char*>(data.data()), std::streamsize(count) * 4);
  if (!in) throw util::Exception("LeaveValues: truncated read");
  return data;
}

uint32_t read_u32(std::ifstream& in) {
  uint32_t v = 0;
  in.read(reinterpret_cast<char*>(&v), 4);
  if (!in) throw util::Exception("LeaveValues: truncated header read");
  return v;
}

// Fills subtree_words[i] with the number of accepting arcs at or below arc i,
// for every arc reachable from `list`. Memoized via `done`, since a minimized
// KWG shares subtrees.
void compute_subtree_words(const std::vector<uint32_t>& nodes, uint32_t list,
                           std::vector<uint32_t>& subtree_words, std::vector<bool>& done) {
  for (uint32_t i = list;; ++i) {
    if (!done[i]) {
      const uint32_t entry = nodes[i];
      const uint32_t child = entry & Dictionary::ARC_MASK;
      if (child != 0) compute_subtree_words(nodes, child, subtree_words, done);
      uint32_t sw = (entry & Dictionary::ACCEPTS_BIT) ? 1u : 0u;
      if (child != 0)
        for (uint32_t c = child;; ++c) {
          sw += subtree_words[c];
          if (nodes[c] & Dictionary::IS_END_BIT) break;
        }
      subtree_words[i] = sw;
      done[i] = true;
    }
    if (nodes[i] & Dictionary::IS_END_BIT) break;
  }
}

}  // namespace

LeaveValues LeaveValues::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw util::Exception("LeaveValues: cannot open {}", path);

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
  lv.root_arc_list_ = nodes.empty() ? 0u : (nodes[0] & Dictionary::ARC_MASK);
  // For the cursor's index accumulation.
  lv.subtree_words_.assign(nodes.size(), 0u);
  if (!nodes.empty()) {
    std::vector<bool> done(nodes.size(), false);
    compute_subtree_words(nodes, nodes[0] & Dictionary::ARC_MASK, lv.subtree_words_, done);
  }
  lv.nodes_ = std::move(nodes);
  lv.values_ = std::move(values);
  lv.content_hash_ = util::file_content_hash(path);
  lv.values_by_leave_.reserve(num_leaves);
  lv.for_each([&](const Rack& leave, float value) { lv.values_by_leave_.emplace(leave, value); });
  return lv;
}

float LeaveValues::lookup(const Rack& leave) const {
  auto it = values_by_leave_.find(leave);
  return it == values_by_leave_.end() ? 0.0f : it->second;
}

}  // namespace scribblez
