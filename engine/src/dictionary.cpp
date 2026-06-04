#include "scribblez/dictionary.h"

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace scribblez {

namespace {

// In-memory trie used only by build_from_words(). Keyed by raw KWG tile value
// (0 = GADDAG separator, 1..26 = letters A..Z), so the same structure serves
// both the DAWG and the GADDAG.
struct TrieNode {
  std::map<uint8_t, std::unique_ptr<TrieNode>> children;
  bool terminal = false;
};

// Insert a sequence of tile values into the trie, marking the final node
// terminal.
void insert(TrieNode* root, const std::vector<uint8_t>& tiles) {
  TrieNode* node = root;
  for (uint8_t t : tiles) {
    auto& ch = node->children[t];
    if (!ch) ch = std::make_unique<TrieNode>();
    node = ch.get();
  }
  node->terminal = true;
}

// Lay out the (non-minimized) trie into the KWG node array, appending nodes and
// returning the arc-list index of `tn`'s children (0 if it has none).
uint32_t layout(std::vector<uint32_t>& nodes, const TrieNode* tn) {
  if (tn->children.empty()) return 0;
  uint32_t first = static_cast<uint32_t>(nodes.size());
  std::vector<uint8_t> tiles;
  tiles.reserve(tn->children.size());
  for (const auto& kv : tn->children) tiles.push_back(kv.first);
  nodes.resize(nodes.size() + tiles.size(), 0u);
  for (size_t i = 0; i < tiles.size(); ++i) {
    const TrieNode* child = tn->children.at(tiles[i]).get();
    uint32_t child_arc = layout(nodes, child);
    uint32_t v = child_arc & Dictionary::ARC_MASK;
    if (child->terminal) v |= Dictionary::ACCEPTS_BIT;
    if (i + 1 == tiles.size()) v |= Dictionary::IS_END_BIT;
    v |= (static_cast<uint32_t>(tiles[i])) << 24;
    nodes[first + i] = v;
  }
  return first;
}

// Validate and normalize a word to A..Z tile values (1-indexed). Returns false
// (and leaves `out` unspecified) for words shorter than 2 or with non-letters.
bool word_to_tiles(const std::string& w, std::vector<uint8_t>& out) {
  if (w.size() < 2) return false;
  out.clear();
  for (char c : w) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    if (c < 'A' || c > 'Z') return false;
    out.push_back(static_cast<uint8_t>(c - 'A' + 1));  // 1-indexed
  }
  return true;
}

}  // namespace

Dictionary::Step Dictionary::step_tile(uint32_t node, uint8_t tile_value) const {
  if (node == 0) return {};  // leaf, no outgoing arcs
  for (uint32_t i = node;; ++i) {
    uint32_t n = nodes_[i];
    if (tile_of(n) == tile_value) {
      return {n & ARC_MASK, true, (n & ACCEPTS_BIT) != 0};
    }
    if (n & IS_END_BIT) return {};
  }
}

Dictionary::Step Dictionary::step(uint32_t node, Letter letter) const {
  return step_tile(node, static_cast<uint8_t>(letter + 1));  // KWG is 1-indexed
}

bool Dictionary::contains(const std::string& word) const {
  if (word.size() < 2) return false;
  uint32_t node = root_;
  bool acc = false;
  for (size_t k = 0; k < word.size(); ++k) {
    char c = word[k];
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    if (c < 'A' || c > 'Z') return false;
    Letter L = static_cast<Letter>(c - 'A');
    Step s = step(node, L);
    if (!s.valid) return false;
    acc = s.accepts;
    node = s.next;
    if (k + 1 < word.size() && node == 0) return false;
  }
  return acc;
}

Dictionary Dictionary::load_kwg(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open KWG file: " + path);
  in.seekg(0, std::ios::end);
  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size <= 0 || size % 4 != 0) {
    throw std::runtime_error("Bad KWG file size for: " + path);
  }
  Dictionary d;
  d.nodes_.resize(static_cast<size_t>(size) / 4);
  in.read(reinterpret_cast<char*>(d.nodes_.data()), size);
  if (!in) throw std::runtime_error("Failed to read KWG file: " + path);
  // KWG is little-endian uint32. This loader assumes a little-endian host
  // (true on x86-64 / aarch64); add a byte-swap fallback if you target BE.
  if (d.nodes_.size() < 2) {
    throw std::runtime_error("KWG file too small: " + path);
  }
  d.root_ = d.nodes_[0] & ARC_MASK;          // DAWG root.
  d.gaddag_root_ = d.nodes_[1] & ARC_MASK;   // GADDAG root.
  return d;
}

Dictionary Dictionary::build_from_words(const std::vector<std::string>& words) {
  TrieNode dawg;
  TrieNode gaddag;
  std::vector<uint8_t> tiles;
  for (const auto& w : words) {
    if (!word_to_tiles(w, tiles)) continue;

    // DAWG: the word itself.
    insert(&dawg, tiles);

    // GADDAG: rev(c1..ci) + SEP + c(i+1..n) for i = 1..n-1, and the fully
    // reversed word (no separator) for i = n. This matches the wolges/Macondo
    // encoding (verified against real .kwg files).
    const int n = static_cast<int>(tiles.size());
    std::vector<uint8_t> path;
    path.reserve(n + 1);
    for (int i = 1; i <= n; ++i) {
      path.clear();
      for (int j = i - 1; j >= 0; --j) path.push_back(tiles[j]);  // rev(c1..ci)
      if (i < n) {
        path.push_back(SEPARATOR);
        for (int j = i; j < n; ++j) path.push_back(tiles[j]);      // c(i+1..n)
      }
      insert(&gaddag, path);
    }
  }

  Dictionary d;
  // Reserve the two-slot KWG header: ArcIndex(0) is the DAWG root, ArcIndex(1)
  // is the GADDAG root. Lay out both tries into the shared node array.
  d.nodes_.push_back(0);
  d.nodes_.push_back(0);
  uint32_t dawg_root = layout(d.nodes_, &dawg);
  uint32_t gaddag_root = layout(d.nodes_, &gaddag);
  d.nodes_[0] = dawg_root & ARC_MASK;
  d.nodes_[1] = gaddag_root & ARC_MASK;
  d.root_ = dawg_root;
  d.gaddag_root_ = gaddag_root;
  return d;
}

}  // namespace scribblez
