#include "test_support.h"

#include <array>
#include <fstream>

namespace scribblez {
namespace test_support {

Dictionary tiny_dict() {
  return Dictionary::build_from_words({"CAT", "CATS", "AT",     "AS",     "BAT", "BATS", "HE",
                                       "TO",  "ON",   "NO",     "IT",     "IS",  "OAT",  "OATS",
                                       "HAT", "HATS", "RAT",    "RATS",   "DOG", "GOD",  "GO",
                                       "OD",  "DO",   "AERIES", "PARTIED"});
}

Rack rack_from(const std::string& s) {
  Rack r;
  for (char c : s) {
    if (c == '?')
      r.add(BLANK);
    else
      r.add(Tile::from_char(c));
  }
  return r;
}

Move make_play(int row, int col, bool horizontal, std::initializer_list<Glyph> gs) {
  std::array<Glyph, RACK_SIZE> played{};
  int n = 0;
  for (Glyph g : gs) {
    if (n >= RACK_SIZE) break;
    played[n++] = g;
  }
  const int start = horizontal ? row : col;
  const int lane0 = horizontal ? col : row;
  uint16_t mask = 0;
  for (int i = 0; i < n; ++i) mask |= static_cast<uint16_t>(1u << (lane0 + i));
  return Move::play(horizontal, start, mask, /*score=*/0, played.data(), n);
}

Move make_play_full(int row, int col, bool horizontal, uint16_t rel_mask, uint16_t score,
                    std::initializer_list<Glyph> gs) {
  std::array<Glyph, RACK_SIZE> played{};
  int n = 0;
  for (Glyph g : gs) {
    if (n >= RACK_SIZE) break;
    played[n++] = g;
  }
  const int start = horizontal ? row : col;
  const int lane0 = horizontal ? col : row;
  uint16_t mask = static_cast<uint16_t>(rel_mask << lane0);
  return Move::play(horizontal, start, mask, score, played.data(), n);
}

KlvFixture write_synthetic_klv(const std::filesystem::path& dir) {
  std::filesystem::path p = dir / "synthetic.klv2";
  std::ofstream f(p, std::ios::binary | std::ios::trunc);

  auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto write_f32 = [&](float v) { f.write(reinterpret_cast<const char*>(&v), 4); };

  // kwg_node_count = 4
  write_u32(4);
  // Node 0: root; arc_index=1, is_end=1, accepts=0, tile=0
  write_u32((0u << 24) | (1u << 22) | (0u << 23) | 1u);
  // Node 1: ? (blank); arc_index=0, is_end=0, accepts=1, tile=0
  write_u32((0u << 24) | (0u << 22) | (1u << 23) | 0u);
  // Node 2: A; arc_index=0, is_end=0, accepts=1, tile=1
  write_u32((1u << 24) | (0u << 22) | (1u << 23) | 0u);
  // Node 3: B; arc_index=0, is_end=1, accepts=1, tile=2
  write_u32((2u << 24) | (1u << 22) | (1u << 23) | 0u);
  // num_leaves = 3, then leave values in word order: ?=12.0, A=1.5, B=-2.5
  write_u32(3);
  write_f32(12.0f);
  write_f32(1.5f);
  write_f32(-2.5f);
  CHECK(f);
  return KlvFixture{p};
}

}  // namespace test_support
}  // namespace scribblez
