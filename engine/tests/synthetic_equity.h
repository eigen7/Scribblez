#pragma once

// A HastyEquity built from a synthetic leave table, for tests that need moves
// ranked by static equity but not the real numbers behind the ranking. It
// spares them the Macondo data mount, and keeps the .klv2 byte layout in one
// place rather than in each suite that needs one.
//
// The table prices exactly three leaves -- blank 12.0, A 1.5, B -2.5 -- and
// every other leave at 0, which is enough to separate candidates while keeping
// what separates them obvious in a failing test.

#include "lexicon/hasty_equity.h"
#include "util/exception.h"

#include <cstdint>
#include <filesystem>
#include <fstream>

namespace scribblez::testing {

// Writes the synthetic leave and pre-endgame files into `dir` and initializes
// the HastyEquity singleton from them.
inline void install_synthetic_hasty_equity(const std::filesystem::path& dir) {
  const std::filesystem::path klv = dir / "synthetic.klv2";
  std::ofstream f(klv, std::ios::binary | std::ios::trunc);
  auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto write_f32 = [&](float v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  write_u32(4);                                          // kwg_node_count
  write_u32((0u << 24) | (1u << 22) | (0u << 23) | 1u);  // root
  write_u32((0u << 24) | (0u << 22) | (1u << 23) | 0u);  // ? (blank)
  write_u32((1u << 24) | (0u << 22) | (1u << 23) | 0u);  // A
  write_u32((2u << 24) | (1u << 22) | (1u << 23) | 0u);  // B
  write_u32(3);                                          // num_leaves
  write_f32(12.0f);
  write_f32(1.5f);
  write_f32(-2.5f);
  if (!f.good()) throw util::Exception("failed writing synthetic leave file");
  f.close();  // flush to disk before HastyEquity::init reopens the file to read

  const std::filesystem::path peg = dir / "peg.json";
  std::ofstream(peg) << "[]";
  HastyEquity::init(klv.string(), peg.string());
}

}  // namespace scribblez::testing
