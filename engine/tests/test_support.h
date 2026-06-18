#pragma once

#include "scribblez/dictionary.h"
#include "scribblez/glyph.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"
#include "scribblez/tile.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <string>

// Hard assertion for the hand-rolled test binaries: prints the failed condition
// and aborts the process with a nonzero status.
#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::cerr << "CHECK failed: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

namespace scribblez {
namespace test_support {

// A 25-word in-memory lexicon shared by the movegen and equity tests.
Dictionary tiny_dict();

// Parse a rack string; '?' denotes a blank.
Rack rack_from(const std::string& s);

// Build a PLAY Move from a starting square, direction, and ordered new glyphs.
// Every glyph is treated as newly placed (sufficient for empty-board setups);
// the square mask is absolute over the play's lane and the score is 0.
Move make_play(int row, int col, bool horizontal, std::initializer_list<Glyph> gs);

// Build a PLAY Move with an explicit per-tile layout. `rel_mask` is the play's
// mask relative to its first lane cell (bit 0 == the start cell); `gs` are the
// newly placed glyphs in word order (count must equal popcount(rel_mask)).
Move make_play_full(int row, int col, bool horizontal, uint16_t rel_mask, uint16_t score,
                    std::initializer_list<Glyph> gs);

// A synthetic Macondo .klv2 leave file written under `dir`. Encodes three
// single-tile leaves: blank "?" = 12.0, "A" = 1.5, "B" = -2.5 (every other
// leave looks up as 0).
struct KlvFixture {
  std::filesystem::path path;
};
KlvFixture write_synthetic_klv(const std::filesystem::path& dir);

}  // namespace test_support
}  // namespace scribblez
