#pragma once

// The leave table (.ltab): every leave drawable from a full bag, for each
// length 1..6, with its static leave value -- the rows the leave inference
// model (docs/plans/deep_bot.md, "The leave inference model") scores, sums
// and samples over. A 7-tile leave arises only from a pass and is drawn from
// the prior alone, so it has no table.
//
// Every consumer reads the same file rather than re-enumerating, so the row
// order -- which sampling's pairing across candidates and the parity tests
// depend on -- cannot drift between the engine and Python.
//
// File layout (little-endian, packed):
//
//   [LeaveTableHeader                                         80 B]
//   [LeaveTableRow x row_counts[0]   (length-1 leaves)        35 B each]
//   ...
//   [LeaveTableRow x row_counts[5]   (length-6 leaves)        35 B each]
//
// Within a length, rows are in enumerate_leaves order (belief/leave_prior.h).

#include "game/tile.h"
#include "lexicon/hasty_equity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace scribblez::belief {

// "LTAB" in little-endian (bytes 'L','T','A','B' on disk).
inline constexpr uint32_t kLeaveTableMagic = 0x4241544Cu;
// Covers the row layout and the row order. Readers accept only this version.
inline constexpr uint16_t kLeaveTableFormatVersion = 1;

inline constexpr int kLeaveTableMaxLength = RACK_SIZE - 1;
inline constexpr size_t kLeaveTableLexiconSize = 16;
// The 16 hex digits of a util::content_hash, without a terminator.
inline constexpr size_t kLeaveTableHashSize = 16;

#pragma pack(push, 1)

struct LeaveTableHeader {
  uint32_t magic;                        // kLeaveTableMagic
  uint16_t version;                      // kLeaveTableFormatVersion
  uint16_t flags;                        // 0
  char lexicon[kLeaveTableLexiconSize];  // the lexicon's name, NUL-padded
  // util::content_hash of the .klv2 the values came from.
  char klv_hash[kLeaveTableHashSize];
  // util::content_hash of all the rows' bytes: the table's identity, which
  // models trained against it stamp, so no consumer hashes the file itself.
  char content_id[kLeaveTableHashSize];
  std::array<uint32_t, kLeaveTableMaxLength> row_counts;  // [k - 1]: length-k rows
};
static_assert(sizeof(LeaveTableHeader) == 80, "LeaveTableHeader must be 80 bytes");

struct LeaveTableRow {
  std::array<uint8_t, TILE_KINDS> counts;  // indexed by Tile code; the blank is 26
  uint32_t mask;                           // bit l set iff counts[l] > 0
  float value;                             // HastyEquity::leave_value
};
static_assert(sizeof(LeaveTableRow) == 35, "LeaveTableRow must be 35 bytes");

#pragma pack(pop)

class LeaveTable {
 public:
  // From the initialized `equity`'s leave values. `lexicon` is recorded, not
  // read, and must fit kLeaveTableLexiconSize.
  static LeaveTable build(const HastyEquity& equity, const std::string& lexicon);

  // Throws util::Exception on a wrong magic or format version, or a file
  // whose size disagrees with its row counts.
  static LeaveTable load(const std::string& path);

  void write(const std::string& path) const;

  const LeaveTableHeader& header() const { return header_; }
  std::string lexicon() const;
  std::string klv_hash() const { return {header_.klv_hash, kLeaveTableHashSize}; }
  std::string content_id() const { return {header_.content_id, kLeaveTableHashSize}; }

  // The length-k rows, 1 <= k <= kLeaveTableMaxLength.
  std::span<const LeaveTableRow> rows(int k) const;

 private:
  LeaveTable(const LeaveTableHeader& header, std::vector<LeaveTableRow> rows);

  LeaveTableHeader header_;
  std::vector<LeaveTableRow> rows_;                         // every length, length 1 first
  std::array<size_t, kLeaveTableMaxLength + 1> offsets_{};  // rows of length k start at [k - 1]
};

}  // namespace scribblez::belief
