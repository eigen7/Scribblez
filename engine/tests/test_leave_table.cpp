// The leave table (belief/leave_table.h): its rows match the leave enumeration
// and the leave values they came from, the header identifies its inputs, a
// written table loads back identically, and load refuses a file it cannot
// trust. The row counts depend only on the tile distribution, so a synthetic
// leave table stands in for the real one and no data mount is needed.

#include "belief/leave_prior.h"
#include "belief/leave_table.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/hasty_equity.h"
#include "synthetic_equity.h"
#include "temp_dir.h"
#include "util/exception.h"
#include "util/hash.h"
#include "util/io.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

using namespace scribblez;
using namespace scribblez::belief;

namespace {

class LeaveTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = scribblez::testing::make_temp_dir("leave_table");
    scribblez::testing::install_synthetic_hasty_equity(dir_);
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }

  LeaveTable build() const { return LeaveTable::build(HastyEquity::instance(), "NWL23"); }
  std::string path(const std::string& name) const { return (dir_ / name).string(); }

  // Writes the built table, applies `edit` to its bytes, and saves the result.
  template <typename F>
  std::string write_edited(F&& edit) const {
    const std::string clean = path("clean.ltab");
    build().write(clean);
    std::vector<char> bytes = util::read_file_bytes(clean);
    edit(bytes);
    const std::string edited = path("edited.ltab");
    std::ofstream(edited, std::ios::binary).write(bytes.data(), std::streamsize(bytes.size()));
    return edited;
  }

  std::filesystem::path dir_;
};

std::span<const char> bytes_of(std::span<const LeaveTableRow> rows) {
  return {reinterpret_cast<const char*>(rows.data()), rows.size_bytes()};
}

TEST_F(LeaveTableTest, RowCountsPerLength) {
  const LeaveTable table = build();
  const std::array<size_t, kLeaveTableMaxLength> expected = {27, 373, 3509, 25254, 148150, 737311};
  for (int k = 1; k <= kLeaveTableMaxLength; ++k) {
    EXPECT_EQ(table.rows(k).size(), expected[k - 1]) << "length " << k;
    EXPECT_EQ(table.header().row_counts[k - 1], expected[k - 1]) << "length " << k;
  }
  EXPECT_THROW(table.rows(0), util::Exception);
  EXPECT_THROW(table.rows(kLeaveTableMaxLength + 1), util::Exception);
}

TEST_F(LeaveTableTest, RowsAreTheEnumeratedLeavesWithTheirValues) {
  const LeaveTable table = build();
  const HastyEquity& equity = HastyEquity::instance();
  for (int k = 1; k <= kLeaveTableMaxLength; ++k) {
    const std::vector<ScoredLeave> leaves = enumerate_leaves(TileCounts::full_distribution(), k);
    const std::span<const LeaveTableRow> rows = table.rows(k);
    ASSERT_EQ(rows.size(), leaves.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      const TileCounts counts = leaves[i].leave.counts();
      int size = 0;
      for (int l = 0; l < TILE_KINDS; ++l) {
        ASSERT_EQ(rows[i].counts[l], counts.count(Tile::of(l))) << "length " << k << " row " << i;
        ASSERT_EQ(bool(rows[i].mask >> l & 1u), rows[i].counts[l] > 0);
        size += rows[i].counts[l];
      }
      ASSERT_EQ(size, k);
      ASSERT_EQ(rows[i].mask >> TILE_KINDS, 0u);
      ASSERT_EQ(rows[i].value, float(equity.leave_value(leaves[i].leave)));
    }
  }
}

TEST_F(LeaveTableTest, HeaderIdentifiesItsInputs) {
  const LeaveTable table = build();
  EXPECT_EQ(table.header().magic, kLeaveTableMagic);
  EXPECT_EQ(table.header().version, kLeaveTableFormatVersion);
  EXPECT_EQ(table.header().flags, 0);
  EXPECT_EQ(table.lexicon(), "NWL23");
  EXPECT_EQ(table.klv_hash(), util::file_content_hash(path("synthetic.klv2")));
  std::vector<char> payload;
  for (int k = 1; k <= kLeaveTableMaxLength; ++k) {
    const std::span<const char> b = bytes_of(table.rows(k));
    payload.insert(payload.end(), b.begin(), b.end());
  }
  EXPECT_EQ(table.content_id(), util::content_hash(payload));
}

TEST_F(LeaveTableTest, WriteLoadRoundTrip) {
  const LeaveTable built = build();
  built.write(path("t.ltab"));
  const LeaveTable loaded = LeaveTable::load(path("t.ltab"));
  EXPECT_EQ(std::memcmp(&built.header(), &loaded.header(), sizeof(LeaveTableHeader)), 0);
  for (int k = 1; k <= kLeaveTableMaxLength; ++k) {
    const std::span<const char> a = bytes_of(built.rows(k));
    const std::span<const char> b = bytes_of(loaded.rows(k));
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0) << "length " << k;
  }
}

TEST_F(LeaveTableTest, LoadRejectsWrongMagic) {
  const std::string p = write_edited([](std::vector<char>& b) { b[0] = 'X'; });
  EXPECT_THROW(LeaveTable::load(p), util::Exception);
}

TEST_F(LeaveTableTest, LoadRejectsWrongVersion) {
  const std::string p = write_edited([](std::vector<char>& b) {
    const uint16_t v = kLeaveTableFormatVersion + 1;
    std::memcpy(b.data() + offsetof(LeaveTableHeader, version), &v, sizeof(v));
  });
  EXPECT_THROW(LeaveTable::load(p), util::Exception);
}

TEST_F(LeaveTableTest, LoadRejectsTruncatedFile) {
  EXPECT_THROW(LeaveTable::load(write_edited([](std::vector<char>& b) { b.pop_back(); })),
               util::Exception);
  EXPECT_THROW(LeaveTable::load(write_edited([](std::vector<char>& b) { b.resize(10); })),
               util::Exception);
}

}  // namespace
