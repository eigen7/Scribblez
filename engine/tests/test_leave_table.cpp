// The leave table (belief/leave_table.h): its rows are the leave enumeration
// with the leave values they came from, and it records which leave values
// those were. The row counts depend only on the tile distribution, so a
// synthetic leave table stands in for the real one and no data mount is
// needed.

#include "belief/leave_prior.h"
#include "belief/leave_table.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/hasty_equity.h"
#include "synthetic_equity.h"
#include "temp_dir.h"
#include "util/exception.h"
#include "util/hash.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <span>
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

  std::filesystem::path dir_;
};

TEST_F(LeaveTableTest, RowCountsPerLength) {
  const LeaveTable table(HastyEquity::instance());
  const std::array<size_t, kLeaveTableMaxLength> expected = {27, 373, 3509, 25254, 148150, 737311};
  for (int k = 1; k <= kLeaveTableMaxLength; ++k) {
    EXPECT_EQ(table.rows(k).size(), expected[k - 1]) << "length " << k;
  }
  EXPECT_THROW(table.rows(0), util::Exception);
  EXPECT_THROW(table.rows(kLeaveTableMaxLength + 1), util::Exception);
}

TEST_F(LeaveTableTest, RowsAreTheEnumeratedLeavesWithTheirValues) {
  const LeaveTable table(HastyEquity::instance());
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

TEST_F(LeaveTableTest, RecordsItsLeaveValues) {
  const LeaveTable table(HastyEquity::instance());
  EXPECT_EQ(table.klv_hash(), util::file_content_hash((dir_ / "synthetic.klv2").string()));
}

}  // namespace
