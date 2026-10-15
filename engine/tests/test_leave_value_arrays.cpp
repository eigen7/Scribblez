// LeaveValueArrays (lexicon/leave_value_arrays.h): every leave of each length,
// with the value LeaveValues gives it, sorted by value, in 8-byte rows; and
// refusal of leave values that lack a drawable leave. The full-table cases
// read HastyBot's NWL23 leave values and skip without the Macondo checkout.

#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "synthetic_equity.h"
#include "temp_dir.h"
#include "util/exception.h"
#include "util/hash.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <set>

using namespace scribblez;

namespace {

const std::string kLeaves = HastyEquity::default_leaves_path("NWL23");

class LeaveValueArraysTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (std::filesystem::exists(kLeaves)) values_ = new LeaveValues(LeaveValues::load(kLeaves));
  }
  void SetUp() override {
    if (!values_) GTEST_SKIP() << "no HastyBot leave values at " << kLeaves;
  }
  static LeaveValues* values_;
};
LeaveValues* LeaveValueArraysTest::values_ = nullptr;

Rack rack_of(const LeaveValueRow& row) {
  Rack r;
  for (uint8_t t : row.tiles) {
    if (t != LeaveValueRow::kNoTile) r.add(Tile::of(t));
  }
  return r;
}

TEST_F(LeaveValueArraysTest, RowCountsPerLength) {
  const LeaveValueArrays arrays(*values_);
  const std::array<size_t, kLeaveArraysMaxLength> expected = {27, 373, 3509, 25254, 148150, 737311};
  for (int k = 1; k <= kLeaveArraysMaxLength; ++k) {
    EXPECT_EQ(arrays.of_length(k).size(), expected[k - 1]) << "length " << k;
  }
  EXPECT_THROW(arrays.of_length(0), util::Exception);
  EXPECT_THROW(arrays.of_length(kLeaveArraysMaxLength + 1), util::Exception);
}

TEST_F(LeaveValueArraysTest, RowsAreDistinctLeavesWithTheirValuesSortedByValue) {
  const LeaveValueArrays arrays(*values_);
  for (int k = 1; k <= kLeaveArraysMaxLength; ++k) {
    std::set<std::array<uint8_t, kLeaveArraysMaxLength>> seen;
    const LeaveValueRow* prev = nullptr;
    for (const LeaveValueRow& row : arrays.of_length(k)) {
      const Rack leave = rack_of(row);
      ASSERT_EQ(leave.size(), k);
      for (int i = 1; i < k; ++i) ASSERT_LE(row.tiles[i - 1], row.tiles[i]);
      for (int i = k; i < kLeaveArraysMaxLength; ++i)
        ASSERT_EQ(row.tiles[i], LeaveValueRow::kNoTile);
      // Rounded to the nearest hundredth; a tie may round either way in float.
      ASSERT_NEAR(row.value(), values_->lookup(leave), 0.501f / LeaveValueRow::kValueScale);
      ASSERT_TRUE(seen.insert(row.tiles).second);
      if (prev) ASSERT_LE(prev->centi_value, row.centi_value);
      prev = &row;
    }
  }
}

TEST_F(LeaveValueArraysTest, RecordsWhichLeaveValues) {
  const LeaveValueArrays arrays(*values_);
  EXPECT_EQ(arrays.content_hash(), util::file_content_hash(kLeaves));
  EXPECT_EQ(arrays.content_hash(), values_->content_hash());
}

// A table missing drawable leaves (the synthetic one prices three) would
// change every probability the model normalizes over, so it is refused.
TEST(LeaveValueArrays, RefusesIncompleteLeaveValues) {
  const std::filesystem::path dir = scribblez::testing::make_temp_dir("leave_value_arrays");
  scribblez::testing::install_synthetic_hasty_equity(dir);
  EXPECT_THROW(LeaveValueArrays arrays(HastyEquity::instance().leave_values()), util::Exception);
  std::filesystem::remove_all(dir);
}

}  // namespace
