// LeaveValueArrays (lexicon/leave_value_arrays.h): every leave of each length,
// with the value LeaveValues gives it, in LeaveValues' own order, in 8-byte rows; and
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
#include <vector>

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

Rack rack_of(const LeaveValueArrays::Row& row) {
  Rack r;
  for (uint8_t t : row.tiles) {
    if (t != LeaveValueArrays::Row::kNoTile) r.add(Tile::of(t));
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

TEST_F(LeaveValueArraysTest, RowsAreTheLeavesWithTheirValuesInFileOrder) {
  const LeaveValueArrays arrays(*values_);
  std::array<std::vector<Rack>, kLeaveArraysMaxLength> order;
  values_->for_each([&](const Rack& leave, float) {
    if (leave.size() <= kLeaveArraysMaxLength) order[leave.size() - 1].push_back(leave);
  });
  for (int k = 1; k <= kLeaveArraysMaxLength; ++k) {
    const auto rows = arrays.of_length(k);
    ASSERT_EQ(rows.size(), order[k - 1].size());
    for (size_t i = 0; i < rows.size(); ++i) {
      const LeaveValueArrays::Row& row = rows[i];
      const Rack leave = rack_of(row);
      for (int l = 0; l < TILE_KINDS; ++l) {
        ASSERT_EQ(leave.counts().count(Tile::of(l)), order[k - 1][i].counts().count(Tile::of(l)))
          << "length " << k << " row " << i;
      }
      for (int j = 1; j < k; ++j) ASSERT_LE(row.tiles[j - 1], row.tiles[j]);
      for (int j = k; j < kLeaveArraysMaxLength; ++j) {
        ASSERT_EQ(row.tiles[j], LeaveValueArrays::Row::kNoTile);
      }
      // Rounded to the nearest hundredth; a tie may round either way in float.
      ASSERT_NEAR(row.value(), values_->lookup(leave), 0.00501f);
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
