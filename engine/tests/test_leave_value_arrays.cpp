// LeaveValueArrays (lexicon/leave_value_arrays.h): every leave of each length,
// in LeaveValues' own order, as per-slot tile codes and float values that
// decode back to exactly the leaves and values LeaveValues holds; and refusal
// of leave values that lack a drawable leave. The full-table cases read
// HastyBot's NWL23 leave values and skip without the Macondo checkout.

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

TEST_F(LeaveValueArraysTest, RowCountsPerLength) {
  const LeaveValueArrays arrays(*values_);
  const std::array<size_t, kLeaveArraysMaxLength> expected = {27, 373, 3509, 25254, 148150, 737311};
  for (int k = 1; k <= kLeaveArraysMaxLength; ++k) {
    EXPECT_EQ(arrays.rows(k), expected[k - 1]) << "length " << k;
    EXPECT_EQ(arrays.values(k).size(), expected[k - 1]);
    for (int slot = 0; slot < k; ++slot) EXPECT_EQ(arrays.codes(k, slot).size(), expected[k - 1]);
  }
  EXPECT_THROW(arrays.rows(0), util::Exception);
  EXPECT_THROW(arrays.rows(kLeaveArraysMaxLength + 1), util::Exception);
  EXPECT_THROW(arrays.codes(3, 3), util::Exception);
}

// Each row's codes name its tiles in ascending letter order, each copy of a
// letter numbered from 0, and its value is LeaveValues'; rows follow
// LeaveValues::for_each.
TEST_F(LeaveValueArraysTest, RowsDecodeToTheLeavesInFileOrder) {
  const LeaveValueArrays arrays(*values_);
  std::array<std::vector<Rack>, kLeaveArraysMaxLength> order;
  values_->for_each([&](const Rack& leave, float) {
    if (leave.size() <= kLeaveArraysMaxLength) order[leave.size() - 1].push_back(leave);
  });
  constexpr int kCopies = LeaveValueArrays::kCopiesPerTile;
  for (int k = 1; k <= kLeaveArraysMaxLength; ++k) {
    ASSERT_EQ(arrays.rows(k), order[k - 1].size());
    for (size_t i = 0; i < arrays.rows(k); ++i) {
      std::array<int, TILE_KINDS> counts{};
      int prev = -1;
      for (int slot = 0; slot < k; ++slot) {
        const int code = arrays.codes(k, slot)[i];
        const int letter = code / kCopies;
        ASSERT_GE(code, prev) << "length " << k << " row " << i;
        ASSERT_EQ(code % kCopies, counts[letter]) << "copies number from 0 in order";
        ++counts[letter];
        prev = code;
      }
      const Rack& leave = order[k - 1][i];
      for (int l = 0; l < TILE_KINDS; ++l) {
        ASSERT_EQ(counts[l], leave.counts().count(Tile::of(l))) << "length " << k << " row " << i;
      }
      ASSERT_EQ(arrays.values(k)[i], values_->lookup(leave));
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
