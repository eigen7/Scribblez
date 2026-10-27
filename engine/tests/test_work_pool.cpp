// util::parallel_for and util::parallel_lanes (util/work_pool.h): every item
// runs exactly once, nested loops finish, a body's exception reaches the
// caller, and a budget of one runs everything on the calling thread.

#include "util/work_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace scribblez;

namespace {

// Restores the process's CPU budget, which these cases change.
class WorkPoolTest : public ::testing::Test {
 protected:
  void TearDown() override { util::set_worker_count(saved_); }

  const int saved_ = util::worker_count();
};

TEST_F(WorkPoolTest, ParallelForRunsEveryItemOnce) {
  util::set_worker_count(4);
  std::vector<std::atomic<int>> runs(1000);
  util::parallel_for(runs.size(), [&](size_t i) { ++runs[i]; });
  for (const std::atomic<int>& r : runs) EXPECT_EQ(r.load(), 1);
}

TEST_F(WorkPoolTest, NestedLoopsFinish) {
  util::set_worker_count(3);
  std::atomic<int> total{0};
  util::parallel_for(8, [&](size_t) { util::parallel_for(50, [&](size_t) { ++total; }); });
  EXPECT_EQ(total.load(), 400);
}

TEST_F(WorkPoolTest, ExceptionReachesCaller) {
  util::set_worker_count(4);
  std::atomic<int> finished{0};
  const auto body = [&](size_t i) {
    if (i == 7) throw std::runtime_error("item 7");
    ++finished;
  };
  EXPECT_THROW(util::parallel_for(20, body), std::runtime_error);
  EXPECT_EQ(finished.load(), 19);
}

TEST_F(WorkPoolTest, BudgetOfOneRunsOnCaller) {
  util::set_worker_count(1);
  const std::thread::id caller = std::this_thread::get_id();
  std::atomic<int> elsewhere{0};
  util::parallel_for(100, [&](size_t) { elsewhere += std::this_thread::get_id() != caller; });
  EXPECT_EQ(elsewhere.load(), 0);
}

TEST_F(WorkPoolTest, LanesClaimEveryItemOnce) {
  util::set_worker_count(4);
  const size_t n = 500;
  ASSERT_EQ(util::lane_count(n), 4u);
  ASSERT_EQ(util::lane_count(2), 2u);
  std::vector<std::atomic<int>> runs(n);
  std::vector<std::atomic<int>> per_lane(util::lane_count(n));
  util::parallel_lanes(n, [&](util::LaneCursor& cursor) {
    ASSERT_LT(cursor.lane(), per_lane.size());
    for (size_t i; cursor.next(i);) {
      ++runs[i];
      ++per_lane[cursor.lane()];
    }
  });
  int total = 0;
  for (const std::atomic<int>& r : runs) EXPECT_EQ(r.load(), 1);
  for (const std::atomic<int>& c : per_lane) total += c.load();
  EXPECT_EQ(total, int(n));
}

}  // namespace
