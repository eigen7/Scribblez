// util::parallel_for and util::parallel_lanes (util/work_pool.h): every item
// runs exactly once, an idle pool thread joins the caller, nested loops finish,
// a body's exception reaches the caller, and a budget of one runs everything on
// the calling thread.

#include "util/work_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
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

// Each of two items waits for the other to start, which only a second thread
// can satisfy: the caller alone would run them one after the other.
TEST_F(WorkPoolTest, AnIdlePoolThreadJoinsTheCaller) {
  util::set_worker_count(2);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  std::atomic<int> started{0};
  std::atomic<int> met{0};
  util::parallel_for(2, [&](size_t) {
    ++started;
    while (started.load() < 2 && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    met += started.load() == 2;
  });
  EXPECT_EQ(met.load(), 2);
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

// Shrinking the budget parks pool threads an earlier, wider loop spawned.
TEST_F(WorkPoolTest, BudgetOfOneRunsOnCaller) {
  util::set_worker_count(4);
  util::parallel_for(100, [](size_t) {});
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

TEST_F(WorkPoolTest, ALanesExceptionReachesCaller) {
  util::set_worker_count(4);
  const auto lane = [](util::LaneCursor& cursor) {
    for (size_t i; cursor.next(i);)
      if (i == 7) throw std::runtime_error("item 7");
  };
  EXPECT_THROW(util::parallel_lanes(20, lane), std::runtime_error);
}

}  // namespace
