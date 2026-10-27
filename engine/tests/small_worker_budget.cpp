// Caps every test binary's CPU budget (util/work_pool.h) at two threads: test
// processes run several at a time beside other work, so each stays near one
// core (py/run_tests.py), while its parallel loops still cross threads. A test
// that checks a result does not depend on the thread count sets its own budgets
// with a WorkerBudget (worker_budget.h).

#include "util/work_pool.h"

#include <gtest/gtest.h>

namespace {

class SmallWorkerBudget : public ::testing::Environment {
 public:
  void SetUp() override { scribblez::util::set_worker_count(2); }
};

const ::testing::Environment* const kSmallWorkerBudget =
  ::testing::AddGlobalTestEnvironment(new SmallWorkerBudget);

}  // namespace
