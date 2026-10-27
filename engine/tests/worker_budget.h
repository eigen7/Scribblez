#pragma once

// Sets the process's CPU budget (util/work_pool.h) for one scope and restores
// it after, for tests that check a result does not depend on the thread count.

#include "util/work_pool.h"

namespace scribblez {

class WorkerBudget {
 public:
  explicit WorkerBudget(int n) : saved_(util::worker_count()) { util::set_worker_count(n); }
  ~WorkerBudget() { util::set_worker_count(saved_); }

  WorkerBudget(const WorkerBudget&) = delete;
  WorkerBudget& operator=(const WorkerBudget&) = delete;

 private:
  int saved_;
};

}  // namespace scribblez
