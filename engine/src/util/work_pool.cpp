#include "util/work_pool.h"

#include "util/misc.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

namespace scribblez::util {

namespace {

// One parallel_for call. Workers and the caller claim items through `next`;
// the caller waits for `finished` to reach n. Held by shared_ptr, so a worker
// that claims past the end after the caller returned touches only this, never
// the caller's body.
struct Batch {
  Batch(size_t n, const std::function<void(size_t)>& body) : n(n), body(&body) {}

  bool claim(size_t& item);
  void run(size_t item);
  void drain();
  void wait();

  const size_t n;
  const std::function<void(size_t)>* body;
  std::atomic<size_t> next{0};
  std::mutex mutex;
  std::condition_variable done;
  size_t finished = 0;
  std::exception_ptr error;
};

bool Batch::claim(size_t& item) {
  item = next.fetch_add(1, std::memory_order_relaxed);
  return item < n;
}

void Batch::run(size_t item) {
  std::exception_ptr thrown;
  try {
    (*body)(item);
  } catch (...) {
    thrown = std::current_exception();
  }
  std::lock_guard<std::mutex> lock(mutex);
  if (thrown && !error) error = thrown;
  if (++finished == n) done.notify_all();
}

void Batch::drain() {
  for (size_t item; claim(item);) run(item);
}

void Batch::wait() {
  std::unique_lock<std::mutex> lock(mutex);
  done.wait(lock, [this] { return finished == n; });
  if (error) std::rethrow_exception(error);
}

// Worker i serves only while i < size - 1 (the caller is the size-th thread),
// so shrinking the budget parks the surplus threads rather than ending them.
class WorkPool {
 public:
  static WorkPool& instance();

  void set_size(int n);
  int size();
  void run(size_t n, const std::function<void(size_t)>& body);

 private:
  explicit WorkPool(int size) : size_(std::max(size, 1)) {}

  void publish(const std::shared_ptr<Batch>& batch);
  void retire(const std::shared_ptr<Batch>& batch);
  void spawn_up_to(int workers);  // requires mutex_
  void serve(int index);

  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Batch>> queue_;
  int size_;
  int spawned_ = 0;
};

// Never destroyed: its threads are detached, so process exit never waits on a
// worker that is blocked (on the GPU, say).
WorkPool& WorkPool::instance() {
  static WorkPool* pool = new WorkPool(default_thread_count());
  return *pool;
}

void WorkPool::set_size(int n) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_ = std::max(n, 1);
  wake_.notify_all();
}

int WorkPool::size() {
  std::lock_guard<std::mutex> lock(mutex_);
  return size_;
}

// A single item runs inline: there is nothing to share.
void WorkPool::run(size_t n, const std::function<void(size_t)>& body) {
  if (n == 0) return;
  auto batch = std::make_shared<Batch>(n, body);
  if (n > 1) publish(batch);
  batch->drain();
  if (n > 1) retire(batch);
  batch->wait();
}

void WorkPool::publish(const std::shared_ptr<Batch>& batch) {
  std::lock_guard<std::mutex> lock(mutex_);
  spawn_up_to(size_ - 1);
  queue_.push_back(batch);
  wake_.notify_all();
}

void WorkPool::retire(const std::shared_ptr<Batch>& batch) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::erase(queue_, batch);
}

void WorkPool::spawn_up_to(int workers) {
  while (spawned_ < workers) std::thread(&WorkPool::serve, this, spawned_++).detach();
}

void WorkPool::serve(int index) {
  while (true) {
    std::shared_ptr<Batch> batch;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return index < size_ - 1 && !queue_.empty(); });
      batch = queue_.front();
    }
    batch->drain();
    retire(batch);
  }
}

}  // namespace

void set_worker_count(int n) { WorkPool::instance().set_size(n); }

int worker_count() { return WorkPool::instance().size(); }

void parallel_for(size_t n, const std::function<void(size_t)>& body) {
  WorkPool::instance().run(n, body);
}

bool LaneCursor::next(size_t& item) {
  item = next_->fetch_add(1, std::memory_order_relaxed);
  return item < n_;
}

size_t lane_count(size_t n) { return std::min(n, size_t(worker_count())); }

void parallel_lanes(size_t n, const std::function<void(LaneCursor&)>& lane) {
  std::atomic<size_t> next{0};
  parallel_for(lane_count(n), [&](size_t k) {
    if (next.load(std::memory_order_relaxed) >= n) return;
    LaneCursor cursor(k, n, &next);
    lane(cursor);
  });
}

}  // namespace scribblez::util
