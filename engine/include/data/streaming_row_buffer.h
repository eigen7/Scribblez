#pragma once

// The producer/consumer hand-off at the heart of the streaming training
// pipeline. The Python trainer owns N float buffers ("slots") and passes their
// addresses in. Several C++ producer threads fill the current slot at distinct
// row indices; a full slot is published to the single consumer (the training
// loop) and producers advance to the next, BLOCKING if it has not been released
// yet. Production therefore stays at most N slots ahead of consumption,
// saturating CPU (game generation) and GPU (training) at once.
//
// Concurrency design:
//   * A global monotonic counter hands each producer a row index r, which maps
//     to slot (r / rows_per_slot) % N, row r % rows_per_slot, and the base row
//     of the slot-fill it belongs to.
//   * Backpressure: a producer for row r blocks until its slot's base equals
//     r's. release_slot() advances a slot's base by rows_per_slot * N, the next
//     fill mapping there, so a slow consumer stalls producers and a runaway
//     producer can never clobber a slot the consumer still holds.
//   * Seal-exactly-once: commit_row() bumps a per-slot completion counter, and
//     the thread whose bump reaches rows_per_slot publishes the slot. Exactly
//     one bump hits the target, so this holds even though claim order and
//     completion order differ.
//
// All waits are predicate waits, so there are no lost wakeups, and no lock is
// held across the expensive row encode between claim_row and commit_row.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace scribblez {
namespace binlog {

// Throughput / backpressure counters. Growing producer_blocked_ns means
// producers wait for free slots, so the consumer/GPU is the bottleneck; growing
// consumer_blocked_ns means the reverse.
struct RingStats {
  int64_t rows_committed = 0;
  int64_t slots_published = 0;
  int64_t producer_blocked_ns = 0;
  int64_t consumer_blocked_ns = 0;
};

class StreamingRowBuffer {
 public:
  static constexpr uint64_t kNoRow = ~uint64_t(0);

  // `slots` points at `num_slots` caller-owned buffers, each at least
  // rows_per_slot * row_floats floats and valid for this object's lifetime.
  StreamingRowBuffer(float* const* slots, int num_slots, int rows_per_slot, int row_floats);

  StreamingRowBuffer(const StreamingRowBuffer&) = delete;
  StreamingRowBuffer& operator=(const StreamingRowBuffer&) = delete;

  // ---- producer side ----
  // Blocks until the next row's slot is free for this fill. Returns the global
  // row index, or kNoRow if the buffer stopped while waiting.
  uint64_t claim_row();

  float* row_dest(uint64_t r) const {
    return slots_[slot_of(r)] + static_cast<int64_t>(row_in(r)) * row_floats_;
  }

  // Publishes the slot to the consumer once its last row is committed.
  void commit_row(uint64_t r);

  // ---- consumer side ----
  // Blocks; -1 if the buffer is stopped and no more slots will come.
  int wait_full_slot();

  void release_slot(int slot);

  // Wake every blocked producer and the consumer. claim_row returns kNoRow and
  // wait_full_slot returns -1 thereafter.
  void stop();

  RingStats stats() const;

  int rows_per_slot() const { return rows_per_slot_; }
  int num_slots() const { return num_slots_; }

 private:
  int slot_of(uint64_t r) const { return static_cast<int>((r / rows_per_slot_) % num_slots_); }
  int row_in(uint64_t r) const { return static_cast<int>(r % rows_per_slot_); }

  std::vector<float*> slots_;
  int num_slots_;
  int rows_per_slot_;
  int row_floats_;

  std::atomic<uint64_t> next_row_{0};

  mutable std::mutex m_;
  std::condition_variable cv_producer_;  // producers wait for a free slot
  std::condition_variable cv_consumer_;  // consumer waits for a full slot
  std::vector<uint64_t> slot_base_;      // current fill's base row index, per slot
  std::vector<int> filled_count_;        // rows committed into the current fill, per slot
  std::deque<int> ready_;                // sealed slots awaiting the consumer
  bool stopped_ = false;

  std::atomic<int64_t> rows_committed_{0};
  std::atomic<int64_t> slots_published_{0};
  std::atomic<int64_t> producer_blocked_ns_{0};
  std::atomic<int64_t> consumer_blocked_ns_{0};
};

}  // namespace binlog
}  // namespace scribblez
