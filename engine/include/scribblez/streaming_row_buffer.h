#pragma once

// StreamingRowBuffer is the producer/consumer hand-off at the heart of the
// streaming training pipeline. The Python trainer owns N float buffers ("slots"),
// each sized rows_per_slot * row_floats, and passes their addresses in. Multiple
// C++ producer threads fill the current slot at distinct row indices; once a
// slot is full it is published to the single consumer (the Python training
// loop), and producers advance to the next slot, BLOCKING if it has not yet been
// released. This bounds production to at most N slots ahead of consumption,
// saturating CPU (game generation) and GPU (training) concurrently.
//
// Concurrency design (deadlock/race-free):
//   * A global monotonic counter hands each producer a row index r. For r:
//       slot   = (r / rows_per_slot) % N
//       row_in =  r % rows_per_slot
//       base   = r - row_in              (the slot-fill r belongs to)
//   * Backpressure: a producer for row r blocks until its slot's `base` equals
//     r's base. release_slot() advances a slot's base by rows_per_slot * N (the
//     next fill that maps to that slot), so a slow consumer stalls producers and
//     a runaway producer can never clobber a slot still held by the consumer.
//   * Seal-exactly-once: commit_row() increments a per-slot completion counter;
//     the thread whose increment reaches rows_per_slot seals the slot (publishes
//     it to the consumer). This is correct even though claim order != completion
//     order, because exactly one increment hits the target.
//
// All waits are predicate waits, so there are no lost wakeups; no lock is held
// during the (expensive) row encode that happens between claim_row/commit_row.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace scribblez {
namespace binlog {

// Throughput / backpressure counters sampled by the trainer. producer_blocked_ns
// growing means producers wait for free slots (the consumer/GPU is the
// bottleneck); consumer_blocked_ns growing means the consumer waits for full
// slots (the producers/CPU are the bottleneck).
struct RingStats {
  int64_t rows_committed = 0;
  int64_t slots_published = 0;
  int64_t producer_blocked_ns = 0;
  int64_t consumer_blocked_ns = 0;
};

class StreamingRowBuffer {
 public:
  // Sentinel returned by claim_row() when the buffer is stopped.
  static constexpr uint64_t kNoRow = ~uint64_t(0);

  // `slots` points at `num_slots` caller-owned buffers, each at least
  // rows_per_slot * row_floats floats. The pointers must stay valid for the
  // buffer's lifetime.
  StreamingRowBuffer(float* const* slots, int num_slots, int rows_per_slot, int row_floats);

  StreamingRowBuffer(const StreamingRowBuffer&) = delete;
  StreamingRowBuffer& operator=(const StreamingRowBuffer&) = delete;

  // ---- producer side ----
  // Claim the next row, blocking until its slot is free for this fill. Returns
  // the global row index, or kNoRow if the buffer was stopped while waiting.
  uint64_t claim_row();

  // Destination float* for a claimed row (start of `row_floats` floats).
  float* row_dest(uint64_t r) const {
    return slots_[slot_of(r)] + static_cast<int64_t>(row_in(r)) * row_floats_;
  }

  // Mark a claimed row written. Publishes the slot to the consumer once the
  // last row of the slot is committed.
  void commit_row(uint64_t r);

  // ---- consumer side ----
  // Block until a full slot is ready; returns its index, or -1 if the buffer is
  // stopped (no more slots will come).
  int wait_full_slot();

  // Return a consumed slot to the producers.
  void release_slot(int slot);

  // Wake every blocked producer and the consumer; claim_row returns kNoRow and
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
