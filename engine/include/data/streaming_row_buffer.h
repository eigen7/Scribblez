#pragma once

// The ring buffer between the C++ game producers and the Python training loop
// in streaming training (arena/streaming_game_producer.h). The trainer owns N
// float buffers ("slots") and passes their addresses in. Producer threads fill
// rows of the current slot concurrently; a full slot goes to the single
// consumer, and producers move on to the next slot, blocking until the
// consumer has released it. Production thus runs at most N slots ahead of
// training, keeping both the CPU (game generation) and the GPU (training)
// busy.
//
// A producer calls claim_row(), writes the row at row_dest(), then calls
// commit_row(). No lock is held while it encodes the row.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace scribblez {
namespace binlog {

// Throughput and backpressure counters. Growing producer_blocked_ns means the
// consumer (GPU) is the bottleneck; growing consumer_blocked_ns means the
// producers (CPU) are.
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
  // Blocks until the next row's slot is free. Returns the row's global index,
  // or kNoRow once the buffer is stopped.
  uint64_t claim_row();

  float* row_dest(uint64_t r) const {
    return slots_[slot_of(r)] + int64_t(row_in(r)) * row_floats_;
  }

  // Hands the slot to the consumer once all its rows are committed.
  void commit_row(uint64_t r);

  // ---- consumer side ----
  // Blocks until a slot is full and returns it, or -1 once the buffer is
  // stopped. The consumer must release_slot() it when done.
  int wait_full_slot();

  void release_slot(int slot);

  // Unblocks every producer and the consumer; see claim_row and
  // wait_full_slot.
  void stop();

  RingStats stats() const;

  int rows_per_slot() const { return rows_per_slot_; }
  int num_slots() const { return num_slots_; }

 private:
  int slot_of(uint64_t r) const { return (r / rows_per_slot_) % num_slots_; }
  int row_in(uint64_t r) const { return r % rows_per_slot_; }

  std::vector<float*> slots_;
  int num_slots_;
  int rows_per_slot_;
  int row_floats_;

  std::atomic<uint64_t> next_row_{0};

  mutable std::mutex m_;
  std::condition_variable cv_producer_;  // producers wait for a free slot
  std::condition_variable cv_consumer_;  // consumer waits for a full slot
  std::vector<uint64_t> slot_base_;      // per slot: first global row of its current fill
  std::vector<int> filled_count_;        // per slot: rows committed to its current fill
  std::deque<int> ready_;                // sealed slots awaiting the consumer
  bool stopped_ = false;

  std::atomic<int64_t> rows_committed_{0};
  std::atomic<int64_t> slots_published_{0};
  std::atomic<int64_t> producer_blocked_ns_{0};
  std::atomic<int64_t> consumer_blocked_ns_{0};
};

}  // namespace binlog
}  // namespace scribblez
