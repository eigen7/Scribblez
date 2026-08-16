#include "data/streaming_row_buffer.h"

#include <chrono>

namespace scribblez {
namespace binlog {

namespace {

int64_t ns_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
    .count();
}

}  // namespace

StreamingRowBuffer::StreamingRowBuffer(float* const* slots, int num_slots, int rows_per_slot,
                                       int row_floats)
    : slots_(slots, slots + num_slots),
      num_slots_(num_slots),
      rows_per_slot_(rows_per_slot),
      row_floats_(row_floats),
      slot_base_(size_t(num_slots)),
      filled_count_(size_t(num_slots), 0) {
  // Slot s initially serves the fill whose base is s * rows_per_slot, so the
  // first N fills (rows [0, N*rows_per_slot)) proceed without backpressure.
  for (int s = 0; s < num_slots_; ++s) {
    slot_base_[s] = uint64_t(s) * rows_per_slot_;
  }
}

uint64_t StreamingRowBuffer::claim_row() {
  const uint64_t r = next_row_.fetch_add(1, std::memory_order_acq_rel);
  const int slot = slot_of(r);
  const uint64_t base = r - row_in(r);

  std::unique_lock<std::mutex> lock(m_);
  const auto t0 = std::chrono::steady_clock::now();
  cv_producer_.wait(lock, [&] { return slot_base_[slot] == base || stopped_; });
  producer_blocked_ns_.fetch_add(ns_since(t0), std::memory_order_relaxed);
  if (stopped_) return kNoRow;
  return r;
}

void StreamingRowBuffer::commit_row(uint64_t r) {
  const int slot = slot_of(r);
  std::lock_guard<std::mutex> lock(m_);
  rows_committed_.fetch_add(1, std::memory_order_relaxed);
  if (++filled_count_[slot] == rows_per_slot_) {
    ready_.push_back(slot);
    slots_published_.fetch_add(1, std::memory_order_relaxed);
    cv_consumer_.notify_one();
  }
}

int StreamingRowBuffer::wait_full_slot() {
  std::unique_lock<std::mutex> lock(m_);
  const auto t0 = std::chrono::steady_clock::now();
  cv_consumer_.wait(lock, [&] { return !ready_.empty() || stopped_; });
  consumer_blocked_ns_.fetch_add(ns_since(t0), std::memory_order_relaxed);
  if (stopped_) return -1;
  const int slot = ready_.front();
  ready_.pop_front();
  return slot;
}

void StreamingRowBuffer::release_slot(int slot) {
  std::lock_guard<std::mutex> lock(m_);
  filled_count_[slot] = 0;
  slot_base_[slot] += uint64_t(rows_per_slot_) * num_slots_;
  cv_producer_.notify_all();
}

void StreamingRowBuffer::stop() {
  std::lock_guard<std::mutex> lock(m_);
  stopped_ = true;
  cv_producer_.notify_all();
  cv_consumer_.notify_all();
}

RingStats StreamingRowBuffer::stats() const {
  RingStats s;
  s.rows_committed = rows_committed_.load(std::memory_order_relaxed);
  s.slots_published = slots_published_.load(std::memory_order_relaxed);
  s.producer_blocked_ns = producer_blocked_ns_.load(std::memory_order_relaxed);
  s.consumer_blocked_ns = consumer_blocked_ns_.load(std::memory_order_relaxed);
  return s;
}

}  // namespace binlog
}  // namespace scribblez
