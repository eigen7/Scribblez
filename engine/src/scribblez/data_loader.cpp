#include "scribblez/data_loader.h"

#include "scribblez/binary_log.h"
#include "scribblez/block_decoder.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace scribblez {
namespace binlog {

namespace {

// Read `expected_size` bytes from `path` into a fresh heap buffer.
std::unique_ptr<char[]> read_whole_file(const std::string& path, int64_t expected_size) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "DataLoader: failed to open " << path << "\n";
    return nullptr;
  }
  std::unique_ptr<char[]> buf(new char[expected_size]);
  f.read(buf.get(), expected_size);
  if (!f) {
    std::cerr << "DataLoader: short read on " << path << "\n";
    return nullptr;
  }
  return buf;
}

}  // namespace

// ===========================================================================
// DataLoader: construction / registry
// ===========================================================================

DataLoader::DataLoader(const Params& params) : params_(params) {
  if (params_.num_worker_threads < 1) params_.num_worker_threads = 1;
  if (params_.num_prefetch_threads < 1) params_.num_prefetch_threads = 1;
  if (params_.memory_budget < static_cast<int64_t>(sizeof(FileHeader))) {
    params_.memory_budget = 256LL * 1024 * 1024;
  }
}

DataLoader::~DataLoader() = default;

void DataLoader::add_file(const std::string& path, int64_t num_positions, int64_t file_size) {
  auto f = std::make_shared<DataFile>();
  f->path = path;
  f->num_positions = num_positions;
  f->file_size = file_size;
  std::lock_guard<std::mutex> lock(mu_);
  f->chrono_start = total_positions_;
  f->chrono_end = total_positions_ + num_positions;
  total_positions_ = f->chrono_end;
  files_chrono_.push_back(std::move(f));
}

int64_t DataLoader::num_positions() const {
  std::lock_guard<std::mutex> lock(mu_);
  return total_positions_;
}

int DataLoader::num_files() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(files_chrono_.size());
}

int64_t DataLoader::resident_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return resident_bytes_;
}

// ===========================================================================
// load(): main entry point
// ===========================================================================

void DataLoader::load(int64_t start, int64_t stop, bool post_move, bool apply_symmetry,
                      float* output) {
  if (output == nullptr) throw std::invalid_argument("DataLoader::load: output is null");

  // ---- snapshot the file registry ----------------------------------------
  std::vector<std::shared_ptr<DataFile>> files;
  int64_t total;
  {
    std::lock_guard<std::mutex> lock(mu_);
    files.assign(files_chrono_.begin(), files_chrono_.end());
    total = total_positions_;
  }
  if (stop > total) stop = total;
  if (start < 0) start = 0;
  if (start >= stop) {
    throw std::invalid_argument("DataLoader::load: empty range after clamp");
  }

  const int64_t n_rows = stop - start;

  // ---- per-row flip bits (one per output row) ----------------------------
  std::vector<uint8_t> per_row_flips(static_cast<size_t>(n_rows), 0);
  if (apply_symmetry) {
    std::mt19937_64 rng(std::random_device{}());
    std::bernoulli_distribution coin(0.5);
    for (int64_t i = 0; i < n_rows; ++i) per_row_flips[i] = coin(rng) ? 1u : 0u;
  }

  // ---- carve the global range into per-file contiguous slices ------------
  std::vector<WorkUnit> units;
  std::vector<std::shared_ptr<DataFile>> needed_files;
  for (auto& f : files) {
    if (f->chrono_end <= start) continue;
    if (f->chrono_start >= stop) break;
    const int64_t slice_start = std::max(f->chrono_start, start);
    const int64_t slice_stop = std::min(f->chrono_end, stop);
    WorkUnit u;
    u.file = f.get();
    u.post_move = post_move;
    u.local_start = slice_start - f->chrono_start;
    u.n_rows = slice_stop - slice_start;
    u.output_row_start = slice_start - start;
    u.flips = per_row_flips.data() + (slice_start - start);
    units.push_back(std::move(u));
    needed_files.push_back(f);
  }

  // ---- load any not-yet-resident files in parallel -----------------------
  // We hold the registry lock briefly to check which files need loading;
  // the actual disk I/O happens lock-free into per-file buffers. resident_bytes_
  // is updated atomically by each loader (it's protected by mu_, but each
  // loader takes the lock only at the end to publish its buffer).
  std::vector<DataFile*> to_load;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& f : needed_files) {
      if (!f->buffer) to_load.push_back(f.get());
    }
    // Make budget room for the new loads. Don't evict anything in
    // `needed_files` -- it's about to be (or already is) referenced.
    int64_t incoming = 0;
    for (DataFile* f : to_load) incoming += f->file_size;
    std::unordered_set<DataFile*> keep;
    for (auto& f : needed_files) keep.insert(f.get());
    // Evict oldest-first until resident_bytes_ + incoming <= budget.
    for (auto& f : files_chrono_) {
      if (resident_bytes_ + incoming <= params_.memory_budget) break;
      if (!f->buffer) continue;
      if (keep.count(f.get())) continue;
      resident_bytes_ -= f->file_size;
      f->buffer.reset();
    }
  }

  if (!to_load.empty()) {
    load_files_in_parallel(to_load);
  }

  // Sanity: every needed file is now loaded.
  for (auto& f : needed_files) {
    if (!f->buffer) {
      throw std::runtime_error("DataLoader::load: failed to load file " + f->path);
    }
  }

  // ---- decode work units in parallel -------------------------------------
  decode_units_in_parallel(units, output);

  // ---- post-load eviction: bring resident set back under budget ----------
  // (Best-effort. Files referenced by this load are eligible for eviction
  // now that we're done with them.)
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& f : files_chrono_) {
      if (resident_bytes_ <= params_.memory_budget) break;
      if (!f->buffer) continue;
      resident_bytes_ -= f->file_size;
      f->buffer.reset();
    }
  }
}

// ===========================================================================
// Parallel I/O + decode helpers
// ===========================================================================

void DataLoader::file_loader_loop(std::atomic<std::size_t>& next_idx,
                                  std::vector<DataFile*>& to_load) {
  while (true) {
    const std::size_t i = next_idx.fetch_add(1, std::memory_order_acq_rel);
    if (i >= to_load.size()) return;
    DataFile* f = to_load[i];
    auto buf = read_whole_file(f->path, f->file_size);
    std::lock_guard<std::mutex> lock(mu_);
    f->buffer = std::move(buf);
    resident_bytes_ += f->file_size;
  }
}

void DataLoader::decode_unit_loop(std::atomic<std::size_t>& next_unit,
                                  const std::vector<WorkUnit>& units, float* output) {
  // One BlockDecoder per worker thread: holds reusable scratch buffers
  // (and, eventually, a loaded Dictionary) across all units it picks up.
  BlockDecoder bd;
  while (true) {
    const std::size_t i = next_unit.fetch_add(1, std::memory_order_acq_rel);
    if (i >= units.size()) return;
    const WorkUnit& u = units[i];
    bd.decode(u.file->buffer.get(), u.file->path, u.local_start, u.n_rows, u.flips, u.post_move,
              u.output_row_start, output);
  }
}

void DataLoader::load_files_in_parallel(std::vector<DataFile*>& to_load) {
  std::atomic<std::size_t> next_idx{0};
  const int n_threads =
    std::min<int>(params_.num_prefetch_threads, static_cast<int>(to_load.size()));
  std::vector<std::thread> pool;
  pool.reserve(n_threads);
  for (int t = 0; t < n_threads - 1; ++t) {
    pool.emplace_back(&DataLoader::file_loader_loop, this, std::ref(next_idx), std::ref(to_load));
  }
  file_loader_loop(next_idx, to_load);  // current thread participates
  for (auto& t : pool) t.join();
}

void DataLoader::decode_units_in_parallel(const std::vector<WorkUnit>& units, float* output) {
  std::atomic<std::size_t> next_unit{0};
  const int n_threads = std::min<int>(params_.num_worker_threads, static_cast<int>(units.size()));
  std::vector<std::thread> pool;
  pool.reserve(n_threads);
  for (int t = 0; t < n_threads - 1; ++t) {
    pool.emplace_back(&DataLoader::decode_unit_loop, this, std::ref(next_unit), std::cref(units),
                      output);
  }
  decode_unit_loop(next_unit, units, output);  // current thread participates
  for (auto& t : pool) t.join();
}

}  // namespace binlog
}  // namespace scribblez
