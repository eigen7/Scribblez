#include "scribblez/data_loader.h"

#include "scribblez/binary_log.h"
#include "scribblez/block_decoder.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
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

DataLoader::~DataLoader() { wait_prefetch(); }

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

// ===========================================================================
// LRU management
// ===========================================================================

void DataLoader::touch_lru(DataFile* file) {
  // Move `file` to the back of the LRU list (most recently used).
  auto it = std::find(lru_order_.begin(), lru_order_.end(), file);
  if (it != lru_order_.end()) {
    lru_order_.erase(it);
  }
  lru_order_.push_back(file);
}

void DataLoader::evict_until_budget(DataFile* keep) {
  // Evict from front of LRU (least recently used) until under budget.
  // Stop if we've cycled through all entries without evicting (only protected
  // files remain).
  size_t scanned = 0;
  while (resident_bytes_ > params_.memory_budget && !lru_order_.empty()) {
    if (scanned >= lru_order_.size()) break;  // no evictable files remain
    DataFile* victim = lru_order_.front();
    lru_order_.pop_front();
    if (victim == keep || !victim->buffer) {
      // Can't evict this one; put it back at the end and continue scanning.
      lru_order_.push_back(victim);
      ++scanned;
      continue;
    }
    resident_bytes_ -= victim->file_size;
    victim->buffer.reset();
    scanned = 0;  // made progress; reset scan counter
  }
}

bool DataLoader::ensure_resident(DataFile* file) {
  // Must be called with mu_ held.
  if (file->buffer) {
    touch_lru(file);
    return true;
  }
  // Need to load -- evict first to make room.
  evict_until_budget(file);

  // Release mu_ during I/O (file identity is stable since we hold shared_ptr).
  mu_.unlock();
  auto buf = read_whole_file(file->path, file->file_size);
  mu_.lock();

  if (!buf) return false;
  file->buffer = std::move(buf);
  resident_bytes_ += file->file_size;
  touch_lru(file);
  evict_until_budget(file);
  return true;
}

// ===========================================================================
// Prefetch
// ===========================================================================

void DataLoader::start_prefetch(DataFile* file) {
  wait_prefetch();  // join any prior prefetch
  if (!file || file->buffer) return;
  {
    std::lock_guard<std::mutex> lock(prefetch_mu_);
    prefetch_target_ = file;
    prefetch_done_ = false;
  }
  prefetch_thread_ = std::thread([this, file]() {
    auto buf = read_whole_file(file->path, file->file_size);
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (buf && !file->buffer) {
        file->buffer = std::move(buf);
        resident_bytes_ += file->file_size;
        touch_lru(file);
      }
    }
    {
      std::lock_guard<std::mutex> lock(prefetch_mu_);
      prefetch_done_ = true;
    }
    prefetch_cv_.notify_one();
  });
}

void DataLoader::wait_prefetch() {
  if (prefetch_thread_.joinable()) {
    prefetch_thread_.join();
  }
  prefetch_target_ = nullptr;
}

// ===========================================================================
// V2 epoch-based streaming API
// ===========================================================================

int DataLoader::epoch_start(const EpochConfig& config) {
  wait_prefetch();

  std::lock_guard<std::mutex> lock(epoch_mu_);

  epoch_config_ = config;
  epoch_cursor_ = 0;

  // Snapshot and shuffle files.
  {
    std::lock_guard<std::mutex> flock(mu_);
    epoch_files_.assign(files_chrono_.begin(), files_chrono_.end());
  }

  if (epoch_files_.empty() || config.batch_size <= 0) {
    epoch_active_ = false;
    epoch_order_.clear();
    epoch_flips_.clear();
    return 0;
  }

  // Shuffle file order using the seed.
  std::mt19937_64 file_rng(config.seed);
  std::shuffle(epoch_files_.begin(), epoch_files_.end(), file_rng);

  // Build the flattened iteration order: for each file in shuffled order,
  // generate a permutation of its local positions.
  int64_t total = 0;
  for (auto& f : epoch_files_) total += f->num_positions;

  epoch_order_.clear();
  epoch_order_.reserve(static_cast<size_t>(total));

  for (int fi = 0; fi < static_cast<int>(epoch_files_.size()); ++fi) {
    const int64_t n = epoch_files_[fi]->num_positions;
    std::vector<int64_t> perm(static_cast<size_t>(n));
    std::iota(perm.begin(), perm.end(), int64_t{0});
    // Each file gets a deterministic sub-seed.
    std::mt19937_64 pos_rng(config.seed ^ static_cast<uint64_t>(fi + 1));
    std::shuffle(perm.begin(), perm.end(), pos_rng);
    for (int64_t p : perm) {
      epoch_order_.push_back(EpochPosition{fi, p});
    }
  }

  // Pre-compute flip bits for the entire epoch.
  epoch_flips_.resize(static_cast<size_t>(total));
  if (config.apply_symmetry) {
    std::mt19937_64 flip_rng(config.seed ^ 0xDEADBEEFCAFEF00DULL);
    std::bernoulli_distribution coin(0.5);
    for (size_t i = 0; i < epoch_flips_.size(); ++i) {
      epoch_flips_[i] = coin(flip_rng) ? 1u : 0u;
    }
  } else {
    std::fill(epoch_flips_.begin(), epoch_flips_.end(), 0u);
  }

  epoch_active_ = true;
  return static_cast<int>(total / config.batch_size);
}

int DataLoader::load_batch(float* output) {
  if (!output) throw std::invalid_argument("DataLoader::load_batch: output is null");

  std::lock_guard<std::mutex> lock(epoch_mu_);
  if (!epoch_active_) return 0;

  const int64_t total = static_cast<int64_t>(epoch_order_.size());
  if (epoch_cursor_ >= total) {
    epoch_active_ = false;
    return 0;
  }

  const int64_t batch_start = epoch_cursor_;
  const int64_t batch_end = std::min(batch_start + epoch_config_.batch_size, total);
  const int n_rows = static_cast<int>(batch_end - batch_start);
  epoch_cursor_ = batch_end;

  // Determine which files this batch touches and identify the next file
  // for prefetch.
  std::unordered_set<int> needed_file_idxs;
  for (int64_t i = batch_start; i < batch_end; ++i) {
    needed_file_idxs.insert(epoch_order_[i].file_idx);
  }

  // Identify next file for prefetch (first file touched by the next batch).
  DataFile* prefetch_file = nullptr;
  if (batch_end < total) {
    int next_fi = epoch_order_[batch_end].file_idx;
    if (!epoch_files_[next_fi]->buffer &&
        needed_file_idxs.find(next_fi) == needed_file_idxs.end()) {
      prefetch_file = epoch_files_[next_fi].get();
    }
  }

  // Ensure all needed files are resident.
  wait_prefetch();

  {
    std::lock_guard<std::mutex> flock(mu_);
    // First pass: load all needed files. We must be careful not to evict
    // a file we need for this batch when loading another. So we temporarily
    // mark all needed files as "keep" by loading them without evicting
    // each other.
    std::vector<DataFile*> batch_files;
    for (int fi : needed_file_idxs) {
      batch_files.push_back(epoch_files_[fi].get());
    }

    for (DataFile* f : batch_files) {
      if (f->buffer) {
        touch_lru(f);
        continue;
      }
      // Evict, but protect all batch files.
      // Evict from LRU front, skipping any file in batch_files.
      size_t scanned = 0;
      while (resident_bytes_ > params_.memory_budget && !lru_order_.empty() &&
             scanned < lru_order_.size()) {
        DataFile* victim = lru_order_.front();
        lru_order_.pop_front();
        bool is_needed = false;
        for (DataFile* bf : batch_files) {
          if (victim == bf) {
            is_needed = true;
            break;
          }
        }
        if (is_needed || !victim->buffer) {
          lru_order_.push_back(victim);
          ++scanned;
          continue;
        }
        resident_bytes_ -= victim->file_size;
        victim->buffer.reset();
        scanned = 0;
      }

      // Load the file (release mu_ during I/O).
      mu_.unlock();
      auto buf = read_whole_file(f->path, f->file_size);
      mu_.lock();
      if (!buf) {
        throw std::runtime_error("DataLoader::load_batch: failed to load file " + f->path);
      }
      f->buffer = std::move(buf);
      resident_bytes_ += f->file_size;
      touch_lru(f);
    }
  }

  // Start prefetch for the next file (after ensuring current files are loaded).
  if (prefetch_file) {
    start_prefetch(prefetch_file);
  }

  // Build work units and decode. For epoch batches, each row is potentially
  // from a different local offset, so we create one WorkUnit per row. To
  // allow parallel decode, we group consecutive rows touching the same file.
  struct BatchRow {
    int file_idx;
    int64_t local_pos;
    uint8_t flip;
    int output_idx;
  };
  std::vector<BatchRow> rows(static_cast<size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    const auto& ep = epoch_order_[batch_start + i];
    rows[i] = {ep.file_idx, ep.local_pos, epoch_flips_[batch_start + i], i};
  }

  // Sort by file_idx for locality (groups consecutive same-file rows).
  std::sort(rows.begin(), rows.end(),
            [](const BatchRow& a, const BatchRow& b) { return a.file_idx < b.file_idx; });

  // Decode rows using a BlockDecoder.
  BlockDecoder bd;
  for (const auto& row : rows) {
    DataFile* file = epoch_files_[row.file_idx].get();
    assert(file->buffer);
    // Decode a single row.
    uint8_t flip = row.flip;
    bd.decode(file->buffer.get(), file->path, row.local_pos, /*n_rows=*/1, &flip,
              epoch_config_.post_move, /*output_row_start=*/row.output_idx, output);
  }

  // Post-batch eviction.
  {
    std::lock_guard<std::mutex> flock(mu_);
    evict_until_budget(prefetch_file);
  }

  if (batch_end >= total) {
    epoch_active_ = false;
  }

  return n_rows;
}

}  // namespace binlog
}  // namespace scribblez
