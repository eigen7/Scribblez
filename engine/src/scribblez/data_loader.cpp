#include "scribblez/data_loader.h"

#include "scribblez/binary_log.h"
#include "scribblez/input_encoder.h"
#include "scribblez/label_encoder.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

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

// In-place chunked Fisher-Yates shuffle of `n` rows, each `row_size` floats.
void chunked_shuffle(float* base, int64_t n, int row_size, std::mt19937_64& rng) {
  std::vector<float> tmp(row_size);
  const size_t bytes = static_cast<size_t>(row_size) * sizeof(float);
  for (int64_t i = n - 1; i > 0; --i) {
    std::uniform_int_distribution<int64_t> dist(0, i);
    const int64_t j = dist(rng);
    if (i == j) continue;
    std::memcpy(tmp.data(), base + i * row_size, bytes);
    std::memcpy(base + i * row_size, base + j * row_size, bytes);
    std::memcpy(base + j * row_size, tmp.data(), bytes);
  }
}

// Decode the positions named by `local_indices` (within the file whose buffer
// is `buf`) directly into `output`. Assumes the buffer is loaded and stable
// for the duration of the call. `flips[i]` selects whether output row
// (output_row_start + i) gets the diagonal symmetry applied.
void decode_block(const char* buf, const std::string& path, const int64_t* local_indices,
                  const uint8_t* flips, size_t n_indices, int64_t output_row_start, float* output) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf);
  if (hdr->magic != kMagic) {
    std::cerr << "DataLoader: bad magic in " << path << "\n";
    return;
  }
  if (hdr->version != kVersion) {
    std::cerr << "DataLoader: version mismatch in " << path << " (file=" << hdr->version
              << " code=" << kVersion << ")\n";
    return;
  }
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const uint32_t num_games = hdr->num_games;

  // Prefix sum: prefix[g] == total positions in games [0, g).
  std::vector<uint32_t> prefix(num_games + 1, 0);
  for (uint32_t g = 0; g < num_games; ++g) {
    prefix[g + 1] = prefix[g] + metas[g].num_positions;
  }

  uint32_t game_cursor = 0;
  for (size_t i = 0; i < n_indices; ++i) {
    const int64_t li = local_indices[i];
    while (game_cursor + 1 < num_games && prefix[game_cursor + 1] <= li) ++game_cursor;
    const GameMetadata& gm = metas[game_cursor];
    const uint32_t intra = static_cast<uint32_t>(li) - prefix[game_cursor];
    const PositionRecord* rec = reinterpret_cast<const PositionRecord*>(
      buf + gm.start_offset + intra * sizeof(PositionRecord));

    float* row = output + (output_row_start + static_cast<int64_t>(i)) * kRowFloats;
    encode_input(*rec, /*apply_flip=*/flips[i] != 0, row);

    // Build a GameLogView over this game's moves blob (which sits in the
    // file right after the game's PositionRecord blob).
    const Move* moves = reinterpret_cast<const Move*>(buf + gm.start_offset + gm.data_size);
    GameLogView view{};
    view.moves = moves;
    view.num_turns = static_cast<int>(gm.num_turns);
    view.turn_index = rec->move_number;
    view.kind = static_cast<PositionKind>(rec->position_kind);
    view.active_player = rec->active_player;
    view.final_score_p0 = gm.final_score_p0;
    view.final_score_p1 = gm.final_score_p1;
    view.apply_flip = flips[i] != 0;

    float* heads[kNumLabelHeads] = {
      row + kInputFloats,
      row + kInputFloats + kWldFloats,
      row + kInputFloats + kWldFloats + kScoreDiffFloats,
    };
    encode_labels(view, heads);
  }
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

void DataLoader::load(int64_t window_start, int64_t window_end, int n_samples, bool apply_symmetry,
                      float* output) {
  if (n_samples <= 0) return;
  if (output == nullptr) throw std::invalid_argument("DataLoader::load: output is null");

  // ---- snapshot the file registry ----------------------------------------
  std::vector<std::shared_ptr<DataFile>> files;
  int64_t total;
  {
    std::lock_guard<std::mutex> lock(mu_);
    files.assign(files_chrono_.begin(), files_chrono_.end());
    total = total_positions_;
  }
  if (window_end > total) window_end = total;
  if (window_start < 0) window_start = 0;
  if (window_start >= window_end) {
    throw std::invalid_argument("DataLoader::load: empty window after clamp");
  }

  // ---- draw n_samples global indices uniformly with replacement ----------
  std::random_device rd;
  std::mt19937_64 rng(static_cast<uint64_t>(rd()) ^ 0x9E3779B97F4A7C15ULL);
  std::uniform_int_distribution<int64_t> dist(window_start, window_end - 1);
  std::vector<int64_t> global_indices(n_samples);
  for (int i = 0; i < n_samples; ++i) global_indices[i] = dist(rng);

  // ---- bucket by file ----------------------------------------------------
  std::sort(global_indices.begin(), global_indices.end());
  // Per-row flip bits (one per output row, aligned with global_indices after
  // the sort). When apply_symmetry is false they're all zero.
  std::vector<uint8_t> per_row_flips(n_samples, 0);
  if (apply_symmetry) {
    std::bernoulli_distribution coin(0.5);
    for (int i = 0; i < n_samples; ++i) per_row_flips[i] = coin(rng) ? 1u : 0u;
  }

  std::vector<WorkUnit> units;
  std::vector<std::shared_ptr<DataFile>> needed_files;
  size_t cursor = 0;
  int64_t row_cursor = 0;
  for (auto& f : files) {
    if (cursor >= global_indices.size()) break;
    if (global_indices[cursor] >= f->chrono_end) continue;
    WorkUnit u;
    u.file = f.get();
    u.output_row_start = row_cursor;
    while (cursor < global_indices.size() && global_indices[cursor] < f->chrono_end) {
      u.local_indices.push_back(global_indices[cursor] - f->chrono_start);
      u.flips.push_back(per_row_flips[cursor]);
      ++cursor;
    }
    row_cursor += static_cast<int64_t>(u.local_indices.size());
    units.push_back(std::move(u));
    needed_files.push_back(f);
  }
  assert(cursor == global_indices.size());
  assert(row_cursor == n_samples);

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
    std::atomic<size_t> next_idx{0};
    auto loader = [&]() {
      while (true) {
        const size_t i = next_idx.fetch_add(1, std::memory_order_acq_rel);
        if (i >= to_load.size()) return;
        DataFile* f = to_load[i];
        auto buf = read_whole_file(f->path, f->file_size);
        std::lock_guard<std::mutex> lock(mu_);
        f->buffer = std::move(buf);
        resident_bytes_ += f->file_size;
      }
    };
    const int n_threads =
      std::min<int>(params_.num_prefetch_threads, static_cast<int>(to_load.size()));
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (int t = 0; t < n_threads - 1; ++t) pool.emplace_back(loader);
    loader();  // current thread participates
    for (auto& t : pool) t.join();
  }

  // Sanity: every needed file is now loaded.
  for (auto& f : needed_files) {
    if (!f->buffer) {
      throw std::runtime_error("DataLoader::load: failed to load file " + f->path);
    }
  }

  // ---- decode work units in parallel -------------------------------------
  {
    std::atomic<size_t> next_unit{0};
    auto decoder = [&]() {
      while (true) {
        const size_t i = next_unit.fetch_add(1, std::memory_order_acq_rel);
        if (i >= units.size()) return;
        const WorkUnit& u = units[i];
        decode_block(u.file->buffer.get(), u.file->path, u.local_indices.data(), u.flips.data(),
                     u.local_indices.size(), u.output_row_start, output);
      }
    };
    const int n_threads = std::min<int>(params_.num_worker_threads, static_cast<int>(units.size()));
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (int t = 0; t < n_threads - 1; ++t) pool.emplace_back(decoder);
    decoder();
    for (auto& t : pool) t.join();
  }

  // ---- shuffle output rows so per-file grouping doesn't bias minibatches -
  chunked_shuffle(output, n_samples, kRowFloats, rng);

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

}  // namespace binlog
}  // namespace scribblez
