#pragma once

// Multithreaded loader that streams sampled training rows from a set of
// .slog files into a caller-provided float buffer. Modelled on the
// AlphaZeroArcade DataLoader but specialized for the Scribblez binary log
// format (which has fixed-size, self-contained PositionRecords; no
// per-frame history walk is needed).
//
// Lifecycle
// ---------
//   1. Construct with a Params (memory budget, thread counts).
//   2. Register every existing .slog file in chronological order via
//      add_file(). The loader assumes a strict newest-last add order.
//   3. Per training epoch, call load(window_start, window_end, n_samples,
//      apply_symmetry, output_buffer). Returns when output_buffer is fully
//      populated and shuffled.
//   4. Optionally call add_file() between load()s as new self-play data
//      arrives.
//
// Output row layout (row_size_floats() floats per row, n_samples rows)
// -------------------------------------------------------------------
//   [ input_floats:    kInputFloats              ]
//   [ wld onehot:      kWldFloats              3 ]  // [win, draw, loss] (POV)
//   [ score_diff:      kScoreDiffFloats        1 ]  // final_active - final_opp
//   [ opp_next_place:  kOppNextPlacementFloats 225 ]  // 15x15 binary mask
//
// (Label layout is owned by label_encoder.h; the constants are re-exported
// from this header for downstream convenience.)
//
// Concurrency model
// -----------------
// load() is synchronous: it spawns up to `num_prefetch_threads` workers to
// read any not-yet-resident files from disk in parallel, then spawns up to
// `num_worker_threads` decoders to parse PositionRecords and write directly
// into disjoint slices of the caller's output buffer. Both pools are
// per-call (created/joined inside load()); thread-creation cost (~10us per
// thread) is dwarfed by an entire epoch's worth of work. A future API
// extension can add prefetch_next(...) -> std::future to overlap epoch N's
// I/O with epoch N-1's training without changing this core API.
//
// Resident files are managed against a `memory_budget`. After each load(),
// the loader evicts files (chronological-oldest-first, skipping any still
// referenced by the just-completed load) until total resident bytes <=
// budget.

#include "scribblez/binary_log.h"
#include "scribblez/input_encoder.h"
#include "scribblez/label_encoder.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

// `kInputFloats` is owned by input_encoder.h; the label constants
// (kWldFloats / kScoreDiffFloats / kLabelFloats) are owned by
// label_encoder.h and re-included above.

inline constexpr int kRowFloats = kInputFloats + kLabelFloats;

class DataLoader {
 public:
  struct Params {
    int64_t memory_budget = 256LL * 1024 * 1024;  // 256 MB resident buffers
    int num_worker_threads = 4;                   // decoder pool size per load()
    int num_prefetch_threads = 2;                 // disk-I/O pool size per load()
  };

  explicit DataLoader(const Params& params);
  ~DataLoader();

  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;

  // Register one file. Caller must invoke in chronological (oldest-first)
  // order; the loader treats the most-recently-added file as the "newest".
  // `num_positions` and `file_size` must match the on-disk header.
  void add_file(const std::string& path, int64_t num_positions, int64_t file_size);

  // Total positions across all currently registered files.
  int64_t num_positions() const;

  // Number of registered files.
  int num_files() const;

  // Total bytes currently resident in memory across loaded file buffers.
  int64_t resident_bytes() const;

  // Decode `n_samples` rows into `output` (must have capacity for at least
  // n_samples * row_size_floats() floats). Indices are drawn uniformly with
  // replacement from the master-array slice [window_start, window_end);
  // window_end is clamped to num_positions(). Rows are written in
  // file-grouped order then chunked-shuffled, so each output row is a
  // uniform random sample.
  //
  // If `apply_symmetry` is true, each output row independently gets a fair
  // coin flip: a 0 keeps the row in canonical orientation; a 1 transposes
  // every spatial plane across the main diagonal. Heads 0 (WLD) and 1
  // (score diff) are flip-invariant; head 2 (opp next placement) is also
  // transposed in lockstep with the input planes so it stays aligned.
  void load(int64_t window_start, int64_t window_end, int n_samples, bool apply_symmetry,
            float* output);

  static constexpr int row_size_floats() { return kRowFloats; }
  static constexpr int input_size_floats() { return kInputFloats; }
  static constexpr int label_size_floats() { return kLabelFloats; }

 private:
  // One registered file. Owns its in-memory buffer once loaded.
  struct DataFile {
    std::string path;
    int64_t num_positions = 0;
    int64_t file_size = 0;
    int64_t chrono_start = 0;  // master-array index of this file's first row
    int64_t chrono_end = 0;
    std::unique_ptr<char[]> buffer;  // nullptr iff not loaded
  };

  // One unit of decoder work: a contiguous block of output rows fed by one
  // file. local_indices are positions-within-the-file (0..num_positions-1),
  // sorted ascending so the decoder walks the file linearly. `flips[i]`
  // selects whether output row (output_row_start + i) gets the diagonal
  // symmetry applied (1) or not (0); always zero-filled when load() was
  // called with apply_symmetry=false.
  struct WorkUnit {
    DataFile* file = nullptr;
    std::vector<int64_t> local_indices;
    std::vector<uint8_t> flips;  // aligned with local_indices
    int64_t output_row_start = 0;
  };

  Params params_;

  // Single mutex guards the file registry and resident_bytes_. load() takes
  // this lock only at entry (to snapshot files / mutate buffers / update
  // counters) and at exit (for eviction); the parallel I/O and decode phases
  // run lock-free since each thread owns disjoint files / output rows.
  mutable std::mutex mu_;
  std::deque<std::shared_ptr<DataFile>> files_chrono_;  // oldest first
  int64_t total_positions_ = 0;
  int64_t resident_bytes_ = 0;
};

}  // namespace binlog
}  // namespace scribblez
