#pragma once

// Multithreaded loader that streams sampled training rows from a set of
// .slog files into a caller-provided float buffer.
//
// Lifecycle
// ---------
//   1. Construct with a Params (memory budget, thread counts).
//   2. Register every existing .slog file in chronological order via
//      add_file(). The loader assumes a strict newest-last add order.
//   3. To consume the master array, call load(start, stop, post_move,
//      apply_symmetry, output_buffer). Rows are written sequentially into
//      `output_buffer` (capacity at least (stop - start) rows) -- one row
//      per game in the chronological window. `post_move` selects which
//      snapshot the encoder takes for each game's sampled turn: pre-move
//      (POV = active player about to move) or post-move (POV = same
//      player, after the move is applied but before the draw).
//   4. Optionally call add_file() between load()s as new self-play data
//      arrives.
//
// Sampling model
// --------------
// Each .slog file records one (turn-index) sample point per game, picked
// at write-time. The loader does NOT shuffle: it walks the master array
// in chronological order. Callers wanting a train/test split slice the
// global index range into disjoint ranges; callers wanting shuffling
// shuffle the resulting tensor on their end.
//
// Output row layout (row_size_floats() floats per row)
// ----------------------------------------------------
//   [ input_floats:    kInputFloats              ]
//   [ wld onehot:      kWldFloats              3   ]  // [win, draw, loss] (POV)
//   [ score_diff pdf:  kScoreDiffFloats        801 ]  // one-hot over clipped bins
//   [ opp_next_place:  kOppNextPlacementFloats 225 ]  // 15x15 binary mask
//
// (Label layout is owned by training_targets.h; the constants are
// re-exported from this header for downstream convenience.)
//
// Concurrency model
// -----------------
// load() is synchronous: it spawns up to `num_prefetch_threads` workers to
// read any not-yet-resident files from disk in parallel, then spawns up to
// `num_worker_threads` decoders to replay games and write decoded sample
// rows directly into disjoint slices of the caller's output buffer. Both
// pools are per-call (created/joined inside load()); thread-creation cost
// (~10us per thread) is dwarfed by an entire epoch's worth of work. A
// future API extension can add prefetch_next(...) -> std::future to overlap
// epoch N's I/O with epoch N-1's training without changing this core API.
//
// Resident files are managed against a `memory_budget`. After each load(),
// the loader evicts files (chronological-oldest-first, skipping any still
// referenced by the just-completed load) until total resident bytes <=
// budget.

#include "scribblez/input_encoder.h"
#include "scribblez/training_targets.h"

#include <atomic>
#include <cstddef>
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
// training_targets.h and re-included above.

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

  // Decode rows [start, stop) of the master array into `output`, in
  // chronological order (one row per row index). `output` must have
  // capacity for at least (stop - start) * row_size_floats() floats.
  // `stop` is clamped to num_positions(); requires start < stop after
  // clamping.
  //
  // `post_move` selects which snapshot the encoder takes for each game's
  // sampled turn: false = pre-move (POV = player about to move, just
  // before the move is applied); true = post-move (POV = same player,
  // immediately after the move is applied but BEFORE they draw
  // replacement tiles).
  //
  // If `apply_symmetry` is true, each output row independently gets a fair
  // coin flip: a 0 keeps the row in canonical orientation; a 1 transposes
  // every spatial plane across the main diagonal. Heads 0 (WLD) and 1
  // (score diff) are flip-invariant; head 2 (opp next placement) is also
  // transposed in lockstep with the input planes so it stays aligned.
  void load(int64_t start, int64_t stop, bool post_move, bool apply_symmetry, float* output);

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

  // One unit of decoder work: the contiguous slice [local_start,
  // local_start + n_rows) within one file, written into the rows
  // [output_row_start, output_row_start + n_rows) of the caller's buffer.
  // `flips` is an n_rows-long array of {0,1}, aligned with the rows.
  struct WorkUnit {
    DataFile* file = nullptr;
    bool post_move = false;
    int64_t local_start = 0;
    int64_t n_rows = 0;
    int64_t output_row_start = 0;
    const uint8_t* flips = nullptr;  // points into a load()-local vector
  };

  // Parallel I/O worker: pulls indices off `next_idx` and reads each
  // `to_load[i]` into memory, publishing the resulting buffer under `mu_`.
  // Invoked by load_files_in_parallel(); not intended for outside use.
  void file_loader_loop(std::atomic<std::size_t>& next_idx, std::vector<DataFile*>& to_load);

  // Parallel decode worker: pulls indices off `next_unit` and decodes each
  // WorkUnit into `output` using a thread-local BlockDecoder. Invoked by
  // decode_units_in_parallel(); not intended for outside use.
  void decode_unit_loop(std::atomic<std::size_t>& next_unit, const std::vector<WorkUnit>& units,
                        float* output);

  // Pool drivers. Each spawns up to `n_threads` workers (current thread
  // included) running the corresponding *_loop above, then joins them.
  void load_files_in_parallel(std::vector<DataFile*>& to_load);
  void decode_units_in_parallel(const std::vector<WorkUnit>& units, float* output);

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
