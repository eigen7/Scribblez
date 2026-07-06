#pragma once

// Multithreaded loader that streams sampled training rows from a set of
// .slog files into a caller-provided float buffer.
//
// Lifecycle
// ---------
//   1. Construct with a Params (memory budget, thread counts).
//   2. Register every existing .slog file in chronological order via
//      add_file(). The loader assumes a strict newest-last add order.
//   3. Call epoch_start() to begin an epoch, then repeatedly call
//      load_batch() to consume batches until it returns 0.
//   4. Optionally call add_file() between epochs as new self-play data
//      arrives.
//
// Sampling model
// --------------
// Each .slog file records one (turn-index) sample point per game, picked
// at write-time. epoch_start() selects which rows the epoch contains, then
// applies a single global shuffle over all selected rows across all files
// (deterministically seeded by the caller-provided seed), so a batch draws
// uniformly from the whole epoch rather than from one file's games.
//
// Output row layout (row_size_floats() floats per row)
// ----------------------------------------------------
//   [ input_floats:    input_floats(spec)          ]
//   [ wld onehot:      kWldFloats              3   ]  // [win, draw, loss] (POV)
//   [ score_diff:      kScoreDiffFloats        1   ]  // observed final diff (clipped)
//   [ opp_next_place:  kOppNextPlacementFloats  225 ]  // 15x15 binary mask
//   [ self_next_place: kSelfNextPlacementFloats 225 ]  // mover's next placement
//   [ opp_win_place:   kOppWinPlacementFloats   225 ]  // opp_next_place x opp-won
//   [ self_win_place:  kSelfWinPlacementFloats  225 ]  // self_next_place x mover-won
//
// (Label layout is owned by training_targets.h; the constants are
// re-exported from this header for downstream convenience.)
//
// Memory management
// -----------------
// Files are loaded on-demand and evicted when total resident bytes exceed
// the memory budget. A background prefetch loop pre-loads upcoming files
// while worker threads decode the current batch.
//
// Architecture
// ------------
// The implementation is divided into focused inner classes, following AZA's
// DataLoader pattern:
//
//   DataFile        — one registered .slog file; owns the in-memory buffer
//   WorkUnit        — (DataFile*, local positions, output offset) for one file
//   ThreadTable     — tracks available thread IDs with blocking allocate
//   PrefetchThread  — persistent thread that loads a file on demand
//   FileManager     — owns all DataFiles, LRU eviction, prefetch orchestration
//   WorkerThread    — persistent thread that decodes WorkUnits via BlockDecoder
//   WorkManager     — distributes WorkUnits to WorkerThreads
//   SamplingManager — builds shuffled epoch plan, slices into batches

#include "scribblez/block_decoder.h"
#include "scribblez/input_encoder.h"
#include "scribblez/max_move_per_lane_task.h"
#include "scribblez/training_targets.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace scribblez {
namespace binlog {

// The input width is spec-dependent (input_encoder.h's layout registry); the
// label constants (kWldFloats / kScoreDiffFloats / kLabelFloats) are owned by
// training_targets.h and re-included above.

// A (game, turn) sample location within a single .slog file: the expansion of a
// flat position index back into the game it belongs to and the turn within it.
struct GameTurn {
  uint32_t game_idx;
  uint16_t turn_idx;
};

class DataLoader {
 public:
  struct Params {
    // Input-encoding configuration the decoders encode with (lexicon +
    // feature blocks). The dict is required and must outlive the loader.
    InputEncodingSpec spec{nullptr, false};
    // Which training row the loader decodes, and (for the lane task) that a game
    // expands over all its turns rather than only the eligible prefix.
    DecodeTask task = DecodeTask::kPostMoveValue;
    int64_t memory_budget = 256LL * 1024 * 1024;  // 256 MB resident buffers
    int num_worker_threads = 4;                   // decoder pool size
    int num_prefetch_threads = 2;                 // disk-I/O pool size
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

  // =========================================================================
  // Epoch-based streaming API
  // =========================================================================

  struct EpochConfig {
    int batch_size = 256;
    bool post_move = true;
    bool apply_symmetry = true;
    uint64_t seed = 42;

    // Per-game turn subsampling. 0 (the default) trains on every eligible turn
    // of every game -- the full expanded epoch. k > 0 draws k turns per game per
    // epoch (clamped to the game's eligible-turn count), yielding a smaller
    // epoch of ~k rows per game; with k == 1 no two rows in the epoch share a
    // game, so a batch is decorrelated. Each game has a fixed pseudo-random turn
    // ordering (seeded by file path + game index, independent of `seed`);
    // `epoch_index` selects the length-k window of that ordering, so successive
    // epochs cover distinct turns until the ordering wraps. Over E epochs each
    // game thus contributes min(E * k, its eligible-turn count) distinct
    // positions.
    int turns_per_game = 0;
    int epoch_index = 0;
  };

  // Begin a new epoch. Returns the number of complete batches that will be
  // yielded (the last partial batch, if any, is also yielded -- so the
  // caller will get num_batches + (1 if remainder else 0) calls to
  // load_batch before it returns 0).
  int epoch_start(const EpochConfig& config);

  // Fill `output` with the next batch. Returns the number of rows written
  // (== batch_size for full batches, < batch_size for the final partial
  // batch, 0 when the epoch is exhausted). `output` must have capacity for
  // at least batch_size * row_size_floats() floats.
  int load_batch(float* output);

  int row_size_floats() const {
    return params_.task == DecodeTask::kMaxMovePerLane ? MaxMovePerLaneTask::kRowFloats
                                                       : input_floats(params_.spec) + kLabelFloats;
  }
  int input_size_floats() const {
    return params_.task == DecodeTask::kMaxMovePerLane ? MaxMovePerLaneTask::kInputFloats
                                                       : input_floats(params_.spec);
  }
  int label_size_floats() const {
    return params_.task == DecodeTask::kMaxMovePerLane ? MaxMovePerLaneTask::kLabelFloats
                                                       : kLabelFloats;
  }

  // =========================================================================
  // Inner classes
  // =========================================================================

  // One registered .slog file. Owns the in-memory buffer once loaded.
  //
  // A file's "positions" are its expanded training rows: one per included turn
  // across all games, read from the file header at construction.
  // `expand_all_turns` chooses which turns count -- every turn (the lane task)
  // or only each game's eligible region (GameMetadata's [eligible_begin,
  // eligible_end), the value task). sample_to_game_turn() maps a flat position
  // index back to the (game, turn) pair it stands for.
  class DataFile {
   public:
    DataFile(const std::string& path, int64_t num_positions, int64_t file_size,
             bool expand_all_turns);
    ~DataFile();

    const std::string& path() const { return path_; }
    int64_t num_positions() const { return num_positions_; }
    int64_t file_size() const { return file_size_; }
    bool is_loaded() const;

    // Loads the file contents into memory (blocking I/O).
    void load();

    // Unloads the buffer if loaded. Returns bytes freed (0 if not loaded).
    int64_t unload();

    // Blocks until the file is loaded, then returns a pointer to the buffer.
    const char* buffer() const;

    // Map a flat position index in [0, num_positions()) to the (game_idx,
    // turn_idx) it expands to, using the per-game index read at construction.
    GameTurn sample_to_game_turn(int64_t sample_index) const;

    // Per-game index, read from the file header + metadata table at
    // construction (no resident body required).
    int64_t num_games() const { return num_games_; }
    // Number of turns game `game` expands into (all turns or the eligible
    // prefix, per the file's expand_all_turns), i.e. its count of flat rows.
    int turns_in_game(int64_t game) const {
      return static_cast<int>(cumulative_turns_[game + 1] - cumulative_turns_[game]);
    }
    int64_t game_base(int64_t game) const { return cumulative_turns_[game]; }

   private:
    std::string path_;
    int64_t num_positions_;
    int64_t file_size_;
    int64_t num_games_ = 0;

    // Per-game prefix sums of included-turn counts (size num_games_ + 1), read
    // from the file's metadata table at construction; cumulative_turns_[g] is the
    // first flat position index of game g and cumulative_turns_.back() is
    // num_positions_.
    std::vector<int64_t> cumulative_turns_;

    // Per-game turn index that game g's first flat position stands for:
    // eligible_begin for the value task, 0 when expanding all turns.
    std::vector<uint8_t> first_turns_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    char* buffer_ = nullptr;
  };

  // A batch of rows to decode from one file.
  struct WorkUnit {
    DataFile* file;
    std::vector<int64_t> local_positions;  // positions within the file
    std::vector<uint8_t> flips;            // per-row flip bits
    std::vector<int> output_indices;       // where each row lands in output
  };

  // Tracks available thread IDs with blocking allocate/release.
  class ThreadTable {
   public:
    explicit ThreadTable(int n_threads);

    void mark_as_available(int id);

    // Blocks until a thread is available. Returns -1 if quitting.
    int allocate_thread();

    // Blocks until all threads are available (or quitting).
    void wait_until_all_available();

    void quit();

   private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::vector<int> available_ids_;
    int n_threads_;
    bool quitting_ = false;
  };

  // A persistent thread that loads a DataFile on demand.
  class PrefetchThread {
   public:
    PrefetchThread(ThreadTable* table, int id);
    ~PrefetchThread();

    void quit();
    void schedule_prefetch(DataFile* file);

   private:
    void loop();

    ThreadTable* table_;
    int id_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::thread thread_;
    DataFile* file_ = nullptr;
    bool quitting_ = false;
  };

  // Owns all DataFiles, manages LRU eviction and prefetch orchestration.
  class FileManager {
   public:
    FileManager(int64_t memory_budget, int num_prefetch_threads, bool expand_all_turns);
    ~FileManager();

    void append(const std::string& path, int64_t num_positions, int64_t file_size);

    int64_t num_positions() const;
    int num_files() const;
    int64_t memory_usage() const;

    // Snapshots the file list (thread-safe).
    std::vector<DataFile*> snapshot_files() const;

    // Adds a file to the unload queue (called by WorkerThread after decode).
    void add_to_unload_queue(DataFile* file);

    // Sorts work_units by loaded-first, enqueues unloaded files for
    // prefetching, and trims files that are no longer needed.
    void prepare_work_units(std::deque<WorkUnit>& work_units);

    // Resets the prefetch loop between epochs.
    void reset_prefetch_loop();

   private:
    enum Instruction : int8_t { kUnload, kLoad, kWait, kQuit };

    Instruction get_next_instruction() const;
    void prefetch_loop();
    void exit_prefetch_loop();

    int64_t memory_budget_;
    bool expand_all_turns_;  // passed to each DataFile: expand all turns vs the eligible prefix

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::thread prefetch_loop_thread_;
    bool quitting_ = false;

    std::vector<PrefetchThread*> prefetch_threads_;
    ThreadTable thread_table_;

    std::deque<DataFile*> load_queue_;
    std::deque<DataFile*> unload_queue_;
    int active_file_count_ = 0;

    int64_t num_positions_ = 0;
    std::deque<DataFile*> all_files_;  // chronological order
    int64_t memory_usage_ = 0;
  };

  // A persistent thread that decodes WorkUnits using a BlockDecoder.
  class WorkerThread {
   public:
    WorkerThread(FileManager* file_manager, ThreadTable* table, int id,
                 const InputEncodingSpec& spec, DecodeTask task);
    ~WorkerThread();

    void quit();
    void schedule_work(WorkUnit unit, const EpochConfig& config, float* output);

   private:
    void loop();
    void do_work();

    FileManager* file_manager_;
    ThreadTable* table_;
    int id_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::thread thread_;
    WorkUnit unit_;
    EpochConfig config_;
    float* output_ = nullptr;
    bool quitting_ = false;
    bool has_work_ = false;

    BlockDecoder decoder_;
  };

  // Distributes WorkUnits to a pool of WorkerThreads.
  class WorkManager {
   public:
    WorkManager(FileManager* file_manager, int num_threads, const InputEncodingSpec& spec,
                DecodeTask task);
    ~WorkManager();

    // Processes all work units. Blocks until all are complete.
    void process(std::deque<WorkUnit>& work_units, const EpochConfig& config, float* output);

   private:
    std::vector<WorkerThread*> workers_;
    ThreadTable thread_table_;
  };

  // Builds the shuffled epoch plan and slices it into per-batch WorkUnits.
  class SamplingManager {
   public:
    // Builds the full epoch iteration order from the file list.
    void build_epoch(const std::vector<DataFile*>& files, const EpochConfig& config);

    // Pops the next batch of work units. Returns the number of rows in
    // this batch (0 = epoch exhausted).
    int next_batch(std::deque<WorkUnit>& work_units, const std::vector<DataFile*>& files);

    int64_t total_positions() const { return total_positions_; }

   private:
    struct EpochPosition {
      int file_idx;
      int64_t local_pos;
    };

    // Collect order_ for an all-turns epoch: every flat position of every file,
    // grouped by file. build_epoch shuffles order_ globally afterwards.
    void collect_full_order(const std::vector<DataFile*>& files);

    // Collect order_ for a subsampled epoch: config.turns_per_game turns drawn
    // per game, grouped by file. build_epoch shuffles order_ globally afterwards.
    void collect_sampled_order(const std::vector<DataFile*>& files, const EpochConfig& config);

    // Append one game's sampled turns to order_. `n` is the game's eligible-turn
    // count, `base` its first flat position; the turns are picked from a fixed
    // per-game ordering seeded by `file_key` and `game`, windowed by epoch_index.
    void append_game_turns(int file_idx, int64_t game, int n, int64_t base, uint64_t file_key,
                           int turns_per_game, int epoch_index);

    // Fill flips_ (size total_positions_) with per-row diagonal-flip bits.
    void build_flips(const EpochConfig& config);

    std::vector<EpochPosition> order_;
    std::vector<uint8_t> flips_;
    int64_t cursor_ = 0;
    int batch_size_ = 0;
    int64_t total_positions_ = 0;
  };

 private:
  Params params_;
  FileManager file_manager_;
  WorkManager work_manager_;
  SamplingManager sampling_manager_;

  std::mutex epoch_mu_;
  bool epoch_active_ = false;
  EpochConfig epoch_config_;
  std::vector<DataFile*> epoch_files_;  // snapshot at epoch_start
};

}  // namespace binlog
}  // namespace scribblez
