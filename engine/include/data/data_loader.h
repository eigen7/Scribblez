#pragma once

// Multithreaded loader that streams sampled training rows from a set of .slog
// files into a caller-provided float buffer.
//
// Register every .slog file in chronological order (the loader assumes a strict
// newest-last add order), then alternate epoch_start() with load_batch() until
// it returns 0. New self-play data can be added between epochs.
//
// epoch_start() selects the epoch's rows and applies one global shuffle across
// all files, deterministically seeded, so a batch draws uniformly from the whole
// epoch rather than from one file's games. Files load on demand and are evicted
// once resident bytes exceed the memory budget, with a background prefetch loop
// pulling in upcoming files while the workers decode the current batch.
//
// A row is input_floats(spec) input floats followed by the label block that
// training_targets.h owns (whose constants this header re-exports).
//
// The implementation splits into focused inner classes, following AZA's
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

#include "data/block_decoder.h"
#include "encoding/input_encoder.h"
#include "training/max_move_per_lane_task.h"
#include "training/training_targets.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace scribblez {
namespace binlog {

// A (game, turn) sample location within a single .slog file: a flat position
// index expanded back into the game it belongs to and the turn within it.
struct GameTurn {
  uint32_t game_idx;
  uint16_t turn_idx;
};

class DataLoader {
 public:
  struct Params {
    // spec.dict is required and must outlive the loader. `task` fixes both the
    // row decoded and which of a game's turns expand into rows.
    InputEncodingSpec spec{nullptr, false};
    DecodeTask task = DecodeTask::kPositionEval;
    int64_t memory_budget = 256LL * 1024 * 1024;  // resident buffers
    int num_worker_threads = 4;                   // decoder pool size
    int num_prefetch_threads = 2;                 // disk-I/O pool size
  };

  explicit DataLoader(const Params& params);
  ~DataLoader();

  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;

  // Oldest-first; the loader treats the most-recently-added file as the
  // newest. `num_positions` and `file_size` must match the on-disk header.
  void add_file(const std::string& path, int64_t num_positions, int64_t file_size);

  // Totals across all registered files.
  int64_t num_positions() const;
  int num_files() const;

  int64_t resident_bytes() const;

  // =========================================================================
  // Epoch-based streaming API
  // =========================================================================

  struct EpochConfig {
    int batch_size = 256;
    bool post_move = true;
    bool apply_symmetry = true;
    uint64_t seed = 42;

    // Per-game turn subsampling. 0 trains on every eligible turn of every game.
    // k > 0 draws k turns per game per epoch (clamped to its eligible-turn
    // count), so at k == 1 no two rows of an epoch share a game and a batch is
    // decorrelated. Each game has a fixed pseudo-random turn ordering, seeded by
    // file path and game index independently of `seed`, of which `epoch_index`
    // selects the length-k window -- so successive epochs cover distinct turns
    // until the ordering wraps, and over E epochs a game contributes
    // min(E * k, its eligible-turn count) distinct positions.
    int turns_per_game = 0;
    int epoch_index = 0;
  };

  // Returns the number of complete batches the epoch will yield; a final
  // partial batch, if any, is yielded on top of those.
  int epoch_start(const EpochConfig& config);

  // The rows written -- short on the final batch, 0 once the epoch is
  // exhausted. `output` needs room for batch_size * row_size_floats() floats.
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

  // One registered .slog file, owning the in-memory buffer once loaded.
  //
  // A file's "positions" are its expanded training rows, one per included turn
  // across all games. `expand_all_turns` chooses which turns count: every turn
  // (the lane task) or only each game's eligible region (the value task).
  class DataFile {
   public:
    DataFile(const std::string& path, int64_t num_positions, int64_t file_size,
             bool expand_all_turns);
    ~DataFile();

    const std::string& path() const { return path_; }
    int64_t num_positions() const { return num_positions_; }
    int64_t file_size() const { return file_size_; }
    bool is_loaded() const;

    // Reads the file contents into memory; blocking I/O.
    void load();

    // Frees the buffer and returns the bytes freed, or 0 if it was not loaded.
    int64_t unload();

    // Blocks until the file is loaded.
    const char* buffer() const;

    // The (game, turn) a flat position index in [0, num_positions()) stands
    // for.
    GameTurn sample_to_game_turn(int64_t sample_index) const;

    // From the file header and metadata table, needing no resident body.
    int64_t num_games() const { return num_games_; }
    // Flat rows the game expands into.
    int turns_in_game(int64_t game) const {
      return cumulative_turns_[game + 1] - cumulative_turns_[game];
    }
    int64_t game_base(int64_t game) const { return cumulative_turns_[game]; }

   private:
    std::string path_;
    int64_t num_positions_;
    int64_t file_size_;
    int64_t num_games_ = 0;

    // Prefix sums of per-game included-turn counts (size num_games_ + 1):
    // cumulative_turns_[g] is game g's first flat position index, and the last
    // entry is num_positions_.
    std::vector<int64_t> cumulative_turns_;

    // The turn index game g's first flat position stands for: eligible_begin
    // for the value task, 0 when expanding all turns.
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

    // Blocks until all threads are available, or quitting.
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

    // Thread-safe.
    std::vector<DataFile*> snapshot_files() const;

    void add_to_unload_queue(DataFile* file);

    // Sorts work_units loaded-first, enqueues unloaded files for prefetching,
    // and trims files this batch no longer needs.
    void prepare_work_units(std::deque<WorkUnit>& work_units);

    void reset_prefetch_loop();

   private:
    enum Instruction : int8_t { kUnload, kLoad, kWait, kQuit };

    Instruction get_next_instruction() const;
    void prefetch_loop();
    void exit_prefetch_loop();

    int64_t memory_budget_;
    bool expand_all_turns_;  // passed to each DataFile

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

    // Blocks until every unit is complete.
    void process(std::deque<WorkUnit>& work_units, const EpochConfig& config, float* output);

   private:
    std::vector<WorkerThread*> workers_;
    ThreadTable thread_table_;
  };

  // Builds the shuffled epoch plan and slices it into per-batch WorkUnits.
  class SamplingManager {
   public:
    void build_epoch(const std::vector<DataFile*>& files, const EpochConfig& config);

    // The rows in this batch, 0 once the epoch is exhausted.
    int next_batch(std::deque<WorkUnit>& work_units, const std::vector<DataFile*>& files);

    int64_t total_positions() const { return total_positions_; }

   private:
    struct EpochPosition {
      int file_idx;
      int64_t local_pos;
    };

    // Both collect order_ grouped by file, which build_epoch then shuffles
    // globally: every flat position of every file, or config.turns_per_game
    // turns drawn per game.
    void collect_full_order(const std::vector<DataFile*>& files);
    void collect_sampled_order(const std::vector<DataFile*>& files, const EpochConfig& config);

    // `n` is the game's eligible-turn count and `base` its first flat position.
    // The turns come from a fixed per-game ordering seeded by `file_key` and
    // `game`, windowed by epoch_index.
    void append_game_turns(int file_idx, int64_t game, int n, int64_t base, uint64_t file_key,
                           int turns_per_game, int epoch_index);

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
