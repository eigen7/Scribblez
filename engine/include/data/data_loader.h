#pragma once

// Multithreaded loader that streams training rows decoded from .slog files into
// a caller-provided float buffer.
//
// Register files with add_file(), then alternate epoch_start() with
// load_batch() until load_batch() returns 0. Files may be added between epochs
// as self-play produces them.
//
// epoch_start() picks the epoch's rows and shuffles them once, globally and
// deterministically, so each batch draws from the whole epoch rather than from
// one file's games. File bodies load on demand: for each batch, a background
// prefetch loop loads the files it needs, keeping resident bytes within the
// memory budget where it can, and files the batch does not need are evicted.
//
// A kPositionEval row is input_floats(spec) input floats followed by the label
// block defined in training/training_targets.h; kMaxMovePerLane rows use
// MaxMovePerLaneTask's layout.
//
// The inner classes follow AlphaZeroArcade's DataLoader design:
//
//   DataFile        one registered .slog file and its in-memory body
//   WorkUnit        the rows of one batch that come from one file
//   ThreadTable     pool of free thread IDs with blocking allocation
//   PrefetchThread  loads one DataFile at a time on request
//   FileManager     owns the DataFiles; drives prefetching and eviction
//   WorkerThread    decodes WorkUnits with its own BlockDecoder
//   WorkManager     hands WorkUnits to the WorkerThreads
//   SamplingManager builds the shuffled epoch plan and slices it into batches

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

// The (game, turn) within one .slog file that a flat row index stands for.
struct GameTurn {
  uint32_t game_idx;
  uint16_t turn_idx;
};

class DataLoader {
 public:
  struct Params {
    // spec.dict is required and must outlive the loader. `task` fixes both the
    // row layout and which of a game's turns become rows (see DecodeTask).
    InputEncodingSpec spec{nullptr};
    DecodeTask task = DecodeTask::kPositionEval;
    int64_t memory_budget = 256LL * 1024 * 1024;  // bytes of resident file bodies
    int num_worker_threads = 4;
    int num_prefetch_threads = 2;
  };

  explicit DataLoader(const Params& params);
  ~DataLoader();

  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;

  // `file_size` must be the file's size on disk. The row count is read from the file's own header;
  // `num_positions` serves only as a fallback game count (one row per game) if
  // that header cannot be read.
  void add_file(const std::string& path, int64_t num_positions, int64_t file_size);

  // Totals across all registered files.
  int64_t num_positions() const;
  int num_files() const;

  int64_t resident_bytes() const;

  struct EpochConfig {
    int batch_size = 256;
    bool post_move = true;       // kPositionEval: encode the position after the move
    bool apply_symmetry = true;  // transpose each row with probability 1/2
    uint64_t seed = 42;

    // Per-game turn subsampling. 0 trains on every eligible turn of every game.
    // k > 0 takes k turns per game (at most its eligible-turn count), so at
    // k == 1 no two rows of an epoch share a game. Each game has a fixed
    // pseudo-random ordering of its turns, seeded by file path and game index
    // and independent of `seed`; epoch `epoch_index` takes the k-long window at
    // epoch_index * k. Successive epochs therefore see distinct turns until the
    // ordering wraps.
    int turns_per_game = 0;
    int epoch_index = 0;
  };

  // Returns the number of full batches in the epoch. A final partial batch, if
  // any, comes on top of those.
  int epoch_start(const EpochConfig& config);

  // Returns the number of rows written: batch_size, fewer on the final batch,
  // 0 once the epoch is exhausted. `output` needs room for
  // batch_size * row_size_floats() floats. Throws if a file body cannot be read.
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

  // One registered .slog file and, while loaded, its in-memory body.
  //
  // A file's "positions" are its training rows, one per included turn of each
  // game. `expand_all_turns` includes every turn (kMaxMovePerLane); otherwise
  // only each game's eligible region is included (kPositionEval). The row
  // index is built at construction from the header and metadata table, so an
  // epoch can be planned before any body is loaded.
  class DataFile {
   public:
    DataFile(const std::string& path, int64_t num_positions, int64_t file_size,
             bool expand_all_turns);
    ~DataFile();

    const std::string& path() const { return path_; }
    int64_t num_positions() const { return num_positions_; }
    int64_t file_size() const { return file_size_; }
    bool is_loaded() const;

    // Blocking.
    void load();

    // Frees the buffer and returns the bytes freed, or 0 if it was not loaded.
    int64_t unload();

    // Blocks until a load attempt resolves, then returns the body, or nullptr
    // if the file could not be read.
    const char* buffer() const;

    GameTurn sample_to_game_turn(int64_t sample_index) const;

    int64_t num_games() const { return num_games_; }
    // The number of rows game `game` contributes.
    int turns_in_game(int64_t game) const {
      return cumulative_turns_[game + 1] - cumulative_turns_[game];
    }
    int64_t game_base(int64_t game) const { return cumulative_turns_[game]; }

   private:
    std::string path_;
    int64_t num_positions_;
    int64_t file_size_;
    int64_t num_games_ = 0;

    // cumulative_turns_[g] is game g's first row index; the extra last entry is
    // num_positions_.
    std::vector<int64_t> cumulative_turns_;

    // The turn that game g's first row stands for: eligible_begin, or 0 when
    // expanding all turns.
    std::vector<uint8_t> first_turns_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    char* buffer_ = nullptr;
    // Set when the latest load() attempt failed, so buffer() can tell "failed"
    // from "not loaded yet" and return instead of waiting forever.
    bool load_failed_ = false;
  };

  // Records the first file whose body failed to load during a batch. The worker
  // that hits the failure records it and skips its rows; load_batch then throws
  // on the caller's thread.
  class LoadFailureLatch {
   public:
    void reset();
    void record(const std::string& path);
    bool failed() const;
    std::string path() const;

   private:
    mutable std::mutex mutex_;
    bool failed_ = false;
    std::string path_;
  };

  // The rows of one batch that come from one file. The three vectors are
  // parallel, one entry per row.
  struct WorkUnit {
    DataFile* file;
    std::vector<int64_t> local_positions;  // row index within the file
    std::vector<uint8_t> flips;            // nonzero: transpose the board
    std::vector<int> output_indices;       // row index within the batch
  };

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

    // Called once per batch before its WorkUnits are dispatched: evicts queued
    // files this batch does not need, queues the ones it needs that are not
    // loaded, and orders already-loaded files first so workers start at once.
    void prepare_work_units(std::deque<WorkUnit>& work_units);

    void reset_prefetch_loop();

   private:
    enum Instruction : int8_t { kUnload, kLoad, kWait, kQuit };

    Instruction get_next_instruction() const;
    void prefetch_loop();
    void exit_prefetch_loop();

    int64_t memory_budget_;
    bool expand_all_turns_;

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
    std::deque<DataFile*> all_files_;
    int64_t memory_usage_ = 0;
  };

  class WorkerThread {
   public:
    WorkerThread(FileManager* file_manager, ThreadTable* table, LoadFailureLatch* load_failure,
                 int id, const InputEncodingSpec& spec, DecodeTask task);
    ~WorkerThread();

    void quit();
    void schedule_work(WorkUnit unit, const EpochConfig& config, float* output);

   private:
    void loop();
    void do_work();

    FileManager* file_manager_;
    ThreadTable* table_;
    LoadFailureLatch* load_failure_;
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

  class WorkManager {
   public:
    WorkManager(FileManager* file_manager, LoadFailureLatch* load_failure, int num_threads,
                const InputEncodingSpec& spec, DecodeTask task);
    ~WorkManager();

    // Blocks until every unit is complete.
    void process(std::deque<WorkUnit>& work_units, const EpochConfig& config, float* output);

   private:
    std::vector<WorkerThread*> workers_;
    ThreadTable thread_table_;
  };

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

    // Fill order_, grouped by file, with every row of every file or with
    // config.turns_per_game turns per game. build_epoch then shuffles it.
    void collect_full_order(const std::vector<DataFile*>& files);
    void collect_sampled_order(const std::vector<DataFile*>& files, const EpochConfig& config);

    // `n` is the game's row count and `base` its first row index.
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
  // Declared before work_manager_ so it outlives the worker threads that write
  // to it.
  LoadFailureLatch load_failure_;
  WorkManager work_manager_;
  SamplingManager sampling_manager_;

  std::mutex epoch_mu_;
  bool epoch_active_ = false;
  EpochConfig epoch_config_;
  std::vector<DataFile*> epoch_files_;  // snapshot taken by epoch_start
};

}  // namespace binlog
}  // namespace scribblez
