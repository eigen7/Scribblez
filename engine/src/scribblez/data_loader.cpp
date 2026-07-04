#include "scribblez/data_loader.h"

#include "scribblez/binary_log.h"
#include "scribblez/block_decoder.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
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

char* read_whole_file(const std::string& path, int64_t expected_size) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "DataLoader: failed to open " << path << "\n";
    return nullptr;
  }
  char* buf = new char[expected_size];
  f.read(buf, expected_size);
  if (!f) {
    std::cerr << "DataLoader: short read on " << path << "\n";
    delete[] buf;
    return nullptr;
  }
  return buf;
}

// Read the per-game prefix sums of included-turn counts from a .slog file's
// header and metadata table without resident body (cum[g] = first flat row index
// of game g, cum.back() = total expanded rows). `all_turns` counts every turn
// of each game (the lane task); otherwise only the bag-non-empty eligible prefix
// (the value task). Sets `num_games`. On any read failure, returns a
// one-turn-per-game fallback (cum = 0,1,2,...) of size num_games + 1.
std::vector<int64_t> read_cumulative_turns(const std::string& path, int64_t& num_games,
                                           int64_t num_games_fallback, bool all_turns) {
  std::ifstream f(path, std::ios::binary);
  FileHeader hdr{};
  if (f && f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)) && hdr.magic == kMagic) {
    num_games = static_cast<int64_t>(hdr.num_games);
    std::vector<GameMetadata> metas(static_cast<size_t>(num_games));
    if (num_games == 0 || f.read(reinterpret_cast<char*>(metas.data()),
                                 static_cast<std::streamsize>(num_games) * sizeof(GameMetadata))) {
      std::vector<int64_t> cum(static_cast<size_t>(num_games) + 1);
      cum[0] = 0;
      for (int64_t g = 0; g < num_games; ++g) {
        const uint32_t turns = all_turns ? metas[g].num_turns : metas[g].eligible_turns;
        cum[g + 1] = cum[g] + static_cast<int64_t>(turns);
      }
      return cum;
    }
    std::cerr << "DataLoader: could not read metadata table of " << path << "\n";
  } else {
    std::cerr << "DataLoader: could not read header of " << path << "; using caller count\n";
    num_games = num_games_fallback;
  }
  std::vector<int64_t> cum(static_cast<size_t>(num_games) + 1);
  std::iota(cum.begin(), cum.end(), int64_t{0});
  return cum;
}

}  // namespace

// ===========================================================================
// DataFile
// ===========================================================================

DataLoader::DataFile::DataFile(const std::string& path, int64_t num_positions, int64_t file_size,
                               bool expand_all_turns)
    : path_(path), num_positions_(num_positions), file_size_(file_size) {
  // Read the per-game included-turn index from the header + metadata table (no
  // resident body needed), so an epoch can be sized and subsampled before the
  // file body is loaded. num_positions_ is the expanded row count; the
  // caller-passed value is only a fallback.
  cumulative_turns_ = read_cumulative_turns(path_, num_games_, num_positions, expand_all_turns);
  num_positions_ = cumulative_turns_.back();
}

DataLoader::DataFile::~DataFile() { unload(); }

GameTurn DataLoader::DataFile::sample_to_game_turn(int64_t sample_index) const {
  // Find the game whose flat-index range contains sample_index: the last g with
  // cumulative_turns_[g] <= sample_index.
  auto it = std::upper_bound(cumulative_turns_.begin(), cumulative_turns_.end(), sample_index);
  int64_t g = (it - cumulative_turns_.begin()) - 1;
  int64_t turn = sample_index - cumulative_turns_[g];
  return GameTurn{static_cast<uint32_t>(g), static_cast<uint16_t>(turn)};
}

bool DataLoader::DataFile::is_loaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffer_ != nullptr;
}

void DataLoader::DataFile::load() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!buffer_) {
    buffer_ = read_whole_file(path_, file_size_);
  }
  lock.unlock();
  cv_.notify_all();
}

int64_t DataLoader::DataFile::unload() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!buffer_) return 0;
  delete[] buffer_;
  buffer_ = nullptr;
  lock.unlock();
  cv_.notify_all();
  return file_size_;
}

const char* DataLoader::DataFile::buffer() const {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return buffer_ != nullptr; });
  return buffer_;
}

// ===========================================================================
// ThreadTable
// ===========================================================================

DataLoader::ThreadTable::ThreadTable(int n_threads) : n_threads_(n_threads) {
  for (int i = 0; i < n_threads; ++i) {
    available_ids_.push_back(i);
  }
}

void DataLoader::ThreadTable::mark_as_available(int id) {
  std::unique_lock<std::mutex> lock(mutex_);
  available_ids_.push_back(id);
  lock.unlock();
  cv_.notify_one();
}

int DataLoader::ThreadTable::allocate_thread() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return quitting_ || !available_ids_.empty(); });
  if (quitting_) return -1;
  int id = available_ids_.back();
  available_ids_.pop_back();
  return id;
}

void DataLoader::ThreadTable::wait_until_all_available() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock,
           [this] { return quitting_ || static_cast<int>(available_ids_.size()) == n_threads_; });
}

void DataLoader::ThreadTable::quit() {
  std::unique_lock<std::mutex> lock(mutex_);
  quitting_ = true;
  lock.unlock();
  cv_.notify_all();
}

// ===========================================================================
// PrefetchThread
// ===========================================================================

DataLoader::PrefetchThread::PrefetchThread(ThreadTable* table, int id) : table_(table), id_(id) {
  thread_ = std::thread(&PrefetchThread::loop, this);
}

DataLoader::PrefetchThread::~PrefetchThread() { quit(); }

void DataLoader::PrefetchThread::quit() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    quitting_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void DataLoader::PrefetchThread::schedule_prefetch(DataFile* data_file) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    file_ = data_file;
  }
  cv_.notify_all();
}

void DataLoader::PrefetchThread::loop() {
  while (true) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return quitting_ || file_ != nullptr; });
    if (quitting_) return;
    DataFile* f = file_;
    file_ = nullptr;
    lock.unlock();

    f->load();
    table_->mark_as_available(id_);
  }
}

// ===========================================================================
// FileManager
// ===========================================================================

DataLoader::FileManager::FileManager(int64_t memory_budget, int num_prefetch_threads,
                                     bool expand_all_turns)
    : memory_budget_(memory_budget),
      expand_all_turns_(expand_all_turns),
      thread_table_(num_prefetch_threads) {
  for (int i = 0; i < num_prefetch_threads; ++i) {
    prefetch_threads_.push_back(new PrefetchThread(&thread_table_, i));
  }
  prefetch_loop_thread_ = std::thread(&FileManager::prefetch_loop, this);
}

DataLoader::FileManager::~FileManager() {
  thread_table_.quit();
  exit_prefetch_loop();

  for (PrefetchThread* t : prefetch_threads_) delete t;
  if (prefetch_loop_thread_.joinable()) prefetch_loop_thread_.join();
  for (DataFile* f : all_files_) delete f;
}

void DataLoader::FileManager::append(const std::string& path, int64_t num_positions,
                                     int64_t file_size) {
  // DataFile derives its true (expanded) position count from the file header;
  // tally that, not the caller-passed value (which is the game count).
  auto* f = new DataFile(path, num_positions, file_size, expand_all_turns_);
  std::lock_guard<std::mutex> lock(mutex_);
  num_positions_ += f->num_positions();
  all_files_.push_back(f);
}

int64_t DataLoader::FileManager::num_positions() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_positions_;
}

int DataLoader::FileManager::num_files() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(all_files_.size());
}

int64_t DataLoader::FileManager::memory_usage() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return memory_usage_;
}

std::vector<DataLoader::DataFile*> DataLoader::FileManager::snapshot_files() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {all_files_.begin(), all_files_.end()};
}

void DataLoader::FileManager::add_to_unload_queue(DataFile* file) {
  std::unique_lock<std::mutex> lock(mutex_);
  unload_queue_.push_back(file);
  assert(active_file_count_ > 0);
  active_file_count_--;
  lock.unlock();
  cv_.notify_all();
}

void DataLoader::FileManager::prepare_work_units(std::deque<WorkUnit>& work_units) {
  std::unique_lock<std::mutex> lock(mutex_);

  // Determine which files this batch needs.
  std::unordered_set<DataFile*> needed;
  for (const WorkUnit& unit : work_units) needed.insert(unit.file);

  // Drain unload queue: unload files not needed by this batch.
  while (!unload_queue_.empty()) {
    DataFile* f = unload_queue_.front();
    unload_queue_.pop_front();
    if (!needed.count(f)) {
      memory_usage_ -= f->unload();
    }
  }

  load_queue_.clear();
  active_file_count_ = 0;

  // Sort: loaded files first so workers can start immediately.
  std::sort(work_units.begin(), work_units.end(), [](const WorkUnit& a, const WorkUnit& b) {
    return a.file->is_loaded() > b.file->is_loaded();
  });

  for (const WorkUnit& unit : work_units) {
    if (unit.file->is_loaded()) {
      active_file_count_++;
    } else {
      load_queue_.push_back(unit.file);
    }
  }

  lock.unlock();
  cv_.notify_all();
}

void DataLoader::FileManager::reset_prefetch_loop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    quitting_ = true;
  }
  cv_.notify_all();
  if (prefetch_loop_thread_.joinable()) prefetch_loop_thread_.join();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    quitting_ = false;
  }
  prefetch_loop_thread_ = std::thread(&FileManager::prefetch_loop, this);
}

DataLoader::FileManager::Instruction DataLoader::FileManager::get_next_instruction() const {
  if (quitting_) return kQuit;
  if (load_queue_.empty()) return kWait;

  DataFile* file = load_queue_.front();
  assert(!file->is_loaded());

  if (memory_usage_ + file->file_size() <= memory_budget_) return kLoad;
  if (!unload_queue_.empty()) return kUnload;
  if (active_file_count_ > 0) return kWait;

  // Memory budget insufficient for this single file — load anyway.
  return kLoad;
}

void DataLoader::FileManager::prefetch_loop() {
  while (true) {
    std::unique_lock<std::mutex> lock(mutex_);
    Instruction instr = kWait;
    cv_.wait(lock, [&] {
      instr = get_next_instruction();
      return instr != kWait;
    });

    if (instr == kQuit) return;

    if (instr == kLoad) {
      DataFile* file = load_queue_.front();
      int id = thread_table_.allocate_thread();
      if (id < 0) return;  // quitting
      prefetch_threads_[id]->schedule_prefetch(file);
      memory_usage_ += file->file_size();
      load_queue_.pop_front();
      active_file_count_++;
    } else if (instr == kUnload) {
      DataFile* file = unload_queue_.front();
      memory_usage_ -= file->unload();
      unload_queue_.pop_front();
    }
  }
}

void DataLoader::FileManager::exit_prefetch_loop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    quitting_ = true;
  }
  cv_.notify_all();
}

// ===========================================================================
// WorkerThread
// ===========================================================================

DataLoader::WorkerThread::WorkerThread(FileManager* file_manager, ThreadTable* table, int id,
                                       const InputEncodingSpec& spec, DecodeTask task)
    : file_manager_(file_manager), table_(table), id_(id), decoder_(spec, task) {
  thread_ = std::thread(&WorkerThread::loop, this);
}

DataLoader::WorkerThread::~WorkerThread() { quit(); }

void DataLoader::WorkerThread::quit() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    quitting_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void DataLoader::WorkerThread::schedule_work(WorkUnit unit, const EpochConfig& config,
                                             float* output) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    unit_ = std::move(unit);
    config_ = config;
    output_ = output;
    has_work_ = true;
  }
  cv_.notify_all();
}

void DataLoader::WorkerThread::loop() {
  while (true) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return quitting_ || has_work_; });
    if (quitting_) return;

    do_work();
    has_work_ = false;
    DataFile* file = unit_.file;
    lock.unlock();

    table_->mark_as_available(id_);
    file_manager_->add_to_unload_queue(file);
  }
}

void DataLoader::WorkerThread::do_work() {
  if (unit_.local_positions.empty()) return;

  DataFile* file = unit_.file;
  const char* buf = file->buffer();  // blocks until loaded

  for (size_t i = 0; i < unit_.local_positions.size(); ++i) {
    // Each local position is a flat (game, turn) sample index within the file.
    GameTurn game_turn = file->sample_to_game_turn(unit_.local_positions[i]);
    decoder_.decode_one(buf, file->path(), game_turn.game_idx, game_turn.turn_idx,
                        unit_.flips[i] != 0, config_.post_move,
                        /*output_row=*/unit_.output_indices[i], output_);
  }
}

// ===========================================================================
// WorkManager
// ===========================================================================

DataLoader::WorkManager::WorkManager(FileManager* file_manager, int num_threads,
                                     const InputEncodingSpec& spec, DecodeTask task)
    : thread_table_(num_threads) {
  for (int i = 0; i < num_threads; ++i) {
    workers_.push_back(new WorkerThread(file_manager, &thread_table_, i, spec, task));
  }
}

DataLoader::WorkManager::~WorkManager() {
  thread_table_.quit();
  for (WorkerThread* t : workers_) delete t;
}

void DataLoader::WorkManager::process(std::deque<WorkUnit>& work_units, const EpochConfig& config,
                                      float* output) {
  while (!work_units.empty()) {
    int id = thread_table_.allocate_thread();
    if (id < 0) return;  // quitting
    workers_[id]->schedule_work(std::move(work_units.front()), config, output);
    work_units.pop_front();
  }
  thread_table_.wait_until_all_available();
}

// ===========================================================================
// SamplingManager
// ===========================================================================

void DataLoader::SamplingManager::build_epoch(const std::vector<DataFile*>& files,
                                              const EpochConfig& config) {
  cursor_ = 0;
  batch_size_ = config.batch_size;
  order_.clear();

  if (config.turns_per_game <= 0) {
    collect_full_order(files);
  } else {
    collect_sampled_order(files, config);
  }

  // The one and only ordering shuffle: a single global shuffle over every
  // sample of every file. next_batch() slices contiguous windows of order_, so
  // this is what decorrelates a batch -- without it, a batch would draw mostly
  // from one file's games (samples sit in order_ grouped by file). The
  // selection of *which* turns each game contributes is done above and is
  // independent of this shuffle.
  std::mt19937_64 rng(config.seed);
  std::shuffle(order_.begin(), order_.end(), rng);

  total_positions_ = static_cast<int64_t>(order_.size());
  build_flips(config);
}

void DataLoader::SamplingManager::collect_full_order(const std::vector<DataFile*>& files) {
  int64_t total = 0;
  for (auto* f : files) total += f->num_positions();
  order_.reserve(static_cast<size_t>(total));

  // Every flat position of every file, in file-then-position order. build_epoch
  // shuffles order_ globally afterwards.
  for (int fi = 0; fi < static_cast<int>(files.size()); ++fi) {
    const int64_t n = files[fi]->num_positions();
    for (int64_t p = 0; p < n; ++p) {
      order_.push_back(EpochPosition{fi, p});
    }
  }
}

void DataLoader::SamplingManager::collect_sampled_order(const std::vector<DataFile*>& files,
                                                        const EpochConfig& config) {
  // For each file, select config.turns_per_game turns from every game. The
  // resulting rows are left grouped by file; build_epoch shuffles order_
  // globally afterwards.
  for (int fi = 0; fi < static_cast<int>(files.size()); ++fi) {
    DataFile* f = files[fi];
    const uint64_t file_key = std::hash<std::string>{}(f->path());
    for (int64_t g = 0; g < f->num_games(); ++g) {
      append_game_turns(fi, g, f->turns_in_game(g), f->game_base(g), file_key,
                        config.turns_per_game, config.epoch_index);
    }
  }
}

void DataLoader::SamplingManager::append_game_turns(int file_idx, int64_t game, int n, int64_t base,
                                                    uint64_t file_key, int turns_per_game,
                                                    int epoch_index) {
  if (n <= 0) return;

  // A fixed per-game ordering of the game's eligible turns, independent of the
  // epoch, so that epoch e drawing the window [e*k, e*k + k) of it covers turns
  // disjoint from neighboring epochs until the ordering wraps after n turns.
  std::vector<int64_t> turn_order(static_cast<size_t>(n));
  std::iota(turn_order.begin(), turn_order.end(), int64_t{0});
  std::mt19937_64 turn_rng(file_key ^ (static_cast<uint64_t>(game) * 0x9E3779B97F4A7C15ULL));
  std::shuffle(turn_order.begin(), turn_order.end(), turn_rng);

  const int m = std::min(turns_per_game, n);
  for (int j = 0; j < m; ++j) {
    const int64_t idx =
      (static_cast<int64_t>(epoch_index) * turns_per_game + j) % static_cast<int64_t>(n);
    order_.push_back(EpochPosition{file_idx, base + turn_order[static_cast<size_t>(idx)]});
  }
}

void DataLoader::SamplingManager::build_flips(const EpochConfig& config) {
  flips_.resize(static_cast<size_t>(total_positions_));
  if (config.apply_symmetry) {
    std::mt19937_64 flip_rng(config.seed ^ 0xDEADBEEFCAFEF00DULL);
    std::bernoulli_distribution coin(0.5);
    for (size_t i = 0; i < flips_.size(); ++i) {
      flips_[i] = coin(flip_rng) ? 1u : 0u;
    }
  } else {
    std::fill(flips_.begin(), flips_.end(), 0u);
  }
}

int DataLoader::SamplingManager::next_batch(std::deque<WorkUnit>& work_units,
                                            const std::vector<DataFile*>& files) {
  work_units.clear();
  if (cursor_ >= static_cast<int64_t>(order_.size())) return 0;

  const int64_t batch_start = cursor_;
  const int64_t batch_end =
    std::min(batch_start + batch_size_, static_cast<int64_t>(order_.size()));
  const int n_rows = static_cast<int>(batch_end - batch_start);
  cursor_ = batch_end;

  // Group rows by file for locality.
  struct TaggedRow {
    int file_idx;
    int64_t local_pos;
    uint8_t flip;
    int output_idx;
  };
  std::vector<TaggedRow> rows(static_cast<size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    const auto& ep = order_[batch_start + i];
    rows[i] = {ep.file_idx, ep.local_pos, flips_[batch_start + i], i};
  }
  std::sort(rows.begin(), rows.end(),
            [](const TaggedRow& a, const TaggedRow& b) { return a.file_idx < b.file_idx; });

  // Build one WorkUnit per contiguous file group.
  int i = 0;
  while (i < n_rows) {
    int fi = rows[i].file_idx;
    WorkUnit unit;
    unit.file = files[fi];
    while (i < n_rows && rows[i].file_idx == fi) {
      unit.local_positions.push_back(rows[i].local_pos);
      unit.flips.push_back(rows[i].flip);
      unit.output_indices.push_back(rows[i].output_idx);
      ++i;
    }
    work_units.push_back(std::move(unit));
  }

  return n_rows;
}

// ===========================================================================
// DataLoader: top-level
// ===========================================================================

DataLoader::DataLoader(const Params& params)
    : params_(params),
      file_manager_(params.memory_budget, std::max(params.num_prefetch_threads, 1),
                    params.task == DecodeTask::kMaxMovePerLane),
      work_manager_(&file_manager_, std::max(params.num_worker_threads, 1), params.spec,
                    params.task) {}

DataLoader::~DataLoader() = default;

void DataLoader::add_file(const std::string& path, int64_t num_positions, int64_t file_size) {
  file_manager_.append(path, num_positions, file_size);
}

int64_t DataLoader::num_positions() const { return file_manager_.num_positions(); }

int DataLoader::num_files() const { return file_manager_.num_files(); }

int64_t DataLoader::resident_bytes() const { return file_manager_.memory_usage(); }

int DataLoader::epoch_start(const EpochConfig& config) {
  file_manager_.reset_prefetch_loop();

  std::lock_guard<std::mutex> lock(epoch_mu_);
  epoch_config_ = config;

  // Snapshot the files. Their order doesn't matter: SamplingManager globally
  // shuffles the per-sample epoch plan, so file_idx assignment is irrelevant to
  // batch composition.
  epoch_files_ = file_manager_.snapshot_files();

  if (epoch_files_.empty() || config.batch_size <= 0) {
    epoch_active_ = false;
    return 0;
  }

  sampling_manager_.build_epoch(epoch_files_, config);
  epoch_active_ = true;

  int64_t total = sampling_manager_.total_positions();
  return static_cast<int>(total / config.batch_size);
}

int DataLoader::load_batch(float* output) {
  if (!output) throw std::invalid_argument("DataLoader::load_batch: output is null");

  std::lock_guard<std::mutex> lock(epoch_mu_);
  if (!epoch_active_) return 0;

  std::deque<WorkUnit> work_units;
  int n_rows = sampling_manager_.next_batch(work_units, epoch_files_);
  if (n_rows == 0) {
    epoch_active_ = false;
    return 0;
  }

  file_manager_.prepare_work_units(work_units);
  work_manager_.process(work_units, epoch_config_, output);
  file_manager_.reset_prefetch_loop();

  if (n_rows < epoch_config_.batch_size) {
    epoch_active_ = false;
  }

  return n_rows;
}

}  // namespace binlog
}  // namespace scribblez
