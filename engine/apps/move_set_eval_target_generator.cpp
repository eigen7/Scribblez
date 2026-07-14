// Offline generator of move set evaluation model distillation targets (.mset
// sidecars) for .slog self-play data (docs/roadmap2.md, track A).
//
// For every training-eligible position of each game (or a per-game sample),
// the tool replays to the pre-move decision point, draws a STRATIFIED sample
// of the legal candidates, encodes each candidate's post-move state exactly
// as a position evaluation model training row, and evaluates the batch with
// the teacher position evaluation model (TensorRT). Each .slog file gets a
// same-stem .mset sidecar recording the sampled Moves and the teacher's
// readouts (move_set_eval::kTargetFloatsV1); files whose sidecar already
// exists are skipped, so interrupted runs resume by rerunning.
//
// The stratified sample balances the filter's two failure modes: dense
// coverage at the top of the equity ranking (where ranking precision
// matters), a slice of the contention zone, uniform coverage of the tail
// (junk rejection -- and where surprising constructive plays live), and
// exchange candidates (the exchange head starves otherwise). The actually-
// played move is always included. Distillation needs coverage, not
// unbiasedness: the sampler only has to visit a move for the teacher to
// value it honestly.
//
// Threading: encoding a post-move row costs roughly a move generation (the
// contingent-map block), so CPU encoding -- not GPU inference -- is the
// bottleneck. Encoder workers produce whole-position row blocks into a
// bounded queue; a single inference thread packs them into TensorRT batches
// and scatters the readouts back.

#include "agent/agent.h"
#include "data/binary_log.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/nn_evaluation_service.h"
#include "nn/trt_util.h"
#include "selfplay/game_runner.h"
#include "selfplay/sim_runner.h"
#include "training/move_set_eval_target_log.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/progress.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <compare>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;
using binlog::FileHeader;
using binlog::GameMetadata;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  int quota_top = 4;
  int quota_mid = 4;
  int quota_tail = 4;
  int quota_exchange = 2;
  int mid_rank_limit = 32;
  int positions_per_game = 0;  // 0 = every training-eligible turn
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file (a cap makes smoke runs cheap)
};

// Identifies one decision point: (game, turn) within the file being
// processed. Ordering is the file's canonical position order.
struct GamePositionIndex {
  uint32_t game_idx;
  uint32_t turn_idx;

  auto operator<=>(const GamePositionIndex&) const = default;
};

// A position's sampled candidates with their encoded post-move rows, produced
// by an encoder worker and consumed by the inference thread, which fills
// `targets` (num candidates x move_set_eval::kTargetFloatsV1).
struct MoveSet {
  GamePositionIndex pos;
  std::vector<Move> candidates;
  std::vector<float> rows;  // candidates.size() x input_floats(spec)
  std::vector<float> targets;
};

bool move_set_order(const MoveSet& a, const MoveSet& b) { return a.pos < b.pos; }

// Bounded handoff between the encoder pool and the inference thread. Bounding
// keeps memory flat when encoding outpaces the GPU.
class MoveSetQueue {
 public:
  explicit MoveSetQueue(size_t capacity) : capacity_(capacity) {}

  void add_producer();
  void push(MoveSet&& item);
  // Pops one item; returns false when the queue is drained AND every producer
  // has finished.
  bool pop(MoveSet* out);
  void producer_done();

 private:
  size_t capacity_;
  std::mutex mutex_;
  std::condition_variable can_push_;
  std::condition_variable can_pop_;
  std::deque<MoveSet> items_;
  int active_producers_ = 0;
};

void MoveSetQueue::add_producer() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++active_producers_;
}

void MoveSetQueue::push(MoveSet&& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  can_push_.wait(lock, [&] { return items_.size() < capacity_; });
  items_.push_back(std::move(item));
  can_pop_.notify_one();
}

bool MoveSetQueue::pop(MoveSet* out) {
  std::unique_lock<std::mutex> lock(mutex_);
  can_pop_.wait(lock, [&] { return !items_.empty() || active_producers_ == 0; });
  if (items_.empty()) return false;
  *out = std::move(items_.front());
  items_.pop_front();
  can_push_.notify_one();
  return true;
}

void MoveSetQueue::producer_done() {
  std::lock_guard<std::mutex> lock(mutex_);
  --active_producers_;
  can_pop_.notify_all();
}

// Append `count` distinct picks from ranked[lo, hi) (uniform, without
// replacement) that are not already in *out.
void sample_range(const std::vector<Move>& ranked, int lo, int hi, int count, std::mt19937_64& rng,
                  std::vector<Move>* out) {
  std::vector<int> pool;
  for (int i = lo; i < hi && i < static_cast<int>(ranked.size()); ++i) {
    if (std::find(out->begin(), out->end(), ranked[i]) == out->end()) pool.push_back(i);
  }
  std::shuffle(pool.begin(), pool.end(), rng);
  for (int j = 0; j < count && j < static_cast<int>(pool.size()); ++j) {
    out->push_back(ranked[static_cast<size_t>(pool[j])]);
  }
}

// The stratified candidate sample for one position (see the file comment).
// `ranked` is every legal play and exchange, best-equity first; `played` is
// the move the self-play game actually made there.
std::vector<Move> sample_candidates(const std::vector<Move>& ranked, const Move& played,
                                    const Options& opt, std::mt19937_64& rng) {
  std::vector<Move> out;
  out.reserve(
    static_cast<size_t>(1 + opt.quota_top + opt.quota_mid + opt.quota_tail + opt.quota_exchange));
  out.push_back(played);
  const int n = static_cast<int>(ranked.size());

  // Top stratum: the head of the ranking, dense.
  for (int i = 0; i < n && static_cast<int>(out.size()) < 1 + opt.quota_top; ++i) {
    if (std::find(out.begin(), out.end(), ranked[i]) == out.end()) out.push_back(ranked[i]);
  }
  // Contention zone, then the tail, uniform within each.
  sample_range(ranked, opt.quota_top, opt.mid_rank_limit, opt.quota_mid, rng, &out);
  sample_range(ranked, opt.mid_rank_limit, n, opt.quota_tail, rng, &out);

  // Exchange stratum: uniform among the non-PLAY candidates.
  std::vector<Move> non_plays;
  for (const Move& m : ranked) {
    if (m.type() != MoveType::PLAY) non_plays.push_back(m);
  }
  std::shuffle(non_plays.begin(), non_plays.end(), rng);
  int taken = 0;
  for (const Move& m : non_plays) {
    if (taken >= opt.quota_exchange) break;
    if (std::find(out.begin(), out.end(), m) == out.end()) {
      out.push_back(m);
      ++taken;
    }
  }
  return out;
}

// Encoder worker: claims positions, replays to the pre-move state, samples
// candidates, and encodes each candidate's post-move row exactly as a
// post-move-model training row (apply the move to a copy of the replayed
// encoder, rack = the leave, no symmetry flip).
void encode_worker(const char* buf, const Dictionary& dict, const InputEncodingSpec& spec,
                   const Options& opt, const std::vector<GamePositionIndex>& work,
                   std::atomic<size_t>* next, MoveSetQueue* queue) {
  std::vector<TurnRecord> scratch;
  binlog::PositionEncoder encoder(spec);
  const int row_floats = input_floats(spec);
  const int total_tiles = Bag(0).size();

  for (size_t i = next->fetch_add(1); i < work.size(); i = next->fetch_add(1)) {
    const GamePositionIndex& w = work[i];
    const GameLog g = binlog::make_game_view(buf, w.game_idx, scratch, nullptr);
    const int mover = encoder.replay_to_sampled(g, static_cast<int>(w.turn_idx),
                                                /*post_move=*/false);
    const Rack& rack = encoder.rack(mover);
    const Board& board = encoder.enc().board();

    int on_board = 0;
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int c = 0; c < BOARD_SIZE; ++c)
        if (!board.at(r, c).is_empty()) ++on_board;
    const int bag_size = total_tiles - on_board - encoder.rack(0).size() - encoder.rack(1).size();

    // Rank every candidate by HastyBot equity (the opponent's replayed rack is
    // hidden information the ranking must not use), then sample the strata.
    const Rack hidden_opp;
    MoveRequest req{
      board,   dict, rack, hidden_opp, encoder.enc().score(mover), encoder.enc().score(1 - mover),
      bag_size};
    const std::vector<Move> ranked = equity_top_k(req, std::numeric_limits<int>::max());
    std::mt19937_64 rng(util::splitmix64(
      opt.seed ^ util::splitmix64((static_cast<uint64_t>(w.game_idx) << 20) | w.turn_idx)));
    MoveSet item;
    item.pos = w;
    item.candidates = sample_candidates(ranked, g.records[w.turn_idx].move, opt, rng);

    // The cross-check planes read the board's movegen caches; building them on
    // the replayed board (a no-op once valid) lets every candidate copy update
    // them incrementally.
    board.ensure_movegen_caches(dict);
    item.rows.resize(item.candidates.size() * static_cast<size_t>(row_floats));
    for (size_t c = 0; c < item.candidates.size(); ++c) {
      const Move& mv = item.candidates[c];
      Rack leave = rack;
      for (int t = 0; t < mv.num_glyphs(); ++t) leave.remove(mv.glyph(t).rack_tile());
      GameStateEncoder post = encoder.enc();
      post.apply_move(mv);
      post.encode_input(mover, leave, /*apply_flip=*/false,
                        item.rows.data() + c * static_cast<size_t>(row_floats));
    }
    queue->push(std::move(item));
  }
  queue->producer_done();
}

// Inference thread: packs queued positions into TensorRT batches (whole
// positions, up to the batch limit), evaluates, and scatters the teacher's
// readouts into each position's target block.
void inference_loop(nn::NNEvaluationService* service, int row_floats, int batch_size,
                    util::ProgressMeter* meter, MoveSetQueue* queue, std::vector<MoveSet>* done,
                    std::mutex* done_mutex) {
  std::vector<MoveSet> pending;
  std::vector<float> inputs(static_cast<size_t>(batch_size) * row_floats);
  std::vector<nn::Eval> evals(static_cast<size_t>(batch_size));

  auto flush = [&]() {
    if (pending.empty()) return;
    int rows = 0;
    for (const MoveSet& p : pending) {
      std::memcpy(inputs.data() + static_cast<size_t>(rows) * row_floats, p.rows.data(),
                  p.rows.size() * sizeof(float));
      rows += static_cast<int>(p.candidates.size());
    }
    service->evaluate(inputs.data(), rows, evals.data());
    int cursor = 0;
    for (MoveSet& p : pending) {
      p.targets.resize(p.candidates.size() * move_set_eval::kTargetFloatsV1);
      for (size_t c = 0; c < p.candidates.size(); ++c) {
        const nn::Eval& e = evals[static_cast<size_t>(cursor++)];
        float* t = p.targets.data() + c * move_set_eval::kTargetFloatsV1;
        t[0] = e.p_win;
        t[1] = e.p_draw;
        t[2] = e.p_loss;
        t[3] = e.score_diff_mean;
        t[4] = e.score_diff_std;
      }
      // Free the encoded rows now that they are consumed; `done` accumulates
      // a whole file's positions and must not hold every encoding.
      p.rows.clear();
      p.rows.shrink_to_fit();
      meter->add_done();
    }
    std::lock_guard<std::mutex> lock(*done_mutex);
    for (MoveSet& p : pending) done->push_back(std::move(p));
    pending.clear();
  };

  int pending_rows = 0;
  MoveSet item;
  while (queue->pop(&item)) {
    if (pending_rows + static_cast<int>(item.candidates.size()) > batch_size) {
      flush();
      pending_rows = 0;
    }
    pending_rows += static_cast<int>(item.candidates.size());
    pending.push_back(std::move(item));
  }
  flush();
}

// Generate the .mset sidecar for one loaded .slog file.
void process_file(const std::vector<char>& buf, const fs::path& mset_path, const Dictionary& dict,
                  const InputEncodingSpec& spec, nn::NNEvaluationService* service, int batch_size,
                  const std::string& model_hash, const Options& opt, util::ProgressMeter* meter) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  const GameMetadata* metas =
    reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));

  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  std::vector<GamePositionIndex> work;
  for (uint32_t g = 0; g < num_games; ++g) {
    const int begin = metas[g].eligible_begin;
    const int end = metas[g].eligible_end;
    if (begin >= end) continue;
    if (opt.positions_per_game <= 0) {
      for (int t = begin; t < end; ++t) work.push_back({g, static_cast<uint32_t>(t)});
    } else {
      std::vector<uint32_t> turns(static_cast<size_t>(end - begin));
      std::iota(turns.begin(), turns.end(), static_cast<uint32_t>(begin));
      std::mt19937_64 rng(util::splitmix64(opt.seed ^ util::splitmix64(0xC0FFEEull + g)));
      std::shuffle(turns.begin(), turns.end(), rng);
      const int take = std::min<int>(opt.positions_per_game, static_cast<int>(turns.size()));
      for (int i = 0; i < take; ++i) work.push_back({g, turns[i]});
    }
  }
  std::sort(work.begin(), work.end());

  MoveSetQueue queue(/*capacity=*/64);
  std::vector<MoveSet> done;
  std::mutex done_mutex;
  std::atomic<size_t> next{0};
  const int encoders = std::clamp<int>(opt.threads, 1, std::max<size_t>(1, work.size()));
  for (int t = 0; t < encoders; ++t) queue.add_producer();
  std::vector<std::thread> workers;
  for (int t = 0; t < encoders; ++t)
    workers.emplace_back(encode_worker, buf.data(), std::cref(dict), std::cref(spec),
                         std::cref(opt), std::cref(work), &next, &queue);
  const int row_floats = input_floats(spec);
  std::thread gpu(inference_loop, service, row_floats, batch_size, meter, &queue, &done,
                  &done_mutex);
  for (auto& w : workers) w.join();
  gpu.join();

  // Canonical order, independent of thread scheduling.
  std::sort(done.begin(), done.end(), move_set_order);
  move_set_eval::TargetWriter writer(mset_path.string(), move_set_eval::kTargetFloatsV1,
                                     model_hash);
  for (const MoveSet& p : done) {
    writer.add_position(p.pos.game_idx, p.pos.turn_idx, p.candidates, p.targets);
  }
  writer.close();
}

std::vector<char> read_file_bytes(const fs::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(size));
  f.read(bytes.data(), size);
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    nn::NeuralNetParams params;
    po::options_description desc("move_set_eval_target_generator options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each without a .mset sidecar gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "quota-top", po::value<int>(&opt.quota_top)->default_value(opt.quota_top),
      "candidates from the head of the equity ranking")(
      "quota-mid", po::value<int>(&opt.quota_mid)->default_value(opt.quota_mid),
      "candidates sampled from the contention zone (ranks quota-top..mid-rank-limit)")(
      "quota-tail", po::value<int>(&opt.quota_tail)->default_value(opt.quota_tail),
      "candidates sampled uniformly from the remaining ranks")(
      "quota-exchange", po::value<int>(&opt.quota_exchange)->default_value(opt.quota_exchange),
      "exchange candidates")("mid-rank-limit",
                             po::value<int>(&opt.mid_rank_limit)->default_value(opt.mid_rank_limit),
                             "exclusive rank bound of the contention zone")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "eligible turns sampled per game (0 = every eligible turn)")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads),
      "encoder worker threads (encoding, not inference, is the bottleneck)")(
      "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
      "run seed (drives position and stratum sampling)")(
      "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
      "process only the first N games of each file (0 = all); for smoke runs");
    params.add_options(desc);
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);

    const Dictionary& dict = GameRunner::load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());

    nn::NNEvaluationService service(params);
    service.load();
    const InputEncodingSpec spec{&dict, service.contingent_features()};
    const std::string model_hash = nn::content_hash(read_file_bytes(params.onnx_path));

    std::vector<fs::path> slogs;
    if (!opt.slog_files.empty()) {
      for (const std::string& f : opt.slog_files) slogs.emplace_back(f);
    } else if (!opt.slog_dir.empty()) {
      for (const auto& entry : fs::directory_iterator(opt.slog_dir))
        if (entry.path().extension() == ".slog") slogs.push_back(entry.path());
      std::sort(slogs.begin(), slogs.end());
    } else {
      throw std::runtime_error("pass --slog-dir or --slog-file");
    }
    if (slogs.empty()) throw std::runtime_error("no .slog files to process");

    std::vector<std::pair<fs::path, std::vector<char>>> pending;
    uint64_t total_positions = 0;
    for (const fs::path& slog : slogs) {
      fs::path mset = slog;
      mset.replace_extension(".mset");
      if (fs::exists(mset)) continue;
      std::vector<char> buf = read_file_bytes(slog);
      const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
      if (buf.size() < sizeof(FileHeader) || hdr->magic != binlog::kMagic ||
          hdr->version != binlog::kVersion) {
        std::cerr << "  skip (bad header): " << slog.filename().string() << "\n";
        continue;
      }
      const GameMetadata* metas =
        reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));
      uint32_t num_games = hdr->num_games;
      if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
      for (uint32_t g = 0; g < num_games; ++g) {
        const int eligible = metas[g].eligible_end - metas[g].eligible_begin;
        const int per_game =
          opt.positions_per_game <= 0 ? eligible : std::min(opt.positions_per_game, eligible);
        total_positions += static_cast<uint64_t>(std::max(per_game, 0));
      }
      pending.emplace_back(slog, std::move(buf));
    }
    if (pending.empty()) return 0;
    std::cerr << "move set eval targets: " << pending.size() << " file(s), " << total_positions
              << " positions; teacher " << model_hash.substr(0, 12) << ", " << opt.threads
              << " encoder threads\n";

    util::ProgressMeter meter(total_positions, "positions");
    for (const auto& [slog, buf] : pending) {
      fs::path mset = slog;
      mset.replace_extension(".mset");
      process_file(buf, mset, dict, spec, &service, params.max_batch_size, model_hash, opt, &meter);
    }
    meter.finish("move set eval targets");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
