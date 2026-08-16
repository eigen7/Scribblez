// Offline generator of move set evaluation model distillation targets (.mset
// sidecars) for .slog self-play data (docs/roadmap.md, track A).
//
// For every training-eligible position of each game (or a per-game sample),
// the tool replays to the pre-move decision point, selects the candidates to
// label, encodes each candidate's post-move state exactly as a position
// evaluation model training row, and evaluates the batch with the teacher
// position evaluation model (TensorRT). Each .slog file gets a same-stem .mset
// sidecar recording the selected Moves, the teacher's value readouts
// (move_set_eval::kTargetFloatsV1), and -- on stratified runs -- its four
// placement planes per candidate; files whose sidecar already exists are
// skipped, so interrupted runs resume by rerunning.
//
// Two selections, chosen per run by --full-sweep and both defined in
// training/move_set_eval_candidates.h: the stratified sample that feeds
// training, and the capped sweep of every legal candidate that the A3 gate
// metrics are measured on. A sweep run stamps kTargetFlagFullSweep into its
// .mset, which routes the whole file to the held-out side -- swept positions
// are evaluation-only.
//
// Information condition: the teacher's input arm comes from its ONNX metadata,
// and each .mset records the variant of the games it labeled (the source
// .slog's face-up-leaves flag).
//
// Threading: encoding a post-move row costs roughly a move generation (the
// contingent-map block), so CPU encoding -- not GPU inference -- is the
// bottleneck. Encoder workers produce encoded candidate slices into a bounded
// queue; a single inference thread packs them into TensorRT batches and
// scatters the readouts back.

#include "agent/agent.h"
#include "data/binary_log.h"
#include "data/sim_observation_log.h"
#include "data/slog_sampling.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "sim/sim_runner.h"
#include "training/move_set_eval_candidates.h"
#include "training/move_set_eval_target_log.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/progress.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;
using binlog::FileHeader;
using binlog::GameMetadata;

// Candidates per encoded slice. An input row is ~80 KB, so a sweep position's
// worth of them (up to sweep_cap) is far too much to hold, hand off, or stage
// for inference at once; slicing bounds all three. A stratified position stays
// a single slice.
constexpr int kSliceCandidates = 64;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  move_set_eval::StratumQuotas quotas;
  bool use_sobs = false;
  bool full_sweep = false;
  int sweep_cap = 1500;
  int positions_per_game = 0;  // 0 = every training-eligible turn
  int threads = std::max(1u, std::thread::hardware_concurrency());
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file (a cap makes smoke runs cheap)
  // Derived in main() from the inference batch size, not a CLI flag: a slice
  // must fit the inference thread's staging buffer.
  int slice_candidates = kSliceCandidates;
};

using binlog::GamePositionIndex;

// The simmed trajectory candidates of a companion .sobs, keyed by decision
// point -- the moves the stratified selection must force-include (roadmap
// item 4: the value-labeled subset always contains the proposer's sims).
using ForcedCandidates = std::map<GamePositionIndex, std::vector<Move>>;

// Placement-plane floats per candidate: the teacher's four masks, which the
// writer quantizes into the record's plane block.
constexpr int kPlaneFloats = nn::kNumMaskHeads * move_set_eval::kPlaneCells;

// The teacher: the concrete position-eval service, held by type because the
// mask-labelling overload is constrained to specs with aux outputs.
using TeacherService = nn::TrtEvalService<nn::PositionEvaluationSpec>;

// One contiguous run of a position's candidates with their encoded post-move
// rows, produced by an encoder worker and consumed by the inference thread,
// which fills `targets` (candidates.size() x move_set_eval::kTargetFloatsV1)
// and, on a stratified run, `planes` (candidates.size() x kPlaneFloats).
// A position's slices partition its candidate set in order.
struct CandidateSlice {
  GamePositionIndex pos;
  uint32_t first;                // this slice's first candidate within the position's set
  uint32_t position_candidates;  // candidates in the whole position
  uint32_t num_legal_moves;      // the position's legal-move count (swept positions only)
  std::vector<Move> candidates;
  std::vector<float> rows;  // candidates.size() x input_floats(spec)
  std::vector<float> targets;
  std::vector<float> planes;

  bool completes_position() const { return first + candidates.size() == position_candidates; }
};

bool slice_order(const CandidateSlice& a, const CandidateSlice& b) {
  if (a.pos != b.pos) return a.pos < b.pos;
  return a.first < b.first;
}

// Bounded handoff between the encoder pool and the inference thread. Bounding
// keeps memory flat when encoding outpaces the GPU; the bound is in rows rather
// than slices because rows are the memory that matters and a slice's row count
// varies with the selection.
class SliceQueue {
 public:
  explicit SliceQueue(size_t row_budget) : row_budget_(row_budget) {}

  void add_producer();
  void push(CandidateSlice&& item);
  // Pops one slice; returns false when the queue is drained AND every producer
  // has finished.
  bool pop(CandidateSlice* out);
  void producer_done();

 private:
  size_t row_budget_;
  size_t queued_rows_ = 0;
  std::mutex mutex_;
  std::condition_variable can_push_;
  std::condition_variable can_pop_;
  std::deque<CandidateSlice> items_;
  int active_producers_ = 0;
};

void SliceQueue::add_producer() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++active_producers_;
}

void SliceQueue::push(CandidateSlice&& item) {
  const size_t rows = item.candidates.size();
  std::unique_lock<std::mutex> lock(mutex_);
  // An empty queue also admits a slice larger than the whole budget, so no
  // budget setting can deadlock a push.
  can_push_.wait(lock, [&] { return queued_rows_ + rows <= row_budget_ || items_.empty(); });
  queued_rows_ += rows;
  items_.push_back(std::move(item));
  can_pop_.notify_one();
}

bool SliceQueue::pop(CandidateSlice* out) {
  std::unique_lock<std::mutex> lock(mutex_);
  can_pop_.wait(lock, [&] { return !items_.empty() || active_producers_ == 0; });
  if (items_.empty()) return false;
  *out = std::move(items_.front());
  items_.pop_front();
  queued_rows_ -= out->candidates.size();
  can_push_.notify_all();
  return true;
}

void SliceQueue::producer_done() {
  std::lock_guard<std::mutex> lock(mutex_);
  --active_producers_;
  can_pop_.notify_all();
}

// The run's selection over `ranked` -- every legal play and exchange,
// best-equity first -- for the position whose game actually played `played`.
// `forced` carries the position's simmed trajectory candidates, if any.
move_set_eval::Selection select_candidates(const std::vector<Move>& ranked, const Move& played,
                                           const Options& opt, std::mt19937_64& rng,
                                           std::span<const Move> forced) {
  if (opt.full_sweep) return move_set_eval::full_sweep_candidates(ranked, played, opt.sweep_cap);
  return move_set_eval::stratified_candidates(ranked, played, opt.quotas, rng, forced);
}

// Encode `sel`'s candidates for the position `encoder` is replayed to, handing
// the inference thread one slice at a time.
void encode_slices(const binlog::PositionEncoder& encoder, const GameLog& g,
                   const GamePositionIndex& w, int mover, const move_set_eval::Selection& sel,
                   int slice_candidates, int row_floats, SliceQueue* queue) {
  const size_t total = sel.candidates.size();
  for (size_t first = 0; first < total; first += size_t(slice_candidates)) {
    const size_t count = std::min(size_t(slice_candidates), total - first);
    CandidateSlice slice;
    slice.pos = w;
    slice.first = first;
    slice.position_candidates = total;
    slice.num_legal_moves = sel.num_legal_moves;
    slice.candidates.assign(sel.candidates.begin() + ptrdiff_t(first),
                            sel.candidates.begin() + ptrdiff_t(first + count));
    slice.rows.resize(count * size_t(row_floats));
    binlog::encode_candidate_rows(encoder, g, int(w.turn_idx), mover, slice.candidates,
                                  slice.rows.data());
    queue->push(std::move(slice));
  }
}

// Encoder worker: claims positions, replays to the pre-move state, selects
// candidates, and encodes each candidate's post-move row exactly as a
// post-move-model training row.
void encode_worker(const char* buf, const Dictionary& dict, const InputEncodingSpec& spec,
                   const Options& opt, const std::vector<GamePositionIndex>& work,
                   const ForcedCandidates* forced, std::atomic<size_t>* next, SliceQueue* queue) {
  std::vector<TurnRecord> scratch;
  binlog::PositionEncoder encoder(spec);
  const int row_floats = input_floats(spec);

  for (size_t i = next->fetch_add(1); i < work.size(); i = next->fetch_add(1)) {
    const GamePositionIndex& w = work[i];
    const GameLog g = binlog::make_game_view(buf, w.game_idx, scratch, nullptr);
    const int mover = encoder.replay_to_sampled(g, int(w.turn_idx),
                                                /*post_move=*/false);
    const Rack& rack = encoder.rack(mover);
    const Board& board = encoder.enc().board();
    const int bag_size = encoder.bag_size();

    // Rank every candidate by HastyBot equity (the opponent's replayed rack is
    // hidden information the ranking must not use), then select from it.
    const Rack hidden_opp;
    MoveRequest req{
      board,   dict, rack, hidden_opp, encoder.enc().score(mover), encoder.enc().score(1 - mover),
      bag_size};
    const std::vector<Move> ranked = equity_top_k(req, std::numeric_limits<int>::max());
    std::mt19937_64 rng(binlog::position_seed(opt.seed, w.game_idx, w.turn_idx));
    std::span<const Move> forced_here;
    if (forced) {
      const auto it = forced->find(w);
      if (it != forced->end()) forced_here = it->second;
    }
    const move_set_eval::Selection sel =
      select_candidates(ranked, g.records[w.turn_idx].move, opt, rng, forced_here);
    encode_slices(encoder, g, w, mover, sel, opt.slice_candidates, row_floats, queue);
  }
  queue->producer_done();
}

// The inference thread: packs queued slices into TensorRT batches (whole
// slices, up to the batch limit), evaluates, and scatters the teacher's
// readouts into each slice's target block (plus its plane block, on a run that
// labels planes -- a full-sweep run does not, matching its plane-less .mset).
// One instance drives one file's thread; run() is the thread body, and the
// completed slices are read from done() after it joins.
class InferenceLoop {
 public:
  InferenceLoop(TeacherService* service, int row_floats, int batch_size, bool label_planes,
                util::ProgressMeter* meter);

  void run(SliceQueue* queue);

  // Slices in thread-completion order; call after run() returns.
  std::vector<CandidateSlice>& done() { return done_; }

 private:
  void flush();

  TeacherService* service_;
  int row_floats_;
  int batch_size_;
  bool label_planes_;
  util::ProgressMeter* meter_;

  // Staging buffers, sized once for a full batch and reused across flushes.
  std::vector<float> inputs_;
  std::vector<float> wld_buf_;
  std::vector<float> score_diff_buf_;
  std::vector<float> masks_;

  std::vector<CandidateSlice> pending_;
  int pending_rows_ = 0;
  std::vector<CandidateSlice> done_;
};

InferenceLoop::InferenceLoop(TeacherService* service, int row_floats, int batch_size,
                             bool label_planes, util::ProgressMeter* meter)
    : service_(service),
      row_floats_(row_floats),
      batch_size_(batch_size),
      label_planes_(label_planes),
      meter_(meter),
      inputs_(size_t(batch_size) * row_floats),
      wld_buf_(size_t(batch_size) * nn::WldOutput::kRowElems),
      score_diff_buf_(size_t(batch_size) * nn::ScoreDiffOutput::kRowElems),
      masks_(label_planes ? batch_size * kPlaneFloats : 0) {}

void InferenceLoop::run(SliceQueue* queue) {
  CandidateSlice item;
  while (queue->pop(&item)) {
    const int count = item.candidates.size();
    if (pending_rows_ + count > batch_size_) {
      flush();
      pending_rows_ = 0;
    }
    pending_rows_ += count;
    pending_.push_back(std::move(item));
  }
  flush();
}

void InferenceLoop::flush() {
  if (pending_.empty()) return;
  int rows = 0;
  for (const CandidateSlice& p : pending_) {
    std::memcpy(inputs_.data() + rows * row_floats_, p.rows.data(), p.rows.size() * sizeof(float));
    rows += p.candidates.size();
  }
  float* const head_out[] = {wld_buf_.data(), score_diff_buf_.data()};
  service_->evaluate({inputs_.data(), rows}, head_out, label_planes_ ? masks_.data() : nullptr);
  int cursor = 0;
  for (CandidateSlice& p : pending_) {
    const int count = p.candidates.size();
    p.targets.resize(count * move_set_eval::kTargetFloatsV1);
    if (label_planes_) {
      p.planes.assign(masks_.data() + cursor * kPlaneFloats,
                      masks_.data() + (cursor + count) * kPlaneFloats);
    }
    for (int c = 0; c < count; ++c) {
      const float* wld = wld_buf_.data() + size_t(cursor) * nn::WldOutput::kRowElems;
      const float* sd = score_diff_buf_.data() + size_t(cursor) * nn::ScoreDiffOutput::kRowElems;
      ++cursor;
      float* t = p.targets.data() + c * move_set_eval::kTargetFloatsV1;
      t[0] = wld[0];
      t[1] = wld[1];
      t[2] = wld[2];
      t[3] = sd[0];
      t[4] = move_set_eval::clamped_sd_std(sd[1]);
    }
    // Free the encoded rows now that they are consumed; done_ accumulates a
    // whole file's slices and must not hold every encoding.
    p.rows.clear();
    p.rows.shrink_to_fit();
    // The meter counts positions, so only a position's last slice advances it.
    if (p.completes_position()) meter_->add_done();
  }
  for (CandidateSlice& p : pending_) done_.push_back(std::move(p));
  pending_.clear();
}

// Reassemble each position from its slices and write it as one .mset record.
// `slices` arrive in thread-completion order; sorting restores the file's
// canonical position order and each position's candidate order.
void write_positions(std::vector<CandidateSlice>& slices, move_set_eval::TargetWriter* writer) {
  std::sort(slices.begin(), slices.end(), slice_order);
  size_t i = 0;
  while (i < slices.size()) {
    const CandidateSlice& head = slices[i];
    std::vector<Move> candidates;
    std::vector<float> targets;
    std::vector<float> planes;
    candidates.reserve(head.position_candidates);
    targets.reserve(size_t(head.position_candidates) * move_set_eval::kTargetFloatsV1);
    size_t j = i;
    for (; j < slices.size() && slices[j].pos == head.pos; ++j) {
      const CandidateSlice& s = slices[j];
      candidates.insert(candidates.end(), s.candidates.begin(), s.candidates.end());
      targets.insert(targets.end(), s.targets.begin(), s.targets.end());
      planes.insert(planes.end(), s.planes.begin(), s.planes.end());
    }
    RELEASE_ASSERT(candidates.size() == head.position_candidates);
    writer->add_position(head.pos.game_idx, head.pos.turn_idx, candidates, targets, planes,
                         head.num_legal_moves);
    i = j;
  }
}

// Generate the .mset sidecar for one loaded .slog file.
void process_file(const std::vector<char>& buf, const fs::path& mset_path, const Dictionary& dict,
                  const InputEncodingSpec& spec, TeacherService* service, int batch_size,
                  const std::string& model_hash, const Options& opt, const ForcedCandidates* forced,
                  util::ProgressMeter* meter) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  const GameMetadata* metas =
    reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));

  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  std::vector<GamePositionIndex> work;
  for (uint32_t g = 0; g < num_games; ++g) {
    binlog::sample_eligible_turns(metas[g], g, opt.seed, opt.positions_per_game, &work);
  }
  std::sort(work.begin(), work.end());

  // Every trajectory position must be one this run labels, or its sims never
  // receive value labels -- the misconfiguration (mismatched --seed, or a
  // smaller --positions-per-game than the trajectory run's) would otherwise
  // waste the corpus's most expensive rows silently.
  if (forced) {
    for (const auto& [pos, moves] : *forced) {
      if (!std::binary_search(work.begin(), work.end(), pos)) {
        throw util::CleanException(
          "{}: trajectory position (game {}, turn {}) is not in this run's sample; the "
          "generator's --seed and --positions-per-game must contain the trajectory run's",
          mset_path.filename().string(), pos.game_idx, pos.turn_idx);
      }
    }
  }

  // A few batches in flight keeps the GPU fed without letting encoded rows
  // (~80 KB each) accumulate.
  SliceQueue queue(/*row_budget=*/size_t(4 * batch_size));
  std::atomic<size_t> next{0};
  const int encoders = std::clamp<int>(opt.threads, 1, std::max<size_t>(1, work.size()));
  for (int t = 0; t < encoders; ++t) queue.add_producer();
  std::vector<std::thread> workers;
  for (int t = 0; t < encoders; ++t)
    workers.emplace_back(encode_worker, buf.data(), std::cref(dict), std::cref(spec),
                         std::cref(opt), std::cref(work), forced, &next, &queue);
  const int row_floats = input_floats(spec);
  InferenceLoop inference(service, row_floats, batch_size, /*label_planes=*/!opt.full_sweep, meter);
  std::thread gpu(&InferenceLoop::run, &inference, &queue);
  for (auto& w : workers) w.join();
  gpu.join();

  const uint32_t flags = move_set_eval::target_flags_from_slog(hdr->flags) |
                         (opt.full_sweep ? move_set_eval::kTargetFlagFullSweep : 0u);
  // Stratified (training) files carry the teacher's placement planes; a
  // full-sweep file does not -- it is evaluation-only with value-based metrics,
  // and its positions run to sweep_cap candidates (move_set_eval_target_log.h).
  const uint32_t record_planes = opt.full_sweep ? 0u : move_set_eval::kTargetPlanes;
  move_set_eval::TargetWriter writer(mset_path.string(), move_set_eval::kTargetFloatsV1,
                                     record_planes, model_hash, flags);
  write_positions(inference.done(), &writer);
  writer.close();
}

// The simmed candidates of a trajectory sidecar, keyed by decision point.
ForcedCandidates load_forced_candidates(const fs::path& sobs_path) {
  SimObsReader reader(sobs_path.string());
  ForcedCandidates out;
  for (int i = 0; i < reader.num_positions(); ++i) {
    const SimObsReader::Position p = reader.position(i);
    std::vector<Move>& moves = out[{p.header->game_index, p.header->turn_index}];
    for (uint32_t c = 0; c < p.header->num_candidates; ++c) moves.push_back(p.records[c].move);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    nn::NeuralNetParams<nn::PositionEvaluationSpec> params;
    po::options_description desc("move_set_eval_target_generator options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each without a .mset sidecar gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "sobs", po::bool_switch(&opt.use_sobs),
      "stratified only: force-include each position's simmed candidates from the same-stem "
      ".sobs trajectory sidecar, which must exist (run evidence_trajectory_generator first)")(
      "full-sweep", po::bool_switch(&opt.full_sweep),
      "label every legal candidate (capped by --sweep-cap) instead of the stratified sample; "
      "the .mset is stamped full-sweep, which makes it evaluation-only")(
      "sweep-cap", po::value<int>(&opt.sweep_cap)->default_value(opt.sweep_cap),
      "with --full-sweep, the most plays to label per position, by static-equity rank "
      "(exchanges and the played move are kept beyond it)")(
      "quota-top", po::value<int>(&opt.quotas.top)->default_value(opt.quotas.top),
      "stratified only: candidates from the head of the equity ranking")(
      "quota-mid", po::value<int>(&opt.quotas.mid)->default_value(opt.quotas.mid),
      "stratified only: candidates sampled from the contention zone (ranks "
      "quota-top..mid-rank-limit)")(
      "quota-tail", po::value<int>(&opt.quotas.tail)->default_value(opt.quotas.tail),
      "stratified only: candidates sampled uniformly from the remaining ranks")(
      "quota-exchange", po::value<int>(&opt.quotas.exchange)->default_value(opt.quotas.exchange),
      "stratified only: exchange candidates")(
      "mid-rank-limit",
      po::value<int>(&opt.quotas.mid_rank_limit)->default_value(opt.quotas.mid_rank_limit),
      "stratified only: exclusive rank bound of the contention zone")(
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
    if (opt.use_sobs && opt.full_sweep) {
      throw util::CleanException("--sobs applies to the stratified selection, not --full-sweep");
    }

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());

    // A stratified run labels placement planes, so the teacher's masks must
    // come back from the device (a sweep labels values alone).
    params.copy_aux = !opt.full_sweep;
    TeacherService service(params);
    service.load();
    const InputEncodingSpec spec{&dict, service.contingent_features(), service.opp_leave_input()};
    const std::string model_hash = nn::content_hash(binlog::read_file_bytes(params.onnx_path));
    opt.slice_candidates = std::min(kSliceCandidates, params.max_rows);

    // Games played face up must be labeled by a teacher that sees the leaves.
    // A blind teacher's readouts would describe a game nobody played, and the
    // .mset -- stamped open-leaves after its source log -- would hand the
    // student targets its own input arm cannot account for. The reverse
    // pairing is deliberate and allowed: an open-leaves teacher over a
    // standard corpus is the privileged-teacher instrument.
    const std::vector<binlog::PendingSlog> pending = binlog::load_pending_slogs(
      binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files), ".mset", spec.opp_leave_input,
      "{} was played with face-up leaves; labeling it needs a teacher whose ONNX declares "
      "opp_leave_input");
    if (pending.empty()) return 0;
    uint64_t total_positions = 0;
    for (const binlog::PendingSlog& p : pending)
      total_positions +=
        binlog::count_sampled_positions(p.bytes, opt.positions_per_game, opt.limit_games);
    const std::string selection =
      opt.full_sweep ? std::format("full sweep (cap {})", opt.sweep_cap) : "stratified";
    std::cerr << "move set eval targets: " << pending.size() << " file(s), " << total_positions
              << " positions, " << selection << "; teacher " << model_hash.substr(0, 12) << ", "
              << opt.threads << " encoder threads\n";
    // A swept position costs roughly `sweep_cap` encodes against a stratified
    // one's ~15, so the position count above understates a sweep's work by
    // orders of magnitude. Sweeping every eligible turn is virtually never
    // what a caller wants (the mode exists to label a few positions per game),
    // and the default --positions-per-game asks for exactly that.
    if (opt.full_sweep && opt.positions_per_game <= 0) {
      std::cerr << "  warning: --full-sweep with --positions-per-game=0 sweeps EVERY eligible "
                   "turn, up to "
                << opt.sweep_cap << " candidates each; pass --positions-per-game to bound it\n";
    }

    util::ProgressMeter meter(total_positions, "positions");
    for (const binlog::PendingSlog& p : pending) {
      ForcedCandidates forced;
      if (opt.use_sobs) {
        const fs::path sobs = p.sidecar(".sobs");
        if (!fs::exists(sobs)) {
          throw util::CleanException("--sobs: {} has no .sobs sidecar", p.path.filename().string());
        }
        forced = load_forced_candidates(sobs);
      }
      process_file(p.bytes, p.sidecar(".mset"), dict, spec, &service, params.max_rows, model_hash,
                   opt, opt.use_sobs ? &forced : nullptr, &meter);
    }
    meter.finish("move set eval targets");
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
