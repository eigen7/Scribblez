// Offline generator of evidence trajectories (.sobs sidecars stamped
// kSimObsFlagTrajectory) for .slog self-play data -- the training data of
// docs/sim_residual_feedback.md's evidence conditioning and proves-best head
// (docs/roadmap.md, item 4).
//
// For a sampled subset of each game's training-eligible turns, the tool
// replays to the pre-move decision point and runs the deployment schedule's
// candidate selection: the greedy anchor (the highest-raw-score legal move,
// model-independent by design), then a randomized-length sequence of
// proposals drawn from a temperature-softmax over the move set evaluation
// model's win-equity scores, then one uniform-random draw over the remaining
// legal moves. The uniform tail is appended LAST deliberately: training rows
// pair an evidence prefix with a held-out simmed candidate, so a last-slot
// sim yields proves-best labels at every prefix size while never entering an
// evidence set -- the deployed loop's evidence contains only proposer picks,
// and the tail's job is calibrating the head over the move space's deep tail
// (both directions: hidden gems and never-labeled junk), not shaping the
// conditional.
//
// All of a position's candidates are simmed in one SimRunner call (common
// random numbers), so the trajectory order is pure record bookkeeping: every
// prefix of a position's record array is a valid evidence set. The proposer
// cannot yet condition on evidence mid-trajectory -- the engine serves the
// unconditioned student (roadmap item 5 lands the fusion runtime) -- so the
// proposal distribution is computed once per position and sampled without
// replacement, which is exactly the roadmap's bootstrap proposer.
//
// The student model is required (--model): generation-0 equity-top-K evidence
// is sim_obs_tool's job. Inference runs on a single dedicated thread
// (NeuralNet's one-net-one-thread contract) that position workers round-trip
// through; the sims dominate wall-clock, so the serialization is free.

#include "agent/agent.h"
#include "data/binary_log.h"
#include "data/sim_observation_log.h"
#include "data/slog_sampling.h"
#include "encoding/input_encoder.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"
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
using binlog::GamePositionIndex;

using StudentService = nn::TrtEvalService<nn::MoveSetEvaluationSpec>;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  bool open_leaves = false;
  int rollouts = 200;
  int proposals_min = 2;
  int proposals_max = 8;
  double temperature = 0.05;
  int proposal_pool = 64;
  int positions_per_game = 1;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file (a cap makes smoke runs cheap)
};

// Parallelism is across positions, so each worker's runner is single-threaded
// (see sim_obs_tool for the determinism rationale).
SimRunner::Params sim_params(const Options& opt) {
  SimRunner::Params p;
  p.rollouts = opt.rollouts;
  p.threads = 1;
  return p;
}

// Reject an unusable invocation before any .slog is read and before any
// worker thread exists (a runner-constructor throw inside a worker would
// terminate the process). --positions-per-game is rejected at 0 for the same
// reason as sim_obs_tool: an empty .sobs would stand in for real evidence.
void validate(const Options& opt) {
  SimRunner::validate(sim_params(opt));
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
  if (opt.proposals_min < 0) throw util::CleanException("--proposals-min must be >= 0");
  if (opt.proposals_max < opt.proposals_min) {
    throw util::CleanException("--proposals-max must be >= --proposals-min");
  }
  if (opt.temperature <= 0.0) throw util::CleanException("--temperature must be > 0");
  if (opt.proposal_pool < 1) throw util::CleanException("--proposal-pool must be >= 1");
}

// A completed position: what SimObsWriter::add_position consumes.
struct PositionResult {
  GamePositionIndex pos;
  uint64_t base_seed;
  uint32_t num_legal_moves;
  uint32_t flags;
  std::vector<Move> candidates;  // trajectory order
  std::vector<SimObservation> observations;
};

// Serializes every student evaluation onto one dedicated thread -- the
// NeuralNet contract is one net driven from one thread -- while position
// workers block on their request's completion. The sims are the long pole by
// orders of magnitude, so the round-trip costs nothing at the tool's scale.
class StudentScorer {
 public:
  explicit StudentScorer(StudentService* service) : service_(service) {}

  // Blocks until `wld_out` / `sd_out` are filled for the batch. Called from
  // position workers.
  void score(const float* board_row, const move_set::MoveFeatureArrays* moves, float* wld_out,
             float* sd_out);

  // Thread body; returns once stop() was called and the queue is drained.
  void run();
  void stop();

 private:
  struct Request {
    const float* board_row;
    const move_set::MoveFeatureArrays* moves;
    float* wld_out;
    float* sd_out;
    bool done = false;
  };

  StudentService* service_;
  std::mutex mutex_;
  std::condition_variable queue_cv_;
  std::condition_variable done_cv_;
  std::deque<Request*> queue_;
  bool stopping_ = false;
};

void StudentScorer::score(const float* board_row, const move_set::MoveFeatureArrays* moves,
                          float* wld_out, float* sd_out) {
  Request req{board_row, moves, wld_out, sd_out};
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.push_back(&req);
  queue_cv_.notify_one();
  done_cv_.wait(lock, [&] { return req.done; });
}

void StudentScorer::run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    queue_cv_.wait(lock, [&] { return !queue_.empty() || stopping_; });
    if (queue_.empty()) return;
    Request* req = queue_.front();
    queue_.pop_front();
    lock.unlock();
    float* const head_out[] = {req->wld_out, req->sd_out};
    service_->evaluate({req->board_row, req->moves}, head_out);
    lock.lock();
    req->done = true;
    done_cv_.notify_all();
  }
}

void StudentScorer::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopping_ = true;
  queue_cv_.notify_all();
}

// The anchor: the highest-raw-score candidate, taken off the move list by a
// rule no model can be wrong about. `ranked` is descending static equity, so
// ties resolve to the equity-preferred instance deterministically
// (max_element keeps the first maximum).
size_t anchor_index(const std::vector<Move>& ranked) {
  return std::ranges::max_element(ranked, {}, &Move::score) - ranked.begin();
}

// Per-worker front end over the shared scorer: owns the staging buffers and
// turns one replayed position's ranked candidate set into per-candidate win
// equities -- the proposal scores select_trajectory samples from.
class CandidateScorer {
 public:
  CandidateScorer(const InputEncodingSpec& spec, StudentScorer* scorer)
      : spec_(spec), scorer_(scorer), board_row_(size_t(input_floats(spec))) {}

  // `encoder` must be replayed to the position; `visible_opp` is the leave
  // the information condition exposes (empty when hidden).
  const std::vector<float>& win_equities(const binlog::PositionEncoder& encoder, int mover,
                                         const Rack& visible_opp, const std::vector<Move>& ranked);

 private:
  InputEncodingSpec spec_;
  StudentScorer* scorer_;
  std::vector<float> board_row_;
  move_set::MoveFeatureArrays move_features_;
  std::vector<float> wld_buf_;
  std::vector<float> sd_buf_;
  std::vector<float> win_equity_;
};

const std::vector<float>& CandidateScorer::win_equities(const binlog::PositionEncoder& encoder,
                                                        int mover, const Rack& visible_opp,
                                                        const std::vector<Move>& ranked) {
  const int n = ranked.size();
  // The contingent input planes read the board's move-generation caches;
  // building them here (a no-op once valid) keeps them lexicon-accurate.
  encoder.enc().board().ensure_movegen_caches(*spec_.dict);
  if (spec_.opp_leave_input) {
    encoder.enc().encode_input(mover, encoder.rack(mover), visible_opp, /*apply_flip=*/false,
                               board_row_.data());
  } else {
    encoder.enc().encode_input(mover, encoder.rack(mover), /*apply_flip=*/false, board_row_.data());
  }
  const int score_diff = encoder.enc().score(mover) - encoder.enc().score(1 - mover);
  move_features_.encode(ranked.data(), n, score_diff);
  wld_buf_.resize(size_t(n) * nn::WldOutput::kRowElems);
  sd_buf_.resize(size_t(n) * nn::ScoreDiffOutput::kRowElems);
  scorer_->score(board_row_.data(), &move_features_, wld_buf_.data(), sd_buf_.data());
  win_equity_.resize(size_t(n));
  for (int c = 0; c < n; ++c) {
    const float* wld = wld_buf_.data() + size_t(c) * nn::WldOutput::kRowElems;
    win_equity_[size_t(c)] = wld[0] + 0.5f * wld[1];
  }
  return win_equity_;
}

// The trajectory's candidate indices into `ranked`, in sim order: anchor,
// then up to a sampled count of temperature-softmax proposals over the
// student's win equities, then (when any move remains) one uniform draw.
// Sets *uniform_tail accordingly.
std::vector<size_t> select_trajectory(const std::vector<Move>& ranked,
                                      const std::vector<float>& win_equity, const Options& opt,
                                      std::mt19937_64& rng, util::SoftmaxSampler& sampler,
                                      bool* uniform_tail) {
  const size_t n = ranked.size();
  std::vector<size_t> chosen{anchor_index(ranked)};
  std::vector<char> taken(n, 0);
  taken[chosen[0]] = 1;

  // The student's ranking, best first: the proposal pool is its unsimmed head.
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return win_equity[a] > win_equity[b]; });

  std::uniform_int_distribution<int> length(opt.proposals_min, opt.proposals_max);
  const int proposals = length(rng);
  std::vector<double> pool_scores;
  std::vector<size_t> pool_index;
  for (int p = 0; p < proposals; ++p) {
    pool_scores.clear();
    pool_index.clear();
    for (size_t i : order) {
      if (taken[i]) continue;
      pool_scores.push_back(win_equity[i]);
      pool_index.push_back(i);
      if (int(pool_index.size()) >= opt.proposal_pool) break;
    }
    if (pool_index.empty()) break;
    const int j = sampler.sample(pool_scores, int(pool_scores.size()), opt.temperature, rng);
    chosen.push_back(pool_index[size_t(j)]);
    taken[pool_index[size_t(j)]] = 1;
  }

  std::vector<size_t> unsimmed;
  for (size_t i = 0; i < n; ++i) {
    if (!taken[i]) unsimmed.push_back(i);
  }
  *uniform_tail = !unsimmed.empty();
  if (*uniform_tail) {
    std::uniform_int_distribution<size_t> pick(0, unsimmed.size() - 1);
    chosen.push_back(unsimmed[pick(rng)]);
  }
  return chosen;
}

// Worker: claims positions off the shared index and fills results[i] --
// replay, rank, score, select, sim. Each worker owns its replay scratch, its
// scoring buffers, and a single-threaded SimRunner; student evaluations
// round-trip through the shared scorer.
void position_worker(const char* buf, const Dictionary& dict, const InputEncodingSpec& spec,
                     const Options& opt, const std::vector<GamePositionIndex>& work,
                     std::atomic<size_t>* next, StudentScorer* scorer,
                     std::vector<PositionResult>* results, util::ProgressMeter* meter) {
  std::vector<TurnRecord> scratch;
  binlog::PositionEncoder encoder(spec);
  const SimRunner runner(dict, sim_params(opt));
  CandidateScorer candidate_scorer(spec, scorer);
  util::SoftmaxSampler sampler;

  for (size_t i = next->fetch_add(1); i < work.size(); i = next->fetch_add(1)) {
    const GamePositionIndex& w = work[i];
    const GameLog g = binlog::make_game_view(buf, w.game_idx, scratch, nullptr);
    const int mover = encoder.replay_to_sampled(g, int(w.turn_idx),
                                                /*post_move=*/false);
    SimPosition pos;
    pos.board = encoder.enc().board();
    pos.scores = {encoder.enc().score(0), encoder.enc().score(1)};
    pos.mover = mover;
    pos.rack = encoder.rack(mover);
    if (opt.open_leaves) {
      pos.opp_leave = binlog::opp_leave_from_replay(g, int(w.turn_idx), encoder.rack(1 - mover));
    }

    // Hidden mode: the opponent's replayed rack is ground truth the mover
    // cannot see, so neither the ranking nor the student input may use it.
    const Rack hidden_opp;
    const Rack& visible_opp = opt.open_leaves ? pos.opp_leave : hidden_opp;
    MoveRequest ranking_req{
      pos.board,         dict, pos.rack, visible_opp, pos.scores[mover], pos.scores[1 - mover],
      encoder.bag_size()};
    const std::vector<Move> ranked = equity_top_k(ranking_req, std::numeric_limits<int>::max());
    const std::vector<float>& win_equity =
      candidate_scorer.win_equities(encoder, mover, visible_opp, ranked);

    PositionResult& res = (*results)[i];
    res.pos = w;
    res.base_seed = binlog::position_seed(opt.seed, w.game_idx, w.turn_idx);
    res.num_legal_moves = ranked.size();
    // The trajectory draws come from their own stream so adding a proposal
    // never perturbs the rollout seeds (base_seed feeds SimRunner directly).
    std::mt19937_64 rng(util::splitmix64(res.base_seed ^ 0x7A6A11EC70ull));
    bool uniform_tail = false;
    const std::vector<size_t> chosen =
      select_trajectory(ranked, win_equity, opt, rng, sampler, &uniform_tail);
    res.flags = uniform_tail ? kSimObsPosFlagUniformTail : 0u;
    res.candidates.reserve(chosen.size());
    for (size_t idx : chosen) res.candidates.push_back(ranked[idx]);
    res.observations = runner.run(pos, res.candidates, res.base_seed);
    meter->add_done();
  }
}

// Generate the trajectory .sobs sidecar for one loaded .slog file.
void process_file(const std::vector<char>& buf, const fs::path& sobs_path, const Dictionary& dict,
                  const InputEncodingSpec& spec, StudentService* service,
                  const std::string& proposer_hash, const Options& opt,
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

  std::vector<PositionResult> results(work.size());
  std::atomic<size_t> next{0};
  StudentScorer scorer(service);
  std::thread gpu(&StudentScorer::run, &scorer);
  std::vector<std::thread> workers;
  const int threads = std::clamp<int>(opt.threads, 1, std::max<size_t>(1, work.size()));
  for (int t = 0; t < threads; ++t)
    workers.emplace_back(position_worker, buf.data(), std::cref(dict), std::cref(spec),
                         std::cref(opt), std::cref(work), &next, &scorer, &results, meter);
  for (auto& w : workers) w.join();
  scorer.stop();
  gpu.join();

  // The work list is sorted by (game, turn) and results are indexed by work
  // slot, so the output is canonically ordered and byte-stable across thread
  // counts.
  const uint32_t flags = kSimObsFlagTrajectory | (opt.open_leaves ? kSimObsFlagOpenLeaves : 0u);
  SimObsWriter writer(sobs_path.string(), flags, proposer_hash);
  for (const PositionResult& r : results) {
    writer.add_position(r.pos.game_idx, r.pos.turn_idx, r.candidates, r.observations,
                        uint32_t(opt.rollouts), r.base_seed, r.num_legal_moves, r.flags);
  }
  writer.close();
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    nn::NeuralNetParams<nn::MoveSetEvaluationSpec> params;
    po::options_description desc("evidence_trajectory_generator options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each without a .sobs sidecar gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "open-leaves", po::bool_switch(&opt.open_leaves),
      "sim and score with the opponent's retained leave known -- required for face-up-leaves "
      "games, and must match the model's input arm")(
      "rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
      "Monte-Carlo rollouts per candidate")(
      "proposals-min", po::value<int>(&opt.proposals_min)->default_value(opt.proposals_min),
      "least model proposals per trajectory (the randomized length's lower bound)")(
      "proposals-max", po::value<int>(&opt.proposals_max)->default_value(opt.proposals_max),
      "most model proposals per trajectory")(
      "temperature", po::value<double>(&opt.temperature)->default_value(opt.temperature),
      "softmax temperature over the model's win-equity scores (win-equity units)")(
      "proposal-pool", po::value<int>(&opt.proposal_pool)->default_value(opt.proposal_pool),
      "proposals are drawn from the model's top-N unsimmed candidates")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "eligible turns sampled per game")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers")(
      "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
      "run seed; MUST match the target generator's --seed for its sampled positions to "
      "contain this tool's (the forced-candidate labeling relies on it)")(
      "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
      "process only the first N games of each file (0 = all); for smoke runs");
    params.add_options(desc);
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());

    StudentService service(params);
    service.load();
    if (service.opp_leave_input() != opt.open_leaves) {
      throw util::CleanException(
        "the model's input arm (opp_leave_input={}) must match --open-leaves={}: a mismatched "
        "proposer would score moves under an information condition the sims do not share",
        service.opp_leave_input(), opt.open_leaves);
    }
    const InputEncodingSpec spec{&dict, service.contingent_features(), service.opp_leave_input()};
    const std::string proposer_hash = nn::content_hash(binlog::read_file_bytes(params.onnx_path));

    // Games played face up must be simmed face up (see sim_obs_tool for why
    // the reverse pairing is allowed).
    const std::vector<binlog::PendingSlog> pending = binlog::load_pending_slogs(
      binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files), ".sobs", opt.open_leaves,
      "{} was played with face-up leaves; pass --open-leaves to sim it");
    if (pending.empty()) return 0;
    uint64_t total_positions = 0;
    for (const binlog::PendingSlog& p : pending)
      total_positions +=
        binlog::count_sampled_positions(p.bytes, opt.positions_per_game, opt.limit_games);
    std::cerr << "evidence trajectories: " << pending.size() << " file(s), " << total_positions
              << " positions; anchor + " << opt.proposals_min << ".." << opt.proposals_max
              << " proposals + uniform tail x " << opt.rollouts << " rollouts, proposer "
              << proposer_hash.substr(0, 12) << ", " << opt.threads << " threads\n";

    util::ProgressMeter meter(total_positions, "positions");
    for (const binlog::PendingSlog& p : pending) {
      process_file(p.bytes, p.sidecar(".sobs"), dict, spec, &service, proposer_hash, opt, &meter);
    }
    meter.finish("evidence trajectories");
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
