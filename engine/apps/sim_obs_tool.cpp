// Offline generator of Monte-Carlo sim observations (.sobs sidecars) for
// .slog self-play data -- the sim-evidence inputs of
// docs/sim_residual_feedback.md and the data source for its kill-test.
//
// For a sampled subset of each game's training-eligible turns, the tool
// replays the game to the pre-move decision point, ranks the legal candidates
// (plays and exchanges) by HastyBot static equity, and runs SimRunner over the
// top-K (common random numbers, HastyBot rollouts to a natural end). Each
// processed .slog file gets a same-stem .sobs sidecar. The input is either
// --slog-dir (every .slog in the directory; files whose sidecar already
// exists are skipped, so an interrupted run resumes by rerunning) or one or
// more explicit --slog-file arguments (what generate_kill_test_data.py
// passes, having already selected the files missing a sidecar). Progress
// across all positions of the invocation renders as a bar on stderr (TTY
// only), and a timing summary prints at the end.
//
// Because the self-play games are HastyBot's own, the equity-argmax candidate
// (rank 0) is the move that was actually played at the position, so each
// position's evidence set contains the played move's own sim.

#include "scribblez/agent.h"
#include "scribblez/binary_log.h"
#include "scribblez/cli.h"
#include "scribblez/dictionary.h"
#include "scribblez/game_runner.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/position_encoder.h"
#include "scribblez/sim_observation_log.h"
#include "scribblez/sim_runner.h"
#include "util/hardware.h"
#include "util/progress.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
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
  bool open_leaves = false;
  int rollouts = 200;
  int top_k = 10;
  int positions_per_game = 1;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file (a cap makes smoke runs cheap)
};

// One sampled decision point of the file being processed.
struct PositionWork {
  uint32_t game_idx;
  uint32_t turn_idx;
};

bool position_order(const PositionWork& a, const PositionWork& b) {
  return a.game_idx != b.game_idx ? a.game_idx < b.game_idx : a.turn_idx < b.turn_idx;
}

// A completed position: what SimObsWriter::add_position consumes.
struct PositionResult {
  uint32_t game_idx;
  uint32_t turn_idx;
  uint64_t base_seed;
  std::vector<Move> candidates;
  std::vector<SimObservation> observations;
};

// splitmix64 finalizer: decorrelates the structured (seed, game, turn) triple
// into the rollout base seed.
uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t position_seed(uint64_t run_seed, uint32_t game_idx, uint32_t turn_idx) {
  return mix64(run_seed ^ mix64((static_cast<uint64_t>(game_idx) << 20) | turn_idx));
}

// Sample --positions-per-game eligible turns of game `gm` (without
// replacement, deterministically from the run seed) into `out`.
void sample_positions(const GameMetadata& gm, uint32_t game_idx, const Options& opt,
                      std::vector<PositionWork>* out) {
  const int begin = gm.eligible_begin;
  const int end = gm.eligible_end;
  if (begin >= end) return;
  std::vector<uint32_t> turns(static_cast<size_t>(end - begin));
  std::iota(turns.begin(), turns.end(), static_cast<uint32_t>(begin));
  std::mt19937_64 rng(mix64(opt.seed ^ mix64(0xC0FFEEull + game_idx)));
  std::shuffle(turns.begin(), turns.end(), rng);
  const int take = std::min<int>(opt.positions_per_game, static_cast<int>(turns.size()));
  for (int i = 0; i < take; ++i) out->push_back({game_idx, turns[i]});
}

// Worker: claims positions off the shared index and fills results[i]. Each
// worker owns its replay scratch and a single-threaded SimRunner (parallelism
// is across positions, which utilizes cores better than within-position
// threading and keeps every position's sims deterministic regardless of the
// worker count).
void position_worker(const char* buf, const Dictionary& dict, const Options& opt,
                     const std::vector<PositionWork>& work, std::atomic<size_t>* next,
                     std::vector<PositionResult>* results, util::ProgressMeter* meter) {
  std::vector<TurnRecord> scratch;
  binlog::PositionEncoder encoder(InputEncodingSpec{&dict, false});
  SimRunner::Params params;
  params.rollouts = opt.rollouts;
  params.threads = 1;
  const SimRunner runner(dict, params);
  const int total_tiles = Bag(0).size();

  for (size_t i = next->fetch_add(1); i < work.size(); i = next->fetch_add(1)) {
    const PositionWork& w = work[i];
    const GameLog g = binlog::make_game_view(buf, w.game_idx, scratch, nullptr);
    const int mover = encoder.replay_to_sampled(g, static_cast<int>(w.turn_idx),
                                                /*post_move=*/false);
    SimPosition pos;
    pos.board = encoder.enc().board();
    pos.scores = {encoder.enc().score(0), encoder.enc().score(1)};
    pos.mover = mover;
    pos.rack = encoder.rack(mover);
    // Open leaves: the replay knows both the opponent's rack and the draws
    // that followed their last move, so their retained leave -- the
    // Bayesian-inferable part -- is exact; their replenishments stay hidden
    // and are sampled per rollout.
    if (opt.open_leaves) {
      pos.opp_leave =
        binlog::opp_leave_from_replay(g, static_cast<int>(w.turn_idx), encoder.rack(1 - mover));
    }

    int on_board = 0;
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int c = 0; c < BOARD_SIZE; ++c)
        if (!pos.board.at(r, c).is_empty()) ++on_board;
    const int bag_size = total_tiles - on_board - encoder.rack(0).size() - encoder.rack(1).size();

    // Hidden mode: the opponent's replayed rack is ground truth the mover
    // cannot see, so the candidate ranking must not use it. Open-leaves mode
    // legitimately reveals the retained leave (only equity's endgame
    // adjustments read it).
    const Rack hidden_opp;
    MoveRequest ranking_req{pos.board,         dict,
                            pos.rack,          opt.open_leaves ? pos.opp_leave : hidden_opp,
                            pos.scores[mover], pos.scores[1 - mover],
                            bag_size};

    PositionResult& res = (*results)[i];
    res.game_idx = w.game_idx;
    res.turn_idx = w.turn_idx;
    res.base_seed = position_seed(opt.seed, w.game_idx, w.turn_idx);
    res.candidates = equity_top_k(ranking_req, opt.top_k);
    res.observations = runner.run(pos, res.candidates, res.base_seed);
    meter->add_done();
  }
}

// Generate the .sobs sidecar for one loaded .slog file.
void process_file(const std::vector<char>& buf, const fs::path& sobs_path, const Dictionary& dict,
                  const Options& opt, util::ProgressMeter* meter) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  const GameMetadata* metas =
    reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));

  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  std::vector<PositionWork> work;
  for (uint32_t g = 0; g < num_games; ++g) sample_positions(metas[g], g, opt, &work);
  std::sort(work.begin(), work.end(), position_order);

  std::vector<PositionResult> results(work.size());
  std::atomic<size_t> next{0};
  std::vector<std::thread> workers;
  const int threads = std::clamp<int>(opt.threads, 1, std::max<size_t>(1, work.size()));
  for (int t = 0; t < threads; ++t)
    workers.emplace_back(position_worker, buf.data(), std::cref(dict), std::cref(opt),
                         std::cref(work), &next, &results, meter);
  for (auto& w : workers) w.join();

  // The work list is sorted by (game, turn) and results are indexed by work
  // slot, so the output is canonically ordered and byte-stable across thread
  // counts.
  SimObsWriter writer(sobs_path.string(), opt.open_leaves ? kSimObsFlagOpenLeaves : 0);
  for (const PositionResult& r : results) {
    writer.add_position(r.game_idx, r.turn_idx, r.candidates, r.observations,
                        static_cast<uint32_t>(opt.rollouts), r.base_seed);
  }
  writer.close();
}

// The number of positions the tool would sample from `buf` -- the same
// per-game arithmetic sample_positions applies, for sizing the progress bar
// before any sims run.
uint64_t count_positions(const std::vector<char>& buf, const Options& opt) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  const GameMetadata* metas =
    reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));
  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  uint64_t total = 0;
  for (uint32_t g = 0; g < num_games; ++g) {
    const int eligible = metas[g].eligible_end - metas[g].eligible_begin;
    total += static_cast<uint64_t>(std::clamp(eligible, 0, opt.positions_per_game));
  }
  return total;
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
    po::options_description desc("sim_obs_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each without a .sobs sidecar gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "open-leaves", po::bool_switch(&opt.open_leaves),
      "sim with the opponent's retained leave known (their replenishment draws stay "
      "hidden and sampled) -- the open-leaves information condition; recorded in the "
      ".sobs header flags")("rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
                            "Monte-Carlo rollouts per candidate")(
      "top-k", po::value<int>(&opt.top_k)->default_value(opt.top_k),
      "candidates simmed per position (HastyBot-equity ranked)")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "eligible turns sampled per game")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers")(
      "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
      "run seed (drives position sampling and rollout seeds)")(
      "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
      "process only the first N games of each file (0 = all); for smoke runs");
    Lexicon::instance().add_options(desc);
    parse_command_line(argc, argv, desc);

    const Dictionary& dict = GameRunner::load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());

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

    // Load every pending file's bytes up front so the progress bar's total is
    // known before the sims start. Pending batches are a handful of files, so
    // holding them resident is cheap next to the sim work.
    std::vector<std::pair<fs::path, std::vector<char>>> pending;
    uint64_t total_positions = 0;
    for (const fs::path& slog : slogs) {
      fs::path sobs = slog;
      sobs.replace_extension(".sobs");
      if (fs::exists(sobs)) continue;
      std::vector<char> buf = read_file_bytes(slog);
      const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
      if (buf.size() < sizeof(FileHeader) || hdr->magic != binlog::kMagic ||
          hdr->version != binlog::kVersion) {
        std::cerr << "  skip (bad header): " << slog.filename().string() << "\n";
        continue;
      }
      total_positions += count_positions(buf, opt);
      pending.emplace_back(slog, std::move(buf));
    }
    if (pending.empty()) return 0;
    std::cerr << "sim-obs: " << pending.size() << " file(s), " << total_positions << " positions; "
              << opt.top_k << " candidates x " << opt.rollouts << " rollouts, " << opt.threads
              << " threads\n";

    util::ProgressMeter meter(total_positions, "positions");
    for (const auto& [slog, buf] : pending) {
      fs::path sobs = slog;
      sobs.replace_extension(".sobs");
      process_file(buf, sobs, dict, opt, &meter);
    }
    meter.finish("sim-obs");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
