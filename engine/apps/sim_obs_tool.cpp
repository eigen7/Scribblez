// Offline generator of Monte-Carlo sim observations (.sobs sidecars) for
// .slog self-play data -- the sim-evidence inputs of
// docs/sim_residual_feedback.md and the data source for its kill-test.
//
// For a sampled subset of each game's training-eligible turns, the tool
// replays the game to the pre-move decision point, ranks the legal candidates
// (plays and exchanges) by HastyBot static equity, and runs SimRunner over the
// top-K (common random numbers, HastyBot rollouts to a natural end). Every
// .slog file in --slog-dir gets a same-stem .sobs sidecar; files whose sidecar
// already exists are skipped, so an interrupted run resumes by rerunning.
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
  int rollouts = 200;
  int top_k = 10;
  int positions_per_game = 1;
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
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

// The mover-POV candidate set at the decision point, ranked by HastyBot static
// equity (best first): every legal play and exchange, or a lone PASS when
// nothing is legal. Equity is evaluated with an empty opponent rack -- the
// opponent's tiles are hidden from the mover mid-game, and the sims run only
// at bag-non-empty positions where the pre-endgame adjustments that read the
// opponent rack are dormant.
std::vector<Move> top_k_candidates(const MoveRequest& req, int k) {
  std::vector<Move> candidates = generate_legal_plays(req);
  const std::vector<Move> exchanges = generate_legal_exchanges(req);
  candidates.insert(candidates.end(), exchanges.begin(), exchanges.end());
  if (candidates.empty()) return {Move::pass()};

  const std::vector<double> vals = HastyEquity::instance().equities(
    candidates, req.board, req.bag_size, req.opp_rack, req.my_rack);
  const int n = static_cast<int>(candidates.size());
  const int keep = std::min(k, n);
  std::vector<int> idx(static_cast<size_t>(n));
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                    [&](int a, int b) { return vals[a] > vals[b]; });
  std::vector<Move> top;
  top.reserve(static_cast<size_t>(keep));
  for (int j = 0; j < keep; ++j) top.push_back(candidates[static_cast<size_t>(idx[j])]);
  return top;
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
                     std::vector<PositionResult>* results) {
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

    int on_board = 0;
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int c = 0; c < BOARD_SIZE; ++c)
        if (!pos.board.at(r, c).is_empty()) ++on_board;
    const int bag_size = total_tiles - on_board - encoder.rack(0).size() - encoder.rack(1).size();

    // The opponent's replayed rack is ground truth the mover cannot see; the
    // candidate ranking must not use it.
    const Rack hidden_opp;
    MoveRequest ranking_req{
      pos.board, dict, pos.rack, hidden_opp, pos.scores[mover], pos.scores[1 - mover], bag_size};

    PositionResult& res = (*results)[i];
    res.game_idx = w.game_idx;
    res.turn_idx = w.turn_idx;
    res.base_seed = position_seed(opt.seed, w.game_idx, w.turn_idx);
    res.candidates = top_k_candidates(ranking_req, opt.top_k);
    res.observations = runner.run(pos, res.candidates, res.base_seed);
  }
}

// Generate the .sobs sidecar for one loaded .slog file.
void process_file(const std::vector<char>& buf, const fs::path& sobs_path, const Dictionary& dict,
                  const Options& opt) {
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
                         std::cref(work), &next, &results);
  for (auto& w : workers) w.join();

  // The work list is sorted by (game, turn) and results are indexed by work
  // slot, so the output is canonically ordered and byte-stable across thread
  // counts.
  SimObsWriter writer(sobs_path.string());
  for (const PositionResult& r : results) {
    writer.add_position(r.game_idx, r.turn_idx, r.candidates, r.observations,
                        static_cast<uint32_t>(opt.rollouts), r.base_seed);
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
    po::options_description desc("sim_obs_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir)->required(),
      "directory of .slog files; each gets a same-stem .sobs sidecar")(
      "rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
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
    for (const auto& entry : fs::directory_iterator(opt.slog_dir))
      if (entry.path().extension() == ".slog") slogs.push_back(entry.path());
    std::sort(slogs.begin(), slogs.end());
    if (slogs.empty()) throw std::runtime_error("no .slog files in " + opt.slog_dir);
    std::cerr << slogs.size() << " .slog files; " << opt.top_k << " candidates x " << opt.rollouts
              << " rollouts, " << opt.positions_per_game << " position(s)/game, " << opt.threads
              << " threads\n";

    for (const fs::path& slog : slogs) {
      fs::path sobs = slog;
      sobs.replace_extension(".sobs");
      if (fs::exists(sobs)) {
        std::cerr << "  skip (exists): " << sobs.filename().string() << "\n";
        continue;
      }
      const std::vector<char> buf = read_file_bytes(slog);
      const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
      if (buf.size() < sizeof(FileHeader) || hdr->magic != binlog::kMagic ||
          hdr->version != binlog::kVersion) {
        std::cerr << "  skip (bad header): " << slog.filename().string() << "\n";
        continue;
      }
      process_file(buf, sobs, dict, opt);
      std::cerr << "  wrote " << sobs.filename().string() << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
