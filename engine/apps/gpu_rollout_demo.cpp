// Demo of the hasty-rollout task on the GPU against the CPU reference:
// reaches a position by HastyBot self-play, runs N rollouts on both, checks
// them rollout by rollout (same draws, so same moves and outcomes), and
// prints the throughput of each. docs/plans/gpu_hasty_rollouts.md.

#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/game.h"
#include "game/game_log.h"
#include "game/move.h"
#include "gpu/hasty_rollouts_gpu.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "sim/hasty_rollout_task.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace scribblez {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// The position after `plies` plies of HastyBot self-play: the last mover is
// "us" and holds their leave.
HastyRolloutPosition self_play_position(const Dictionary& dict, uint64_t seed, int plies) {
  HastyBot p0({.thread_id = 0, .name = "p0"});
  HastyBot p1({.thread_id = 0, .name = "p1"});
  Game game(p0, p1, dict, seed);
  game.set_max_plies(plies);
  game.play();
  const GameLog log = game.log();
  if (!game.truncated() || log.num_records < 1) {
    throw std::runtime_error(std::format("the game ended before {} plies", plies));
  }
  const int us = log.records[log.num_records - 1].player;
  return make_rollout_position(game.board(), game.leave(us), game.score(us), game.score(1 - us),
                               seed);
}

bool same_move(const rollout_abi::PackedMove& a, const rollout_abi::PackedMove& b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

void print_mismatch(const HastyRolloutPosition& pos, uint32_t i,
                    const rollout_abi::RolloutResult& cpu, const rollout_abi::RolloutResult& gpu) {
  const RolloutRacks racks = rollout_racks(pos, i);
  std::cout << std::format("  rollout {}: opp rack {} our rack {} | diff cpu {} gpu {}\n", i,
                           racks.opp.to_string(), racks.mine.to_string(), cpu.diff, gpu.diff);
  Board board = pos.board;
  for (int ply = 0; ply < 2; ++ply) {
    const Move mc = unpack_move(cpu.moves[ply]), mg = unpack_move(gpu.moves[ply]);
    std::cout << std::format("    ply {}: cpu [{}] gpu [{}]{}\n", ply, describe_move(mc, board),
                             describe_move(mg, board),
                             same_move(cpu.moves[ply], gpu.moves[ply]) ? "" : "  <-- differs");
    if (mc.type() == MoveType::PLAY) board.apply(mc);
  }
}

int run(int argc, char** argv) {
  namespace po = boost::program_options;
  uint64_t seed = 1;
  int plies = 10, rollouts = 65536, threads = int(std::thread::hardware_concurrency());
  po::options_description desc("gpu_rollout_demo options");
  desc.add_options()("help,h", "show this help message and exit");
  desc.add_options()("seed", po::value<uint64_t>(&seed)->default_value(seed),
                     "seed of the self-play game and of the rollout draws");
  desc.add_options()("plies", po::value<int>(&plies)->default_value(plies),
                     "HastyBot self-play plies that reach the position");
  desc.add_options()("rollouts,n", po::value<int>(&rollouts)->default_value(rollouts),
                     "rollouts per backend");
  desc.add_options()("threads,t", po::value<int>(&threads)->default_value(threads),
                     "CPU reference threads");
  Lexicon::instance().add_options(desc);
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }
  po::notify(vm);

  const Dictionary& dict = Lexicon::instance().dict();
  HastyEquity::ensure_initialized(Lexicon::instance().name());

  const HastyRolloutPosition pos = self_play_position(dict, seed, plies);
  std::cout << pos.board.to_string();
  std::cout << std::format("our leave {} | scores us {} opp {} | unseen {} (opp rack {}, bag {})\n",
                           pos.leave.to_string(), pos.my_score, pos.opp_score, pos.pool.size(),
                           rollout_abi::opp_rack_draw(pos.pool.size()),
                           pos.pool.size() - rollout_abi::opp_rack_draw(pos.pool.size()));
  int blank_racks = 0;
  for (int i = 0; i < rollouts; ++i) {
    const RolloutRacks r = rollout_racks(pos, uint32_t(i));
    blank_racks += (r.opp.blanks() > 0 ? 1 : 0) + (r.mine.blanks() > 0 ? 1 : 0);
  }
  std::cout << std::format("{} rollouts; {:.1f}% of the drawn racks hold a blank\n\n", rollouts,
                           100.0 * blank_racks / (2.0 * rollouts));

  auto t0 = Clock::now();
  const std::vector<rollout_abi::RolloutResult> cpu =
    run_hasty_rollouts_cpu(pos, dict, rollouts, threads);
  const double cpu_seconds = seconds_since(t0);

  t0 = Clock::now();
  GpuHastyRollouts gpu(dict, HastyEquity::instance());
  const double gpu_setup = seconds_since(t0);
  // The first launch includes module load; time the second.
  gpu.run(pos, dict, rollouts);
  const std::vector<rollout_abi::RolloutResult> res = gpu.run(pos, dict, rollouts);
  const GpuHastyRollouts::Timings& tm = gpu.last_timings();

  double cpu_mean = 0, gpu_mean = 0;
  int diff_mismatch = 0, move_mismatch = 0, overflow = 0, game_over = 0, shown = 0;
  for (int i = 0; i < rollouts; ++i) {
    cpu_mean += cpu[size_t(i)].diff;
    gpu_mean += res[size_t(i)].diff;
    overflow += res[size_t(i)].pad;
    game_over += cpu[size_t(i)].game_over;
    const bool bad_diff = cpu[size_t(i)].diff != res[size_t(i)].diff;
    const bool bad_move = !same_move(cpu[size_t(i)].moves[0], res[size_t(i)].moves[0]) ||
                          !same_move(cpu[size_t(i)].moves[1], res[size_t(i)].moves[1]);
    diff_mismatch += bad_diff;
    move_mismatch += bad_move;
    if ((bad_diff || bad_move) && shown < 8) {
      if (shown == 0) std::cout << "mismatches:\n";
      print_mismatch(pos, uint32_t(i), cpu[size_t(i)], res[size_t(i)]);
      ++shown;
    }
  }
  cpu_mean /= rollouts;
  gpu_mean /= rollouts;

  std::cout << std::format("\nmean score differential: cpu {:+.3f} gpu {:+.3f}\n", cpu_mean,
                           gpu_mean);
  std::cout << std::format(
    "agreement: {} of {} rollouts identical ({} outcome mismatches, {} move mismatches); "
    "{} games ended within the rollout; {} span-list overflows\n",
    rollouts - std::max(diff_mismatch, move_mismatch), rollouts, diff_mismatch, move_mismatch,
    game_over, overflow);
  std::cout << std::format("\n{:<28}{:>12}{:>16}\n", "", "seconds", "rollouts/s");
  std::cout << std::format("{:<28}{:>12.4f}{:>16.0f}\n", std::format("cpu ({} threads)", threads),
                           cpu_seconds, rollouts / cpu_seconds);
  std::cout << std::format("{:<28}{:>12.4f}{:>16.0f}\n", "gpu kernel", tm.kernel_seconds,
                           rollouts / tm.kernel_seconds);
  std::cout << std::format(
    "{:<28}{:>12.4f}{:>16.0f}\n", "gpu kernel + transfers",
    tm.upload_seconds + tm.kernel_seconds + tm.download_seconds,
    rollouts / (tm.upload_seconds + tm.kernel_seconds + tm.download_seconds));
  std::cout << std::format("gpu tables: {:.1f} MB uploaded in {:.3f} s (setup {:.3f} s)\n",
                           gpu.table_bytes() / 1e6, gpu.table_upload_seconds(), gpu_setup);
  std::cout << std::format("speedup (kernel vs cpu): {:.1f}x\n", cpu_seconds / tm.kernel_seconds);
  std::cout << "kernel time by phase:";
  for (const GpuHastyRollouts::PhaseShare& ps : gpu.last_phase_shares()) {
    std::cout << std::format("  {} {:.0f}%", ps.name, 100.0 * ps.share);
  }
  std::cout << "\n";
  const std::vector<GpuHastyRollouts::CountValue> counts = gpu.last_counts();
  std::cout << "per ply:";
  for (const GpuHastyRollouts::CountValue& cv : counts) {
    if (cv.name == std::string("plies")) continue;
    std::cout << std::format("  {} {:.0f}", cv.name, double(cv.value) / double(counts[0].value));
  }
  std::cout << "\n";
  return (diff_mismatch || move_mismatch) ? 1 : 0;
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  try {
    return scribblez::run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "gpu_rollout_demo: " << e.what() << "\n";
    return 2;
  }
}
