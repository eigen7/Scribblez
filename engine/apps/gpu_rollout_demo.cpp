// Demo of the hasty-rollout task on the GPU against the CPU reference:
// reaches a position by HastyBot self-play, makes the mover's top candidates
// one group (its rollouts seating known opponent leaves of every length), runs
// N rollouts of each candidate on both, checks them task by task (same draws,
// so same moves and outcomes), and prints the throughput of each. With
// --leaf-model it also checks the leaf model's TensorRT values; the kernel's
// own checks run in the test suite (gpu_kernel_tests, gpu/rollout_check.h).
// docs/plans/gpu_hasty_rollouts.md.

#include "encoding/leaf_eval_input_encoder.h"
#include "game/move.h"
#include "gpu/gpu_rollout_values.h"
#include "gpu/hasty_rollouts_gpu.h"
#include "gpu/rollout_check.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/trt_util.h"
#include "sim/hasty_rollout_task.h"
#include "sim/map_residuals.h"
#include "sim/rollout_values.h"
#include "sim/sim_maps.h"
#include "sim/sim_runner.h"
#include "util/work_pool.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace scribblez {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

using LeafEncoder = LeafEvalInputEncoder;

// The host route over `leaves`: encoded and staged from host memory
// (GpuRolloutValues::predict, the sims' prior), timed. Returns each leaf's
// win value.
struct HostRoute {
  std::vector<LeafPrediction> predictions;
  std::vector<double> win_values;
  double seconds = 0;  // the encoding and predict() with the upload
};

HostRoute run_host_route(GpuRolloutValues& values, const std::vector<CpuLeaf>& leaves) {
  HostRoute out;
  const auto t0 = Clock::now();
  out.predictions = values.predict(leaves);
  out.seconds = seconds_since(t0);
  for (const LeafPrediction& p : out.predictions) out.win_values.push_back(win_value(p.wld));
  return out;
}

// The tiles the network expects placed over the rest of the game from
// `leaf`: the placement classes' probability summed over the empty squares.
double expected_placements(const LeafPrediction& p, const Board& board) {
  double placed = 0;
  for (int sq = 0; sq < kBoardCells; ++sq) {
    if (!board.at(sq / BOARD_SIZE, sq % BOARD_SIZE).is_empty()) continue;
    float probs[rollout_abi::kFateClasses] = {};
    class_probabilities(p.ownership.data() + sq, rollout_abi::kFateClasses, kBoardCells, probs);
    for (int c = 0; c < rollout_abi::kFateClasses - 1; ++c) placed += probs[c];
  }
  return placed;
}

// A consistency read on the network itself over the first `leaves` tasks
// whose rollout seats a full known leave for the opponent: the tiles it
// expects placed from the start (both leaves known there) against the tiles
// the rollout's plies placed plus what it expects from the leaf. The two
// should agree: the rollout only moves the prediction's clock.
struct PlacementBalance {
  int rollouts = 0;
  double from_start = 0, by_plies = 0, from_leaf = 0;  // means
};

PlacementBalance placement_balance(GpuRolloutValues& values, const RolloutBatch& batch,
                                   const std::vector<CpuLeaf>& leaves,
                                   const std::vector<rollout_abi::RolloutFates>& fates,
                                   const HostRoute& host) {
  PlacementBalance out;
  std::vector<CpuLeaf> starts;
  std::vector<size_t> which;
  for (size_t k = 0; k < leaves.size(); ++k) {
    const rollout_abi::RolloutTask& t = batch.tasks[k];
    const RolloutStart& start = batch.starts[t.position];
    const RolloutGroup& group = batch.groups[start.group];
    if (t.rollout >= group.opp_leaves.size() || group.opp_leaves[t.rollout].size() != RACK_SIZE)
      continue;
    if (fates[k].went_out != rollout_abi::kWentOutOpen) continue;
    starts.push_back({.board = start.board,
                      .leaves = {start.leave, group.opp_leaves[t.rollout]},
                      .scores = {start.my_score, start.opp_score}});
    which.push_back(k);
  }
  const std::vector<LeafPrediction> priors = values.predict(starts);
  for (size_t j = 0; j < which.size(); ++j) {
    const size_t k = which[j];
    ++out.rollouts;
    out.from_start += expected_placements(priors[j], starts[j].board);
    int placed = 0;
    for (const int8_t fate : fates[k].fate)
      placed += fate >= 0 && fate < rollout_abi::kFateStaysEmpty;
    out.by_plies += placed;
    out.from_leaf += expected_placements(host.predictions[k], leaves[k].board);
  }
  if (out.rollouts > 0) {
    out.from_start /= out.rollouts;
    out.by_plies /= out.rollouts;
    out.from_leaf /= out.rollouts;
  }
  return out;
}

// The device-completed maps against the host route's completion of the
// same leaves: for every start whose tasks all lie among the first
// `host.predictions.size()` (start-major, so the first starts), the CPU
// reference's fates summed and completed from the host route's heads, each
// task's completion in ascending task order as the device adds them.
struct CompletionCheck {
  int starts = 0, within = 0;
  std::string first_diff;
};

CompletionCheck check_completion(const RolloutBatch& batch,
                                 const std::vector<rollout_abi::RolloutFates>& fates,
                                 const std::vector<rollout_abi::RolloutResult>& results,
                                 const HostRoute& host,
                                 const std::vector<rollout_abi::SimMaps>& gpu, double tolerance) {
  const TasksByStart by_start = tasks_by_start(batch);
  std::vector<rollout_abi::SimMaps> cpu = sum_sim_maps(fates, by_start);
  std::vector<rollout_abi::LeafHeadRows> heads(batch.tasks.size(), rollout_abi::LeafHeadRows{});
  for (size_t k = 0; k < host.predictions.size(); ++k)
    if (!results[k].game_over) heads[k] = host.predictions[k].heads();
  complete_sim_maps(cpu, fates, by_start, heads);
  CompletionCheck out;
  for (size_t s = 0; s < cpu.size(); ++s) {
    bool covered = true;
    for (uint32_t i = by_start.offsets[s]; i < by_start.offsets[s + 1]; ++i)
      covered = covered && by_start.tasks[i] < host.predictions.size();
    if (!covered) continue;
    ++out.starts;
    const std::string diff = sim_maps_difference(cpu[s], gpu[s], float(tolerance));
    if (diff.empty())
      ++out.within;
    else if (out.first_diff.empty())
      out.first_diff = std::format("start {}: {}", s, diff);
  }
  return out;
}

// The largest completion_gap of the starts' maps, as a share of the start's
// rollouts.
double relative_completion_gap(const std::vector<rollout_abi::SimMaps>& maps) {
  double gap = 0;
  for (const rollout_abi::SimMaps& m : maps)
    gap = std::max(gap, double(completion_gap(m)) / std::max(1.0f, m.sims));
  return gap;
}

// The batch through GpuRolloutValues (the kernel, the device encoder and the
// leaf network), then the checks: the device rows of the first leaves
// against the CPU encoder's on the CPU reference's end states, those leaves'
// win values against the host route's, and the maps the network completed
// for counting every rollout once. Returns 1 unless the rows match, the
// values agree and the maps count true.
int evaluate_leaves(GpuRolloutValues& values, const RolloutBatch& batch, const Dictionary& dict,
                    const std::vector<CpuLeaf>& cpu_leaves,
                    const std::vector<rollout_abi::RolloutFates>& cpu_fates,
                    const std::string& onnx_path) {
  values.run(batch);  // the first run builds the TensorRT context's buffers
  const BatchValues valued = values.run(batch);
  const std::vector<RolloutOutcome>& outcomes = valued.outcomes;
  const double maps_gap = relative_completion_gap(valued.maps);
  const GpuRolloutValues::Timings& tm = values.last_timings();
  const auto t0 = Clock::now();
  const std::vector<float> cpu_rows = cpu_leaf_rows(cpu_leaves, dict);
  const double cpu_seconds = seconds_since(t0);
  const RowCheck check = compare_leaf_rows(
    cpu_rows, device_leaf_rows(values.rollouts(), int(cpu_leaves.size()), values.net().max_rows()));
  const HostRoute host = run_host_route(values, cpu_leaves);
  double max_value_diff = 0, mean_win = 0;
  for (size_t i = 0; i < outcomes.size(); ++i) mean_win += outcomes[i].win_value();
  for (size_t i = 0; i < cpu_leaves.size(); ++i) {
    if (outcomes[i].result.game_over) continue;
    max_value_diff =
      std::max(max_value_diff, std::fabs(outcomes[i].win_value() - host.win_values[i]));
  }
  // FP16 runs of one engine at different batch sizes may pick different
  // kernels; beyond this the routes disagree on more than rounding.
  constexpr double kValueTolerance = 1e-2;
  std::vector<rollout_abi::RolloutResult> results;
  for (const RolloutOutcome& o : outcomes) results.push_back(o.result);
  const CompletionCheck completion =
    check_completion(batch, cpu_fates, results, host, valued.maps, kValueTolerance);
  const PlacementBalance balance = placement_balance(values, batch, cpu_leaves, cpu_fates, host);
  const double n = double(outcomes.size());
  std::cout << std::format(
    "\nleaf model {}:\n  mean win value from the chooser's side: {:.4f}\n"
    "  rollouts {:.4f} s + leaves {:.4f} s -> {:.0f} valued rollouts/s (leaves alone {:.0f}/s)\n"
    "  device rows vs CPU encoder: {} of {} identical, max |diff| {}\n"
    "  win values vs the host route on those leaves: max |diff| {:.2g}\n"
    "  completed maps: largest gap from counting every rollout once {:.2g} of the rollouts\n"
    "  completed maps vs the host route's completion: {} of {} starts within {:.0e}\n",
    onnx_path, mean_win / n, tm.rollout_seconds, tm.leaf_seconds,
    n / (tm.rollout_seconds + tm.leaf_seconds), tm.leaves / tm.leaf_seconds, check.rows_identical,
    check.checked, check.max_abs_diff, max_value_diff, maps_gap, completion.within,
    completion.starts, kValueTolerance);
  if (!completion.first_diff.empty()) std::cout << "  first: " << completion.first_diff << "\n";
  std::cout << std::format(
    "  tiles the network expects placed over the rest of the game, over {} rollouts with both "
    "leaves known at the start: {:.1f} from the start; {:.1f} placed by the plies + {:.1f} from "
    "the leaf\n",
    balance.rollouts, balance.from_start, balance.by_plies, balance.from_leaf);
  std::cout << std::format(
    "  the host route: CPU encoder {:.0f} rows/s per thread, predict with upload {:.0f} evals/s\n",
    check.checked / cpu_seconds, check.checked / host.seconds);
  if (check.rows_identical != check.checked)
    std::cout << "  first differences:" << check.first_diffs << "\n";
  // Each rollout adds probabilities summing to 1; the float sums over
  // thousands of rollouts round off parts in a million of the count.
  constexpr double kMapsTolerance = 1e-3;
  const bool ok = check.rows_identical == check.checked && max_value_diff <= kValueTolerance &&
                  maps_gap <= kMapsTolerance && completion.within == completion.starts;
  return ok ? 0 : 1;
}

// How many starts' device maps differ from the CPU reference's.
int differing_maps(const std::vector<rollout_abi::SimMaps>& cpu,
                   const std::vector<rollout_abi::SimMaps>& gpu) {
  int n = 0;
  for (size_t s = 0; s < cpu.size(); ++s) n += !sim_maps_difference(cpu[s], gpu[s]).empty();
  return n;
}

struct Agreement {
  int diff_mismatch = 0, move_mismatch = 0, overflow = 0, game_over = 0;
};

Agreement compare_results(const RolloutBatch& batch,
                          const std::vector<rollout_abi::RolloutResult>& cpu,
                          const std::vector<rollout_abi::RolloutResult>& gpu) {
  Agreement a;
  int shown = 0;
  for (size_t k = 0; k < cpu.size(); ++k) {
    a.overflow += gpu[k].pad;
    a.game_over += cpu[k].game_over;
    const bool bad_diff = cpu[k].diff != gpu[k].diff;
    const bool bad_move =
      !same_move(cpu[k].moves[0], gpu[k].moves[0]) || !same_move(cpu[k].moves[1], gpu[k].moves[1]);
    a.diff_mismatch += bad_diff;
    a.move_mismatch += bad_move;
    if ((bad_diff || bad_move) && shown < 8) {
      if (shown == 0) std::cout << "mismatches:\n";
      std::cout << describe_mismatch(batch, k, cpu[k], gpu[k]);
      ++shown;
    }
  }
  return a;
}

void print_kernel_profile(const GpuHastyRollouts& gpu) {
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
}

int run(int argc, char** argv) {
  namespace po = boost::program_options;
  uint64_t seed = 1;
  int plies = 10, rollouts = 4096, num_candidates = 15,
      threads = int(std::thread::hardware_concurrency());
  int leaf_check = 2048;
  std::string dump_rows;
  std::string leaf_model, leaf_precision = "fp16";
  int leaf_rows = nn::LeafEvalSpec::kDefaultMaxRows;
  po::options_description desc("gpu_rollout_demo options");
  desc.add_options()("help,h", "show this help message and exit");
  desc.add_options()("seed", po::value<uint64_t>(&seed)->default_value(seed),
                     "seed of the self-play game, the known leaves and the rollout draws");
  desc.add_options()("plies", po::value<int>(&plies)->default_value(plies),
                     "HastyBot self-play plies that reach the position");
  desc.add_options()("candidates", po::value<int>(&num_candidates)->default_value(num_candidates),
                     "the mover's top moves by static equity (plus the best exchange)");
  desc.add_options()("rollouts,n", po::value<int>(&rollouts)->default_value(rollouts),
                     "rollouts per candidate");
  desc.add_options()("threads,t", po::value<int>(&threads)->default_value(threads),
                     "CPU reference threads");
  desc.add_options()("leaf-model", po::value<std::string>(&leaf_model),
                     "a deep_bot_leaf_eval ONNX export to value every rollout with, on the GPU");
  desc.add_options()("leaf-check", po::value<int>(&leaf_check)->default_value(leaf_check),
                     "leaves whose device-built rows and values are checked against the CPU");
  desc.add_options()("dump-rows", po::value<std::string>(&dump_rows),
                     "write the CPU encoder's rows of the checked leaves to this file, as raw "
                     "float32 (leaf-check x LeafEvalInputEncoder::kInputFloats), for a look at "
                     "them from Python");
  desc.add_options()("leaf-precision",
                     po::value<std::string>(&leaf_precision)->default_value(leaf_precision),
                     "TensorRT precision for the leaf model: fp32, fp16 or bf16");
  desc.add_options()("leaf-rows", po::value<int>(&leaf_rows)->default_value(leaf_rows),
                     "rows per leaf network call");
  Lexicon::instance().add_options(desc);
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }
  po::notify(vm);
  for (const auto& [name, value] : {std::pair{"rollouts", rollouts},
                                    {"candidates", num_candidates},
                                    {"threads", threads},
                                    {"leaf-check", leaf_check},
                                    {"leaf-rows", leaf_rows}}) {
    if (value < 1) throw po::error(std::format("--{} must be at least 1", name));
  }
  util::set_worker_count(threads);

  const Dictionary& dict = Lexicon::instance().dict();
  HastyEquity::ensure_initialized(Lexicon::instance().name());

  const DecisionPoint at = self_play_position(dict, seed, plies);
  const std::vector<Move> moves = rollout_candidates(at, dict, num_candidates);
  RolloutBatch batch = candidate_batch(at, moves, rollouts, seed);
  batch.sum_maps = true;
  std::cout << at.board.to_string();
  std::cout << std::format("to move: rack {} | scores {} to {} | unseen {}\n", at.rack.to_string(),
                           at.my_score, at.opp_score, batch.groups[0].pool.size());
  int blank_racks = 0;
  for (const rollout_abi::RolloutTask& task : batch.tasks) {
    const RolloutRacks r = rollout_racks(batch, task);
    blank_racks += (r.opp.blanks() > 0 ? 1 : 0) + (r.mine.blanks() > 0 ? 1 : 0);
  }
  std::cout << std::format(
    "{} candidates x {} rollouts (seating known opponent leaves of 0 to 7 tiles); {:.1f}% of the "
    "drawn racks hold a blank\n\n",
    moves.size(), rollouts, 100.0 * blank_racks / (2.0 * double(batch.tasks.size())));

  auto t0 = Clock::now();
  std::vector<CpuLeaf> cpu_leaves(
    leaf_model.empty() ? 0 : std::min(size_t(leaf_check), batch.tasks.size()));
  std::vector<rollout_abi::RolloutFates> cpu_fates;
  const std::vector<rollout_abi::RolloutResult> cpu =
    run_hasty_rollouts_cpu(batch, dict, &cpu_leaves, &cpu_fates);
  const double cpu_seconds = seconds_since(t0);

  t0 = Clock::now();
  GpuHastyRollouts gpu(dict, HastyEquity::instance());
  const double gpu_setup = seconds_since(t0);
  // The first launch includes module load; time the second.
  gpu.run(batch, dict);
  const std::vector<rollout_abi::RolloutResult> res = gpu.run(batch, dict);
  const std::vector<rollout_abi::SimMaps> gpu_maps = gpu.maps();
  const GpuHastyRollouts::Timings& tm = gpu.last_timings();
  const Agreement a = compare_results(batch, cpu, res);
  const int maps_differ = differing_maps(sum_sim_maps(cpu_fates, tasks_by_start(batch)), gpu_maps);

  const double n = double(batch.tasks.size());
  std::cout << std::format(
    "\nagreement: {} of {} rollouts identical ({} outcome mismatches, {} move mismatches); "
    "{} games ended within the rollout; {} span-list overflows; sim maps of {} of {} starts "
    "identical\n",
    int(n) - std::max(a.diff_mismatch, a.move_mismatch), int(n), a.diff_mismatch, a.move_mismatch,
    a.game_over, a.overflow, int(gpu_maps.size()) - maps_differ, gpu_maps.size());
  std::cout << std::format("\n{:<28}{:>12}{:>16}\n", "", "seconds", "rollouts/s");
  std::cout << std::format("{:<28}{:>12.4f}{:>16.0f}\n", std::format("cpu ({} threads)", threads),
                           cpu_seconds, n / cpu_seconds);
  std::cout << std::format("{:<28}{:>12.4f}{:>16.0f}\n", "gpu kernel", tm.kernel_seconds,
                           n / tm.kernel_seconds);
  const double with_transfers = tm.upload_seconds + tm.kernel_seconds + tm.download_seconds;
  std::cout << std::format("{:<28}{:>12.4f}{:>16.0f}\n", "gpu kernel + transfers", with_transfers,
                           n / with_transfers);
  std::cout << std::format("gpu tables: {:.1f} MB uploaded in {:.3f} s (setup {:.3f} s)\n",
                           gpu.table_bytes() / 1e6, gpu.table_upload_seconds(), gpu_setup);
  std::cout << std::format("speedup (kernel vs cpu): {:.1f}x\n", cpu_seconds / tm.kernel_seconds);
  print_kernel_profile(gpu);
  int leaf_status = 0;
  if (!leaf_model.empty()) {
    GpuRolloutValues values(dict, HastyEquity::instance(),
                            {.leaf_onnx = leaf_model,
                             .precision = nn::parse_precision(leaf_precision),
                             .leaf_rows = leaf_rows});
    leaf_status = evaluate_leaves(values, batch, dict, cpu_leaves, cpu_fates, leaf_model);
    if (!dump_rows.empty()) {
      const std::vector<float> rows = cpu_leaf_rows(cpu_leaves, dict);
      std::ofstream out(dump_rows, std::ios::binary);
      out.write(reinterpret_cast<const char*>(rows.data()), std::streamsize(rows.size() * 4));
    }
  }
  return (a.diff_mismatch || a.move_mismatch || maps_differ || leaf_status != 0) ? 1 : 0;
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
