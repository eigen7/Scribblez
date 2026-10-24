// Demo of the hasty-rollout task on the GPU against the CPU reference:
// reaches a position by HastyBot self-play, makes the mover's top candidates
// one group (every other rollout seating a known opponent leave), runs N
// rollouts of each candidate on both, checks them task by task (same draws,
// so same moves and outcomes), and prints the throughput of each.
// docs/plans/gpu_hasty_rollouts.md.

#include "agent/hasty_bot.h"
#include "encoding/leaf_eval_input_encoder.h"
#include "game/board.h"
#include "game/game.h"
#include "game/game_log.h"
#include "game/move.h"
#include "gpu/gpu_rollout_values.h"
#include "gpu/hasty_rollouts_gpu.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/trt_util.h"
#include "sim/hasty_rollout_task.h"
#include "sim/rollout_values.h"
#include "sim/sim_runner.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <random>
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

// The decision point after `plies` plies of HastyBot self-play: the board,
// the mover's rack and the scores from the mover's side.
struct DecisionPoint {
  Board board;
  Rack rack;
  int my_score = 0, opp_score = 0;
};

DecisionPoint self_play_position(const Dictionary& dict, uint64_t seed, int plies) {
  HastyBot p0({.thread_id = 0, .name = "p0"});
  HastyBot p1({.thread_id = 0, .name = "p1"});
  Game game(p0, p1, dict, seed);
  game.set_max_plies(plies);
  game.play();
  const GameLog log = game.log();
  if (!game.truncated() || log.num_records < 1) {
    throw std::runtime_error(std::format("the game ended before {} plies", plies));
  }
  const int mover = 1 - log.records[log.num_records - 1].player;
  return {game.board(), game.rack(mover), game.score(mover), game.score(1 - mover)};
}

// The mover's top `count` moves by static equity, plus the best exchange if
// the cut has none.
std::vector<Move> candidates(const DecisionPoint& at, const Dictionary& dict, int count) {
  const MoveRequest req{at.board,
                        dict,
                        at.rack,
                        Rack{},
                        at.my_score,
                        at.opp_score,
                        at.board.pov_bag_size(at.rack.size())};
  const std::vector<Move> ranked = equity_top_k(req, std::numeric_limits<int>::max());
  std::vector<Move> out(ranked.begin(), ranked.begin() + std::min<size_t>(ranked.size(), count));
  const auto is_exchange = [](const Move& m) { return m.type() == MoveType::EXCHANGE; };
  if (std::none_of(out.begin(), out.end(), is_exchange)) {
    const auto it = std::find_if(ranked.begin(), ranked.end(), is_exchange);
    if (it != ranked.end()) out.push_back(*it);
  }
  return out;
}

// Every other rollout seats a known opponent leave of 1 to 6 tiles from the
// pool; the rest draw the opponent's whole rack.
std::vector<Rack> known_leaves(const TileCounts& pool, int rollouts, uint64_t seed) {
  std::vector<Tile> tiles;
  for (int k = 0; k < TILE_KINDS; ++k)
    for (int c = 0; c < pool.count(Tile::of(k)); ++c) tiles.push_back(Tile::of(k));
  std::mt19937_64 rng(seed);
  std::vector<Rack> out(static_cast<size_t>(rollouts));
  for (int i = 0; i < rollouts; i += 2) {
    std::shuffle(tiles.begin(), tiles.end(), rng);
    const int n = std::min<int>(int(tiles.size()) - RACK_SIZE, 1 + int(rng() % 6));
    for (int j = 0; j < n; ++j) out[size_t(i)].add(tiles[size_t(j)]);
  }
  return out;
}

RolloutBatch candidate_batch(const DecisionPoint& at, const std::vector<Move>& moves, int rollouts,
                             uint64_t seed) {
  RolloutBatch batch;
  batch.groups.push_back(make_rollout_group(at.board, at.rack, seed));
  batch.groups[0].opp_leaves = known_leaves(batch.groups[0].pool, rollouts, seed);
  for (const Move& m : moves) {
    batch.starts.push_back(make_rollout_start(at.board, at.rack, at.my_score, at.opp_score, m, 0));
    batch.add_rollouts(uint32_t(batch.starts.size() - 1), 0, uint32_t(rollouts));
  }
  return batch;
}

bool same_move(const rollout_abi::PackedMove& a, const rollout_abi::PackedMove& b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

void print_mismatch(const RolloutBatch& batch, size_t k, const rollout_abi::RolloutResult& cpu,
                    const rollout_abi::RolloutResult& gpu) {
  const rollout_abi::RolloutTask& task = batch.tasks[k];
  const RolloutRacks racks = rollout_racks(batch, task);
  std::cout << std::format(
    "  candidate {} rollout {}: opp rack {} our rack {} | diff cpu {} gpu {}\n", task.position,
    task.rollout, racks.opp.to_string(), racks.mine.to_string(), cpu.diff, gpu.diff);
  Board board = batch.starts[task.position].board;
  for (int ply = 0; ply < 2; ++ply) {
    const Move mc = unpack_move(cpu.moves[ply]), mg = unpack_move(gpu.moves[ply]);
    std::cout << std::format("    ply {}: cpu [{}] gpu [{}]{}\n", ply, describe_move(mc, board),
                             describe_move(mg, board),
                             same_move(cpu.moves[ply], gpu.moves[ply]) ? "" : "  <-- differs");
    if (mc.type() == MoveType::PLAY) board.apply(mc);
  }
}

using LeafEncoder = LeafEvalInputEncoder;

// The CPU encoder's rows for `leaves`, kInputFloats each.
struct CpuRows {
  std::vector<float> rows;
  double seconds = 0;  // encoding alone, one thread
};

CpuRows encode_on_cpu(const std::vector<CpuLeaf>& leaves, const Dictionary& dict) {
  CpuRows out;
  out.rows.resize(leaves.size() * LeafEncoder::kInputFloats);
  const LeaveValues& lv = HastyEquity::instance().leave_values();
  for (size_t i = 0; i < leaves.size(); ++i) {
    const CpuLeaf& leaf = leaves[i];
    const LeafEvalInput in{&leaf.board, leaf.leaves[0], leaf.leaves[1],
                           leaf.scores[0] - leaf.scores[1]};
    const auto t0 = Clock::now();
    LeafEncoder::encode(in, dict, lv, &out.rows[i * LeafEncoder::kInputFloats]);
    out.seconds += seconds_since(t0);
  }
  return out;
}

// The device encoder's rows for leaves [0, n) of the last run, laid out as
// the CPU writes them.
std::vector<float> device_rows(GpuHastyRollouts& gpu, int n, int batch) {
  std::vector<float> spatial(size_t(n) * LeafEncoder::kSpatialFloats);
  std::vector<float> scalar(size_t(n) * LeafEncoder::kScalarFloats);
  for (int first = 0; first < n; first += batch) {
    gpu.encode_leaves_to_host(first, std::min(batch, n - first),
                              &spatial[size_t(first) * LeafEncoder::kSpatialFloats],
                              &scalar[size_t(first) * LeafEncoder::kScalarFloats]);
  }
  std::vector<float> rows(size_t(n) * LeafEncoder::kInputFloats);
  for (size_t i = 0; i < size_t(n); ++i) {
    float* row = &rows[i * LeafEncoder::kInputFloats];
    std::copy_n(&spatial[i * LeafEncoder::kSpatialFloats], LeafEncoder::kSpatialFloats, row);
    std::copy_n(&scalar[i * LeafEncoder::kScalarFloats], LeafEncoder::kScalarFloats,
                row + LeafEncoder::kSpatialFloats);
  }
  return rows;
}

std::string describe_element(size_t row, int k, float cpu, float gpu) {
  if (k < LeafEncoder::kSpatialFloats) {
    return std::format(" [row {} plane {} cell {}: cpu {} gpu {}]", row, k / kBoardCells,
                       k % kBoardCells, cpu, gpu);
  }
  return std::format(" [row {} scalar {}: cpu {} gpu {}]", row, k - LeafEncoder::kSpatialFloats,
                     cpu, gpu);
}

struct RowCheck {
  int checked = 0, rows_identical = 0;
  float max_abs_diff = 0;
  std::string first_diffs;  // up to 8 elements of the first differing row
};

RowCheck compare_rows(const std::vector<float>& cpu, const std::vector<float>& gpu) {
  RowCheck check;
  check.checked = int(cpu.size() / LeafEncoder::kInputFloats);
  for (size_t i = 0; i < size_t(check.checked); ++i) {
    const float* c = &cpu[i * LeafEncoder::kInputFloats];
    const float* g = &gpu[i * LeafEncoder::kInputFloats];
    const bool first_differing = check.first_diffs.empty();
    bool same = true;
    int shown = 0;
    for (int k = 0; k < LeafEncoder::kInputFloats; ++k) {
      const float d = std::fabs(c[k] - g[k]);
      if (d == 0) continue;
      same = false;
      check.max_abs_diff = std::max(check.max_abs_diff, d);
      if (first_differing && shown++ < 8) check.first_diffs += describe_element(i, k, c[k], g[k]);
    }
    check.rows_identical += same ? 1 : 0;
  }
  return check;
}

// The host route over `rows`: staged from host memory, predicted, timed.
// Returns each row's win value.
struct HostRoute {
  std::vector<double> win_values;
  double seconds = 0;  // predict() with the upload
};

HostRoute run_host_route(nn::NeuralNet<nn::LeafEvalSpec>& net, const std::vector<float>& rows) {
  HostRoute out;
  const int n = int(rows.size() / LeafEncoder::kInputFloats);
  for (int first = 0; first < n; first += net.max_rows()) {
    const int count = std::min(net.max_rows(), n - first);
    for (int i = 0; i < count; ++i) {
      nn::stage_board_row(net, &rows[size_t(first + i) * LeafEncoder::kInputFloats], i);
    }
    const auto t0 = Clock::now();
    net.predict(count);
    out.seconds += seconds_since(t0);
    const float* z = net.host<nn::WldOutput>();
    for (int i = 0; i < count; ++i) out.win_values.push_back(win_value(wld_from_logits(z + 3 * i)));
  }
  return out;
}

// The batch through GpuRolloutValues (the kernel, the device encoder and the
// leaf network), then the checks: the device rows of the first leaves
// against the CPU encoder's on the CPU reference's end states, and those
// leaves' win values against the host route's. Returns 1 unless the rows
// match and the values agree.
int evaluate_leaves(GpuRolloutValues& values, const RolloutBatch& batch, const Dictionary& dict,
                    const std::vector<CpuLeaf>& cpu_leaves, const std::string& onnx_path) {
  values.run(batch);  // the first run builds the TensorRT context's buffers
  const std::vector<RolloutOutcome> outcomes = values.run(batch);
  const GpuRolloutValues::Timings& tm = values.last_timings();
  const CpuRows cpu = encode_on_cpu(cpu_leaves, dict);
  const RowCheck check = compare_rows(
    cpu.rows, device_rows(values.rollouts(), int(cpu_leaves.size()), values.net().max_rows()));
  const HostRoute host = run_host_route(values.net(), cpu.rows);
  double max_value_diff = 0, mean_win = 0;
  for (size_t i = 0; i < outcomes.size(); ++i) mean_win += outcomes[i].win_value();
  for (size_t i = 0; i < cpu_leaves.size(); ++i) {
    if (outcomes[i].result.game_over) continue;
    max_value_diff =
      std::max(max_value_diff, std::fabs(outcomes[i].win_value() - host.win_values[i]));
  }
  const double n = double(outcomes.size());
  std::cout << std::format(
    "\nleaf model {}:\n  mean win value from the chooser's side: {:.4f}\n"
    "  rollouts {:.4f} s + leaves {:.4f} s -> {:.0f} valued rollouts/s (leaves alone {:.0f}/s)\n"
    "  device rows vs CPU encoder: {} of {} identical, max |diff| {}\n"
    "  win values vs the host route on those leaves: max |diff| {:.2g}\n",
    onnx_path, mean_win / n, tm.rollout_seconds, tm.leaf_seconds,
    n / (tm.rollout_seconds + tm.leaf_seconds), n / tm.leaf_seconds, check.rows_identical,
    check.checked, check.max_abs_diff, max_value_diff);
  std::cout << std::format(
    "  the host route: CPU encoder {:.0f} rows/s per thread, predict with upload {:.0f} evals/s\n",
    check.checked / cpu.seconds, check.checked / host.seconds);
  if (check.rows_identical != check.checked)
    std::cout << "  first differences:" << check.first_diffs << "\n";
  // FP16 runs of one engine at different batch sizes may pick different
  // kernels; beyond this the routes disagree on more than rounding.
  constexpr double kValueTolerance = 1e-2;
  return check.rows_identical == check.checked && max_value_diff <= kValueTolerance ? 0 : 1;
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
      print_mismatch(batch, k, cpu[k], gpu[k]);
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

  const Dictionary& dict = Lexicon::instance().dict();
  HastyEquity::ensure_initialized(Lexicon::instance().name());

  const DecisionPoint at = self_play_position(dict, seed, plies);
  at.board.ensure_movegen_caches(dict);
  const std::vector<Move> moves = candidates(at, dict, num_candidates);
  const RolloutBatch batch = candidate_batch(at, moves, rollouts, seed);
  std::cout << at.board.to_string();
  std::cout << std::format("to move: rack {} | scores {} to {} | unseen {}\n", at.rack.to_string(),
                           at.my_score, at.opp_score, batch.groups[0].pool.size());
  int blank_racks = 0;
  for (const rollout_abi::RolloutTask& task : batch.tasks) {
    const RolloutRacks r = rollout_racks(batch, task);
    blank_racks += (r.opp.blanks() > 0 ? 1 : 0) + (r.mine.blanks() > 0 ? 1 : 0);
  }
  std::cout << std::format(
    "{} candidates x {} rollouts (every other one seats a known opponent leave); {:.1f}% of the "
    "drawn racks hold a blank\n\n",
    moves.size(), rollouts, 100.0 * blank_racks / (2.0 * double(batch.tasks.size())));

  auto t0 = Clock::now();
  std::vector<CpuLeaf> cpu_leaves(
    leaf_model.empty() ? 0 : std::min(size_t(leaf_check), batch.tasks.size()));
  const std::vector<rollout_abi::RolloutResult> cpu =
    run_hasty_rollouts_cpu(batch, dict, threads, &cpu_leaves);
  const double cpu_seconds = seconds_since(t0);

  t0 = Clock::now();
  GpuHastyRollouts gpu(dict, HastyEquity::instance());
  const double gpu_setup = seconds_since(t0);
  // The first launch includes module load; time the second.
  gpu.run(batch, dict);
  const std::vector<rollout_abi::RolloutResult> res = gpu.run(batch, dict);
  const GpuHastyRollouts::Timings& tm = gpu.last_timings();
  const Agreement a = compare_results(batch, cpu, res);

  const double n = double(batch.tasks.size());
  std::cout << std::format(
    "\nagreement: {} of {} rollouts identical ({} outcome mismatches, {} move mismatches); "
    "{} games ended within the rollout; {} span-list overflows\n",
    int(n) - std::max(a.diff_mismatch, a.move_mismatch), int(n), a.diff_mismatch, a.move_mismatch,
    a.game_over, a.overflow);
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
    leaf_status = evaluate_leaves(values, batch, dict, cpu_leaves, leaf_model);
  }
  return (a.diff_mismatch || a.move_mismatch || leaf_status != 0) ? 1 : 0;
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
