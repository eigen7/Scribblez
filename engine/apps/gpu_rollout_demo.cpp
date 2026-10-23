// Demo of the hasty-rollout task on the GPU against the CPU reference:
// reaches a position by HastyBot self-play, runs N rollouts on both, checks
// them rollout by rollout (same draws, so same moves and outcomes), and
// prints the throughput of each. docs/plans/gpu_hasty_rollouts.md.

#include "agent/hasty_bot.h"
#include "encoding/leaf_eval_input_encoder.h"
#include "game/board.h"
#include "game/game.h"
#include "game/game_log.h"
#include "game/move.h"
#include "gpu/hasty_rollouts_gpu.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/onnx_metadata.h"
#include "nn/trt_util.h"
#include "sim/hasty_rollout_task.h"
#include "util/io.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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

// The leaf eval model over every rollout's leaf: rows built on the device
// straight into the net's input buffers, batches of the net's max rows. The
// first `check` leaves are also encoded on the CPU (LeafEvalInputEncoder on
// the CPU reference's end states), compared element by element, and run
// through the host route for comparison.
using LeafNet = nn::NeuralNet<nn::LeafEvalSpec>;
using LeafEncoder = LeafEvalInputEncoder;

// P(win) + P(draw) / 2 from a row's win/draw/loss logits.
double win_probability(const float* z) {
  const float m = std::max({z[0], z[1], z[2]});
  const double e0 = std::exp(z[0] - m), e1 = std::exp(z[1] - m), e2 = std::exp(z[2] - m);
  return (e0 + 0.5 * e1) / (e0 + e1 + e2);
}

struct DeviceRoute {
  double mean_win = 0;  // from our side, over unfinished rollouts
  int evaluated = 0;
  double encode_seconds = 0, predict_seconds = 0;
};

DeviceRoute run_device_route(LeafNet& net, GpuHastyRollouts& gpu,
                             const std::vector<rollout_abi::RolloutResult>& results) {
  DeviceRoute route;
  auto* d_spatial = static_cast<float*>(net.device_ptr(nn::SpatialInput::kName));
  auto* d_scalar = static_cast<float*>(net.device_ptr(nn::ScalarInput::kName));
  const int n = int(results.size());
  for (int first = 0; first < n; first += net.max_rows()) {
    const int count = std::min(net.max_rows(), n - first);
    auto t0 = Clock::now();
    gpu.encode_leaves(first, count, d_spatial, d_scalar);
    route.encode_seconds += seconds_since(t0);
    t0 = Clock::now();
    net.predict_device_inputs(count);
    route.predict_seconds += seconds_since(t0);
    const float* wld = net.host<nn::WldOutput>();
    for (int i = 0; i < count; ++i) {
      if (results[size_t(first + i)].game_over) continue;
      route.mean_win += win_probability(wld + 3 * i);
      ++route.evaluated;
    }
  }
  if (route.evaluated > 0) route.mean_win /= route.evaluated;
  return route;
}

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

// The device encoder's rows for leaves [0, n), laid out as the CPU writes them.
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

// predict() over `rows` uploaded from host memory: the host route's
// inference, timed.
double time_host_route(LeafNet& net, const std::vector<float>& rows) {
  const int n = int(rows.size() / LeafEncoder::kInputFloats);
  double seconds = 0;
  for (int first = 0; first < n; first += net.max_rows()) {
    const int count = std::min(net.max_rows(), n - first);
    for (int i = 0; i < count; ++i) {
      nn::stage_board_row(net, &rows[size_t(first + i) * LeafEncoder::kInputFloats], i);
    }
    const auto t0 = Clock::now();
    net.predict(count);
    seconds += seconds_since(t0);
  }
  return seconds;
}

// Every leaf through the model on the device, then the CPU checks; the
// return code is 1 unless the device rows match the CPU encoder's.
int evaluate_leaves(GpuHastyRollouts& gpu, const std::string& onnx_path, nn::Precision precision,
                    const Dictionary& dict, double kernel_seconds,
                    const std::vector<rollout_abi::RolloutResult>& results,
                    const std::vector<CpuLeaf>& cpu_leaves) {
  nn::check_leave_values(nn::parse_onnx_metadata(util::read_file_bytes(onnx_path)), onnx_path,
                         HastyEquity::instance().leave_values().content_hash());
  nn::NeuralNetParams<nn::LeafEvalSpec> params;
  params.onnx_path = onnx_path;
  params.precision = precision;
  LeafNet net(params);
  net.load();
  const DeviceRoute dev = run_device_route(net, gpu, results);
  const CpuRows cpu = encode_on_cpu(cpu_leaves, dict);
  const RowCheck check =
    compare_rows(cpu.rows, device_rows(gpu, int(cpu_leaves.size()), net.max_rows()));
  const double host_predict_seconds = time_host_route(net, cpu.rows);

  const double n = double(results.size());
  const double total = dev.encode_seconds + dev.predict_seconds;
  std::cout << std::format(
    "\nleaf model {}:\n  mean P(win) from our side over {} unfinished rollouts: {:.4f}\n"
    "  encode {:.4f} s + predict {:.4f} s = {:.4f} s -> {:.0f} leaf evals/s (kernel + leaf: "
    "{:.0f} rollouts/s)\n  device rows vs CPU encoder: {} of {} identical, max |diff| {}\n",
    onnx_path, dev.evaluated, dev.mean_win, dev.encode_seconds, dev.predict_seconds, total,
    n / total, n / (kernel_seconds + total), check.rows_identical, check.checked,
    check.max_abs_diff);
  std::cout << std::format(
    "  the host route on those {} leaves: CPU encoder {:.0f} rows/s per thread, predict with "
    "upload {:.0f} evals/s (device route: encoder {:.0f} rows/s, predict {:.0f} evals/s)\n",
    check.checked, check.checked / cpu.seconds, check.checked / host_predict_seconds,
    n / dev.encode_seconds, n / dev.predict_seconds);
  if (check.rows_identical == check.checked) return 0;
  std::cout << "  first differences:" << check.first_diffs << "\n";
  return 1;
}

int run(int argc, char** argv) {
  namespace po = boost::program_options;
  uint64_t seed = 1;
  int plies = 10, rollouts = 65536, threads = int(std::thread::hardware_concurrency());
  int leaf_check = 2048;
  std::string leaf_model, leaf_precision = "fp16";
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
  desc.add_options()("leaf-model", po::value<std::string>(&leaf_model),
                     "a deep_bot_leaf_eval ONNX export to evaluate every leaf with, on the GPU");
  desc.add_options()("leaf-check", po::value<int>(&leaf_check)->default_value(leaf_check),
                     "leaves whose device-built rows are checked against the CPU encoder");
  desc.add_options()("leaf-precision",
                     po::value<std::string>(&leaf_precision)->default_value(leaf_precision),
                     "TensorRT precision for the leaf model: fp32, fp16 or bf16");
  Lexicon::instance().add_options(desc);
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }
  po::notify(vm);
  if (leaf_check < 1) throw po::error("--leaf-check must be at least 1");

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
  std::vector<CpuLeaf> cpu_leaves(leaf_model.empty() ? 0 : size_t(std::min(leaf_check, rollouts)));
  const std::vector<rollout_abi::RolloutResult> cpu =
    run_hasty_rollouts_cpu(pos, dict, rollouts, threads, &cpu_leaves);
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
  const int leaf_status = leaf_model.empty()
                            ? 0
                            : evaluate_leaves(gpu, leaf_model, nn::parse_precision(leaf_precision),
                                              dict, tm.kernel_seconds, res, cpu_leaves);
  return (diff_mismatch || move_mismatch || leaf_status != 0) ? 1 : 0;
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
