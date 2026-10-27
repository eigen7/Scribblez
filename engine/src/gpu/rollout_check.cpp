#include "gpu/rollout_check.h"

#include "agent/hasty_bot.h"
#include "encoding/board_planes.h"
#include "encoding/leaf_eval_input_encoder.h"
#include "game/game.h"
#include "game/game_log.h"
#include "gpu/hasty_rollouts_gpu.h"
#include "lexicon/hasty_equity.h"
#include "sim/sim_runner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace scribblez {

namespace {

using LeafEncoder = LeafEvalInputEncoder;

bool is_exchange(const Move& m) { return m.type() == MoveType::EXCHANGE; }

std::string describe_element(size_t row, int k, float cpu, float gpu) {
  if (k < LeafEncoder::kSpatialFloats) {
    return std::format(" [row {} plane {} cell {}: cpu {} gpu {}]", row, k / kBoardCells,
                       k % kBoardCells, cpu, gpu);
  }
  return std::format(" [row {} scalar {}: cpu {} gpu {}]", row, k - LeafEncoder::kSpatialFloats,
                     cpu, gpu);
}

}  // namespace

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
  DecisionPoint at{game.board(), game.rack(mover), game.score(mover), game.score(1 - mover)};
  at.board.ensure_movegen_caches(dict);
  return at;
}

std::vector<Move> rollout_candidates(const DecisionPoint& at, const Dictionary& dict, int count) {
  const MoveRequest req{at.board,
                        dict,
                        at.rack,
                        Rack{},
                        at.my_score,
                        at.opp_score,
                        at.board.pov_bag_size(at.rack.size())};
  const std::vector<Move> ranked = equity_top_k(req, std::numeric_limits<int>::max());
  std::vector<Move> out(ranked.begin(), ranked.begin() + std::min<size_t>(ranked.size(), count));
  if (std::none_of(out.begin(), out.end(), is_exchange)) {
    const auto it = std::find_if(ranked.begin(), ranked.end(), is_exchange);
    if (it != ranked.end()) out.push_back(*it);
  }
  return out;
}

std::vector<Rack> known_leaves(const TileCounts& pool, int rollouts, uint64_t seed) {
  std::vector<Tile> tiles;
  for (int k = 0; k < TILE_KINDS; ++k)
    for (int c = 0; c < pool.count(Tile::of(k)); ++c) tiles.push_back(Tile::of(k));
  std::mt19937_64 rng(seed);
  std::vector<Rack> out(static_cast<size_t>(rollouts));
  for (int i = 0; i < rollouts; ++i) {
    std::shuffle(tiles.begin(), tiles.end(), rng);
    const int n = std::min<int>(int(tiles.size()) - RACK_SIZE, i % (RACK_SIZE + 1));
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

std::string describe_mismatch(const RolloutBatch& batch, size_t k,
                              const rollout_abi::RolloutResult& cpu,
                              const rollout_abi::RolloutResult& gpu) {
  const rollout_abi::RolloutTask& task = batch.tasks[k];
  const RolloutRacks racks = rollout_racks(batch, task);
  std::string out = std::format(
    "  candidate {} rollout {}: opp rack {} our rack {} | diff cpu {} gpu {}\n", task.position,
    task.rollout, racks.opp.to_string(), racks.mine.to_string(), cpu.diff, gpu.diff);
  Board board = batch.starts[task.position].board;
  for (int ply = 0; ply < 2; ++ply) {
    const Move mc = unpack_move(cpu.moves[ply]), mg = unpack_move(gpu.moves[ply]);
    out += std::format("    ply {}: cpu [{}] gpu [{}]{}\n", ply, describe_move(mc, board),
                       describe_move(mg, board),
                       same_move(cpu.moves[ply], gpu.moves[ply]) ? "" : "  <-- differs");
    if (mc.type() == MoveType::PLAY) board.apply(mc);
  }
  return out;
}

std::vector<float> cpu_leaf_rows(const std::vector<CpuLeaf>& leaves, const Dictionary& dict) {
  std::vector<float> rows(leaves.size() * LeafEncoder::kInputFloats);
  const LeaveValues& lv = HastyEquity::instance().leave_values();
  for (size_t i = 0; i < leaves.size(); ++i) {
    const CpuLeaf& leaf = leaves[i];
    const LeafEvalInput in{&leaf.board, leaf.leaves[0], leaf.leaves[1],
                           leaf.scores[0] - leaf.scores[1]};
    LeafEncoder::encode(in, dict, lv, &rows[i * LeafEncoder::kInputFloats]);
  }
  return rows;
}

std::vector<float> device_leaf_rows(GpuHastyRollouts& gpu, int n, int batch) {
  std::vector<float> spatial(size_t(n) * LeafEncoder::kSpatialFloats);
  std::vector<float> scalar(size_t(n) * LeafEncoder::kScalarFloats);
  std::vector<uint32_t> tasks(static_cast<size_t>(n));
  std::iota(tasks.begin(), tasks.end(), 0u);
  for (int first = 0; first < n; first += batch) {
    gpu.encode_leaves_to_host(&tasks[size_t(first)], std::min(batch, n - first),
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

RowCheck compare_leaf_rows(const std::vector<float>& cpu, const std::vector<float>& gpu) {
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

}  // namespace scribblez
