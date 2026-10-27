// The GPU rollout kernel on the GPU, against its CPU reference rollout for
// rollout (gpu/rollout_check.h): the same draws must give the same moves and
// outcomes, and the leaf rows the device encodes must equal the CPU encoder's,
// over positions from the opening to the game's end, every candidate kind and
// known opponent leaves of every length up to a full rack. Skips without the
// NWL23 lexicon and HastyBot's leave values.

#include "gpu/hasty_rollouts_gpu.h"
#include "gpu/rollout_check.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/hasty_rollout_task.h"

#include <gtest/gtest.h>

#include <format>
#include <fstream>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

constexpr uint64_t kSeed = 1;
constexpr int kCandidates = 6;                  // plus the best exchange
constexpr int kRollouts = 4 * (RACK_SIZE + 1);  // each leave length four times

bool have_data() {
  if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) return false;
  if (!std::ifstream(HastyEquity::default_leaves_path("NWL23")).good()) return false;
  HastyEquity::ensure_initialized("NWL23");
  return true;
}

const Dictionary& nwl23() {
  static const Dictionary dict = Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG);
  return dict;
}

// The kernel with its tables on the device, uploaded once for the binary
// (~2 s with the CUDA context, most of this binary's time).
GpuHastyRollouts& gpu() {
  static GpuHastyRollouts g(nwl23(), HastyEquity::instance());
  return g;
}

RolloutBatch batch_at(int plies) {
  const DecisionPoint at = self_play_position(nwl23(), kSeed, plies);
  return candidate_batch(at, rollout_candidates(at, nwl23(), kCandidates), kRollouts, kSeed);
}

constexpr int kMismatchesShown = 4;

// How many tasks' GPU results differ from the CPU reference's, the first few
// described; empty when none do.
std::string mismatches(const RolloutBatch& batch,
                       const std::vector<rollout_abi::RolloutResult>& cpu,
                       const std::vector<rollout_abi::RolloutResult>& gpu) {
  std::string shown;
  int count = 0;
  for (size_t k = 0; k < cpu.size(); ++k) {
    const bool same = cpu[k].diff == gpu[k].diff && same_move(cpu[k].moves[0], gpu[k].moves[0]) &&
                      same_move(cpu[k].moves[1], gpu[k].moves[1]);
    if (same) continue;
    if (count++ < kMismatchesShown) shown += describe_mismatch(batch, k, cpu[k], gpu[k]);
  }
  if (count == 0) return "";
  return std::format("{} of {} rollouts differ, first:\n{}", count, cpu.size(), shown);
}

class GpuRolloutKernelTest : public ::testing::TestWithParam<int> {};

}  // namespace

// The opening, the middle game, and a late position where many rollouts end
// the game.
TEST_P(GpuRolloutKernelTest, RolloutsMatchTheCpuReference) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const RolloutBatch batch = batch_at(GetParam());
  const auto cpu = run_hasty_rollouts_cpu(batch, nwl23());
  const auto res = gpu().run(batch, nwl23());
  ASSERT_EQ(res.size(), cpu.size());
  EXPECT_EQ(mismatches(batch, cpu, res), "");
  for (const rollout_abi::RolloutResult& r : res) EXPECT_EQ(r.pad, 0) << "span-list overflow";
}

INSTANTIATE_TEST_SUITE_P(Plies, GpuRolloutKernelTest, ::testing::Values(2, 10, 16));

// Every rollout seating a full rack as the opponent's known leave: its leave
// after a pass, as Alice's after-reply sims seat it (agent/bob_subtree.h).
TEST(GpuRolloutKernel, SeatsAFullRackAsTheOpponentsLeave) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  RolloutBatch batch = batch_at(10);
  std::vector<Rack>& leaves = batch.groups[0].opp_leaves;
  std::vector<Rack> full;
  for (const Rack& leave : leaves)
    if (leave.size() == RACK_SIZE) full.push_back(leave);
  ASSERT_FALSE(full.empty());
  for (size_t i = 0; i < leaves.size(); ++i) leaves[i] = full[i % full.size()];
  const auto cpu = run_hasty_rollouts_cpu(batch, nwl23());
  EXPECT_EQ(mismatches(batch, cpu, gpu().run(batch, nwl23())), "");
}

// The opponent's exchanged tiles going back after its refill, as in Alice's
// sims after Bob's reply when her candidate was an exchange: three tiles out
// of the pool, returned by every start.
TEST(GpuRolloutKernel, PutsTheOpponentsExchangeBackAfterItsRefill) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  RolloutBatch batch = batch_at(10);
  RolloutGroup& group = batch.groups[0];
  group.opp_leaves.clear();
  Rack returned;
  for (int k = 0; k < TILE_KINDS && returned.size() < 3; ++k)
    if (group.pool.count(Tile::of(k)) > 0) returned.add(Tile::of(k));
  ASSERT_TRUE(group.pool.remove(returned.counts()));
  for (RolloutStart& start : batch.starts) start.opp_returned = returned;
  const auto cpu = run_hasty_rollouts_cpu(batch, nwl23());
  EXPECT_EQ(mismatches(batch, cpu, gpu().run(batch, nwl23())), "");
}

TEST(GpuRolloutKernel, DeviceLeafRowsMatchTheCpuEncoder) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const RolloutBatch batch = batch_at(16);
  std::vector<CpuLeaf> leaves(batch.tasks.size());
  run_hasty_rollouts_cpu(batch, nwl23(), &leaves);
  gpu().run(batch, nwl23());
  const RowCheck check = compare_leaf_rows(
    cpu_leaf_rows(leaves, nwl23()), device_leaf_rows(gpu(), int(leaves.size()), /*batch=*/64));
  EXPECT_EQ(check.rows_identical, check.checked) << check.first_diffs;
}
