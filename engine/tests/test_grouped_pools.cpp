// Grouped pools on the CPU reference with the synthetic leaf (the suites hide
// the GPU): a group's pools are Bob's full pools at each (move, rack), the
// same racks under every move; its maps are the leading replies' first
// rollouts, the pool's own deals, at each of the file's counts; and the
// .group sidecar round-trips with each pool's values cut to the maps'
// count. Skips without the NWL23 lexicon and HastyBot's leave values.

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "data/group_log.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "nwl23_equity.h"
#include "sim/grouped_pools.h"
#include "sim/pool_labeler.h"
#include "sim/rollout_values.h"
#include "sim/sim_maps.h"
#include "temp_dir.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

bool have_data() {
  if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) return false;
  return scribblez::testing::install_nwl23_hasty_equity();
}

const Dictionary& nwl23() {
  static const Dictionary dict = Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG);
  return dict;
}

const BeliefModel kNoModel;  // Bob draws Alice's whole rack from the unseen pool
constexpr int kMoves = 2;
constexpr GroupParams kParams{.quotas = {.top = kMoves, .mid = 0, .tail = 0, .exchange = 0},
                              .racks = 3,
                              .map_sims = 4,
                              .map_steps = 2,
                              .rollouts = 6,
                              .blank_prune_points = 40};

// A midgame parent, Alice to move, from a few plies of HastyBot self-play.
GroupParent parent_after(uint64_t seed, int plies) {
  HastyBot p0({.thread_id = 0, .name = "p0"});
  HastyBot p1({.thread_id = 0, .name = "p1"});
  Game game(p0, p1, nwl23(), seed);
  game.set_max_plies(plies);
  game.play();
  EXPECT_TRUE(game.truncated());
  const int mover = plies % 2;
  GroupParent out;
  out.at = {.game_idx = 0, .turn_idx = uint32_t(plies)};
  out.position.board = game.board();
  out.position.board.ensure_movegen_caches(nwl23());
  out.position.scores = {game.score(0), game.score(1)};
  out.position.mover = mover;
  out.position.rack = game.rack(mover);
  out.bag = out.position.board.pov_bag_size(out.position.rack.size());
  const MoveRequest req{
    out.position.board,    nwl23(), out.position.rack, Rack{}, game.score(mover),
    game.score(1 - mover), out.bag};
  out.moves = equity_top_k(req, kMoves);
  for (size_t i = 0; i < out.moves.size(); ++i) out.equity_ranks.push_back(int32_t(i));
  return out;
}

// Runs `jobs` through a labeler with `params`, calling back each record by
// its index.
void label(RolloutValues& values, const PoolLabeler::Params& params, std::vector<PoolJob> jobs,
           const std::function<void(size_t, PoolPosition&&)>& done) {
  PoolLabeler labeler(values, params);
  size_t handed = 0;
  size_t finished = 0;
  labeler.run(
    [&]() -> std::optional<PoolJob> {
      if (handed == jobs.size()) return std::nullopt;
      return std::move(jobs[handed++]);
    },
    [&](PoolPosition&& p) { done(finished++, std::move(p)); });
}

// A group built and labeled through both stages.
struct Built {
  std::unique_ptr<CpuRolloutValues> values;
  std::unique_ptr<GroupBuild> group;
};

Built build_group(uint64_t seed) {
  Built out;
  out.values = std::make_unique<CpuRolloutValues>(nwl23(), synthetic_leaf());
  out.group = std::make_unique<GroupBuild>(parent_after(seed, 6), kNoModel, kParams, nwl23(), seed);
  label(*out.values, {.rollouts = kParams.rollouts, .race = true}, out.group->pool_jobs(),
        [&](size_t i, PoolPosition&& p) { out.group->pool_labeled(i, std::move(p)); });
  label(*out.values, {.rollouts = kParams.map_sims, .race = false, .sum_maps = true},
        out.group->map_jobs(), [&](size_t i, PoolPosition&& p) { out.group->map_labeled(i, p); });
  return out;
}

}  // namespace

TEST(GroupedPools, PoolsAreBobsFullPoolsAtEachMoveAndRack) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Built b = build_group(/*seed=*/3);
  const GroupRecord& g = b.group->record();
  ASSERT_EQ(g.moves.size(), size_t(kMoves));
  ASSERT_EQ(g.equity_ranks, (std::vector<int32_t>{0, 1}));
  ASSERT_EQ(g.racks.size(), size_t(kParams.racks));
  ASSERT_EQ(g.pools.size(), g.moves.size() * g.racks.size());
  for (const Rack& r : g.racks) EXPECT_EQ(r.size(), RACK_SIZE);
  for (const PoolPosition& p : g.pools) {
    EXPECT_EQ(p.at.turn_idx, 6u);
    // Every legal move but the pruned blank moves, and the pass beyond them.
    EXPECT_GT(p.moves.size(), 1u);
    EXPECT_GE(p.num_legal_moves + 1, uint32_t(p.moves.size()));
    EXPECT_EQ(std::count_if(p.moves.begin(), p.moves.end(),
                            [](const Move& m) { return m.type() == MoveType::PASS; }),
              1);
    ASSERT_EQ(p.labels.size(), p.moves.size());
    double q = 0;
    for (const PoolLabel& l : p.labels) {
      q += l.q;
      EXPECT_GE(l.rollouts, 1);
      EXPECT_LE(l.rollouts, kParams.rollouts);
    }
    EXPECT_NEAR(q, 1.0, 1e-6);
    EXPECT_FALSE(p.opp_rack.modelled);
    EXPECT_TRUE(p.maps.empty());
  }
}

TEST(GroupedPools, MapsAreTheLeadingRepliesFirstRolloutsAtEachCount) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Built b = build_group(/*seed=*/3);
  const GroupRecord& g = b.group->record();
  ASSERT_FALSE(g.maps.empty());
  EXPECT_EQ(g.maps.size() % size_t(kParams.map_steps), 0u);
  std::set<std::pair<int, int>> pools_with_maps;
  for (const GroupMap& m : g.maps) {
    EXPECT_TRUE(m.sims == 2 || m.sims == 4) << m.sims;
    EXPECT_EQ(m.maps.sims, float(m.sims));
    const PoolPosition& p = g.pools[size_t(m.move * kParams.racks + m.rack)];
    ASSERT_LT(size_t(m.reply), p.moves.size());
    // The leading replies ran every rollout: the race never stopped them.
    EXPECT_EQ(p.labels[size_t(m.reply)].rollouts, kParams.rollouts);
    pools_with_maps.insert({m.move, m.rack});
  }
  EXPECT_EQ(pools_with_maps.size(), g.pools.size());
  // A map is the pool's own first rollouts of the reply: the same group and
  // start, the same tasks in the same order, give the same sums.
  const GroupMap& m = g.maps.front();
  const PoolPosition& p = g.pools[size_t(m.move * kParams.racks + m.rack)];
  GroupBuild again(parent_after(/*seed=*/3, 6), kNoModel, kParams, nwl23(), /*seed=*/3);
  std::vector<PoolJob> jobs = again.pool_jobs();
  PoolJob& job = jobs[size_t(m.move * kParams.racks + m.rack)];
  ASSERT_EQ(job.record.moves, p.moves);
  RolloutBatch batch{.groups = {job.group}, .starts = {job.starts[size_t(m.reply)]}, .tasks = {}};
  batch.starts[0].group = 0;
  batch.add_rollouts(0, 0, uint32_t(m.sims));
  batch.sum_maps = true;
  const BatchValues values = b.values->run(batch);
  ASSERT_EQ(values.maps.size(), 1u);
  EXPECT_EQ(sim_maps_difference(values.maps[0], m.maps), "");
}

TEST(GroupedPools, TheSidecarRoundTripsWithValuesCutToTheMapsCount) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Built b = build_group(/*seed=*/5);
  const GroupRecord& g = b.group->record();
  const std::filesystem::path dir = scribblez::testing::make_temp_dir("grouped_pools");
  const std::string path = (dir / "chunk.group").string();
  PoolSources sources;
  sources.leaf_model.tag = "synthetic";
  sources.rollouts = 6;
  {
    GroupWriter writer(path, sources, kParams.map_sims, kParams.map_steps);
    writer.add_group(g);
    writer.close();
  }
  const GroupReader reader(path);
  const GroupFileHeader& h = reader.header();
  EXPECT_EQ(h.version, kGroupFormatVersion);
  EXPECT_EQ(h.flags, 0);
  EXPECT_EQ(h.rollouts, 6u);
  EXPECT_EQ(h.map_sims, 4u);
  EXPECT_EQ(h.map_steps, 2u);
  EXPECT_EQ(PoolSources::of(h.leaf_model, h.leave_model, int(h.rollouts)), sources);
  ASSERT_EQ(reader.num_groups(), 1);
  const GroupReader::Group& r = reader.group(0);
  EXPECT_EQ(r.header->turn_index, 6u);
  ASSERT_EQ(r.header->num_moves, g.moves.size());
  for (size_t a = 0; a < g.moves.size(); ++a) {
    EXPECT_EQ(r.moves[a], g.moves[a]);
    EXPECT_EQ(r.equity_ranks[a], g.equity_ranks[a]);
  }
  ASSERT_EQ(r.header->num_racks, g.racks.size());
  for (size_t i = 0; i < g.racks.size(); ++i) {
    for (int t = 0; t < TILE_KINDS; ++t)
      EXPECT_EQ(int(r.racks[i].counts[size_t(t)]), g.racks[i].count(Tile::of(t)));
  }
  ASSERT_EQ(r.pools.size(), g.pools.size());
  for (size_t i = 0; i < g.pools.size(); ++i) {
    const PoolPosition& p = g.pools[i];
    const PoolPositionView& v = r.pools[i];
    ASSERT_EQ(v.header->num_candidates, p.moves.size());
    const float* values = v.rollout_values;
    const float* want = p.rollout_values.data();
    for (size_t c = 0; c < p.moves.size(); ++c) {
      EXPECT_EQ(v.candidates[c].move, p.moves[c]);
      EXPECT_EQ(v.candidates[c].rollouts, p.labels[c].rollouts);
      const int kept = std::min(p.labels[c].rollouts, kParams.map_sims);
      for (int k = 0; k < kept; ++k) EXPECT_EQ(values[k], want[k]);
      values += kept;
      want += p.labels[c].rollouts;
    }
  }
  ASSERT_EQ(r.maps.size(), g.maps.size());
  for (size_t m = 0; m < g.maps.size(); ++m) {
    EXPECT_EQ(r.maps[m]->move, g.maps[m].move);
    EXPECT_EQ(r.maps[m]->rack, g.maps[m].rack);
    EXPECT_EQ(int(r.maps[m]->reply), g.maps[m].reply);
    EXPECT_EQ(int(r.maps[m]->sims), g.maps[m].sims);
    EXPECT_EQ(sim_maps_difference(GroupReader::maps_of(r.maps[m]), g.maps[m].maps), "");
  }
  std::filesystem::remove_all(dir);
}
