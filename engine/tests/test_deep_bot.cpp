// DeepBot under both reply models, on the CPU reference with the synthetic
// leaf (the suites hide the GPU): it plays whole games, deterministically for a
// seed; asks the leave inference model for Alice's belief from her own view
// and, in Bob's subtrees, for Bob's with her private past hidden; Bob replies
// from his own equity top N; and --subtree-dump writes a line per turn. Skips
// without the NWL23 lexicon and HastyBot's leave values.

#include "agent/alice_subtree.h"
#include "agent/bob_subtree.h"
#include "agent/deep_bot.h"
#include "agent/hasty_bot.h"
#include "belief/leave_params.h"
#include "encoding/leave_inference_input_encoder.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "sim/pool_jobs.h"
#include "sim/rollout_values.h"
#include "temp_dir.h"
#include "util/exception.h"

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

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

DeepBot::Params small_params(uint64_t seed, const std::string& reply = "subtree") {
  DeepBot::Params p{.thread_id = 0,
                    .name = "DeepBot",
                    .leaf_model = "synthetic",
                    .candidates = 2,
                    .late_candidates = 3,
                    .reply = reply,
                    .rollouts = 8,
                    .bob_racks = 3,
                    .sims_per_rack = 2,
                    .seed = seed};
  p.subtree = {.opp_racks = 3, .candidates = 2, .late_candidates = 2};
  p.second = {.opp_racks = 3, .candidates = 2, .late_candidates = 2};
  return p;
}

using Encoder = LeaveInferenceInputEncoder;
constexpr size_t kHiddenFlag = Encoder::kSpatialFloats + Encoder::kCurrentScalars - 1;

DeepBot::Backend cpu_backend(belief::LeaveParamsFn predict = {},
                             std::shared_ptr<const LeaveValueArrays> table = {}) {
  return {std::make_unique<CpuRolloutValues>(nwl23(), synthetic_leaf()), std::move(predict),
          std::move(table)};
}

// A game of `deepbot` against HastyBot from a fixed deal; returns its moves.
std::vector<Move> play(DeepBot& deepbot) {
  HastyBot hasty({.thread_id = 0, .name = "HastyBot"});
  Game g(deepbot, hasty, nwl23(), /*seed=*/1);
  g.play();
  const GameLogStorage log = g.extract_log();
  std::vector<Move> moves;
  for (const TurnRecord& t : log.turns) moves.push_back(t.move);
  return moves;
}

}  // namespace

TEST(DeepBot, PlaysWholeGamesTheSameForASeed) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  DeepBot a(small_params(3), cpu_backend());
  DeepBot b(small_params(3), cpu_backend());
  const std::vector<Move> first = play(a);
  EXPECT_GT(first.size(), 10u);
  EXPECT_EQ(play(b), first);
  // The empty bag's turns fall back to HastyBot's move: the game ends.
}

TEST(DeepBot, AsksTheModelForAlicesBeliefAtEachModelledTurn) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto table = std::make_shared<const LeaveValueArrays>(
    LeaveValues::load(HastyEquity::default_leaves_path("NWL23")));
  int calls = 0;
  belief::LeaveParamsFn predict = [&](const float* rows, int count, belief::LeaveParams* out) {
    ++calls;
    ASSERT_EQ(count, 1);
    // Alice's own view of her position: her private history is not hidden.
    EXPECT_EQ(rows[kHiddenFlag], 0.0f);
    out[0] = belief::LeaveParams{};
  };
  DeepBot bot(small_params(4, "hasty"), cpu_backend(predict, table));
  play(bot);
  EXPECT_GE(calls, 5);  // every turn but the first, a bingo's or a pass's, and the endgame
}

// In Bob's subtrees the model is asked once per of Alice's candidates, for
// every rack she sampled for Bob at once, and each row hides Alice's private
// past: it is Alice's model of Bob's view.
TEST(DeepBot, AsksTheModelForBobsBeliefWithAlicesPastHidden) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto table = std::make_shared<const LeaveValueArrays>(
    LeaveValues::load(HastyEquity::default_leaves_path("NWL23")));
  int bob_rows = 0;
  belief::LeaveParamsFn predict = [&](const float* rows, int count, belief::LeaveParams* out) {
    for (int k = 0; k < count; ++k) {
      bob_rows += rows[size_t(k) * Encoder::kInputFloats + kHiddenFlag] == 1.0f;
      out[k] = belief::LeaveParams{};
    }
  };
  DeepBot bot(small_params(4), cpu_backend(predict, table));
  play(bot);
  EXPECT_GT(bob_rows, 10);
}

// Bob's reply at each rack is one of his own static-equity top N, chosen by
// his sims; the dump reports it per turn as one JSON line, naming its game.
// With --subtree-candidates 0 every candidate that leaves tiles in the bag is
// searched.
TEST(DeepBot, TheSubtreeDumpWritesALinePerSearchedTurn) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const auto dir = scribblez::testing::make_temp_dir("scribblez_test_subtree_dump");
  DeepBot::Params p = small_params(7);
  p.subtree_candidates = 0;
  p.subtree_dump = (dir / "dump.jsonl").string();
  DeepBot bot(p, cpu_backend());
  play(bot);
  std::ifstream in(p.subtree_dump);
  int lines = 0;
  int emptying = 0;
  for (std::string line; std::getline(in, line); ++lines) {
    const boost::json::object turn = boost::json::parse(line).as_object();
    EXPECT_EQ(turn.at("agent").as_string(), "DeepBot");
    EXPECT_EQ(turn.at("game").as_int64(), 1);
    for (const boost::json::value& v : turn.at("candidates").as_array()) {
      const boost::json::object& c = v.as_object();
      if (!c.at("searched").as_bool()) {
        // Only a candidate that empties the bag goes unsearched: it keeps its
        // hasty value.
        EXPECT_EQ(c.at("subtree_value").as_double(), c.at("hasty_value").as_double());
        ++emptying;
        continue;
      }
      const double value = c.at("subtree_value").as_double();
      EXPECT_TRUE(value >= 0.0 && value <= 1.0);
      const boost::json::object& r = c.at("replies").as_object();
      const boost::json::array& ranks = r.at("rank_hist").as_array();
      EXPECT_LE(ranks.size(), size_t(p.subtree.late_candidates));
      int64_t racks = 0;
      for (const boost::json::value& n : ranks) racks += n.as_int64();
      EXPECT_EQ(racks, int64_t(p.bob_racks));
    }
  }
  EXPECT_GT(lines, 5);
  EXPECT_GT(emptying, 0);  // this seed's game reaches one
  std::filesystem::remove_all(dir);
}

// With --subtree-candidates 1, a first pass of hasty sims leaves one
// candidate for Bob's subtree, the best by its hasty value of those that do
// not empty the bag; those others have no subtree value, one that empties the
// bag keeps its hasty value, and the pick is the searched one or one that
// empties the bag.
TEST(DeepBot, OnlyTheFirstPassesBestGetBobsSubtree) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const auto dir = scribblez::testing::make_temp_dir("scribblez_test_subtree_halving");
  DeepBot::Params p = small_params(8);
  p.candidates = 3;
  p.subtree_candidates = 1;
  p.subtree_dump = (dir / "dump.jsonl").string();
  DeepBot bot(p, cpu_backend());
  play(bot);
  std::ifstream in(p.subtree_dump);
  int turns = 0;
  for (std::string line; std::getline(in, line); ++turns) {
    const boost::json::object turn = boost::json::parse(line).as_object();
    const boost::json::array& candidates = turn.at("candidates").as_array();
    int searched = 0;
    double searched_hasty = 0;
    double best_pruned_hasty = 0;
    for (const boost::json::value& v : candidates) {
      const boost::json::object& c = v.as_object();
      const double hasty = c.at("hasty_value").as_double();
      if (c.at("searched").as_bool()) {
        ++searched;
        searched_hasty = hasty;
      } else if (c.at("subtree_value").is_null()) {
        best_pruned_hasty = std::max(best_pruned_hasty, hasty);
      } else {
        EXPECT_EQ(c.at("subtree_value").as_double(), hasty);
      }
    }
    EXPECT_LE(searched, 1);
    if (searched == 1) {
      EXPECT_GE(searched_hasty, best_pruned_hasty);
    }
    const auto pick = size_t(turn.at("subtree_pick").as_int64());
    EXPECT_FALSE(candidates[pick].as_object().at("subtree_value").is_null());
  }
  EXPECT_GT(turns, 5);
  std::filesystem::remove_all(dir);
}

TEST(DeepBot, RefusesAnUnknownReplyModelAndADumpWithoutSubtrees) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  EXPECT_THROW(DeepBot(small_params(1, "bogus"), cpu_backend()), util::CleanException);
  DeepBot::Params p = small_params(1, "hasty");
  p.subtree_dump = "dump.jsonl";
  EXPECT_THROW(DeepBot(p, cpu_backend()), util::CleanException);
  p = small_params(1, "alice-subtree");
  p.subtree_dump = "dump.jsonl";
  EXPECT_THROW(DeepBot(p, cpu_backend()), util::CleanException);
  p = small_params(1, "alice-subtree");
  p.second.candidates = 0;
  EXPECT_THROW(DeepBot(p, cpu_backend()), util::CleanException);
}

TEST(DeepBot, PlaysWholeGamesTheSameForASeedUnderAliceSubtree) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  DeepBot a(small_params(3, "alice-subtree"), cpu_backend());
  DeepBot b(small_params(3, "alice-subtree"), cpu_backend());
  const std::vector<Move> first = play(a);
  EXPECT_GT(first.size(), 10u);
  EXPECT_EQ(play(b), first);
}

// At her second turns Alice asks the model about Bob's leave after his reply,
// a row per rack where the bag hides it, each with Bob's private past hidden:
// her view of a move of his she did not see him choose.
TEST(DeepBot, AsksTheModelForAlicesSecondTurnBeliefWithBobsPastHidden) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto table = std::make_shared<const LeaveValueArrays>(
    LeaveValues::load(HastyEquity::default_leaves_path("NWL23")));
  int hidden_rows = 0;
  belief::LeaveParamsFn predict = [&](const float* rows, int count, belief::LeaveParams* out) {
    for (int k = 0; k < count; ++k) {
      hidden_rows += rows[size_t(k) * Encoder::kInputFloats + kHiddenFlag] == 1.0f;
      out[k] = belief::LeaveParams{};
    }
  };
  DeepBot bot(small_params(4, "alice-subtree"), cpu_backend(predict, table));
  play(bot);
  EXPECT_GT(hidden_rows, 10);
}

namespace {

// The racks rollout k of `job` deals: Alice's, then Bob's.
RolloutRacks after_reply_racks(const PoolJob& job, uint32_t k) {
  const RolloutBatch batch{.groups = {job.group}, .starts = job.starts, .tasks = {}};
  return rollout_racks(batch, {.position = 0, .rollout = k});
}

// `rack` less `part`, which it must hold.
Rack without(Rack rack, const Rack& part) {
  for (int i = 0; i < part.size(); ++i) EXPECT_TRUE(rack.remove(part.tiles()[size_t(i)]));
  return rack;
}

bool holds(const Rack& rack, const Rack& part) {
  for (int k = 0; k < TILE_KINDS; ++k)
    if (rack.count(Tile::of(k)) < part.count(Tile::of(k))) return false;
  return true;
}

}  // namespace

// Alice's replenishment in her sims after Bob's reply comes from the bag as
// she found it: an exchange's tiles went back only after she drew. Here she
// exchanges the only J, K, Q, X and Z, so no rack of hers may hold one.
TEST(DeepBot, AliceNeverRedrawsTheTilesSheExchanged) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  Board board;
  board.ensure_movegen_caches(nwl23());
  const Rack alice = Rack::from_string("JKQXZAE");
  const RolloutStart exchanged = make_rollout_start(
    board, alice, 0, 0, Move::exchange(Rack::from_string("JKQXZ").counts()), /*group=*/0);
  const Rack bob = Rack::from_string("BCDFGHI");
  const PoolJob job = after_reply_job(exchanged, bob, Move::pass(), /*sims=*/200, nwl23(), 1);
  for (uint32_t k = 0; k < 200; ++k) {
    const TileCounts rack = after_reply_racks(job, k).opp.counts();
    ASSERT_EQ(rack.size(), RACK_SIZE);
    EXPECT_GE(rack.count(Tile::from_char('A')), 1);
    EXPECT_GE(rack.count(Tile::from_char('E')), 1);
    for (const char c : std::string("JKQXZ")) EXPECT_EQ(rack.count(Tile::from_char(c)), 0) << c;
  }
}

// Alice's candidates play different numbers of tiles, and Bob replies to each
// by exchanging the same three. Under one seed, sim k of every candidate draws
// Bob the same three tiles, and the candidate that plays fewer tiles draws
// Alice part of what the other draws her.
TEST(DeepBot, AlicesCandidatesArePairedOnBothPlayersDrawsAfterBobsReply) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  Board board;
  board.ensure_movegen_caches(nwl23());
  const Rack alice = Rack::from_string("AEINRST");
  const MoveRequest req{board, nwl23(), alice, Rack{}, 0, 0, 100 - 2 * RACK_SIZE};
  std::vector<RolloutStart> applied;
  for (const Move& m : equity_top_k(req, std::numeric_limits<int>::max())) {
    const bool fewer = applied.empty() || m.num_glyphs() < RACK_SIZE - applied.back().leave.size();
    if (m.type() == MoveType::PLAY && fewer)
      applied.push_back(make_rollout_start(board, alice, 0, 0, m, /*group=*/0));
  }
  ASSERT_GE(applied.size(), 2u);
  const Rack bob = Rack::from_string("BCDFGHI");
  const Move reply = Move::exchange(Rack::from_string("BCD").counts());
  constexpr int kSims = 16;
  std::vector<PoolJob> jobs;
  for (const RolloutStart& a : applied)
    jobs.push_back(after_reply_job(a, bob, reply, kSims, nwl23(), /*seed=*/5));
  for (uint32_t k = 0; k < kSims; ++k) {
    const RolloutRacks first = after_reply_racks(jobs[0], k);
    const Rack bob_drew = without(first.mine, Rack::from_string("FGHI"));
    for (size_t c = 1; c < jobs.size(); ++c) {
      const RolloutRacks racks = after_reply_racks(jobs[c], k);
      EXPECT_EQ(without(racks.mine, Rack::from_string("FGHI")), bob_drew) << "candidate " << c;
      const Rack more = without(after_reply_racks(jobs[c - 1], k).opp, applied[c - 1].leave);
      const Rack fewer = without(racks.opp, applied[c].leave);
      EXPECT_TRUE(holds(more, fewer)) << "candidate " << c;
    }
  }
}

namespace {

// Alice to move after `plies` moves of HastyBot self-play from `seed`'s deal.
struct Midgame {
  Board board;
  Rack rack;
  int my_score = 0;
  int opp_score = 0;
};

Midgame midgame(uint64_t seed, int plies) {
  HastyBot p0({.thread_id = 0, .name = "p0"});
  HastyBot p1({.thread_id = 0, .name = "p1"});
  Game game(p0, p1, nwl23(), seed);
  game.set_max_plies(plies);
  game.play();
  EXPECT_TRUE(game.truncated());
  const int mover = plies % 2;
  Midgame out{game.board(), game.rack(mover), game.score(mover), game.score(1 - mover)};
  out.board.ensure_movegen_caches(nwl23());
  return out;
}

}  // namespace

// Bob's reply at a second turn is the move the rollouts' HastyBot plays on the
// same deal, and Alice holds the rack that deal refilled: the alice-subtree's
// model of a HastyBot opponent is the rollouts' own first ply.
TEST(DeepBot, TheSecondTurnIsTheRolloutsFirstPly) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Midgame at = midgame(/*seed=*/3, /*plies=*/6);
  const MoveRequest req{at.board,
                        nwl23(),
                        at.rack,
                        Rack{},
                        at.my_score,
                        at.opp_score,
                        at.board.pov_bag_size(at.rack.size())};
  const std::vector<Move> moves = equity_top_k(req, 3);
  constexpr int kRacks = 24;
  const PoolJob root =
    position_job(sim_position_from(req), moves, /*seed=*/9, nullptr, nwl23(), kRacks);
  const std::vector<Rack> racks = bob_racks(root.group, root.starts[0], kRacks);
  const belief::ReplyPosition position{.board = at.board,
                                       .chooser_rack = at.rack,
                                       .chooser_score = at.my_score,
                                       .opp_score = at.opp_score,
                                       .bag = req.bag_size,
                                       .earlier = {}};
  const AliceTurn turn{position, moves, root.starts, racks, root.group};
  for (size_t c = 0; c < moves.size(); ++c) {
    const std::vector<SecondTurn> second = second_turns(turn, c, nwl23());
    RolloutBatch batch{.groups = {root.group}, .starts = {root.starts[c]}, .tasks = {}};
    batch.starts[0].group = 0;
    batch.add_rollouts(0, 0, kRacks);
    const auto results = run_hasty_rollouts_cpu(batch, nwl23());
    for (size_t i = 0; i < second.size(); ++i) {
      EXPECT_EQ(second[i].bob_rack, racks[i]);
      EXPECT_EQ(second[i].reply, unpack_move(results[i].moves[0])) << c << " " << i;
      EXPECT_EQ(second[i].alice_rack, rollout_racks(batch, batch.tasks[i]).mine);
    }
  }
}
