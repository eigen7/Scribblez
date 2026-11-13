// DeepBot under both reply models, on the CPU reference with the synthetic
// leaf (the suites hide the GPU): it plays whole games, deterministically for a
// seed; asks the leave inference model for Alice's belief from her own view
// and, in Bob's subtrees, for Bob's with her private past hidden; Bob replies
// from his own equity top N, as contenders weighted by their chance of being
// his best; --subtree-dump writes a line per turn, and with --policy-model
// measures a policy-set model against Bob's replies. Skips without the NWL23
// lexicon and HastyBot's leave values.

#include "agent/alice_subtree.h"
#include "agent/bob_subtree.h"
#include "agent/deep_bot.h"
#include "agent/hasty_bot.h"
#include "arena/recorded_turn.h"
#include "belief/leave_params.h"
#include "encoding/leave_inference_input_encoder.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/leave_values.h"
#include "nwl23_equity.h"
#include "sim/pool_jobs.h"
#include "sim/rollout_values.h"
#include "stub_policy_set_service.h"
#include "temp_dir.h"
#include "util/exception.h"
#include "util/math.h"

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
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
  return {.values = std::make_unique<CpuRolloutValues>(nwl23(), synthetic_leaf()),
          .predict = std::move(predict),
          .table = std::move(table),
          .policy = std::nullopt};
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

// Replayed to one of its turns of a recorded game, DeepBot holds what it held
// there in play: it asks the model the same question and picks the same move.
TEST(DeepBot, ReplayedToARecordedTurnItAsksAndPicksAsItDidInPlay) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  auto table = std::make_shared<const LeaveValueArrays>(
    LeaveValues::load(HastyEquity::default_leaves_path("NWL23")));
  std::vector<std::vector<float>> asked;
  belief::LeaveParamsFn predict = [&](const float* rows, int, belief::LeaveParams* out) {
    asked.emplace_back(rows, rows + Encoder::kInputFloats);
    out[0] = belief::LeaveParams{};
  };
  DeepBot live(small_params(5, "hasty"), cpu_backend(predict, table));
  HastyBot hasty({.thread_id = 0, .name = "HastyBot"});
  Game game(live, hasty, nwl23(), /*seed=*/1);
  game.play();
  const GameLogStorage log = game.extract_log();
  const std::vector<std::vector<float>> in_play = std::move(asked);
  asked.clear();

  // DeepBot's fifth turn: by then its query reads its own earlier racks.
  const int turn = 8;
  ASSERT_GT(log.turns[turn].bag_size_before, 0);
  DeepBot replayed(small_params(5, "hasty"), cpu_backend(predict, table));
  const RecordedPick pick = pick_at_recorded_turn(replayed, log.view(), turn, nwl23());
  EXPECT_EQ(pick.pick, log.turns[turn].move);
  EXPECT_EQ(pick.rack, log.turns[turn].rack_before);
  EXPECT_EQ(pick.bag, log.turns[turn].bag_size_before);
  ASSERT_EQ(asked.size(), 1u);  // the earlier turns record, and ask nothing
  EXPECT_NE(std::find(in_play.begin(), in_play.end(), asked[0]), in_play.end());
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
// With a policy model, each searched candidate's dump entry carries how often
// the policy's top move at Bob's position is his subtree's reply. A policy
// that favours the equity-best move agrees exactly where the subtree chose
// HastyBot's move (hasty_share), over the pool and over the simmed candidates
// alike, since the equity-best move is always simmed.
TEST(DeepBot, TheDumpMeasuresAPolicyAgainstBobsReplies) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const auto dir = scribblez::testing::make_temp_dir("scribblez_test_policy_agreement");
  DeepBot::Params p = small_params(7);
  p.subtree_candidates = 0;
  p.subtree_dump = (dir / "dump.jsonl").string();
  p.policy_model = "stub";
  p.dict = &nwl23();
  auto stub = std::make_shared<scribblez::testing::StubPolicySetService>();
  stub->favourite = 0;
  DeepBot::Backend backend = cpu_backend();
  backend.policy = PolicyBot::Backend{
    .policy = stub, .blank_prune_points = 1000, .predict = {}, .table = {}, .cross_checks = {}};
  DeepBot bot(p, std::move(backend));
  play(bot);
  std::ifstream in(p.subtree_dump);
  int searched = 0;
  for (std::string line; std::getline(in, line);) {
    const boost::json::object turn = boost::json::parse(line).as_object();
    for (const boost::json::value& v : turn.at("candidates").as_array()) {
      const boost::json::object& c = v.as_object();
      if (!c.at("searched").as_bool()) continue;
      ++searched;
      const double hasty = c.at("replies").as_object().at("hasty_share").as_double();
      const boost::json::object& policy = c.at("policy").as_object();
      EXPECT_EQ(policy.at("pool_share").as_double(), hasty) << line;
      EXPECT_EQ(policy.at("simmed_share").as_double(), hasty) << line;
    }
  }
  EXPECT_GT(searched, 5);
  EXPECT_GT(stub->calls.size(), size_t(searched));  // one call per (candidate, rack)
}

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
  int with_choice = 0;
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
      // A weighted mean of win probabilities, within rounding of the weights.
      const double value = c.at("subtree_value").as_double();
      EXPECT_TRUE(value >= 0.0 && value <= 1.0 + 1e-9) << line;
      const boost::json::object& r = c.at("replies").as_object();
      const boost::json::array& ranks = r.at("rank_hist").as_array();
      EXPECT_LE(ranks.size(), size_t(p.subtree.late_candidates));
      int64_t racks = 0;
      for (const boost::json::value& n : ranks) racks += n.as_int64();
      EXPECT_EQ(racks, int64_t(p.bob_racks));
      const double contenders = r.at("contenders_mean").as_double();
      EXPECT_GE(contenders, 1.0);
      EXPECT_LE(contenders, double(p.sims_per_rack));
      const double choice = r.at("choice_share").as_double();
      EXPECT_GE(choice, 0.0);
      EXPECT_LE(choice, 1.0);
      // Two of Bob's moves at a rack can also be worth the same to Alice.
      EXPECT_GE(r.at("payoff_range_mean").as_double(), 0.0);
      with_choice += choice > 0;
    }
    // A turn whose candidates all empty the bag runs only the first pass.
    const boost::json::object& sims = turn.at("sims").as_object();
    const bool searched_any =
      turn.at("subtree_pick").as_int64() >= 0 &&
      std::any_of(
        turn.at("candidates").as_array().begin(), turn.at("candidates").as_array().end(),
        [](const boost::json::value& v) { return v.as_object().at("searched").as_bool(); });
    EXPECT_GT(sims.at("first_pass").as_int64(), 0);
    EXPECT_EQ(sims.at("subtrees").as_int64() > 0, searched_any) << line;
    EXPECT_EQ(sims.at("after_replies").as_int64() > 0, searched_any) << line;
  }
  EXPECT_GT(lines, 5);
  EXPECT_GT(emptying, 0);  // this seed's game reaches one
  // Three samples per rack leave Bob's sims unsure somewhere.
  EXPECT_GT(with_choice, 0);
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
  p = small_params(1);
  p.policy_model = "policy.onnx";  // without a dump, nothing would read it
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
  const PoolJob job = after_reply_job(exchanged, bob, {Move::pass()}, /*sims=*/{200}, nwl23(), 1);
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
    jobs.push_back(after_reply_job(a, bob, {reply}, {kSims}, nwl23(), /*seed=*/5));
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
  const std::vector<Rack> racks = other_racks(root.group, root.starts[0], kRacks);
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

namespace {

// A labeled pool of `q.size()` moves, move k with chance q[k] of being best.
PoolPosition labeled_pool(const std::vector<double>& q) {
  PoolPosition out;
  for (size_t k = 0; k < q.size(); ++k) {
    out.moves.push_back(Move::exchange(Rack::from_string(std::string(k + 1, 'A')).counts()));
    out.labels.push_back({.win_prob = 0.5, .paired_se = 0, .q = q[k], .rollouts = 1});
  }
  return out;
}

std::vector<int> ranks_of(const std::vector<SubtreeMove>& contenders) {
  std::vector<int> out;
  for (const SubtreeMove& m : contenders) out.push_back(m.equity_rank);
  return out;
}

std::vector<int> sims_of(const std::vector<SubtreeMove>& contenders) {
  std::vector<int> out;
  for (const SubtreeMove& m : contenders) out.push_back(m.sims);
  return out;
}

}  // namespace

// Alice's sims at a rack go to Bob's moves by the Sainte-Laguë rule over
// their chances of being his best, likeliest first; the moves that get any
// are his contenders, their chances renormalized over them. A likelier move
// never gets fewer sims, where cumulative rounding would give 0.31, 0.29,
// 0.25, 0.15 at ten sims 4, 2, 3, 1. chosen_move is the likeliest alone.
TEST(DeepBot, AlicesSimsAreApportionedAmongBobsMovesByTheirChanceOfBeingBest) {
  const PoolPosition pool = labeled_pool({0.05, 0.6, 0.3, 0.05});
  const std::vector<SubtreeMove> all = contenders(pool.moves, pool, /*sims=*/12);
  EXPECT_EQ(ranks_of(all), (std::vector<int>{1, 2, 0, 3}));
  EXPECT_EQ(sims_of(all), (std::vector<int>{7, 3, 1, 1}));
  EXPECT_DOUBLE_EQ(all[0].weight, 0.6);
  EXPECT_DOUBLE_EQ(all[0].q, 0.6);
  EXPECT_EQ(all[0].candidates, 4);
  const std::vector<SubtreeMove> two = contenders(pool.moves, pool, /*sims=*/4);
  EXPECT_EQ(ranks_of(two), (std::vector<int>{1, 2}));
  EXPECT_EQ(sims_of(two), (std::vector<int>{3, 1}));
  EXPECT_DOUBLE_EQ(two[0].weight, 0.6 / 0.9);
  EXPECT_DOUBLE_EQ(two[1].weight, 0.3 / 0.9);
  const SubtreeMove chosen = chosen_move(pool.moves, pool);
  EXPECT_EQ(chosen.equity_rank, 1);
  EXPECT_DOUBLE_EQ(chosen.weight, 1.0);
  const PoolPosition close = labeled_pool({0.31, 0.29, 0.25, 0.15});
  EXPECT_EQ(sims_of(contenders(close.moves, close, /*sims=*/10)), (std::vector<int>{3, 3, 3, 1}));
  // Ties go to the lower equity rank, as they did when the likeliest alone
  // was Bob's reply.
  const PoolPosition flat = labeled_pool({0.08, 0.08, 0.08});
  EXPECT_EQ(ranks_of(contenders(flat.moves, flat, /*sims=*/1)), (std::vector<int>{0}));
  EXPECT_EQ(sims_of(contenders(flat.moves, flat, /*sims=*/4)), (std::vector<int>{2, 1, 1}));
}

// One more sim only adds to the split, and a larger share never holds fewer.
TEST(DeepBot, ApportioningIsMonotoneInSharesAndInTheTotal) {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> shares(size_t(2 + trial % 5));
    for (double& s : shares) s = u(rng);
    std::sort(shares.rbegin(), shares.rend());
    std::vector<int> before(shares.size(), 0);
    for (int total = 1; total <= 20; ++total) {
      const std::vector<int> split = util::apportion(shares, total);
      EXPECT_EQ(std::accumulate(split.begin(), split.end(), 0), total);
      for (size_t k = 0; k < split.size(); ++k) {
        EXPECT_GE(split[k], before[k]);
        if (k > 0) {
          EXPECT_GE(split[k - 1], split[k]);
        }
      }
      before = split;
    }
  }
}

// Alice's value of a candidate is the mean over Bob's racks of her payoffs
// after his contenders, weighted by their weights, each contender getting its
// own sims; with one contender per rack it is the mean payoff, and its
// payoff is the same with the others' sims beside it.
TEST(DeepBot, AlicesValueWeightsHerPayoffsAfterBobsContenders) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 lexicon or leave values";
  const Midgame at = midgame(/*seed=*/5, /*plies=*/8);
  const MoveRequest req{at.board,
                        nwl23(),
                        at.rack,
                        Rack{},
                        at.my_score,
                        at.opp_score,
                        at.board.pov_bag_size(at.rack.size())};
  const std::vector<Move> moves = equity_top_k(req, 2);
  constexpr int kRacks = 3;
  const PoolJob root =
    position_job(sim_position_from(req), moves, /*seed=*/11, nullptr, nwl23(), kRacks);
  const std::vector<Rack> racks = other_racks(root.group, root.starts[0], kRacks);
  const belief::ReplyPosition position{.board = at.board,
                                       .chooser_rack = at.rack,
                                       .chooser_score = at.my_score,
                                       .opp_score = at.opp_score,
                                       .bag = req.bag_size,
                                       .earlier = {}};
  const AliceTurn turn{position, moves, root.starts, racks, root.group};
  // Bob's reply at each rack: his equity top two, weighted 3 to 1, with
  // three of Alice's sims after the first and one after the second.
  std::vector<std::vector<BobReply>> replies(moves.size(), std::vector<BobReply>(kRacks));
  for (size_t c = 0; c < moves.size(); ++c) {
    for (size_t i = 0; i < kRacks; ++i) {
      const std::vector<Move> his = mover_candidates(
        mover_position(root.starts[c], racks[i]), {.candidates = 2, .late_candidates = 2}, nwl23());
      ASSERT_EQ(his.size(), 2u);
      replies[c][i] = {{.move = his[0], .weight = 0.75, .sims = 3},
                       {.move = his[1], .weight = 0.25, .sims = 1}};
    }
  }
  CpuRolloutValues values(nwl23(), synthetic_leaf());
  const std::vector<size_t> which{0, 1};
  const ReplyPayoffs weighted =
    payoffs_after_replies(turn, which, replies, values, nwl23(), /*seed=*/2);
  EXPECT_EQ(weighted.sims, int64_t(moves.size()) * kRacks * (3 + 1));
  for (size_t c : which) {
    double expect = 0;
    for (size_t i = 0; i < kRacks; ++i) {
      ASSERT_EQ(weighted.payoffs[c][i].size(), 2u);
      for (const double p : weighted.payoffs[c][i]) {
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
      }
      expect += (0.75 * weighted.payoffs[c][i][0] + 0.25 * weighted.payoffs[c][i][1]) / kRacks;
    }
    EXPECT_NEAR(weighted.values[c], expect, 1e-12);
  }
  // With residuals: one summary per (candidate, rack, contender), its sims
  // the contender's, over the squares empty after the reply.
  const ReplyPayoffs with_residuals =
    payoffs_after_replies(turn, which, replies, values, nwl23(), /*seed=*/2, /*residuals=*/true);
  for (size_t c : which) {
    ASSERT_EQ(with_residuals.residuals[c].size(), size_t(kRacks));
    for (size_t i = 0; i < kRacks; ++i) {
      ASSERT_EQ(with_residuals.residuals[c][i].size(), 2u);
      for (size_t k = 0; k < 2; ++k) {
        const MapResidual& r = with_residuals.residuals[c][i][k];
        EXPECT_EQ(r.sims, replies[c][i][k].sims);
        EXPECT_GT(r.squares, 150);
        EXPECT_LT(r.squares, kBoardCells);
        EXPECT_GE(r.points_noise, 0.0);
        EXPECT_GE(r.ownership_noise, 0.0);
        EXPECT_TRUE(std::isfinite(r.points_r2 + r.ownership_r2 + r.end_r2 + r.going_out_r2));
      }
      EXPECT_DOUBLE_EQ(with_residuals.payoffs[c][i][0], weighted.payoffs[c][i][0]);
    }
  }
  // One contender, weight 1: the mean of its payoffs, the same three sims.
  for (auto& by_rack : replies)
    for (BobReply& r : by_rack) {
      r.resize(1);
      r[0].weight = 1;
    }
  const ReplyPayoffs single =
    payoffs_after_replies(turn, which, replies, values, nwl23(), /*seed=*/2);
  for (size_t c : which) {
    double expect = 0;
    for (size_t i = 0; i < kRacks; ++i) {
      ASSERT_EQ(single.payoffs[c][i].size(), 1u);
      EXPECT_DOUBLE_EQ(single.payoffs[c][i][0], weighted.payoffs[c][i][0]);
      expect += single.payoffs[c][i][0] / kRacks;
    }
    EXPECT_NEAR(single.values[c], expect, 1e-12);
  }
}
