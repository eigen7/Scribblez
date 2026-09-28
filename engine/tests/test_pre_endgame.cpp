// PreEndgameSolver, the port of Macondo's pre-endgame solver, and its wiring
// into EndgameTurnPolicy. Most cases evaluate leaves with a scripted oracle, so
// the solver's tallies can be checked against a brute-force reference; one runs
// the real SolverEndgameOracle. Every case needs the NWL23 lexicon and
// Macondo's leave table, and skips without them.

#include "agent/best_bot.h"
#include "agent/endgame_agent.h"
#include "agent/hasty_bot.h"
#include "endgame/endgame_oracle.h"
#include "endgame/endgame_solver.h"
#include "endgame/pre_endgame_solver.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "hasty_positions.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "move_key.h"

#include <gtest/gtest.h>

#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace scribblez;
using scribblez::testing::find_position;
using scribblez::testing::key_set;
using scribblez::testing::move_key;
using scribblez::testing::Position;

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

// Evaluates every leaf with a fixed function of the position and counts calls.
class ScriptedOracle : public EndgameOracle {
 public:
  explicit ScriptedOracle(std::function<EndgameVerdict(const EndgameState&)> f)
      : f_(std::move(f)) {}
  EndgameVerdict evaluate(const EndgameState& state, int /*effort*/, EndgameGoal) override {
    ++calls;
    return f_(state);
  }
  int calls = 0;

 private:
  std::function<EndgameVerdict(const EndgameState&)> f_;
};

// A crude but deterministic leaf value for the side to move: the score, plus
// the face value each side still has to shed.
EndgameVerdict rack_spread(const EndgameState& s) {
  const int32_t v = s.my_score - s.opp_score + s.opp_rack.point_value() - s.my_rack.point_value();
  return {.cls = (v > 0) - (v < 0), .spread = v, .proven = true};
}

// A position with one tile in the bag, from the first game at or after seed
// `from` that reaches one.
Position one_in_bag(uint64_t from) {
  for (uint64_t seed = from;; ++seed) {
    const std::optional<Position> p =
      find_position(nwl23(), seed, [](int bag) { return bag == 1; });
    if (p) return *p;
  }
}

PreEndgamePosition peg_position(const Position& p) {
  return {&nwl23(), p.board, p.rack, Rack{}, p.my_score, p.opp_score, 0};
}

// The points a play that empties the bag earns under `oracle`, drawing each
// unseen tile in turn: the opponent, holding the rest, then faces a known
// endgame.
double reference_points(const Position& p, const Move& m, EndgameOracle& oracle) {
  const TileCounts unseen = p.board.unseen_tiles(p.rack);
  double points = 0.0;
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    const int count = unseen.count(t);
    if (count == 0) continue;
    Board board = p.board;
    board.apply(m);
    Rack mine = p.rack;
    for (int i = 0; i < m.num_glyphs(); ++i) mine.remove(m.glyph(i).rack_tile());
    mine.add(t);
    TileCounts opp = unseen;
    opp.remove(t);
    const EndgameState opp_to_move{
      &nwl23(), board, Rack::from_counts(opp), mine, p.opp_score, p.my_score + m.score(), 0};
    const int ours = -oracle.evaluate(opp_to_move, 1, EndgameGoal::kClass).cls;
    points += ours > 0 ? count : ours == 0 ? count / 2.0 : 0.0;
  }
  return points;
}

const PreEndgameSolver::RankedPlay* find_play(const std::vector<PreEndgameSolver::RankedPlay>& r,
                                              MoveType type) {
  for (const auto& p : r)
    if (p.move.type() == type) return &p;
  return nullptr;
}

}  // namespace

// Every play that empties the bag and was scored over all eight draws has
// exactly the reference's points, and none beats the solver's pick.
TEST(PreEndgameSolver, ScoresBagEmptyingPlaysOverEveryDraw) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / leaves";
  const Position p = one_in_bag(3);
  ScriptedOracle oracle(rack_spread);
  const std::vector<PreEndgameSolver::RankedPlay> ranked =
    PreEndgameSolver(oracle).solve(peg_position(p), {.max_effort = 1});
  ASSERT_FALSE(ranked.empty());

  ScriptedOracle ref(rack_spread);
  int checked = 0;
  for (const auto& r : ranked) {
    if (r.move.type() != MoveType::PLAY) continue;
    const double want = reference_points(p, r.move, ref);
    EXPECT_LE(want, ranked.front().points + 1e-9) << move_key(r.move);
    if (r.total == RACK_SIZE + 1) {
      EXPECT_DOUBLE_EQ(r.points, want) << move_key(r.move);
      ++checked;
    }
  }
  EXPECT_GT(checked, 0);
}

// The pass is searched too: the opponent's replies, then our nested turn.
TEST(PreEndgameSolver, SearchesThePass) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / leaves";
  ScriptedOracle oracle(rack_spread);
  const std::vector<PreEndgameSolver::RankedPlay> ranked =
    PreEndgameSolver(oracle).solve(peg_position(one_in_bag(5)), {.max_effort = 1});
  const PreEndgameSolver::RankedPlay* pass = find_play(ranked, MoveType::PASS);
  ASSERT_NE(pass, nullptr);
  EXPECT_GT(pass->total, 0);
  EXPECT_GE(pass->points, 0.0);
  EXPECT_LE(pass->points, double(pass->total));
}

// A leaf the oracle cannot classify is scored by the sign of its spread
// estimate, or as a loss for the mover, as configured.
TEST(PreEndgameSolver, UnprovenLeavesFollowThePolicy) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / leaves";
  const Position p = one_in_bag(3);
  // Unclassified, and estimated as a loss for whoever is to move: after a
  // bag-emptying play, that is the opponent.
  const auto unknown_loss = [](const EndgameState&) { return EndgameVerdict{.spread = -5}; };
  for (const UnprovenPolicy policy : {UnprovenPolicy::kEstimatedSpread, UnprovenPolicy::kLoss}) {
    ScriptedOracle oracle(unknown_loss);
    const std::vector<PreEndgameSolver::RankedPlay> ranked =
      PreEndgameSolver(oracle).solve(peg_position(p), {.max_effort = 1, .unproven = policy});
    int checked = 0;
    for (const auto& r : ranked) {
      if (r.move.type() != MoveType::PLAY || r.total != RACK_SIZE + 1) continue;
      EXPECT_DOUBLE_EQ(r.points, policy == UnprovenPolicy::kLoss ? 0.0 : RACK_SIZE + 1.0)
        << move_key(r.move);
      ++checked;
    }
    EXPECT_GT(checked, 0);
  }
}

TEST(PreEndgameSolver, RejectsBagsItDoesNotSolve) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / leaves";
  const std::optional<Position> p = find_position(nwl23(), 3, [](int bag) { return bag > 10; });
  ASSERT_TRUE(p);
  ScriptedOracle oracle(rack_spread);
  EXPECT_THROW(PreEndgameSolver(oracle).solve(peg_position(*p), {}), std::exception);
}

// Macondo's BestBot deepens less the further apart the scores are.
TEST(PreEndgameSolver, MacondoEffortSchedule) {
  EXPECT_EQ(PreEndgameSolver::macondo_max_effort(0), 7);
  EXPECT_EQ(PreEndgameSolver::macondo_max_effort(-49), 7);
  EXPECT_EQ(PreEndgameSolver::macondo_max_effort(50), 5);
  EXPECT_EQ(PreEndgameSolver::macondo_max_effort(-60), 4);
  EXPECT_EQ(PreEndgameSolver::macondo_max_effort(85), 3);
  EXPECT_EQ(PreEndgameSolver::macondo_max_effort(-150), 2);
}

// With our solver behind the oracle, the pick is one of the mover's legal
// moves.
TEST(PreEndgameSolver, RunsOnTheSolverOracle) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / leaves";
  const Position p = one_in_bag(4);
  EndgameSolver solver;
  SolverEndgameOracle oracle(solver, /*budget=*/500);
  const std::vector<PreEndgameSolver::RankedPlay> ranked =
    PreEndgameSolver(oracle).solve(peg_position(p), {.max_effort = 1});
  ASSERT_FALSE(ranked.empty());
  const Move best = ranked.front().move;
  if (best.type() == MoveType::PLAY) {
    const std::set<std::string> legal = key_set(generate_legal_plays(p.request(nwl23())));
    EXPECT_TRUE(legal.count(move_key(best)) > 0);
  }
}

// hastybot-endgame leaves the one-tile turn to HastyBot unless asked;
// bestbot-endgame solves it by default, as Macondo's BestBot does.
TEST(PreEndgameSolver, EndgameAgentsWireItIn) {
  if (!have_data()) GTEST_SKIP() << "no NWL23 kwg / leaves";
  EXPECT_FALSE(EndgameAgent<HastyBot>::Params{}.peg.enabled);
  EXPECT_TRUE(EndgameAgent<BestBot>::Params{}.peg.enabled);

  const Position p = one_in_bag(4);
  const MoveRequest req = p.request(nwl23());
  EndgameAgent<HastyBot>::Params params;
  params.base = {.thread_id = 0, .name = "E"};
  EndgameAgent<HastyBot> off(params);
  EXPECT_EQ(off.make_move(req).move, hasty_best_move_wmp(req));

  params.peg.enabled = true;
  params.peg.solver.max_effort = 1;
  params.peg.budget = 200;
  EndgameAgent<HastyBot> on(params);
  const Move m = on.make_move(req).move;
  if (m.type() == MoveType::PLAY) {
    EXPECT_TRUE(key_set(generate_legal_plays(req)).count(move_key(m)) > 0);
  }

  EXPECT_NE(EndgameAgent<HastyBot>::from_spec(
              {"--peg=1", "--peg-budget=300", "--peg-max-effort=3", "--peg-unproven=loss"}, 0, "X"),
            nullptr);
  EXPECT_THROW(EndgameAgent<HastyBot>::from_spec({"--peg-unproven=maybe"}, 0, "X"), std::exception);
}
