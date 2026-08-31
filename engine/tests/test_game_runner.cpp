// GameRunner match-eval plumbing: that --paired plays games 2k and 2k+1 under
// one game seed with the seats swapped, and that --results-file records one
// readable JSON line per game. Drives real (deterministic HastyBot) agents
// through PlayerFactory, which is why this binary links like play_game rather
// than joining scribblez_tests. Needs the real lexicon mount; skips without it.

#include "agent/player_factory.h"
#include "arena/game_runner.h"
#include "lexicon/lexicon.h"
#include "synthetic_equity.h"
#include "util/exception.h"
#include "util/seed_producer.h"

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace scribblez;

namespace {

// One --results-file line, read back.
struct ResultLine {
  uint64_t seed;
  std::array<int, 2> seat_players;
  std::array<int, 2> seat_scores;
  int turns;
};

std::vector<ResultLine> read_result_lines(const std::filesystem::path& path) {
  std::vector<ResultLine> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const boost::json::object o = boost::json::parse(line).as_object();
    const boost::json::array& sp = o.at("seat_players").as_array();
    const boost::json::array& ss = o.at("seat_scores").as_array();
    out.push_back({o.at("seed").to_number<uint64_t>(),
                   {sp[0].to_number<int>(), sp[1].to_number<int>()},
                   {ss[0].to_number<int>(), ss[1].to_number<int>()},
                   o.at("turns").to_number<int>()});
  }
  return out;
}

// The real lexicon plus a synthetic equity table, torn down with the temp
// dir. Skips (not fails) without the lexicon mount, like the other gated
// suites.
class GameRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!std::ifstream(Lexicon::instance().kwg_path()).good()) {
      GTEST_SKIP() << "no lexicon at " << Lexicon::instance().kwg_path();
    }
    tmp_ =
      std::filesystem::temp_directory_path() / ("scribblez_paired_" + std::to_string(::getpid()));
    std::filesystem::create_directories(tmp_);
    scribblez::testing::install_synthetic_hasty_equity(tmp_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_, ec);
  }

  static GameRunner::Params paired_params(int games) {
    GameRunner::Params rp;
    rp.games = games;
    rp.threads = 2;
    rp.paired = true;
    rp.progress_secs = 0;
    return rp;
  }

  static PlayerFactory::Params hasty_players() {
    PlayerFactory::Params pp;
    pp.specs = {"--type=hastybot --name=A", "--type=hastybot --name=B"};
    return pp;
  }

  std::filesystem::path tmp_;
};

}  // namespace

// With deterministic agents the two games of a pair are the same game with the
// player labels exchanged: same per-seat scores and length, opposite seat
// assignment. Distinct pairs get distinct seeds.
TEST_F(GameRunnerTest, PairedGamesShareSeedsAndMirrorSeats) {
  const std::filesystem::path results = tmp_ / "results.jsonl";

  SeedProducer::Params seed_params;
  seed_params.seed = 20260801;
  SeedProducer::instance().seed(seed_params);

  GameRunner::Params rp = paired_params(/*games=*/4);
  rp.results_file = results.string();

  GameRunner runner(rp, hasty_players());
  runner.run();

  const std::vector<ResultLine> lines = read_result_lines(results);
  ASSERT_EQ(lines.size(), 4u);

  std::map<uint64_t, std::vector<ResultLine>> by_seed;
  for (const ResultLine& r : lines) by_seed[r.seed].push_back(r);
  ASSERT_EQ(by_seed.size(), 2u);
  for (const auto& [seed, pair] : by_seed) {
    ASSERT_EQ(pair.size(), 2u) << "seed " << seed;
    EXPECT_EQ(pair[0].seat_players[0], pair[1].seat_players[1]);
    EXPECT_EQ(pair[0].seat_players[1], pair[1].seat_players[0]);
    EXPECT_EQ(pair[0].seat_scores, pair[1].seat_scores);
    EXPECT_EQ(pair[0].turns, pair[1].turns);
  }
}

// An odd --games cannot form pairs; construction must reject it rather than
// leave the last seed half-mirrored.
TEST_F(GameRunnerTest, PairedRequiresEvenGames) {
  EXPECT_THROW(GameRunner(paired_params(/*games=*/3), hasty_players()), util::CleanException);
}

// PlayerFactory's per-type knowledge lives in one trait list (player_factory.cpp).
// These lexicon-free tests pin the list-generated surfaces -- default names, the
// help block, and the unknown-type error -- so each stays covered as types are
// added. Full construction of every type isn't unit-testable (human blocks on
// its Vite server, neural needs a model), so agent building is left to the
// GameRunner suite's real hastybot games above.
namespace {

// Every player type paired with its default display name (no explicit --name).
const std::array<std::pair<const char*, const char*>, 9> kTypeDefaults{{
  {"greedy", "Greedy"},
  {"human", "You"},
  {"hastybot", "HastyBot"},
  {"hastybot-endgame", "EndgameHastyBot"},
  {"mset-sim", "MsetSim"},
  {"neural", "Neural"},
  {"neural-sim", "NeuralSim"},
  {"sim", "SimBot"},
  {"weirdbot", "WeirdBot"},
}};

}  // namespace

// Each type resolves to its trait-list default name, and an explicit --name wins.
TEST(PlayerFactoryTest, DefaultDisplayNames) {
  for (const auto& [type, def] : kTypeDefaults) {
    PlayerSpec spec;
    spec.type = type;
    EXPECT_EQ(spec.display_name(), def) << "type " << type;
    spec.name = "Zed";
    EXPECT_EQ(spec.display_name(), "Zed") << "explicit --name should win for " << type;
  }
}

// An unknown type falls back to the literal type string (no trait entry).
TEST(PlayerFactoryTest, UnknownTypeDisplayNameFallsBack) {
  PlayerSpec spec;
  spec.type = "nope";
  EXPECT_EQ(spec.display_name(), "nope");
}

// The help block is generated from the list: one --type=<t> line per type.
TEST(PlayerFactoryTest, HelpListsEveryType) {
  const std::string help = PlayerFactory::all_player_types_help();
  for (const auto& [type, def] : kTypeDefaults) {
    EXPECT_NE(help.find(std::string("--type=") + type + " "), std::string::npos)
      << "help missing type " << type;
  }
}

// A bad --type is rejected before any agent is built, and the error names every
// valid type (the listing is generated from the same list).
TEST(PlayerFactoryTest, UnknownTypeErrorNamesEveryType) {
  PlayerFactory::Params params;
  params.specs = {"--type=bogus", "--type=greedy"};
  try {
    PlayerFactory::make_players(params, /*thread_id=*/0);
    FAIL() << "expected a CleanException for the bogus type";
  } catch (const util::CleanException& e) {
    const std::string msg = e.what();
    for (const auto& [type, def] : kTypeDefaults) {
      EXPECT_NE(msg.find(type), std::string::npos) << "error missing type " << type;
    }
  }
}
