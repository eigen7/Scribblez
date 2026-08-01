// GameRunner match-eval plumbing: that --paired plays games 2k and 2k+1 under
// one game seed with the seats swapped, and that --results-file records one
// readable JSON line per game. Drives real (deterministic HastyBot) agents
// through PlayerFactory, which is why this binary links like play_game rather
// than joining scribblez_tests. Needs the real lexicon mount; skips without it.

#include "agent/player_factory.h"
#include "lexicon/lexicon.h"
#include "selfplay/game_runner.h"
#include "selfplay/seed_producer.h"
#include "synthetic_equity.h"

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

}  // namespace

// With deterministic agents the two games of a pair are the same game with the
// player labels exchanged: same per-seat scores and length, opposite seat
// assignment. Distinct pairs get distinct seeds.
TEST(GameRunner, PairedGamesShareSeedsAndMirrorSeats) {
  if (!std::ifstream(Lexicon::instance().kwg_path()).good()) {
    GTEST_SKIP() << "no lexicon at " << Lexicon::instance().kwg_path();
  }
  namespace fs = std::filesystem;
  const fs::path tmp =
    fs::temp_directory_path() / ("scribblez_paired_" + std::to_string(::getpid()));
  fs::create_directories(tmp);
  scribblez::testing::install_synthetic_hasty_equity(tmp);
  const fs::path results = tmp / "results.jsonl";

  SeedProducer::Params seed_params;
  seed_params.seed = 20260801;
  SeedProducer::instance().seed(seed_params);

  GameRunner::Params rp;
  rp.games = 4;
  rp.threads = 2;
  rp.paired = true;
  rp.progress_secs = 0;
  rp.results_file = results.string();
  PlayerFactory::Params pp;
  pp.specs = {"--type=hastybot --name=A", "--type=hastybot --name=B"};

  GameRunner runner(rp, pp);
  runner.run();

  const std::vector<ResultLine> lines = read_result_lines(results);
  std::error_code ec;
  fs::remove_all(tmp, ec);
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
