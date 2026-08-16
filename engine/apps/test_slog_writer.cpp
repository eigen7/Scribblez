// Standalone helper that writes test .slog files for integration testing.
// Usage: test_slog_writer <output_dir> <num_games> <games_per_file> [--face-up-leaves]
//
// Plays `num_games` games using a medium-sized in-memory dictionary and a
// hasty-move agent, writing them through BinaryLogWriter into the given
// output directory. Files are split by `games_per_file`. With
// --face-up-leaves the games are played under the variant and the file
// headers are stamped accordingly, mirroring GameRunner.

#include "agent/agent.h"
#include "data/binary_log.h"
#include "game/game.h"
#include "lexicon/dictionary.h"

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

using namespace scribblez;

// Hasty agent: picks the highest-scoring legal move (ties broken randomly).
class HastyAgent : public Agent {
 public:
  HastyAgent(int tid, std::string name, uint64_t seed) : Agent(tid, std::move(name)), rng_(seed) {}

  MoveDecision make_move(const MoveRequest& req) override {
    const std::vector<Move> plays = generate_legal_plays(req);
    if (!plays.empty()) {
      int best = -1;
      for (const auto& m : plays) best = std::max(best, int(m.score()));
      std::vector<const Move*> top;
      for (const auto& m : plays)
        if (int(m.score()) == best) top.push_back(&m);
      std::uniform_int_distribution<size_t> d(0, top.size() - 1);
      return *top[d(rng_)];
    }
    return Move::pass();
  }

 private:
  std::mt19937_64 rng_;
};

static Dictionary medium_dict() {
  return Dictionary::build_from_words({
    "CAT",  "CATS", "AT",   "AS",     "BAT",     "BATS", "HE",   "TO",   "ON",   "NO",
    "IT",   "IS",   "OAT",  "OATS",   "HAT",     "HATS", "RAT",  "RATS", "DOG",  "GOD",
    "GO",   "OD",   "DO",   "AERIES", "PARTIED", "THE",  "THEN", "THIN", "TIN",  "IN",
    "SIN",  "BIN",  "PIN",  "PAN",    "TAN",     "RAN",  "MAN",  "CAN",  "DAN",  "FAN",
    "VAN",  "AN",   "RE",   "ARE",    "CARE",    "DARE", "FARE", "HARE", "MARE", "PARE",
    "RARE", "TARE", "WARE", "OR",     "FOR",     "NOR",  "TOR",  "COR",  "AH",   "OH",
    "OX",   "AX",   "EX",   "ADD",    "ODD",     "ILL",  "ALL",  "EEL",  "ERA",  "ERR",
  });
}

int main(int argc, char* argv[]) {
  const bool face_up = argc == 5 && std::string(argv[4]) == "--face-up-leaves";
  if (argc != 4 && !face_up) {
    std::cerr
      << "Usage: test_slog_writer <output_dir> <num_games> <games_per_file> [--face-up-leaves]\n";
    return 1;
  }
  const std::string out_dir = argv[1];
  const int num_games = std::atoi(argv[2]);
  const int games_per_file = std::atoi(argv[3]);

  if (num_games <= 0 || games_per_file <= 0) {
    std::cerr << "num_games and games_per_file must be > 0\n";
    return 1;
  }

  Dictionary dict = medium_dict();
  binlog::BinaryLogWriter writer(out_dir, games_per_file,
                                 face_up ? binlog::kFlagFaceUpLeaves : uint16_t{0});

  for (int i = 0; i < num_games; ++i) {
    uint64_t seed = 3000 + i;
    HastyAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
    HastyAgent a1(1, "A1", seed ^ 0x2222222222222222ULL);
    Game g(a0, a1, dict, seed);
    g.set_face_up_leaves(face_up);
    g.play();
    writer.append(g.extract_log());
  }

  std::cout << "Wrote " << num_games << " games to " << out_dir << "\n";
  return 0;
}
