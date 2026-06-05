#pragma once

#include "scribblez/agent.h"
#include "scribblez/dictionary.h"
#include "scribblez/player_factory.h"

#include <array>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <string>

// Forward-declared so Params::add_options() can register options without
// pulling boost::program_options into every consumer of this header.
namespace boost::program_options { class options_description; }

namespace scribblez {

// Owns the agents, the GCG output stream, the win/loss tally, and the game
// loop. Plays a series of games on a fixed dictionary, alternating seats
// each game and honoring each agent's EndGameResult to extend (PLAY_AGAIN)
// or shorten (QUIT) the series past the requested `--games` count.
class GameRunner {
 public:
  struct Params {
    std::string kwg_path = "data/lexica/NWL23.kwg";  // lexicon file to load
    int games = 1;            // minimum number of games to play
    std::string out_path;     // empty => stdout
    bool verbose = false;     // per-game + batch summaries to stderr

    void add_options(boost::program_options::options_description& desc);
  };

  // Takes ownership of `players` and loads the lexicon named by
  // params.kwg_path. Pulls its starting seed from SeedProducer::instance()
  // (which the caller is responsible for having reseeded from --seed if
  // reproducibility is desired); the starting seed decides who starts
  // game 1 (low bit) and seeds the bag (seed, seed+1, ...) for successive
  // games. Throws scribblez::Exception on user-visible errors (bad
  // --games, missing lexicon, failed output file), having already printed
  // an explanation to stderr.
  GameRunner(const Params& params, PlayerFactory::Players players);
  ~GameRunner();

  // Run the game loop. Throws on output-file failures; returns normally on
  // a clean end (whether by --games count, PLAY_AGAIN extension, or QUIT).
  void run();

 private:
  // Per-game-and-batch tally, indexed by *player identity* rather than seat
  // (seats alternate every game).
  class Results;

  // One iteration of the loop. Returns false if the loop should terminate.
  bool play_one_game(std::array<int, 2>& player_at_seat, uint64_t game_idx);

  Params params_;
  std::array<std::unique_ptr<Agent>, 2> agents_;
  Dictionary dict_;
  uint64_t seed_;

  std::ofstream of_;   // owns the file iff params_.out_path is non-empty
  std::ostream* out_;  // points at of_ or std::cout (never null)
  std::unique_ptr<Results> results_;
};

}  // namespace scribblez
