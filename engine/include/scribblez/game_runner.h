#pragma once

#include "scribblez/agent.h"
#include "scribblez/dictionary.h"
#include "scribblez/player_factory.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

// Forward-declared so Params::add_options() can register options without
// pulling boost::program_options into every consumer of this header.
namespace boost::program_options { class options_description; }

namespace scribblez {

// Owns the agents, the win/loss tally, and the game loop. Plays a series of
// games on a fixed dictionary, alternating seats each game and honoring each
// agent's EndGameResult to extend (PLAY_AGAIN) or shorten (QUIT) the series
// past the requested `--games` count. Supports parallel execution via a
// thread pool (--threads / -t); human players disable parallelism because they
// own an interactive browser session.
class GameRunner {
 public:
  struct Params {
    // Lexicon name (e.g. "NWL23", "CSW24"). The .kwg is loaded from
    // <lexica_dir>/<lexicon>.kwg. Use the setup_wizard.py "install lexica"
    // step to fetch additional lexica.
    std::string lexicon = "NWL23";
    // Directory holding the .kwg files. Defaults to the Docker mount layout
    // (/workspace/mount/lexica); rarely overridden.
    std::string lexica_dir = "/workspace/mount/lexica";
    int games = 1;            // minimum number of games to play
    std::string log_dir;      // if non-empty, write one <id>.gcg per game here
    int threads = 1;          // number of parallel game threads
    bool verbose = false;     // per-game + batch summaries to stderr

    void add_options(boost::program_options::options_description& desc);

    // Resolved path to the .kwg for params.lexicon.
    std::string kwg_path() const;
  };

  // Constructs the runner from the two Params structs. Validates the params,
  // builds one agent pair per thread (checking parallelism support and
  // downgrading to 1 thread with a warning if any player cannot run in
  // parallel), and loads the lexicon. Pulls its starting seed from
  // SeedProducer::instance() (which the caller is responsible for having
  // reseeded from --seed if reproducibility is desired). Throws
  // scribblez::Exception on user-visible errors (bad --games/--threads,
  // missing lexicon, bad log dir), having already printed an explanation
  // to stderr.
  GameRunner(const Params& runner_params, const PlayerFactory::Params& player_params);
  ~GameRunner();

  // Run the game loop. For threads==1 the loop honors PLAY_AGAIN/QUIT from
  // agents; for threads>1 it plays exactly params.games games in parallel.
  // Returns normally on a clean end.
  void run();

 private:
  // Per-game-and-batch tally, indexed by *player identity* rather than seat
  // (seats alternate every game). Thread-safe via an internal mutex.
  class Results;

  // Play one game using agents_[thread_idx]. Returns the EndGameActions from
  // each seat's agent (only meaningful for the serial/single-thread path).
  std::pair<EndGameAction, EndGameAction> play_one_game(
      int thread_idx, const std::array<int, 2>& seats, uint64_t game_idx);

  Params params_;
  std::vector<PlayerFactory::Players> agents_;
  Dictionary dict_;
  uint64_t seed_;

  std::unique_ptr<Results> results_;
};

}  // namespace scribblez
