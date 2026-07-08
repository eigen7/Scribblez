#include "selfplay/self_play_engine.h"

#include "game/game.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"

#include <cmath>
#include <iostream>
#include <random>

namespace scribblez {

namespace {

// Choose a head-start handicap for one game: pick a player at random and gift
// them P points, P uniform in [0, max]. Returns per-player starting scores.
// Seeded from the game seed so the choice is reproducible; max <= 0 yields
// {0, 0}.
std::array<int, 2> pick_handicap(uint64_t game_seed, int max) {
  if (max <= 0) return {0, 0};
  std::mt19937_64 rng(game_seed);
  const int player = static_cast<int>(rng() & 1ULL);
  const int points = std::uniform_int_distribution<int>(0, max)(rng);
  std::array<int, 2> scores = {0, 0};
  scores[player] = points;
  return scores;
}

// Choose one game's random-opening length: an exponential draw with the given
// mean, rounded to the nearest integer (so ~22% of games get 0 at mean 2 and
// long openings tail off geometrically). Seeded from the game seed (salted to
// decorrelate from pick_handicap's stream) so the choice is reproducible;
// mean <= 0 yields 0 (no random opening).
int pick_random_opening_plies(uint64_t game_seed, double mean) {
  if (mean <= 0) return 0;
  std::mt19937_64 rng(game_seed ^ 0x6C62272E07BB0142ULL);
  const double x = std::exponential_distribution<double>(1.0 / mean)(rng);
  return static_cast<int>(std::lround(x));
}

}  // namespace

SelfPlayEngine::SelfPlayEngine(const Params& params, const PlayerFactory::Params& player_params)
    : params_(params) {
  if (params_.threads < 1) {
    std::cerr << "Error: threads must be >= 1\n";
    throw Exception("threads must be >= 1");
  }
  // Build the first pair to check parallelism support before creating the rest.
  agents_.push_back(PlayerFactory::make_players(player_params, /*thread_id=*/0));
  bool parallel_ok = agents_[0][0]->supports_parallelism() && agents_[0][1]->supports_parallelism();
  if (!parallel_ok && params_.threads > 1) {
    std::cerr << "Warning: a player does not support parallelism; running single-threaded.\n";
    params_.threads = 1;
  }
  for (int i = 1; i < params_.threads; ++i) {
    agents_.push_back(PlayerFactory::make_players(player_params, /*thread_id=*/i));
  }
}

std::array<std::string, 2> SelfPlayEngine::player_names() const {
  return {agents_[0][0]->name(), agents_[0][1]->name()};
}

std::pair<EndGameAction, EndGameAction> SelfPlayEngine::play(int thread_idx,
                                                             const std::array<int, 2>& seats,
                                                             uint64_t game_idx, GameSink& sink) {
  Agent& seat0 = *agents_[thread_idx][seats[0]];
  Agent& seat1 = *agents_[thread_idx][seats[1]];
  const uint64_t game_seed = params_.seed + game_idx;
  Game game(seat0, seat1, Lexicon::instance().dict(), game_seed);
  game.set_initial_scores(pick_handicap(game_seed, params_.handicap_max));
  game.set_random_opening(pick_random_opening_plies(game_seed, params_.random_opening_mean));
  game.play();

  // Hand the finished game's owning log storage to the sink (a move). The Game
  // remains valid for end_game (which reads only the live board/score/rack).
  sink.on_game(game.extract_log(), seats);

  auto r0 = seat0.end_game(game, 0);
  auto r1 = seat1.end_game(game, 1);
  return {r0.action, r1.action};
}

}  // namespace scribblez
