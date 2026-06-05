#pragma once

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

// Forward declaration so Agent::end_game() can take a Game& without pulling
// game.h (which itself includes agent.h) into every translation unit.
class Game;

// Outcome of Agent::end_game(). NONE means the agent has no opinion (the
// default for bots); PLAY_AGAIN / QUIT are produced by the human agent in
// response to the user's choice on the game-over screen.
enum class EndGameAction { NONE, PLAY_AGAIN, QUIT };
struct EndGameResult {
  EndGameAction action = EndGameAction::NONE;
};

// Everything an agent needs in order to choose a move on its turn. Constructed
// by the game loop and passed to Agent::make_move().
struct MoveRequest {
  const Board& board;
  const Rack& my_rack;
  int my_score;
  int opp_score;
  int bag_size;
  int opp_rack_size;              // tiles on the opponent's rack (hidden contents)
  std::vector<Move> legal_plays;  // PLAY moves only; agent may pass/exchange
};

class Agent {
 public:
  virtual ~Agent() = default;
  virtual std::string name() const = 0;
  virtual Move make_move(const MoveRequest& req) = 0;

  // Called once after the game ends, on each seat's agent. The returned
  // EndGameResult tells play_game what to do next: the default no-op result
  // means "no opinion" (the bot batch loop / single-game exit governs).
  // The human web agent overrides this to surface the final position and a
  // Play Again / Quit prompt to the user.
  virtual EndGameResult end_game(const Game& game, int my_seat) {
    (void)game;
    (void)my_seat;
    return {};
  }
};

// Picks the highest-scoring PLAY. If none exists, exchanges the entire rack
// (if the bag has >= RACK_SIZE tiles), otherwise passes. Ties broken randomly
// using the agent's own RNG (seeded by SeedProducer by default, or by an
// explicit --seed=N option).
class GreedyAgent : public Agent {
 public:
  explicit GreedyAgent(const std::string& name = "Greedy");
  GreedyAgent(const std::string& name, uint64_t seed);

  std::string name() const override { return name_; }
  Move make_move(const MoveRequest& req) override;

  // Build a GreedyAgent from `--player "--type=greedy [options]"` tokens
  // (after the factory has stripped --type and --name). `name` is the resolved
  // display name. Throws std::runtime_error on bad input.
  static std::unique_ptr<GreedyAgent> from_spec(const std::vector<std::string>& tokens,
                                                const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  std::string name_;
  std::mt19937_64 rng_;
};

}  // namespace scribblez
