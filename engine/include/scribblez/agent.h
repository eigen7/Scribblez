#pragma once

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <array>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

// Forward declaration so Agent::end_game() can take a Game& without pulling
// game.h (which itself includes agent.h) into every translation unit.
class Game;
class Dictionary;

// Outcome of Agent::end_game(). NONE means the agent has no opinion (the
// default for bots); PLAY_AGAIN / QUIT are produced by the human agent in
// response to the user's choice on the game-over screen.
enum class EndGameAction { NONE, PLAY_AGAIN, QUIT };
struct EndGameResult {
  EndGameAction action = EndGameAction::NONE;
};

// Everything an agent needs in order to choose a move on its turn. Constructed
// by the game loop and passed to Agent::make_move(). The game loop no longer
// pre-computes legal moves: each agent generates the moves it needs from
// `board` + `dict` (most via generate_legal_plays(); HastyBot via its own
// shadow-play search), so an agent only pays for the generation it actually
// uses.
struct MoveRequest {
  const Board& board;
  const Dictionary& dict;
  const Rack& my_rack;
  const Rack& opp_rack;  // opponent's rack (hidden during mid-game; fully
                         // visible at endgame when bag is empty)
  int my_score;
  int opp_score;
  int bag_size;
};

// Generate every legal PLAY for the active player (the GADDAG move generator
// over req.board with req.my_rack). PASS/EXCHANGE are the agent's concern. This
// is the shared path for agents that want the full move list (Greedy, Human,
// the test bots); HastyBot bypasses it with shadow-play.
std::vector<Move> generate_legal_plays(const MoveRequest& req);

// Generate every legal EXCHANGE for the active player: one move per distinct
// non-empty sub-multiset of req.my_rack (duplicate tiles do not multiply the
// list). Empty when the bag has fewer than RACK_SIZE tiles, where exchanging
// is illegal.
std::vector<Move> generate_legal_exchanges(const MoveRequest& req);

// Pick a move for the active player uniformly at random among all legal PLAYs
// and all legal EXCHANGEs; passes only when neither exists. Used by Game's
// random-opening mode to reach off-policy positions that agent self-play would
// never visit.
Move pick_uniform_random_play(const MoveRequest& req, std::mt19937_64& rng);

class Agent {
 public:
  Agent(int thread_id, const std::string& name) : thread_id_(thread_id), name_(name) {}
  virtual ~Agent() = default;

  // Display name shown in logs / UI. Set at construction time and never
  // changed, so this is intentionally non-virtual.
  const std::string& name() const { return name_; }

  // Index of the GameRunner thread this agent runs on (0..threads-1). Agents
  // that need a per-thread external resource (e.g. a Macondo subprocess) key
  // off this id via the corresponding pool, ensuring two agents on the same
  // thread share the resource while different threads never contend.
  int thread_id() const { return thread_id_; }

  virtual Move make_move(const MoveRequest& req) = 0;

  // Called once at the start of each game, before any make_move() on it. Lets a
  // stateful agent reset to a clean starting position. The same Agent instance
  // is reused across a series of games (seats alternate), so this is the reset
  // point. Default: no-op.
  virtual void begin_game() {}

  // Called after every applied move in the game -- the agent's own moves and
  // the opponent's -- in turn order. Lets a stateful agent mirror the full game
  // progression even though make_move() only fires on its own turns (needed,
  // e.g., to maintain a GameStateEncoder whose features depend on both players'
  // most-recent placements). `move` is the move just applied. Default: no-op.
  virtual void observe_move(const Move& move) {}

  // Called once after the game ends, on each seat's agent. The returned
  // EndGameResult tells play_game what to do next: the default no-op result
  // means "no opinion" (the bot batch loop / single-game exit governs).
  // The human web agent overrides this to surface the final position and a
  // Play Again / Quit prompt to the user.
  virtual EndGameResult end_game(const Game& game, int my_seat) { return {}; }

  // Returns true if this agent can safely run in a multi-threaded game loop
  // (i.e. multiple independent instances of this agent class can execute
  // concurrently). Bot agents return true; the human web agent returns false
  // because it owns a browser session and interactive I/O.
  virtual bool supports_parallelism() const { return true; }

 protected:
  int thread_id_;
  std::string name_;
};

// Picks the highest-scoring PLAY. If none exists, exchanges the entire rack
// (if the bag has >= RACK_SIZE tiles), otherwise passes. Ties broken randomly
// using the agent's own RNG (seeded by SeedProducer by default, or by an
// explicit --seed=N option).
class GreedyAgent : public Agent {
 public:
  explicit GreedyAgent(int thread_id, const std::string& name = "Greedy");
  GreedyAgent(int thread_id, const std::string& name, uint64_t seed);

  Move make_move(const MoveRequest& req) override;

  // Build a GreedyAgent from `--player "--type=greedy [options]"` tokens
  // (after the factory has stripped --type and --name). `name` is the resolved
  // display name. Throws std::runtime_error on bad input.
  static std::unique_ptr<GreedyAgent> from_spec(const std::vector<std::string>& tokens,
                                                int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  std::mt19937_64 rng_;
};

}  // namespace scribblez
