#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

class Game;
class Dictionary;

// Outcome of Agent::end_game(). NONE means the agent has no opinion (the
// default for bots); PLAY_AGAIN / QUIT come from the human agent.
enum class EndGameAction { NONE, PLAY_AGAIN, QUIT };
struct EndGameResult {
  EndGameAction action = EndGameAction::NONE;
};

// Everything an agent needs in order to choose a move on its turn. Constructed
// by the game loop and passed to Agent::make_move().
struct MoveRequest {
  const Board& board;
  const Dictionary& dict;
  const Rack& my_rack;
  const Rack& opp_rack;  // visible during the endgame, hidden otherwise
  int my_score;
  int opp_score;
  int bag_size;
};

std::vector<Move> generate_legal_plays(const MoveRequest& req);

// One move per distinct non-empty sub-multiset of the rack. Empty when the bag
// holds fewer than RACK_SIZE tiles, where exchanging is illegal.
std::vector<Move> generate_legal_exchanges(const MoveRequest& req);

// An agent's answer for one turn: the move to play now, optionally annotated
// with a projection of the whole remainder of the game. An agent sets
// projected_remaining_moves only when it can PROVE how the rest of the game
// goes (the endgame solver's proof certificate). A game loop configured to
// respect projections (Game::set_respect_projections) plays them out directly
// instead of prompting the agents further -- the self-play break-out that stops
// spending compute on a decided game. The projection is best-effort: when the
// game is not over after the list is exhausted, the loop resumes prompting.
struct MoveDecision {
  Move move;
  std::vector<Move> projected_remaining_moves;

  // Intentionally implicit, so agents without proof machinery return a Move.
  MoveDecision(const Move& m) : move(m) {}
  MoveDecision(const Move& m, std::vector<Move> projected);
};

// Uniformly at random among all legal PLAYs and EXCHANGEs; passes only when
// neither exists. Game's random-opening mode reaches off-policy positions with
// it.
Move pick_uniform_random_play(const MoveRequest& req, std::mt19937_64& rng);

class Agent {
 public:
  Agent(int thread_id, const std::string& name) : thread_id_(thread_id), name_(name) {}
  virtual ~Agent() = default;

  const std::string& name() const { return name_; }

  // Index of the game thread this agent runs on (0..threads-1). An agent
  // needing a per-thread external resource (e.g. a Macondo subprocess) keys it
  // off this id, so seat-mates share one and distinct threads never contend.
  int thread_id() const { return thread_id_; }

  virtual MoveDecision make_move(const MoveRequest& req) = 0;

  // Called once at the start of each game, before any make_move() on it. One
  // Agent instance is reused across a series of games, so this is where a
  // stateful agent resets.
  virtual void begin_game() {}

  // Called after every applied move of the game, the agent's own and the
  // opponent's, in turn order. Lets a stateful agent mirror the whole game
  // even though make_move() fires only on its own turns.
  virtual void observe_move(const Move& move) {}

  // Called once after the game ends, on each seat's agent. The default result
  // means "no opinion"; the human agent returns the user's Play Again / Quit
  // choice.
  virtual EndGameResult end_game(const Game& game, int my_seat) { return {}; }

  // Whether instances of this agent can run concurrently in a threaded game
  // loop. The human web agent cannot: it owns a browser session.
  virtual bool supports_parallelism() const { return true; }

 protected:
  int thread_id_;
  std::string name_;
};

// Picks the highest-scoring PLAY. If none exists, exchanges the entire rack (if
// the bag has >= RACK_SIZE tiles), otherwise passes. Ties broken randomly.
class GreedyAgent : public Agent {
 public:
  explicit GreedyAgent(int thread_id, const std::string& name = "Greedy");
  GreedyAgent(int thread_id, const std::string& name, uint64_t seed);

  MoveDecision make_move(const MoveRequest& req) override;

  // Build from `--player "--type=greedy [options]"` tokens, with --type and
  // --name already stripped. Throws std::runtime_error on bad input.
  static std::unique_ptr<GreedyAgent> from_spec(const std::vector<std::string>& tokens,
                                                int thread_id, const std::string& name);

  static std::string options_help();

 private:
  std::mt19937_64 rng_;
};

}  // namespace scribblez
