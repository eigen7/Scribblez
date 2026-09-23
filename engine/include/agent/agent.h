#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
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

// Everything an agent is told on its turn.
struct MoveRequest {
  const Board& board;
  const Dictionary& dict;
  const Rack& my_rack;
  // What the mover legitimately knows of the opponent's rack, which is not
  // necessarily all of it. In a face-up-leaves game it is the tiles they kept
  // from their last move (their fresh draws stay hidden); once the bag is
  // empty it is their whole rack; otherwise it is empty. An agent may read it
  // freely but must not assume the opponent holds nothing else.
  const Rack& opp_rack;
  int my_score;
  int opp_score;
  int bag_size;
};

// What an agent is told at the start of a game, before any make_move(). A
// struct, like MoveRequest, so new game-start information needs no signature
// change.
struct BeginGameRequest {
  // Each seat's score before the first move: {0, 0} unless a head-start
  // handicap was set (Game::set_initial_scores). An agent that mirrors the game
  // through a GameStateEncoder must seed it with these, because the training
  // replay seeds its encoder from the handicap the .slog records.
  std::array<int, 2> initial_scores{0, 0};
};

std::vector<Move> generate_legal_plays(const MoveRequest& req);

// One move per distinct non-empty sub-multiset of the rack. Empty when the bag
// holds fewer than RACK_SIZE tiles, where exchanging is illegal.
std::vector<Move> generate_legal_exchanges(const MoveRequest& req);

// An agent's answer for one turn: the move to play, optionally followed by a
// projection of the rest of the game. An agent sets projected_remaining_moves
// only when it can prove how the game ends (the endgame solver's certificate).
// A game loop that respects projections (Game::set_respect_projections) plays
// them out without prompting the agents again, so self-play stops spending
// compute on decided games. If the game is not over when the list runs out,
// the loop resumes prompting.
struct MoveDecision {
  Move move;
  std::vector<Move> projected_remaining_moves;

  // Implicit, so an agent without proof machinery can return a plain Move.
  MoveDecision(const Move& m) : move(m) {}
  MoveDecision(const Move& m, std::vector<Move> projected);
};

// Draws from the legal PLAYs and EXCHANGEs together, passing only when neither
// exists. Game's random-opening mode reaches off-policy positions with it.
Move pick_uniform_random_play(const MoveRequest& req, std::mt19937_64& rng);

class Agent {
 public:
  Agent(int thread_id, const std::string& name) : thread_id_(thread_id), name_(name) {}
  virtual ~Agent() = default;

  const std::string& name() const { return name_; }

  // Index of the game thread this agent runs on (0..threads-1). Both seats of
  // a thread share it, so an agent keys per-thread resources off it (e.g. the
  // pooled EndgameSolver): seat-mates share one, and threads never contend.
  int thread_id() const { return thread_id_; }

  virtual MoveDecision make_move(const MoveRequest& req) = 0;

  // Called at the start of each game. An Agent instance plays many games in
  // sequence, so this is where a stateful agent resets.
  virtual void begin_game(const BeginGameRequest& req) {}

  // Called after every move of the game, both seats', in turn order, so a
  // stateful agent can mirror the whole game.
  virtual void observe_move(const Move& move) {}

  // Called on each seat's agent once the game ends.
  virtual EndGameResult end_game(const Game& game, int my_seat) { return {}; }

  // Whether the game engine may run this agent on several game threads. False
  // for the human web agent, which owns a browser session.
  virtual bool supports_parallelism() const { return true; }

 protected:
  int thread_id_;
  std::string name_;
};

}  // namespace scribblez
