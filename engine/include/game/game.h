#pragma once

#include "agent/agent.h"
#include "game/bag.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <array>
#include <random>

namespace scribblez {

class Game {
 public:
  // The agents are borrowed, so one agent (and any state it holds, such as a
  // web session) can persist across a series of games.
  Game(Agent& p0, Agent& p1, const Dictionary& dict, uint64_t seed);

  // The set_* configuration calls must precede play()/play_from(); each
  // asserts that no move has been made.

  void set_initial_scores(std::array<int, 2> initial_scores);

  // Honor MoveDecision::projected_remaining_moves: once an agent has proven the
  // rest of the game, play its projection instead of prompting the agents.
  // Self-play turns this on to save compute on decided games. Off by default,
  // so interactive play and strength benchmarks always exercise the agents.
  void set_respect_projections(bool on);

  // Play the first `plies` turns uniformly at random instead of asking the
  // agents, to reach off-policy positions (especially unusual leaves) that
  // self-play would never visit. Agents still see these moves via
  // observe_move().
  void set_random_opening(int plies);

  // Play face-up-leaves Scrabble (docs/roadmap.md): the tiles a player keeps
  // after moving are public until their next move; only the replacement draws
  // stay hidden. Each seat sees the other's leave in MoveRequest::opp_rack.
  void set_face_up_leaves(bool on);

  // Stop after `plies` moves: the horizon of a value-truncated rollout
  // (docs/roadmap.md), whose final position is scored by the position
  // evaluation model. A game that ends naturally by then is not truncated, and
  // end-of-game score adjustments apply only on a natural end.
  //
  // The cap never cuts into the endgame: it applies only if the capped move
  // was made with tiles in the bag, else the game plays out. The model trains
  // only on positions after such moves (binlog::eligible_span), so it could
  // not score a later leaf.
  void set_max_plies(int plies);

  void play();

  // Play out from a mid-game position, e.g. a Monte-Carlo rollout. `board` and
  // `scores` are the position; `known_racks[p]` are the tiles each player is
  // known to hold (empty to deal fresh), topped up from `pool` in turn order
  // starting with `to_move`, who plays first. `returned_to_bag` joins the bag
  // only after those refills: it is what a just-made exchange gave up, which
  // later draws may see but neither initial refill may.
  void play_from(const Board& board, std::array<int, 2> scores,
                 const std::array<Rack, 2>& known_racks, const Bag& pool, int to_move,
                 const Rack& returned_to_bag = Rack{});

  // Valid while the Game lives and extract_log() has not been called.
  GameLog log() const { return log_.view(); }

  // After this, log() must not be used.
  GameLogStorage extract_log() { return std::move(log_); }

  // These reflect the final state once play() returns.
  const Board& board() const { return board_; }
  int score(int player) const { return scores_[player]; }
  const Rack& rack(int player) const { return racks_[player]; }
  int bag_size() const { return bag_.size(); }

  // Whether play stopped at the set_max_plies cap rather than a natural end.
  bool truncated() const { return log_.end_reason == "truncated"; }

  // The tiles `player` kept from their most recent move, before drawing. A
  // value-truncated rollout encodes these at the horizon. Until the player
  // first moves, it is their play_from() known rack.
  const Rack& leave(int player) const { return leaves_[player]; }

 private:
  Agent* players_[2];
  const Dictionary& dict_;
  uint64_t seed_;
  Bag bag_;
  Board board_;
  Rack racks_[2];
  std::array<int, 2> scores_{0, 0};
  GameLogStorage log_;
  std::array<Rack, 2> leaves_{};
  bool face_up_leaves_ = false;
  int max_plies_ = 0;  // 0 = no cap
  int random_opening_plies_ = 0;
  std::mt19937_64 opening_rng_;
  bool respect_projections_ = false;

  // `drawn_out`, when non-null, accumulates the tiles drawn.
  void refill_rack(int p, Rack* drawn_out);

  // What the mover can know of the opponent's rack: all of it once the bag is
  // empty, else the public leave under face-up leaves, else nothing.
  const Rack& visible_opp_rack(int mover) const;

  // A uniformly random move during the random opening (tallied in the log),
  // otherwise the seated agent's choice.
  MoveDecision choose_move(int player, const MoveRequest& req);

  // The turn loop shared by play() and play_from().
  void play_loop(int start_player);
};

}  // namespace scribblez
