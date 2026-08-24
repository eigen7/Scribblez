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
  // The Game does not own the agents; the caller keeps them alive across
  // Game instances, so a human or bot persists (with any per-process state
  // like a WebSession or a Macondo subprocess) over a series of games.
  Game(Agent& p0, Agent& p1, const Dictionary& dict, uint64_t seed);

  // Must be called before play(); asserts that no move has been made yet.
  void set_initial_scores(std::array<int, 2> initial_scores);

  // Respect agents' projected_remaining_moves annotations (see MoveDecision):
  // a proven remainder is played out directly instead of prompting the agents
  // further. Self-play generation turns this on, since a decided game is not
  // worth more agent compute; off by default, so interactive play and strength
  // benchmarks always exercise the agents. Must be called before play().
  void set_respect_projections(bool on);

  // Play the first `plies` turns uniformly at random instead of asking the
  // seated agents, reaching off-policy positions -- especially unusual rack
  // leaves -- that agent self-play would never visit. Agents still see the
  // random moves through observe_move(). Must be called before play().
  void set_random_opening(int plies);

  // Play face-up-leaves Scrabble (docs/roadmap.md): the tiles each player
  // retains from their move are public from then until they move again, and
  // only their replenishment draws stay hidden. Both seats see the other's,
  // and each reads it off MoveRequest::opp_rack. Must be called before play().
  void set_face_up_leaves(bool on);

  // Stop after `plies` moves instead of playing to a natural end -- a
  // value-truncated rollout's horizon (docs/roadmap.md item 2). A game that
  // ends naturally at or before the cap is not truncated, and end-of-game
  // score adjustments apply only on a natural end.
  //
  // The cap applies only while the capped move's pre-move bag was non-empty
  // -- the training-eligibility bound (binary_log.h's eligible span): once
  // the endgame starts the game plays out however many plies that takes.
  // Truncation exists to hand the position evaluation model the capped
  // ply's post-move state as a leaf, and an endgame leaf would be scored by
  // a model that has never seen one. Must be called before play().
  void set_max_plies(int plies);

  void play();

  // Play out from a mid-game position, e.g. a Monte-Carlo rollout. `board` and
  // `scores` are the post-move state and `known_racks[p]` each player's already
  // known tiles (empty to deal fresh); both racks are then refilled from `pool`
  // and `to_move` plays first. `returned_to_bag` holds tiles that rejoin the bag
  // only after those refills -- what a just-made exchange surrendered, which
  // future draws may see but neither initial refill may.
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

  // The tiles `player` retained from their most recent move, before any draw
  // -- their post-move pre-draw rack, which a value-truncated rollout's leaf
  // encode reads at the horizon (and, for the other seat, as the
  // opponent-leave input). play_from seeds it with the caller's known_racks
  // entry until the player first moves.
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
  // What each player has publicly retained, empty until they first move. A
  // move sets its mover's entry to the tiles left after it, before any draw.
  std::array<Rack, 2> leaves_{};
  bool face_up_leaves_ = false;
  int max_plies_ = 0;  // 0 = no cap
  int random_opening_plies_ = 0;
  std::mt19937_64 opening_rng_;
  bool respect_projections_ = false;

  // `drawn_out`, when non-null, accumulates the tiles drawn.
  void refill_rack(int p, Rack* drawn_out);

  // What the player to move legitimately knows of their opponent's rack: the
  // whole thing once the bag is empty, else the public leave under face-up
  // leaves, else nothing.
  const Rack& visible_opp_rack(int mover) const;

  // The decision for the current turn: a uniformly-random move while the turn
  // index is within the random opening (which it also tallies in the log), the
  // seated agent's choice otherwise.
  MoveDecision choose_move(int player, const MoveRequest& req);

  // The turn loop shared by play() and play_from(), running from `start_player`
  // to the standard end (out / stalemate / max-turns).
  void play_loop(int start_player);
};

}  // namespace scribblez
