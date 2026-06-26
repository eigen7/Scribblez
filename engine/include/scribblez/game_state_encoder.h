#pragma once

// Stateful Scribblez game-state tracker + win-probability model input
// encoder, scoped to the information set of a single player observing a
// game from the outside.
//
// The class is designed to be owned by an Agent that plays a real game:
// the Agent observes every move (its own and the opponent's), forwarding
// each move to apply_move() as it happens. The Agent never observes the
// opponent's draws or rack contents -- that data is intentionally absent
// from this class. When the Agent needs to evaluate a model on the current
// position, it calls encode_input() and supplies its OWN rack at that
// moment (the Agent always knows its own rack).
//
// Internal state: board, both cumulative scores, both players'
// most-recent moves, the active player index, and the turn index. All of
// this is fully observable to a player at the table.
//
// Information leakage: encode_input()'s output depends only on the
// caller-supplied my_rack plus this class's tracked state -- never on
// hidden opponent tiles. The "unseen pool" scalar feature lumps the bag
// and the opponent's rack together (they are indistinguishable from the
// active player's POV); the partition is recoverable from
// unseen_pool_size via Scrabble's refill-to-7 rule, so no separate
// opp-rack-size scalar is needed.
//
// Replay canonicalization: a default-constructed encoder is in the
// game-start state (empty board, scores 0, both last-moves are PASS,
// active player 0, turn index 0). A head-start handicap can be supplied via
// the initial-scores constructor, which seeds the score accumulator so the
// score-differential feature reflects the handicap from turn 0. apply_move()
// flips active_player and increments turn_index in lock-step with normal play.

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <array>
#include <cstdint>

namespace scribblez {

// Sample kinds within a single game turn. Used by the DataLoader's replay
// decoder to label which of the two per-PLAY-turn samples is which.
enum class PositionKind : uint8_t {
  kPreMove = 0,   // active player is about to play
  kPostMove = 1,  // active player just played; refill has not happened yet
                  // (unseen-pool composition unchanged from pre-move).
                  // Only emitted for PLAY turns.
};

class GameStateEncoder {
 public:
  GameStateEncoder() = default;

  // Seed the score accumulator with a per-player starting handicap.
  explicit GameStateEncoder(std::array<int, 2> initial_scores) : scores_(initial_scores) {}

  // Advance one turn: the *current* active player made `move`. PLAY also
  // updates the board and the active player's cumulative score. The active
  // player then alternates and turn_index increments.
  //
  // This method intentionally takes no draw information -- an outside
  // observer does not see opponent draws, and the encoder never needs to
  // know a player's exact rack to encode (the caller supplies its own rack
  // at encode time).
  void apply_move(const Move& move);

  // --- inspectors ---------------------------------------------------------
  int active_player() const { return active_; }
  int turn_index() const { return turn_index_; }
  const Board& board() const { return board_; }
  int score(int p) const { return scores_[p]; }
  const Move& last_move_by(int p) const { return last_move_by_[p]; }

  // --- encoders -----------------------------------------------------------
  // Encode the current state into `out` (kInputFloats long) from `player`'s
  // POV. `my_rack` is `player`'s own rack right now.
  //
  // For a pre-move sample, callers pass player == active_player(). For a
  // post-PLAY sample (the player who just moved, before any draw and before
  // the opp has responded), the canonical sequence is:
  //     enc.apply_move(my_play);  // advances board+score, flips active
  //     enc.encode_input(/*player=*/the_player_who_just_played,
  //                      /*my_rack=*/rack_after_play_pre_draw, ...);
  // The encoder's own active_player() is now the opponent; passing the
  // pre-flip player here keeps the encode anchored to their POV (so labels
  // and last_opp_move both attach correctly to that player).
  void encode_input(int player, const Rack& my_rack, bool apply_flip, float* out) const;

  // Encode as encode_input(), but force the score differential
  // (score(player) - score(opp)) to `score_diff`, leaving every other
  // feature identical. The score feature depends only on this differential,
  // so this isolates it for structural monotonicity probes that sweep the
  // active player's score advantage across an otherwise-fixed position.
  void encode_input_with_score_diff(int player, const Rack& my_rack, int score_diff,
                                    bool apply_flip, float* out) const;

 private:
  Board board_{};
  std::array<int, 2> scores_{0, 0};
  std::array<Move, 2> last_move_by_{};  // default-constructed = PASS
  int active_ = 0;
  int turn_index_ = 0;
};

}  // namespace scribblez
