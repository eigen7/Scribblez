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
// moment (the Agent always knows its own rack) plus the size of the
// opponent's rack (the Agent tracks both players' draw counts implicitly
// from the move stream and the bag's initial size, so opp_rack_size is
// also known to it; we just don't need to track it here).
//
// Internal state: board, both cumulative scores, both players'
// most-recent moves, the active player index, and the turn index. All of
// this is fully observable to a player at the table.
//
// Information leakage: encode_input()'s output depends only on the
// caller-supplied my_rack and opp_rack_size plus this class's tracked
// state -- never on hidden opponent tiles. The "unseen pool" scalar
// feature lumps the bag and the opponent's rack together (they are
// indistinguishable from the active player's POV); only the opp rack
// SIZE is exposed as a separate scalar.
//
// Replay canonicalization: a default-constructed encoder is in the
// game-start state (empty board, scores 0, both last-moves are PASS,
// active player 0, turn index 0). apply_move() flips active_player and
// increments turn_index in lock-step with normal play.

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <array>
#include <cstdint>

namespace scribblez {
namespace binlog {

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
  // Encode the current (pre-move) state into `out` (kInputFloats long) from
  // the active player's POV.
  //   my_rack         -- the active player's own rack right now
  //   opp_rack_size   -- the opponent's rack tile count (the active player
  //                      knows this from the draw stream, but the encoder
  //                      itself doesn't track it)
  void encode_input(const Rack& my_rack, int opp_rack_size, bool apply_flip, float* out) const;

  // Encode a post-PLAY view: as if the active player just applied
  // `play_move` (no draw yet, so the unseen-pool composition is unchanged
  // from pre-move). Does NOT mutate this encoder. `play_move` must be a
  // PLAY-typed move. `my_rack` is the PRE-play rack; the played tiles are
  // removed inside this method.
  void encode_input_post_play(const Move& play_move, const Rack& my_rack, int opp_rack_size,
                              bool apply_flip, float* out) const;

 private:
  Board board_{};
  std::array<int, 2> scores_{0, 0};
  std::array<Move, 2> last_move_by_{};  // default-constructed = PASS
  int active_ = 0;
  int turn_index_ = 0;
};

}  // namespace binlog
}  // namespace scribblez
