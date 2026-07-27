#pragma once

// Stateful game-state tracker and model input encoder, scoped to the
// information set of one player observing a game from the outside. It tracks
// only what a player at the table can see -- board, both scores, both players'
// most-recent moves, the active player, and the turn index -- so an owning
// Agent forwards it every move through apply_move() and supplies its OWN rack
// at encode time. Nothing here depends on hidden opponent tiles.
//
// The "unseen pool" scalar deliberately lumps the bag and the opponent's rack
// together, indistinguishable from the active player's POV; the partition
// follows from Scrabble's refill-to-7 rule, so no opp-rack-size scalar is
// needed.
//
// A default-constructed encoder is in the game-start state, so a replay is just
// apply_move() per turn. The initial-scores constructor seeds a head-start
// handicap, making the score-differential feature reflect it from turn 0.

#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>

namespace scribblez {

class Dictionary;

// TILE_COUNTS minus the tiles on `board` and in `my_rack`: the union of the bag
// and the opponent's rack. Indexed by tile kind (A..Z, then blank).
void compute_unseen_pool(uint8_t out[27], const Board& board, const Rack& my_rack);

// Which of a PLAY turn's two samples a position is.
enum class PositionKind : uint8_t {
  kPreMove = 0,   // active player is about to play
  kPostMove = 1,  // active player just played; refill has not happened yet
                  // (unseen-pool composition unchanged from pre-move).
                  // Only emitted for PLAY turns.
};

class GameStateEncoder {
 public:
  explicit GameStateEncoder(const InputEncodingSpec& spec) : spec_(spec) {}

  // Additionally seed the score accumulator with a per-player handicap.
  GameStateEncoder(const InputEncodingSpec& spec, std::array<int, 2> initial_scores)
      : spec_(spec), scores_(initial_scores) {}

  // Advance one turn: the *current* active player made `move`. It deliberately
  // takes no draw information, an outside observer seeing none.
  void apply_move(const Move& move);

  // --- inspectors ---------------------------------------------------------
  const InputEncodingSpec& spec() const { return spec_; }
  int active_player() const { return active_; }
  int turn_index() const { return turn_index_; }
  const Board& board() const { return board_; }
  int score(int p) const { return scores_[p]; }
  const Move& last_move_by(int p) const { return last_move_by_[p]; }

  // --- encoders -----------------------------------------------------------
  // Encode the current state from `player`'s POV into `out`
  // (input_floats(spec()) long, the spec's blocks in registry order).
  // `my_rack` is `player`'s own rack right now.
  //
  // A pre-move sample passes player == active_player(). A post-PLAY sample
  // (that player, before any draw and before the opponent responds) goes:
  //     enc.apply_move(my_play);
  //     enc.encode_input(the_player_who_just_played, rack_after_play_pre_draw, ...);
  // where active_player() is now the opponent, so passing the pre-flip player
  // keeps the encode anchored to their POV and both labels and last_opp_move
  // attach to them.
  //
  // Aborts if the spec demands an opponent leave; use the overload.
  void encode_input(int player, const Rack& my_rack, bool apply_flip, float* out) const;

  // Additionally encodes `opp_leave` into the kOppLeaveCounts block, for a spec
  // under the open-leaves condition. An empty leave (opponent has not acted, or
  // bingoed) is legitimate and encodes as zeros.
  void encode_input(int player, const Rack& my_rack, const Rack& opp_leave, bool apply_flip,
                    float* out) const;

  // As encode_input(), but forcing the score differential to `score_diff` and
  // leaving every other feature identical -- isolating it for the structural
  // monotonicity probes that sweep a fixed position's score advantage.
  void encode_input_with_score_diff(int player, const Rack& my_rack, int score_diff,
                                    bool apply_flip, float* out) const;

  // Rewrite an already-encoded row's score differential in place, so a sweep
  // encodes the position once instead of re-running the move-generating encode
  // per step.
  void overwrite_score_diff(int score_diff, float* input_row) const;

 private:
  InputEncodingSpec spec_;
  Board board_{};
  std::array<int, 2> scores_{0, 0};
  std::array<Move, 2> last_move_by_{};  // default-constructed = PASS
  int active_ = 0;
  int turn_index_ = 0;
};

}  // namespace scribblez
