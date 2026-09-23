#pragma once

// Tracks a game's public state and encodes model input rows from it (layout in
// input_encoder.h). It holds only what either player can see: the board, the
// scores, each player's last move, and whose turn it is. The caller supplies
// the POV player's own rack, and under open leaves the opponent's known leave,
// at encode time. An agent keeps one per game and feeds it every move through
// apply_move(); replaying a log is apply_move() per turn from the start state.
//
// The unseen pool lumps the bag and the opponent's rack together, as the POV
// player cannot tell them apart. The opponent's rack size needs no feature of
// its own: it is always 7, or the whole unseen pool once that is smaller.

#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>

namespace scribblez {

class Dictionary;

// The tiles the holder of `my_rack` cannot see: TILE_COUNTS minus the board
// and `my_rack`, i.e. the bag plus the opponent's rack. Indexed A..Z, blank.
void compute_unseen_pool(uint8_t out[27], const Board& board, const Rack& my_rack);

class GameStateEncoder {
 public:
  explicit GameStateEncoder(const InputEncodingSpec& spec) : spec_(spec) {}

  // Starts from handicap scores, so the score-difference feature reflects them
  // from turn 0.
  GameStateEncoder(const InputEncodingSpec& spec, std::array<int, 2> initial_scores)
      : spec_(spec), scores_(initial_scores) {}

  // Starts mid-game with no move history, e.g. at a Monte-Carlo rollout's
  // decision point. Both last moves start as PASS. Since the placement planes
  // and move-meta scalars read them, encode only after two apply_move() calls
  // have supplied real ones.
  GameStateEncoder(const InputEncodingSpec& spec, const Board& board, std::array<int, 2> scores,
                   int active)
      : spec_(spec), board_(board), scores_(scores), active_(active) {}

  // The active player made `move`. Takes no draw, since nobody else sees it.
  void apply_move(const Move& move);

  // This state with the board and both last moves transposed (see
  // Board::transpose), so encoding the result gives the transposed row. Moves
  // applied afterwards must be in the transposed frame too.
  GameStateEncoder transpose() const;

  const InputEncodingSpec& spec() const { return spec_; }
  int active_player() const { return active_; }
  int turn_index() const { return turn_index_; }
  const Board& board() const { return board_; }
  int score(int p) const { return scores_[p]; }
  const Move& last_move_by(int p) const { return last_move_by_[p]; }

  // Encodes the current state from `player`'s POV into `out`, which needs
  // input_floats(spec()) floats. `my_rack` is `player`'s current rack.
  //
  // `player` need not be the active player. A post-move row encodes the player
  // who just moved, with their rack before drawing:
  //     enc.apply_move(my_move);
  //     enc.encode_input(me, rack_after_move_before_draw, out);
  //
  // Aborts under an open-leaves spec; use the overload.
  void encode_input(int player, const Rack& my_rack, float* out) const;

  // Under any spec. `opp_leave` is read only under an open-leaves spec, so a
  // caller that has the opponent's leave can encode without branching on the
  // spec. It may be empty (the opponent has not moved, or kept nothing).
  void encode_input(int player, const Rack& my_rack, const Rack& opp_leave, float* out) const;

 private:
  InputEncodingSpec spec_;
  Board board_{};
  std::array<int, 2> scores_{0, 0};
  std::array<Move, 2> last_move_by_{};  // PASS until each player moves
  int active_ = 0;
  int turn_index_ = 0;
};

// The post-move row for candidate `mv` from `pre`, from `mover`'s POV.
// `my_rack` is the mover's rack before the move. `opp_leave` is read only
// under an open-leaves spec.
//
// This is the one place a candidate's row is built, so the serving agent and
// the offline target generator cannot disagree on how to encode it.
void encode_post_move_row(const GameStateEncoder& pre, int mover, const Rack& my_rack,
                          const Move& mv, const Rack& opp_leave, float* out);

}  // namespace scribblez
