#pragma once

#include "encoding/input_encoder.h"
#include "game/game_log.h"

#include <functional>
#include <memory>
#include <random>

namespace scribblez {
namespace binlog {

// Turns a finished self-play game into one training row for a particular
// TrainingTask: it picks the turn to sample, then encodes the position there.
// This keeps the streaming producer task-agnostic. It owns a PositionEncoder,
// so each worker thread has its own.
class RowEncoder {
 public:
  virtual ~RowEncoder() = default;

  virtual int row_floats() const = 0;

  // The turn of `view` to sample, or -1 to drop the game. Called before a row
  // is claimed in the StreamingRowBuffer, so a dropped game never holds one.
  virtual int pick_turn(const GameLog& view, std::mt19937_64& rng) = 0;

  virtual void encode(const GameLog& view, int turn, bool transpose, float* dest) = 0;
};

// Called once per streaming producer thread.
using RowEncoderFactory = std::function<std::unique_ptr<RowEncoder>()>;

// Position eval samples an eligible turn (binlog::eligible_span), pre- or
// post-move per `post_move`. Max-move-per-lane samples any turn, endgame
// included, always pre-move.
std::unique_ptr<RowEncoder> make_position_eval_row_encoder(const InputEncodingSpec& spec,
                                                           bool post_move);
std::unique_ptr<RowEncoder> make_max_move_per_lane_row_encoder(const InputEncodingSpec& spec);

}  // namespace binlog
}  // namespace scribblez
