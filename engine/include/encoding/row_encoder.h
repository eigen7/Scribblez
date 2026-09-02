#pragma once

#include "encoding/input_encoder.h"
#include "game/game_log.h"

#include <functional>
#include <memory>
#include <random>

namespace scribblez {
namespace binlog {

// Per-worker policy turning a finished self-play game into one training row:
// choose a sampled turn, then encode that position for a specific TrainingTask.
// Stateful (it owns a PositionEncoder), so the streaming producer builds one
// per worker thread and its sink stays task-agnostic.
class RowEncoder {
 public:
  virtual ~RowEncoder() = default;

  virtual int row_floats() const = 0;

  // The turn of `view` to sample, or -1 to drop the game. Called before a ring
  // slot is claimed, so a dropped game never holds one.
  virtual int pick_turn(const GameLog& view, std::mt19937_64& rng) = 0;

  virtual void encode(const GameLog& view, int turn, bool transpose, float* dest) = 0;
};

// The streaming producer calls this once per thread.
using RowEncoderFactory = std::function<std::unique_ptr<RowEncoder>()>;

// The built-in row encoders. Position-eval samples bag-nonempty turns, with
// `post_move` picking the snapshot; max-move-per-lane samples any turn
// (endgames included) pre-move and enumerates legal moves with the spec's
// lexicon.
std::unique_ptr<RowEncoder> make_position_eval_row_encoder(const InputEncodingSpec& spec,
                                                           bool post_move);
std::unique_ptr<RowEncoder> make_max_move_per_lane_row_encoder(const InputEncodingSpec& spec);

}  // namespace binlog
}  // namespace scribblez
