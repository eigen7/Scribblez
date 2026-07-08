#pragma once

#include "scribblez/game.h"
#include "scribblez/input_encoding_spec.h"

#include <functional>
#include <memory>
#include <random>

namespace scribblez {
namespace binlog {

// Per-worker policy that turns a finished self-play game into one training row:
// choose a sampled turn, then encode that position's row for a specific
// TrainingTask. Stateful (owns a PositionEncoder), so the streaming producer
// builds one per worker thread; the RingBufferGameSink drives it and stays
// task-agnostic.
class RowEncoder {
 public:
  virtual ~RowEncoder() = default;

  // Width of the row this encoder writes (the task's kRowFloats).
  virtual int row_floats() const = 0;

  // Choose a turn of `view` to sample, or -1 to drop the game (no eligible
  // turn). Called before a ring slot is claimed, so a dropped game never holds
  // one.
  virtual int pick_turn(const GameLog& view, std::mt19937_64& rng) = 0;

  // Encode the row for `view` at `turn` into `dest` (row_floats() floats);
  // `flip` applies the diagonal-symmetry augmentation.
  virtual void encode(const GameLog& view, int turn, bool flip, float* dest) = 0;
};

// Builds a fresh per-worker RowEncoder; the streaming producer calls it once per
// thread.
using RowEncoderFactory = std::function<std::unique_ptr<RowEncoder>()>;

// Built-in row encoders. Both take the run's input-encoding spec (lexicon +
// feature blocks).
//   * position-eval: the win-probability task; samples bag-nonempty turns; the
//     `post_move` flag picks the pre-move vs post-move snapshot.
//   * max-move-per-lane: the per-lane best-move task; samples any turn (incl. endgames)
//     pre-move, and additionally enumerates legal moves with the spec's lexicon.
std::unique_ptr<RowEncoder> make_position_eval_row_encoder(const InputEncodingSpec& spec,
                                                           bool post_move);
std::unique_ptr<RowEncoder> make_max_move_per_lane_row_encoder(const InputEncodingSpec& spec);

}  // namespace binlog
}  // namespace scribblez
