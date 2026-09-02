#include "encoding/row_encoder.h"

#include "data/binary_log.h"  // pick_sampled_turn, pick_any_turn
#include "encoding/position_encoder.h"
#include "training/max_move_per_lane_task.h"
#include "training/training_task.h"

namespace scribblez {
namespace binlog {

namespace {

// Win-probability rows: bag-nonempty turn sampling + PositionEvalTask encoding.
class PositionEvalRowEncoder : public RowEncoder {
 public:
  PositionEvalRowEncoder(const InputEncodingSpec& spec, bool post_move)
      : post_move_(post_move), row_floats_(PositionEvalTask::row_floats(spec)), pos_(spec) {}

  int row_floats() const override { return row_floats_; }

  int pick_turn(const GameLog& view, std::mt19937_64& rng) override {
    return pick_sampled_turn(view, rng);
  }

  void encode(const GameLog& view, int turn, bool transpose, float* dest) override {
    pos_.encode_row<PositionEvalTask>(view, turn, post_move_, transpose, dest);
  }

 private:
  bool post_move_;
  int row_floats_;
  PositionEncoder pos_;
};

// Max-move-per-lane rows: uniform all-turn sampling + MaxMovePerLaneTask encoding (always
// pre-move; the lexicon drives the per-lane move enumeration).
class MaxMovePerLaneRowEncoder : public RowEncoder {
 public:
  explicit MaxMovePerLaneRowEncoder(const InputEncodingSpec& spec) : pos_(spec) {}

  int row_floats() const override { return MaxMovePerLaneTask::kRowFloats; }

  int pick_turn(const GameLog& view, std::mt19937_64& rng) override {
    return pick_any_turn(view, rng);
  }

  void encode(const GameLog& view, int turn, bool transpose, float* dest) override {
    pos_.encode_row<MaxMovePerLaneTask>(view, turn, /*post_move=*/false, transpose, dest);
  }

 private:
  PositionEncoder pos_;
};

}  // namespace

std::unique_ptr<RowEncoder> make_position_eval_row_encoder(const InputEncodingSpec& spec,
                                                           bool post_move) {
  return std::make_unique<PositionEvalRowEncoder>(spec, post_move);
}

std::unique_ptr<RowEncoder> make_max_move_per_lane_row_encoder(const InputEncodingSpec& spec) {
  return std::make_unique<MaxMovePerLaneRowEncoder>(spec);
}

}  // namespace binlog
}  // namespace scribblez
