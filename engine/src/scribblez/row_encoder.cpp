#include "scribblez/row_encoder.h"

#include "scribblez/binary_log.h"  // pick_sampled_turn, pick_any_turn
#include "scribblez/max_move_per_lane_task.h"
#include "scribblez/position_encoder.h"
#include "scribblez/training_task.h"

namespace scribblez {
namespace binlog {

namespace {

// Win-probability rows: bag-nonempty turn sampling + PostMoveValueTask encoding.
class PostMoveValueRowEncoder : public RowEncoder {
 public:
  PostMoveValueRowEncoder(const Dictionary& dict, bool post_move)
      : post_move_(post_move), pos_(dict) {}

  int row_floats() const override { return PostMoveValueTask::kRowFloats; }

  int pick_turn(const GameLog& view, std::mt19937_64& rng) override {
    return pick_sampled_turn(view, rng);
  }

  void encode(const GameLog& view, int turn, bool flip, float* dest) override {
    pos_.encode_row<PostMoveValueTask>(view, turn, post_move_, flip, dest);
  }

 private:
  bool post_move_;
  PositionEncoder pos_;
};

// Max-move-per-lane rows: uniform all-turn sampling + MaxMovePerLaneTask encoding (always
// pre-move; the lexicon drives the per-lane move enumeration).
class MaxMovePerLaneRowEncoder : public RowEncoder {
 public:
  explicit MaxMovePerLaneRowEncoder(const Dictionary& dict) : pos_(dict) {}

  int row_floats() const override { return MaxMovePerLaneTask::kRowFloats; }

  int pick_turn(const GameLog& view, std::mt19937_64& rng) override {
    return pick_any_turn(view, rng);
  }

  void encode(const GameLog& view, int turn, bool flip, float* dest) override {
    pos_.encode_row<MaxMovePerLaneTask>(view, turn, /*post_move=*/false, flip, dest);
  }

 private:
  PositionEncoder pos_;
};

}  // namespace

std::unique_ptr<RowEncoder> make_post_move_value_row_encoder(const Dictionary& dict,
                                                             bool post_move) {
  return std::make_unique<PostMoveValueRowEncoder>(dict, post_move);
}

std::unique_ptr<RowEncoder> make_max_move_per_lane_row_encoder(const Dictionary& dict) {
  return std::make_unique<MaxMovePerLaneRowEncoder>(dict);
}

}  // namespace binlog
}  // namespace scribblez
