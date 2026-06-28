#include "scribblez/row_encoder.h"

#include "scribblez/binary_log.h"  // pick_sampled_turn, pick_any_turn
#include "scribblez/lexical_task.h"
#include "scribblez/position_encoder.h"
#include "scribblez/training_task.h"

namespace scribblez {
namespace binlog {

namespace {

// Win-probability rows: bag-nonempty turn sampling + PostMoveTask encoding.
class PostMoveRowEncoder : public RowEncoder {
 public:
  explicit PostMoveRowEncoder(bool post_move) : post_move_(post_move) {}

  int row_floats() const override { return PostMoveTask::kRowFloats; }

  int pick_turn(const GameLog& view, std::mt19937_64& rng) override {
    return pick_sampled_turn(view, rng);
  }

  void encode(const GameLog& view, int turn, bool flip, float* dest) override {
    pos_.encode_row<PostMoveTask>(view, turn, post_move_, flip, dest);
  }

 private:
  bool post_move_;
  PositionEncoder pos_;
};

// Lexical rows: uniform all-turn sampling + LexicalTask encoding (always
// pre-move; the lexicon drives the per-lane move enumeration).
class LexicalRowEncoder : public RowEncoder {
 public:
  explicit LexicalRowEncoder(const Dictionary& dict) : pos_(&dict) {}

  int row_floats() const override { return LexicalTask::kRowFloats; }

  int pick_turn(const GameLog& view, std::mt19937_64& rng) override {
    return pick_any_turn(view, rng);
  }

  void encode(const GameLog& view, int turn, bool flip, float* dest) override {
    pos_.encode_row<LexicalTask>(view, turn, /*post_move=*/false, flip, dest);
  }

 private:
  PositionEncoder pos_;
};

}  // namespace

std::unique_ptr<RowEncoder> make_post_move_row_encoder(bool post_move) {
  return std::make_unique<PostMoveRowEncoder>(post_move);
}

std::unique_ptr<RowEncoder> make_lexical_row_encoder(const Dictionary& dict) {
  return std::make_unique<LexicalRowEncoder>(dict);
}

}  // namespace binlog
}  // namespace scribblez
