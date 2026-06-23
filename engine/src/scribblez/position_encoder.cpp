#include "scribblez/position_encoder.h"

#include "scribblez/input_encoder.h"
#include "scribblez/training_targets.h"

#include <cassert>
#include <cstdint>

namespace scribblez {
namespace binlog {

namespace {

// Replace all glyphs the move places with empty tiles in `rack` (PLAY: the
// tiles being placed; EXCHANGE: the tiles being swapped out).
void remove_played_or_exchanged(Rack& rack, const Move& m) {
  const int n = m.num_glyphs();
  for (int i = 0; i < n; ++i) {
    [[maybe_unused]] bool ok = rack.remove(m.glyph(i).rack_tile());
    assert(ok);
  }
}

}  // namespace

int PositionEncoder::replay_to_sampled(const GameLog& g, int sampled_turn, bool post_move) {
  enc_ = GameStateEncoder{g.initial_scores};
  racks_[0] = g.initial_racks[0];
  racks_[1] = g.initial_racks[1];

  // Replay turns [0, sampled_turn) silently (apply move + remove played tiles +
  // add drawn tiles). The mover of turn k alternates from active=0.
  for (int k = 0; k < sampled_turn; ++k) {
    const int mover = enc_.active_player();
    const Move& move = g.records[k].move;
    if (move.type() == MoveType::PLAY || move.type() == MoveType::EXCHANGE) {
      remove_played_or_exchanged(racks_[mover], move);
    }
    enc_.apply_move(move);
    for (Tile t : g.records[k].drawn.tiles()) {
      if (t.is_empty()) break;
      racks_[mover].add(t);
    }
  }

  // At this point enc_ holds the pre-move state for turn `sampled_turn`.
  const int mover = enc_.active_player();
  if (post_move) {
    const Move& move = g.records[sampled_turn].move;
    if (move.type() == MoveType::PLAY || move.type() == MoveType::EXCHANGE) {
      remove_played_or_exchanged(racks_[mover], move);
    }
    enc_.apply_move(move);
    // racks_[mover] is now the pre-draw rack; do NOT add records[sampled_turn].drawn.
  }
  return mover;
}

void PositionEncoder::encode_row(const GameLog& g, int sampled_turn, bool post_move, bool flip,
                                 float* out_row) {
  const int mover = replay_to_sampled(g, sampled_turn, post_move);
  enc_.encode_input(mover, racks_[mover], flip, out_row);

  // The "opponent next move" -- whichever upcoming turn was played by
  // (1 - mover). Pre-move: that's turn sampled_turn+1. Post-move: enc_ has
  // advanced past sampled_turn (active is now opp), whose next move is also at
  // sampled_turn+1.
  const int next_idx = sampled_turn + 1;
  TargetInputs view{};
  if (next_idx < g.num_records) {
    view.next_move = g.records[next_idx].move;
    view.has_next_move = true;
  }
  view.active_player = mover;
  view.final_score_p0 = g.final_scores[0];
  view.final_score_p1 = g.final_scores[1];
  view.apply_flip = flip;
  AllTargets::encode_all(view, out_row + kInputFloats);
}

void PositionEncoder::encode_score_diff_sweep(const GameLog& g, int sampled_turn, bool post_move,
                                              int diff_lo, int diff_hi, float* out) {
  const int mover = replay_to_sampled(g, sampled_turn, post_move);
  int64_t i = 0;
  for (int d = diff_lo; d <= diff_hi; ++d, ++i) {
    enc_.encode_input_with_score_diff(mover, racks_[mover], d, /*apply_flip=*/false,
                                      out + i * kInputFloats);
  }
}

}  // namespace binlog
}  // namespace scribblez
