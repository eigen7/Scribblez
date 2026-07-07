#include "scribblez/position_encoder.h"

#include "scribblez/input_encoder.h"

#include <cassert>
#include <cstdint>
#include <cstring>

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

Rack opp_leave_from_replay(const GameLog& g, int sampled_turn, const Rack& opp_rack_now) {
  // The opponent of the mover at `sampled_turn` acted at sampled_turn - 1 (if
  // at all); every tile in records[sampled_turn - 1].drawn is a replenishment
  // still on their rack (they have not moved since), so subtracting the draws
  // from the current rack leaves exactly the retained leave.
  if (sampled_turn == 0) return Rack{};
  Rack leave = opp_rack_now;
  for (Tile t : g.records[sampled_turn - 1].drawn.tiles()) {
    if (t.is_empty()) break;
    [[maybe_unused]] const bool ok = leave.remove(t);
    assert(ok);
  }
  return leave;
}

int PositionEncoder::replay_to_sampled(const GameLog& g, int sampled_turn, bool post_move) {
  enc_ = GameStateEncoder{spec_, g.initial_scores};
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

EncodeContext PositionEncoder::make_context(const GameLog& g, int sampled_turn, int mover,
                                            bool post_move, bool flip) const {
  EncodeContext ctx{};
  ctx.enc = &enc_;
  ctx.pov_rack = &racks_[mover];
  ctx.opp_known_leave = opp_leave_from_replay(g, sampled_turn, racks_[1 - mover]);
  ctx.active_player = mover;
  ctx.apply_flip = flip;
  ctx.spec = spec_;

  // Each player's next move from the sampled snapshot. The opponent's is turn
  // sampled_turn+1 in both snapshot kinds (pre-move: the mover plays
  // sampled_turn and the opponent answers at +1; post-move: enc_ has advanced
  // past sampled_turn, and the opponent's reply is still at +1). The mover's
  // own next move is the sampled turn itself for a pre-move snapshot, and the
  // turn after the opponent's reply for a post-move one.
  const int opp_idx = sampled_turn + 1;
  if (opp_idx < g.num_records) {
    ctx.opp_next_move = g.records[opp_idx].move;
    ctx.has_opp_next_move = true;
  }
  const int self_idx = post_move ? sampled_turn + 2 : sampled_turn;
  if (self_idx < g.num_records) {
    ctx.self_next_move = g.records[self_idx].move;
    ctx.has_self_next_move = true;
  }
  ctx.final_score_p0 = g.final_scores[0];
  ctx.final_score_p1 = g.final_scores[1];
  return ctx;
}

void PositionEncoder::encode_score_diff_sweep(const GameLog& g, int sampled_turn, bool post_move,
                                              int diff_lo, int diff_hi, float* out) {
  const int mover = replay_to_sampled(g, sampled_turn, post_move);
  // Only the score-diff thermometer varies across the sweep, so the position
  // is fully encoded once (the expensive, move-generating part) and each
  // swept differential is stamped into a copy of that row.
  const int64_t row_floats = input_floats(spec_);
  enc_.encode_input_with_score_diff(mover, racks_[mover], diff_lo, /*apply_flip=*/false, out);
  for (int64_t i = 1; i <= diff_hi - diff_lo; ++i) {
    float* row = out + i * row_floats;
    std::memcpy(row, out, sizeof(float) * static_cast<size_t>(row_floats));
    enc_.overwrite_score_diff(diff_lo + static_cast<int>(i), row);
  }
}

}  // namespace binlog
}  // namespace scribblez
