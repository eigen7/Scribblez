#include "encoding/position_encoder.h"

#include "encoding/input_encoder.h"
#include "game/bag.h"
#include "util/assert.h"

#include <cstdint>
#include <cstring>

namespace scribblez {
namespace binlog {

namespace {

void remove_played_or_exchanged(Rack& rack, const Move& m) {
  const int n = m.num_glyphs();
  for (int i = 0; i < n; ++i) {
    bool ok = rack.remove(m.glyph(i).rack_tile());
    DEBUG_ASSERT(ok);
  }
}

// A move from the game log, which is in the untransposed frame, brought into
// the frame given by `transposed`.
Move in_frame(const Move& m, bool transposed) { return transposed ? m.transpose() : m; }

}  // namespace

Rack opp_leave_from_replay(const GameLog& g, int sampled_turn, const Rack& opp_rack_now) {
  // The opponent last moved at sampled_turn - 1 and has not moved since, so
  // everything they drew then is still on their rack.
  if (sampled_turn == 0) return Rack{};
  Rack leave = opp_rack_now;
  for (Tile t : g.records[sampled_turn - 1].drawn.tiles()) {
    if (t.is_empty()) break;
    const bool ok = leave.remove(t);
    RELEASE_ASSERT(ok);
  }
  return leave;
}

void encode_candidate_rows(const PositionEncoder& encoder, const GameLog& g, int turn, int mover,
                           const std::vector<Move>& candidates, float* out) {
  const GameStateEncoder& pre = encoder.enc();
  const InputEncodingSpec& spec = pre.spec();
  const Rack opp_leave =
    spec.opp_leave_input ? opp_leave_from_replay(g, turn, encoder.rack(1 - mover)) : Rack{};

  // Build the move-generation caches once on the shared board, so each
  // candidate's copy updates them incrementally instead of rebuilding them.
  pre.board().ensure_movegen_caches(*spec.dict);
  const size_t row_floats = input_floats(spec);
  for (size_t c = 0; c < candidates.size(); ++c) {
    encode_post_move_row(pre, mover, encoder.rack(mover), candidates[c], opp_leave,
                         out + c * row_floats);
  }
}

int PositionEncoder::bag_size() const {
  return Bag::kTotalTiles - enc_.board().num_tiles() - racks_[0].size() - racks_[1].size();
}

int PositionEncoder::replay_to_sampled(const GameLog& g, int sampled_turn, bool post_move) {
  enc_ = GameStateEncoder{spec_, g.initial_scores};
  racks_[0] = g.initial_racks[0];
  racks_[1] = g.initial_racks[1];

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

  const int mover = enc_.active_player();
  if (post_move) {
    const Move& move = g.records[sampled_turn].move;
    if (move.type() == MoveType::PLAY || move.type() == MoveType::EXCHANGE) {
      remove_played_or_exchanged(racks_[mover], move);
    }
    enc_.apply_move(move);
    // The post-move position precedes the draw, so records[sampled_turn].drawn
    // is not added.
  }

  return mover;
}

EncodeContext PositionEncoder::make_context(const GameLog& g, int sampled_turn, int mover,
                                            bool post_move) const {
  const bool transposed = enc_.board().transposed();
  EncodeContext ctx{};
  ctx.enc = &enc_;
  ctx.pov_rack = &racks_[mover];
  ctx.opp_known_leave = opp_leave_from_replay(g, sampled_turn, racks_[1 - mover]);
  ctx.active_player = mover;
  ctx.spec = spec_;

  // The opponent's next move is turn sampled_turn + 1 either way. The
  // mover's is the sampled turn itself pre-move, or the turn after the
  // opponent's reply post-move.
  const int opp_idx = sampled_turn + 1;
  if (opp_idx < g.num_records) {
    ctx.opp_next_move = in_frame(g.records[opp_idx].move, transposed);
    ctx.has_opp_next_move = true;
  }
  const int self_idx = post_move ? sampled_turn + 2 : sampled_turn;
  if (self_idx < g.num_records) {
    ctx.self_next_move = in_frame(g.records[self_idx].move, transposed);
    ctx.has_self_next_move = true;
  }
  ctx.final_score_p0 = g.final_scores[0];
  ctx.final_score_p1 = g.final_scores[1];
  return ctx;
}

void PositionEncoder::encode_score_diff_sweep(const GameLog& g, int sampled_turn, bool post_move,
                                              int diff_lo, int diff_hi, float* out) {
  const int mover = replay_to_sampled(g, sampled_turn, post_move);
  // Encode fully once, since that runs move generation, then copy the row and
  // overwrite only the score difference.
  const int64_t row_floats = input_floats(spec_);
  enc_.encode_input_with_score_diff(mover, racks_[mover], diff_lo, out);
  for (int64_t i = 1; i <= diff_hi - diff_lo; ++i) {
    float* row = out + i * row_floats;
    std::memcpy(row, out, sizeof(float) * size_t(row_floats));
    enc_.overwrite_score_diff(diff_lo + int(i), row);
  }
}

}  // namespace binlog
}  // namespace scribblez
