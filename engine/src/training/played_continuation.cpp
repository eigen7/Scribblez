#include "training/played_continuation.h"

#include "encoding/encode_context.h"
#include "game/game_end.h"
#include "training/game_end_targets.h"

#include <algorithm>

namespace scribblez::policy_set {

namespace {

// Turn `turn`'s move of `g` in `transpose`'s frame.
Move played_move(const GameLog& g, int turn, bool transpose) {
  const Move& m = g.records[turn].move;
  return transpose ? m.transpose() : m;
}

}  // namespace

PlayedContinuation played_continuation(const GameLog& g, int turn, const Board& board) {
  const Move played = played_move(g, turn, board.transposed());
  Board after = board;
  if (played.type() == MoveType::PLAY) after.apply(played);
  PlayedContinuation out;
  out.fates = square_fates(g, turn, after);
  EncodeContext ctx;
  ctx.active_player = g.records[turn].player;
  ctx.game_end = game_end(g);
  GoingOutTarget::encode(ctx, &out.going_out);
  EndPointsTarget::encode(ctx, &out.end_points);
  return out;
}

int played_index(const GameLog& g, int turn, bool transpose, const std::vector<Move>& moves) {
  const Move played = played_move(g, turn, transpose);
  const auto it = std::find(moves.begin(), moves.end(), played);
  return it == moves.end() ? -1 : int(it - moves.begin());
}

}  // namespace scribblez::policy_set
