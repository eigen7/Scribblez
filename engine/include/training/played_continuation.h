#pragma once

// What a game record labels at a pool position (docs/plans/deep_bot.md, "The
// search network", training data): the move the game played there is one of
// the pool's candidates, and the game's rest is one continuation after it,
// labelled from the mover's side on the board after it. The targets are the
// leaf network's (encoding/square_fates.h, training/game_end_targets.h), here
// attached to a candidate rather than to a position, so the search network's
// continuation heads pretrain on finished games.

#include "encoding/square_fates.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/move.h"

#include <vector>

namespace scribblez::policy_set {

struct PlayedContinuation {
  // On the post-move board, in the pre-move board's frame; the mover is the
  // evaluator.
  SquareFates fates;
  float going_out;   // GoingOutTarget's class, or its kUnlabelled
  float end_points;  // EndPointsTarget's net adjustment, from the mover's side
};

// The targets after turn `turn`'s move of `g`, where `board` is the position
// before it, in either frame.
PlayedContinuation played_continuation(const GameLog& g, int turn, const Board& board);

// The index of turn `turn`'s move among `moves`, which are in the natural
// frame or, with `transpose`, the transposed one; -1 when the pool holds no
// such candidate (a blank designation the pool pruned).
int played_index(const GameLog& g, int turn, bool transpose, const std::vector<Move>& moves);

}  // namespace scribblez::policy_set
