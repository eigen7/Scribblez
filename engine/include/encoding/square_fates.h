#pragma once

// What becomes of each empty square of a position over the rest of the game:
// who places a tile there, whether that move is a bingo, and the signed points
// of that move. These label the continuation heads of docs/plans/deep_bot.md
// ("The search network"): five-class ownership, and per-square points of the
// whole move through the square, not a share of it.
//
// The evaluator is the player who made the position's last move; signs and
// classes are from that player's side.

#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/game_log.h"

#include <array>
#include <cstdint>

namespace scribblez {

enum class SquareFate : int8_t {
  kFilled = -1,  // occupied at the position; no label
  kEvaluatorBingo = 0,
  kEvaluatorOther = 1,
  kOpponentBingo = 2,
  kOpponentOther = 3,
  kStaysEmpty = 4,
};

inline constexpr int kSquareFateClasses = 5;

struct SquareFates {
  std::array<SquareFate, kBoardCells> fate{};
  // The points of the move that fills the square, positive for the evaluator's
  // moves; 0 for a square that stays empty or is already filled.
  std::array<int16_t, kBoardCells> points{};
};

// The fates of `board`'s squares, where `board` is the position right after
// turn `turn`'s move of `g` (before its draw), in either frame.
SquareFates square_fates(const GameLog& g, int turn, const Board& board);

}  // namespace scribblez
