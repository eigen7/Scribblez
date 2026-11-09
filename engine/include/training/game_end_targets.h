#pragma once

// Row targets from EncodeContext::game_end (game/game_end.h), from the POV
// player's side: the end-of-game heads of docs/plans/deep_bot.md ("The search
// network"). Near the end of a game, who goes out and what the other player is
// left holding can decide it while the board barely changes, which the maps
// do not show.

#include "encoding/encode_context.h"

namespace scribblez {

// Who goes out, as a class index stored as a float: 0 the POV player, 1 the
// opponent, 2 nobody (the game ends on scoreless turns), or -1 for a game cut
// short, which the loss masks.
struct GoingOutTarget {
  static constexpr const char* kName = "going_out";
  static constexpr int kDims[] = {1};
  static constexpr int kClasses = 3;
  static constexpr int kUnlabelled = -1;
  static void encode(const EncodeContext& v, float* out);
};

// The net end-of-game rack adjustment from the POV player's side (GameEnd::net):
// what the player who went out collected from the other's rack, signed by who
// did; the opponent's leftover value minus the POV player's when nobody did.
// Masked like GoingOutTarget.
struct EndPointsTarget {
  static constexpr const char* kName = "end_points";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

}  // namespace scribblez
