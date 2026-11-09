#include "training/game_end_targets.h"

namespace scribblez {

void GoingOutTarget::encode(const EncodeContext& v, float* out) {
  const GameEnd& end = v.game_end;
  switch (end.kind) {
    case GameEndKind::kWentOut:
      *out = end.went_out == v.active_player ? 0.0f : 1.0f;
      return;
    case GameEndKind::kStalemate:
      *out = 2.0f;
      return;
    case GameEndKind::kUnfinished:
      *out = float(kUnlabelled);
      return;
  }
}

void EndPointsTarget::encode(const EncodeContext& v, float* out) {
  *out = float(v.game_end.net(v.active_player));
}

}  // namespace scribblez
