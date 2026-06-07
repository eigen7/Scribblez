#pragma once

// Training-target (label) encoder for the win-probability model. A label row
// is a fixed-width float vector containing both the WLD one-hot (for the
// 3-class cross-entropy head) and the signed final score difference (for the
// Huber-regression head).
//
// Layout (in order, contiguous floats):
//
//   [0] win  | 1.0 if the active player won, else 0.0
//   [1] draw | 1.0 if the active player drew,  else 0.0
//   [2] loss | 1.0 if the active player lost,  else 0.0
//   [3] score_diff = final_score_active - final_score_opp
//
// The WLD entries are mutually exclusive and sum to 1.0.

#include <cstdint>

namespace scribblez {
namespace binlog {

inline constexpr int kWldFloats = 3;
inline constexpr int kScoreDiffFloats = 1;
inline constexpr int kLabelFloats = kWldFloats + kScoreDiffFloats;  // 4

// Encode the label for one position. `out` must point to at least
// kLabelFloats writeable floats; previous contents are overwritten.
//
// `score_active_final` and `score_opp_final` are the game's final cumulative
// scores from the active player's POV at the sampled position.
void encode_labels(int score_active_final, int score_opp_final, float* out);

}  // namespace binlog
}  // namespace scribblez
