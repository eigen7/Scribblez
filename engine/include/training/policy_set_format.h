#pragma once

// The format version of the policy-set model (docs/plans/deep_bot_v0.md,
// "Versions"): its opponent-rack input (belief/opp_rack_summary.h, as the
// model reads it), its equity input, and each form's set-conditioning stage
// and outputs. The board-row and move encodings carry their own versions.
// Bump it whenever one of these changes; checkpoints and exports stamp it,
// and loaders accept only this version.

namespace scribblez {

// 2: the equity input.
// 3: normalized candidate vectors, the slot form's cosine pointer with a learned
//    scale, and the greedy form's linear gain head, in percentage points.
inline constexpr int kPolicySetFormatVersion = 3;

}  // namespace scribblez
