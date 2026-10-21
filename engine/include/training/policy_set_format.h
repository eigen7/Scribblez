#pragma once

// The format version of the policy-set model (docs/plans/deep_bot_v0.md,
// "Versions"): its opponent-rack input (belief/opp_rack_summary.h, as the
// model reads it) and each form's set-conditioning stage and outputs. The
// board-row and move encodings carry their own versions. Bump it whenever one
// of these changes; checkpoints and exports stamp it, and loaders accept only
// this version.

namespace scribblez {

inline constexpr int kPolicySetFormatVersion = 1;

}  // namespace scribblez
