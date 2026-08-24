#pragma once

// The score differential's input representation, shared by every model that
// takes one: the position evaluation model's kScoreDiff scalar block
// (input_encoder.h) and the move set evaluation model's resultant-diff move
// feature (move_set_encoder.h). One owner so the two never drift apart --
// their differentials are on the same scale and mean the same thing, and a
// candidate's resultant differential is meant to be comparable to the board
// trunk's.
//
// The representation is the raw normalized scalar followed by a compact
// nonlinear basis. The raw scalar keeps full arithmetic resolution (nothing is
// binned away, and it stays first so a consumer can still read the
// differential back off the row). The basis exists because the raw scalar
// alone is too smooth: the map from current lead to final outcome is sharp and
// phase-conditional near a decided margin and nearly flat mid-game, and a
// single scalar makes that a smoothness-fighting learn. Measured consequence
// with the scalar alone: the teacher under-regresses current leads toward
// realized finals by ~0.21 points per point of lead, a structural
// miscalibration stable across checkpoints and training ages
// (docs/pov_calibration_bias.md).
//
// The basis is kScoreDiffBasisFloats Gaussian bumps spaced uniformly over the
// squashed coordinate
//
//     u(d) = d / (|d| + kScoreDiffBasisSoftening)
//
// which is odd, monotone, and maps the unbounded differential into (-1, 1).
// Uniform spacing in u is dense in points near 0 and sparse out in the tail --
// centers land near 0, +-5, +-12, +-23, +-40, +-75, +-180 points, with the
// endpoint bumps acting as "decided lead" saturating features. That is the
// resolution profile the differential actually needs: a two-point swing
// matters at +-15 and does not at +-180. Adjacent overlapping bumps also let a
// linear readout build a steep ramp anywhere in range, which is what a phase-
// gated win/loss boundary asks for.

namespace scribblez {

// The differential's normalizing scale (points), applied to the raw scalar.
inline constexpr float kScoreDiffInputScale = 100.0f;

// Bumps in the basis. Odd, so one center sits exactly at a tied score.
inline constexpr int kScoreDiffBasisFloats = 15;

// Points at which the squashed coordinate reaches +-1/2 -- the knob setting how
// much of the basis is spent near a close game.
inline constexpr float kScoreDiffBasisSoftening = 30.0f;

// Raw scalar + basis: the floats every score-differential feature block holds.
inline constexpr int kScoreDiffFeatureFloats = 1 + kScoreDiffBasisFloats;

// Write the kScoreDiffFeatureFloats representation of `score_diff` (the POV
// player's score advantage, in points) at `out`.
void encode_score_diff_features(int score_diff, float* out);

}  // namespace scribblez
