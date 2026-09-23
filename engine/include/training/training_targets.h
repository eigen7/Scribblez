#pragma once

// The label registry for position evaluation training rows. The targets are
// declared in one place, the AllTargets alias below. The row's label layout,
// the decoded row width and the FFI shape table all derive from it, so adding,
// removing or reordering a target is a one-line edit there.
//
// Each target struct exposes:
//   static constexpr const char* kName   display / FFI name
//   static constexpr int kDims[]         tensor shape (any rank >= 1)
//   static void encode(const EncodeContext&, float* out)
//                                        writes product(kDims) floats
//
// Targets are laid out in declaration order.

#include "encoding/encode_context.h"
#include "training/footprint.h"

#include <array>
#include <cstddef>

namespace scribblez {

struct WldTarget {
  static constexpr const char* kName = "wld";
  static constexpr int kDims[] = {3};  // [win, draw, loss]
  static void encode(const EncodeContext& v, float* out);
};

struct ScoreDiffTarget {
  // The final score differential, active player minus opponent. The head
  // predicts its mean and standard deviation (kScoreDiffOutputFloats), each
  // trained by a Huber regression against this one float (see ScoreDiffHead in
  // py/scribblez/position_eval/model.py).
  static constexpr const char* kName = "score_diff";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

// Each placement target is the footprint class (training/footprint.h) of a next
// move: one index in [0, kFootprintClasses) stored as a float, the label for a
// masked softmax cross-entropy head. The move must be in the sampled board's
// frame (asserted), so the class lines up with the spatial planes. Each side's
// heads pair with that side's mask target below. docs/plans/sim_residual_feedback.md
// motivates the four heads.
//
// Plays heads (opp_next, self_next): the played footprint, or kPassClass for an
// EXCHANGE, a PASS, or no move. Win heads (opp_win, self_win): the played
// footprint if that seat went on to win (a draw is not a win), else
// kExtraClass, the not-win outcome. A win head is thus a proper distribution
// over footprints, pass and not-win, whose mass on a footprint is
// Pr[that footprint is played and that seat wins].

struct OppNextPlacementTarget {
  static constexpr const char* kName = "opp_next_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfNextPlacementTarget {
  // The mover's own next move, after the opponent's. Paired with
  // SelfWinPlacementTarget, it lets the network separate "plays there often"
  // from "wins when playing there".
  static constexpr const char* kName = "self_next_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

struct OppWinPlacementTarget {
  // An "opponent danger" signal: where the opponent plays when they go on to
  // win.
  static constexpr const char* kName = "opp_win_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfWinPlacementTarget {
  // A "self opportunity" signal: where the mover plays next when they go on to
  // win.
  static constexpr const char* kName = "self_win_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

// Per-side legality masks over the footprint classes, one float per class
// (1.0 = keep, 0.0 = drive to -inf before the softmax), so the masked softmax
// never spends probability on a structurally illegal footprint. They are sound
// over-approximations (training/footprint_mask.h): close to exact for the opponent, who
// moves next on this board, and invariant to the opponent's move for the mover,
// who plays two plies out. The loss also always keeps the target class, so a
// mask gap cannot produce -log(0).
//
// There is one mask per side rather than per head because a side's plays and
// win heads differ only at kExtraClass. The masks carry the plays-head form
// (kExtraClass illegal), and the loss makes kExtraClass legal for the win head.
// This halves both the mask-building cost on the replay path and the mask bytes
// in the row.

struct OppPlacementMaskTarget {
  static constexpr const char* kName = "opp_placement_mask";
  static constexpr int kDims[] = {kFootprintClasses};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfPlacementMaskTarget {
  static constexpr const char* kName = "self_placement_mask";
  static constexpr int kDims[] = {kFootprintClasses};
  static void encode(const EncodeContext& v, float* out);
};

namespace detail {
template <typename T>
constexpr int target_floats() {
  int n = 1;
  for (int d : T::kDims) n *= d;
  return n;
}
}  // namespace detail

template <typename... Ts>
struct TargetList {
  static constexpr std::size_t size = sizeof...(Ts);
  static constexpr int total_floats = (0 + ... + detail::target_floats<Ts>());

  static constexpr std::array<int, size> floats_per_target = {detail::target_floats<Ts>()...};

  static void encode_all(const EncodeContext& v, float* out);
};

using AllTargets =
  TargetList<WldTarget, ScoreDiffTarget, OppNextPlacementTarget, SelfNextPlacementTarget,
             OppWinPlacementTarget, SelfWinPlacementTarget, OppPlacementMaskTarget,
             SelfPlacementMaskTarget>;

inline constexpr int kNumLabelHeads = AllTargets::size;
inline constexpr int kLabelFloats = AllTargets::total_floats;

inline constexpr int kWldFloats = detail::target_floats<WldTarget>();
inline constexpr int kScoreDiffFloats = detail::target_floats<ScoreDiffTarget>();
// The score-diff head's output width (the Gaussian's mean and standard
// deviation), as against kScoreDiffFloats, the target width.
inline constexpr int kScoreDiffOutputFloats = 2;

inline constexpr int kPlacementClassFloats = detail::target_floats<OppNextPlacementTarget>();
inline constexpr int kPlacementMaskFloats = detail::target_floats<OppPlacementMaskTarget>();
static_assert(kPlacementClassFloats == 1);
static_assert(kPlacementMaskFloats == kFootprintClasses);

}  // namespace scribblez

#include "inlines/training/training_targets.inl"
