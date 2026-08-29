#pragma once

// Training-target (label) registry. The targets a model trains against are
// declared in exactly ONE place, the `AllTargets` alias at the bottom of this
// file, from which everything downstream is derived -- on-disk row layout,
// BlockDecoder emit code, FFI shape advertising, total label floats -- so
// adding, removing, or reordering one is a one-line edit.
//
// Each target struct must expose:
//   * static constexpr const char* kName  -- display / FFI name
//   * static constexpr int kDims[]        -- tensor shape (any rank >= 1)
//   * static void encode(const EncodeContext&, float* out)
//                                         -- writes product(kDims) floats
//
// Targets are emitted in declaration order.

#include "encoding/encode_context.h"
#include "training/footprint.h"

#include <array>
#include <cstddef>

namespace scribblez {

// ---------- per-target structs -----------------------------------------

struct WldTarget {
  static constexpr const char* kName = "wld";
  static constexpr int kDims[] = {3};  // [win, draw, loss]
  static void encode(const EncodeContext& v, float* out);
};

struct ScoreDiffTarget {
  // The observed final differential, the active player's final score minus the
  // opponent's. The head predicts a Gaussian over it (see
  // kScoreDiffOutputFloats) and trains by negative log-likelihood, which scores
  // that pair against this one stored float.
  static constexpr const char* kName = "score_diff";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

// The four placement targets are the FOOTPRINT CLASS INDEX of a next move (see
// training/footprint.h) -- a single class in [0, kFootprintClasses), stored as a
// float and read back as the label for a masked softmax-CE head, replacing the
// former per-cell Bernoulli maps. Emitted in the (optionally diagonally flipped)
// `apply_flip` frame the input encoder used, so the class stays aligned with the
// spatial planes. Each is paired with a legality mask target below.
//
// PLAYS heads (opp_next, self_next): the played footprint, or kPassClass for an
// absent / EXCHANGE / PASS move. WIN heads (opp_win, self_win): the conjunction
// Pr[footprint & that-seat-wins] -- the footprint if that seat went on to win
// (draws counting as not winning), else kExtraClass (the "not-win" outcome), so
// the head is a proper distribution over {footprints} u {pass} u {not-win}.

struct OppNextPlacementTarget {
  // The opponent's next move as a footprint class, in the `apply_flip` frame.
  static constexpr const char* kName = "opp_next_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfNextPlacementTarget {
  // OppNextPlacementTarget's mover-side sibling, over the mover's own next move
  // from the sampled snapshot. The marginal-occupancy partner of
  // SelfWinPlacementTarget, letting the network separate "plays there often"
  // from "wins when playing there" (see docs/sim_residual_feedback.md).
  static constexpr const char* kName = "self_next_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

struct OppWinPlacementTarget {
  // OppNextPlacementTarget conjoined with the opponent going on to win: the
  // footprint class if the opponent won, else kExtraClass. An "opponent danger"
  // signal over move footprints whose occurrence is associated with losing (see
  // docs/sim_residual_feedback.md).
  static constexpr const char* kName = "opp_win_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfWinPlacementTarget {
  // OppWinPlacementTarget's mover-side sibling -- a "self opportunity" signal
  // over the footprints of the mover's winning follow-ups (see
  // docs/sim_residual_feedback.md).
  static constexpr const char* kName = "self_win_placement";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

// The per-head legality masks over the footprint classes, one float per class
// (1.0 = keep, 0.0 = drive to -inf before the softmax), recomputed on replay
// from the sampled board so the masked softmax-CE never spends probability on a
// structurally illegal footprint. Sound over-approximations (see
// training/footprint_mask.h): opp heads exact-ish from the current board (the
// opponent moves next), self heads opp-move-invariant (the mover plays two plies
// out). The loss additionally force-keeps the target class, so a mask can never
// zero out the very footprint it is scored against (-log(0) guard).

struct OppNextPlacementMaskTarget {
  static constexpr const char* kName = "opp_next_placement_mask";
  static constexpr int kDims[] = {kFootprintClasses};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfNextPlacementMaskTarget {
  static constexpr const char* kName = "self_next_placement_mask";
  static constexpr int kDims[] = {kFootprintClasses};
  static void encode(const EncodeContext& v, float* out);
};

struct OppWinPlacementMaskTarget {
  static constexpr const char* kName = "opp_win_placement_mask";
  static constexpr int kDims[] = {kFootprintClasses};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfWinPlacementMaskTarget {
  static constexpr const char* kName = "self_win_placement_mask";
  static constexpr int kDims[] = {kFootprintClasses};
  static void encode(const EncodeContext& v, float* out);
};

// ---------- TargetList: derives everything else from the pack ----------

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

  // In declaration order.
  static constexpr std::array<int, size> floats_per_target = {detail::target_floats<Ts>()...};

  // One target after the next, in declaration order.
  static void encode_all(const EncodeContext& v, float* out);
};

// ---------- The single source of truth ---------------------------------

using AllTargets =
  TargetList<WldTarget, ScoreDiffTarget, OppNextPlacementTarget, SelfNextPlacementTarget,
             OppWinPlacementTarget, SelfWinPlacementTarget, OppNextPlacementMaskTarget,
             SelfNextPlacementMaskTarget, OppWinPlacementMaskTarget, SelfWinPlacementMaskTarget>;

// Constants derived from AllTargets, so code that just wants a size need not
// mention the template or the target struct.
inline constexpr int kNumLabelHeads = AllTargets::size;
inline constexpr int kLabelFloats = AllTargets::total_floats;

inline constexpr int kWldFloats = detail::target_floats<WldTarget>();
inline constexpr int kScoreDiffFloats = detail::target_floats<ScoreDiffTarget>();
// The score-diff head's OUTPUT width -- the Gaussian's mean and standard
// deviation -- as against kScoreDiffFloats, the target the NLL scores them on.
inline constexpr int kScoreDiffOutputFloats = 2;

// The four placement heads' class index (1 float) and legality-mask
// (kFootprintClasses floats) target widths.
inline constexpr int kPlacementClassFloats = detail::target_floats<OppNextPlacementTarget>();
inline constexpr int kPlacementMaskFloats = detail::target_floats<OppNextPlacementMaskTarget>();
static_assert(kPlacementClassFloats == 1);
static_assert(kPlacementMaskFloats == kFootprintClasses);

}  // namespace scribblez

#include "inlines/training/training_targets.inl"
