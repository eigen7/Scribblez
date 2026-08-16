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

struct OppNextPlacementTarget {
  // Cell (r,c) is 1.0 iff the opponent placed a tile there on their next move.
  // All zeros if that move is missing, EXCHANGE, or PASS. Transposed when
  // apply_flip is set, staying aligned with the input's spatial planes.
  static constexpr int kSide = 15;
  static constexpr const char* kName = "opp_next_placement";
  static constexpr int kDims[] = {kSide, kSide};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfNextPlacementTarget {
  // OppNextPlacementTarget's mover-side sibling, over the mover's own next move
  // from the sampled snapshot. The marginal-occupancy partner of
  // SelfWinPlacementTarget, letting the network separate "plays there often"
  // from "wins when playing there" (see docs/sim_residual_feedback.md).
  static constexpr int kSide = 15;
  static constexpr const char* kName = "self_next_placement";
  static constexpr int kDims[] = {kSide, kSide};
  static void encode(const EncodeContext& v, float* out);
};

struct OppWinPlacementTarget {
  // OppNextPlacementTarget conjoined with the opponent going on to win, draws
  // counting as not winning. The head trained against it predicts, per square,
  // Pr[opponent-next-move-occupies AND opponent-wins] -- an "opponent danger"
  // map of spots whose occupation is associated with losing (see
  // docs/sim_residual_feedback.md).
  static constexpr int kSide = 15;
  static constexpr const char* kName = "opp_win_placement";
  static constexpr int kDims[] = {kSide, kSide};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfWinPlacementTarget {
  // OppWinPlacementTarget's mover-side sibling -- a "self opportunity" map of
  // hot spots for the mover's follow-up (see docs/sim_residual_feedback.md).
  static constexpr int kSide = 15;
  static constexpr const char* kName = "self_win_placement";
  static constexpr int kDims[] = {kSide, kSide};
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
             OppWinPlacementTarget, SelfWinPlacementTarget>;

// Constants derived from AllTargets, so code that just wants a size need not
// mention the template or the target struct.
inline constexpr int kNumLabelHeads = AllTargets::size;
inline constexpr int kLabelFloats = AllTargets::total_floats;

inline constexpr int kWldFloats = detail::target_floats<WldTarget>();
inline constexpr int kScoreDiffFloats = detail::target_floats<ScoreDiffTarget>();
// The score-diff head's OUTPUT width -- the Gaussian's mean and standard
// deviation -- as against kScoreDiffFloats, the target the NLL scores them on.
inline constexpr int kScoreDiffOutputFloats = 2;
inline constexpr int kOppNextPlacementFloats = detail::target_floats<OppNextPlacementTarget>();
inline constexpr int kOppNextPlacementSide = OppNextPlacementTarget::kSide;
inline constexpr int kSelfNextPlacementFloats = detail::target_floats<SelfNextPlacementTarget>();
inline constexpr int kOppWinPlacementFloats = detail::target_floats<OppWinPlacementTarget>();
inline constexpr int kSelfWinPlacementFloats = detail::target_floats<SelfWinPlacementTarget>();

}  // namespace scribblez

#include "inlines/training/training_targets.inl"
