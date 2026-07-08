#pragma once

// Training-target (label) registry. The set of targets the model is
// trained against is declared in exactly ONE place: the `AllTargets`
// alias near the bottom of this file. Adding, removing, or reordering a
// target is a one-line edit; everything downstream -- on-disk row layout,
// BlockDecoder emit code, FFI shape advertising, total label floats --
// is derived from that single list.
//
// Each target struct must expose:
//   * static constexpr const char* kName  -- display / FFI name
//   * static constexpr int kDims[]        -- tensor shape (any rank >= 1)
//   * static void encode(const EncodeContext&, float* out)
//                                         -- writes product(kDims) floats
//
// Targets are emitted into the row in declaration order; offsets are
// computed by TargetList::encode_all().

#include "scribblez/encode_context.h"

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
  // Regression target for the score-differential head: the final differential,
  // the active player's final score minus the opponent's. The head predicts the
  // mean and standard deviation of this differential (a Gaussian, see
  // kScoreDiffOutputFloats) and is trained by Gaussian negative log-likelihood,
  // so the single stored target float is just the observed differential the NLL
  // scores against.
  static constexpr const char* kName = "score_diff";
  static constexpr int kDims[] = {1};
  static void encode(const EncodeContext& v, float* out);
};

struct OppNextPlacementTarget {
  // 15x15 binary mask: cell (r,c) is 1.0 iff the opponent placed a tile
  // on (r,c) on their next move. All zeros if the next move is missing,
  // EXCHANGE, or PASS. Transposed across the main diagonal when the context's
  // apply_flip is true (so it stays aligned with the InputEncoder's spatial
  // planes).
  static constexpr int kSide = 15;
  static constexpr const char* kName = "opp_next_placement";
  static constexpr int kDims[] = {kSide, kSide};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfNextPlacementTarget {
  // The mover-side sibling of OppNextPlacementTarget: cell (r,c) is 1.0 iff
  // the mover placed a tile on (r,c) on their own next move from the sampled
  // snapshot. All zeros if that move is missing, EXCHANGE, or PASS. Transposed
  // when apply_flip is true. Serves as the marginal-occupancy partner of
  // SelfWinPlacementTarget, letting the network separate "plays there often"
  // from "wins when playing there" (see docs/sim_residual_feedback.md).
  static constexpr int kSide = 15;
  static constexpr const char* kName = "self_next_placement";
  static constexpr int kDims[] = {kSide, kSide};
  static void encode(const EncodeContext& v, float* out);
};

struct OppWinPlacementTarget {
  // 15x15 conjunction mask: cell (r,c) is 1.0 iff the opponent placed a tile
  // on (r,c) on their next move AND the opponent went on to win the game
  // (draws count as not winning). The head trained against it predicts, per
  // square, Pr[opponent-next-move-occupies AND opponent-wins] -- an
  // "opponent danger" map marking spots whose occupation by the opponent is
  // associated with losing (see docs/sim_residual_feedback.md). All zeros when
  // the next move is missing, EXCHANGE, or PASS, or when the opponent did not
  // win. Transposed when apply_flip is true.
  static constexpr int kSide = 15;
  static constexpr const char* kName = "opp_win_placement";
  static constexpr int kDims[] = {kSide, kSide};
  static void encode(const EncodeContext& v, float* out);
};

struct SelfWinPlacementTarget {
  // The mover-side sibling of OppWinPlacementTarget: cell (r,c) is 1.0 iff the
  // mover placed a tile on (r,c) on their own next move from the sampled
  // snapshot AND went on to win the game -- a "self opportunity" map of hot
  // spots for the mover's follow-up (see docs/sim_residual_feedback.md). All
  // zeros when that move is missing, EXCHANGE, or PASS, or when the mover did
  // not win. Transposed when apply_flip is true.
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

  // Per-target float counts in declaration order.
  static constexpr std::array<int, size> floats_per_target = {detail::target_floats<Ts>()...};

  // Writes total_floats floats starting at `out`, one target after the
  // next in declaration order.
  static void encode_all(const EncodeContext& v, float* out);
};

// ---------- The single source of truth ---------------------------------

using AllTargets =
  TargetList<WldTarget, ScoreDiffTarget, OppNextPlacementTarget, SelfNextPlacementTarget,
             OppWinPlacementTarget, SelfWinPlacementTarget>;

// Convenience constants derived from AllTargets. These exist so code that
// just wants "how many label floats in a row?" doesn't have to mention the
// template.
inline constexpr int kNumLabelHeads = static_cast<int>(AllTargets::size);
inline constexpr int kLabelFloats = AllTargets::total_floats;

// Per-target convenience aliases. Downstream code (tests, FFI, layout
// docs) can refer to a specific target's size without naming the struct.
inline constexpr int kWldFloats = detail::target_floats<WldTarget>();
inline constexpr int kScoreDiffFloats = detail::target_floats<ScoreDiffTarget>();  // 1 (regression)
// The score-diff head's OUTPUT width: the mean and standard deviation of the
// final differential (a Gaussian). Distinct from kScoreDiffFloats, which is the
// 1-float regression target the NLL loss scores those two against.
inline constexpr int kScoreDiffOutputFloats = 2;
inline constexpr int kOppNextPlacementFloats = detail::target_floats<OppNextPlacementTarget>();
inline constexpr int kOppNextPlacementSide = OppNextPlacementTarget::kSide;
inline constexpr int kSelfNextPlacementFloats = detail::target_floats<SelfNextPlacementTarget>();
inline constexpr int kOppWinPlacementFloats = detail::target_floats<OppWinPlacementTarget>();
inline constexpr int kSelfWinPlacementFloats = detail::target_floats<SelfWinPlacementTarget>();

}  // namespace scribblez

#include "inlines/scribblez/training_targets.inl"
