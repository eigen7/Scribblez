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
//   * static void encode(const TargetInputs&, float* out)
//                                         -- writes product(kDims) floats
//
// Targets are emitted into the row in declaration order; offsets are
// computed by TargetList::encode_all().

#include "scribblez/move.h"

#include <array>
#include <cstddef>

namespace scribblez {

// A snapshot of the per-sample inputs needed to compute every target
// head. The caller (BlockDecoder, or a test) populates this from the
// replay state at the sampled position.
struct TargetInputs {
  Move next_move{};  // the opponent's response to the sampled position;
                     // only consulted when has_next_move is true
  bool has_next_move = false;
  int active_player = 0;  // 0 or 1 -- the POV for WLD / score_diff
  int final_score_p0 = 0;
  int final_score_p1 = 0;
  bool apply_flip = false;  // transpose spatial outputs when true

  int final_active() const { return active_player == 0 ? final_score_p0 : final_score_p1; }
  int final_opp() const { return active_player == 0 ? final_score_p1 : final_score_p0; }
};

// ---------- per-target structs -----------------------------------------

struct WldTarget {
  static constexpr const char* kName = "wld";
  static constexpr int kDims[] = {3};  // [win, draw, loss]
  static void encode(const TargetInputs& v, float* out);
};

struct ScoreDiffTarget {
  // Regression target for the score-differential head. The differential (active
  // player's final score minus the opponent's) is clipped (not rejected) to a
  // symmetric range; 400 covers essentially every realistic Scrabble game. The
  // head predicts the mean and standard deviation of this differential (a
  // Gaussian, see kScoreDiffOutputFloats) and is trained by Gaussian negative
  // log-likelihood, so the single stored target float is just the observed
  // (clipped) differential the NLL scores against.
  static constexpr int kClip = 400;
  static constexpr const char* kName = "score_diff";
  static constexpr int kDims[] = {1};
  static void encode(const TargetInputs& v, float* out);
};

struct OppNextPlacementTarget {
  // 15x15 binary mask: cell (r,c) is 1.0 iff the opponent placed a tile
  // on (r,c) on their next move. All zeros if the next move is missing,
  // EXCHANGE, or PASS. Transposed across the main diagonal when
  // view.apply_flip is true (so it stays aligned with the InputEncoder's
  // spatial planes).
  static constexpr int kSide = 15;
  static constexpr const char* kName = "opp_next_placement";
  static constexpr int kDims[] = {kSide, kSide};
  static void encode(const TargetInputs& v, float* out);
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
  static void encode_all(const TargetInputs& v, float* out);
};

// ---------- The single source of truth ---------------------------------

using AllTargets = TargetList<WldTarget, ScoreDiffTarget, OppNextPlacementTarget>;

// Convenience constants derived from AllTargets. These exist so code that
// just wants "how many label floats in a row?" doesn't have to mention the
// template.
inline constexpr int kNumLabelHeads = static_cast<int>(AllTargets::size);
inline constexpr int kLabelFloats = AllTargets::total_floats;

// Per-target convenience aliases. Downstream code (tests, FFI, layout
// docs) can refer to a specific target's size without naming the struct.
inline constexpr int kWldFloats = detail::target_floats<WldTarget>();
inline constexpr int kScoreDiffFloats = detail::target_floats<ScoreDiffTarget>();  // 1 (regression)
inline constexpr int kScoreDiffClip = ScoreDiffTarget::kClip;
// The score-diff head's OUTPUT width: the mean and standard deviation of the
// final differential (a Gaussian). Distinct from kScoreDiffFloats, which is the
// 1-float regression target the NLL loss scores those two against.
inline constexpr int kScoreDiffOutputFloats = 2;
inline constexpr int kOppNextPlacementFloats = detail::target_floats<OppNextPlacementTarget>();
inline constexpr int kOppNextPlacementSide = OppNextPlacementTarget::kSide;

}  // namespace scribblez

#include "inlines/scribblez/training_targets.inl"
