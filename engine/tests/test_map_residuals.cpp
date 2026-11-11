// sim/map_residuals.h on hand-built sums: a start's sims against a prior,
// worked by hand for one empty square, the end of the game and the value.

#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "gpu/rollout_abi.h"
#include "sim/map_residuals.h"
#include "sim/rollout_values.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace scribblez {
namespace {

constexpr int kSq = 7 * BOARD_SIZE + 7;  // the centre, left empty
constexpr int kFilled = 7 * BOARD_SIZE + 8;

// A board with one tile, at kFilled.
Board one_tile_board() {
  Board board;
  const Glyph g = Glyph::of(Tile::of(0));
  board.apply(Move::play(/*horizontal=*/true, 7, 1u << 8, 1, &g, 1));
  return board;
}

// Four rollouts: at kSq, points 10, 10, -20, 0 (classes 1, 1, 3, 4); the
// game ended in two (we went out, +6 and +10), open in the other two.
rollout_abi::SimMaps four_rollouts() {
  rollout_abi::SimMaps m{};
  m.sims = 4;
  m.class_sum[1][kSq] = m.class_sq_sum[1][kSq] = 2;
  m.class_sum[3][kSq] = m.class_sq_sum[3][kSq] = 1;
  m.class_sum[4][kSq] = m.class_sq_sum[4][kSq] = 1;
  m.points_sum[kSq] = 0;
  m.points_sq_sum[kSq] = 100 + 100 + 400;
  m.went_out_sum[0] = m.went_out_sq_sum[0] = 2;
  m.end_points_sum = 16;
  m.end_points_sq_sum = 36 + 100;
  return m;
}

// A prior: ownership at kSq (.1, .2, .3, .15, .25) as logits, with every
// other square's logits a different constant per square (so a read along
// the squares instead of the planes shows), points mean 5 with variance 100
// (in points), going out uniform, end points mean 8, variance 25, win value
// 0.4.
constexpr std::array<float, 5> kOwnershipAtSq = {0.1f, 0.2f, 0.3f, 0.15f, 0.25f};

LeafPrediction uniform_prior() {
  LeafPrediction p{};
  p.wld = {0.3f, 0.2f, 0.5f};
  for (int sq = 0; sq < kBoardCells; ++sq)
    for (int c = 0; c < 5; ++c) p.ownership[size_t(c * kBoardCells + sq)] = float(sq % 7);
  for (int c = 0; c < 5; ++c)
    p.ownership[size_t(c * kBoardCells + kSq)] = std::log(kOwnershipAtSq[size_t(c)]);
  p.square_points[kSq] = 0.05f;
  p.square_points[kBoardCells + kSq] = 0.01f;
  p.end_points = {0.08f, 0.0025f};
  return p;
}

}  // namespace

TEST(MapResiduals, SumsEachLabelledSquaresResidualAndNoise) {
  const Board board = one_tile_board();
  const MapResidual r = map_residual(four_rollouts(), uniform_prior(), board, /*sim_value=*/0.6);
  EXPECT_EQ(r.sims, 4);
  EXPECT_EQ(r.squares, kBoardCells - 1);
  // kSq: mean 0 against 5, variance 600 / 3 = 200; every other empty
  // square: mean 0 against 0, variance 0, prior variance 0.
  // The prior's heads are float32 in units of 100 points, hence the slack.
  constexpr double kTol = 1e-4;
  EXPECT_NEAR(r.points_r2, 25.0, kTol);
  EXPECT_DOUBLE_EQ(r.points_noise, 200.0 / 4);
  EXPECT_DOUBLE_EQ(r.points_var, 200.0);
  EXPECT_DOUBLE_EQ(r.points_leaf_var, 0.0);
  EXPECT_NEAR(r.points_prior_var, 100.0, kTol);
  // kSq's shares (0, .5, 0, .25, .25) against (.1, .2, .3, .15, .25); the
  // other empty squares' shares (0, 0, 0, 0, 0) against uniform .2: 5 x .04.
  const double at_sq = 0.01 + 0.09 + 0.09 + 0.01 + 0.0;
  EXPECT_NEAR(r.ownership_r2, at_sq + (kBoardCells - 2) * 0.2, kTol);
  EXPECT_NEAR(r.ownership_noise, (0.25 + 0.1875 + 0.1875) * 4 / 3 / 4, kTol);
  // The mass moved: at kSq (-.1, .3, -.3, .1, 0), elsewhere -.2 each.
  EXPECT_NEAR(r.ownership_shift[1], 0.3 - (kBoardCells - 2) * 0.2, kTol);
  EXPECT_NEAR(r.ownership_shift[3], 0.1 - (kBoardCells - 2) * 0.2, kTol);
  // The end: mean 4 against 8, variance (136 - 64) / 3 = 24, prior
  // variance 25.
  EXPECT_NEAR(r.end_r2, 16.0, kTol);
  EXPECT_DOUBLE_EQ(r.end_noise, 24.0 / 4);
  EXPECT_DOUBLE_EQ(r.end_var, 24.0);
  EXPECT_DOUBLE_EQ(r.end_leaf_var, 0.0);
  EXPECT_NEAR(r.end_prior_var, 25.0, kTol);
  // Going out: shares (.5, 0, 0) against a third each.
  const double third = 1.0 / 3;
  EXPECT_NEAR(r.going_out_r2, (0.5 - third) * (0.5 - third) + 2 * third * third, kTol);
  EXPECT_NEAR(r.going_out_noise, 0.25 * 4 / 3 / 4, kTol);
  EXPECT_NEAR(r.going_out_shift[0], 0.5 - third, kTol);
  EXPECT_NEAR(r.value_r, 0.6 - 0.4, kTol);
}

}  // namespace scribblez
