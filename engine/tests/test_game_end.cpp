// game/game_end.h and training/game_end_targets.h on hand-built logs: the end
// is read from the final scores' gap over the moves' sum, from either seat.

#include "encoding/encode_context.h"
#include "game/game_end.h"
#include "game/game_log.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "training/game_end_targets.h"

#include <gtest/gtest.h>

#include <array>

namespace scribblez {
namespace {

constexpr std::array<Glyph, 2> kGlyphs = {Glyph::of(Tile::of(0)), Glyph::of(Tile::of(1))};

// A three-move log, a 100-point play by each player and then a pass, with the
// given final scores.
struct EndedGame {
  std::array<TurnRecord, 3> records{};
  GameLog log{};

  explicit EndedGame(std::array<int, 2> final_scores) {
    for (int k = 0; k < 2; ++k) {
      records[k].player = k;
      records[k].move = Move::play(/*horizontal=*/true, 7 + k, 0b11u, 100, kGlyphs.data(), 2);
    }
    records[2].player = 0;
    records[2].move = Move::pass();
    log.records = records.data();
    log.num_records = int(records.size());
    log.final_scores = final_scores;
  }
};

std::array<float, 2> targets(const GameLog& log, int pov) {
  EncodeContext ctx{};
  ctx.active_player = pov;
  ctx.game_end = game_end(log);
  std::array<float, 2> out{};
  GoingOutTarget::encode(ctx, &out[0]);
  EndPointsTarget::encode(ctx, &out[1]);
  return out;
}

TEST(GameEnd, GoingOutGainsTheOpponentsLeftovers) {
  const EndedGame g({100, 116});
  const GameEnd end = game_end(g.log);
  EXPECT_EQ(end.kind, GameEndKind::kWentOut);
  EXPECT_EQ(end.went_out, 1);
  EXPECT_EQ(end.adjustment, (std::array<int, 2>{0, 16}));
  EXPECT_EQ(targets(g.log, 1), (std::array<float, 2>{0.0f, 16.0f}));
  EXPECT_EQ(targets(g.log, 0), (std::array<float, 2>{1.0f, -16.0f}));
}

TEST(GameEnd, StalemateChargesEachTheirOwnRack) {
  const EndedGame g({97, 90});
  const GameEnd end = game_end(g.log);
  EXPECT_EQ(end.kind, GameEndKind::kStalemate);
  EXPECT_EQ(end.went_out, -1);
  EXPECT_EQ(end.adjustment, (std::array<int, 2>{-3, -10}));
  EXPECT_EQ(targets(g.log, 0), (std::array<float, 2>{2.0f, 7.0f}));
  EXPECT_EQ(targets(g.log, 1), (std::array<float, 2>{2.0f, -7.0f}));
}

TEST(GameEnd, NoAdjustmentIsUnlabelled) {
  const EndedGame g({100, 100});
  EXPECT_EQ(game_end(g.log).kind, GameEndKind::kUnfinished);
  EXPECT_EQ(targets(g.log, 0), (std::array<float, 2>{-1.0f, 0.0f}));
  GameLog empty{};
  EXPECT_EQ(game_end(empty).kind, GameEndKind::kUnfinished);
}

}  // namespace
}  // namespace scribblez
