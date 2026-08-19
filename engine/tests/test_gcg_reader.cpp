#include "data/gcg_post_move.h"
#include "data/gcg_reader.h"
#include "game/tile.h"
#include "game/tile_counts.h"

#include <gtest/gtest.h>

#include <string>

namespace scribblez {
namespace {

std::string rack_letters(const ParsedRackSlots& slots) {
  std::string out;
  for (const auto& slot : slots) {
    if (slot.has_value()) out.push_back(slot->to_char());
  }
  return out;
}

ParsedGcgGame parse_or_fail(const std::string& gcg) {
  ParsedGcgGame game;
  std::string error;
  EXPECT_TRUE(read_gcg_text(gcg, &game, &error)) << error;
  return game;
}

// A top-of-file "#Rack1" pragma records a player's current rack for the final
// position; the reader must restore it there instead of leaving an unknown
// hand. Without it, player 0's rack is cleared after their move and the final
// snapshot holds nothing for them.
TEST(GcgReaderTest, InitialRackPragmaRestoresFinalRack) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    "#Rack1 ADEIMRZ\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_FALSE(game.snapshots.empty());
  EXPECT_EQ(rack_letters(game.snapshots.back().racks[0]), "ADEIMRZ");
}

// A "#Rack2" pragma is honored the same way for the second player.
TEST(GcgReaderTest, InitialRackPragmaRestoresSecondPlayerRack) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    "#Rack2 QUARTZY\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_FALSE(game.snapshots.empty());
  EXPECT_EQ(rack_letters(game.snapshots.back().racks[1]), "QUARTZY");
}

// A "#Rack1" pragma emitted after an event line records that player's rack just
// after the event, updating the turn's post-event racks and its snapshot.
TEST(GcgReaderTest, PostEventRackPragmaUpdatesThatTurn) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    "#Rack1 EEIORST\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_EQ(game.turns.size(), 2u);
  EXPECT_EQ(rack_letters(game.turns[0].racks_after_turn[0]), "EEIORST");
  ASSERT_GE(game.snapshots.size(), 2u);
  EXPECT_EQ(rack_letters(game.snapshots[1].racks[0]), "EEIORST");
}

// Absent any pragma, the final snapshot still clears the mover's rack -- the
// baseline the pragma corrects.
TEST(GcgReaderTest, NoRackPragmaLeavesFinalRackCleared) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_FALSE(game.snapshots.empty());
  EXPECT_EQ(rack_letters(game.snapshots.back().racks[0]), "");
}

// A position-set .gcg is read at its final recorded state, the side to move
// holding the rack its #RackN pragma records (read_gcg_endgame's reading, with
// a bag). Under open leaves the opponent's retained leave (their last rack
// minus what they played) is exposed.
TEST(GcgPositionTest, FinalStateWithThePragmaRack) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    "#Rack1 CCCDEEE\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  ParsedGcgPosition p;
  std::string error;
  ASSERT_TRUE(read_gcg_position(gcg, /*open_leaves=*/true, &p, &error)) << error;
  EXPECT_EQ(p.mover, 0);
  EXPECT_EQ(p.rack.to_string(), "CCCDEEE");
  EXPECT_EQ(p.scores[0], 6);
  EXPECT_EQ(p.scores[1], 8);
  EXPECT_EQ(p.turns, 2);
  EXPECT_EQ(p.board.num_tiles(), 6);
  // Bob kept BBBB after playing BBB.
  EXPECT_EQ(p.opp_leave.to_string(), "BBBB");
  // AAA and BBB are on the board; the unseen pool is 100 - 6 - 7 = 87, minus
  // the opponent's assumed-full rack.
  EXPECT_EQ(p.bag_size, 87 - 7);

  ParsedGcgPosition hidden;
  ASSERT_TRUE(read_gcg_position(gcg, /*open_leaves=*/false, &hidden, &error)) << error;
  EXPECT_TRUE(hidden.opp_leave.empty());
}

TEST(GcgPositionTest, RefusesAMissingRackPragma) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";
  ParsedGcgPosition p;
  std::string error;
  EXPECT_FALSE(read_gcg_position(gcg, false, &p, &error));
  EXPECT_NE(error.find("#Rack1"), std::string::npos);
}

// The position-evaluation datasets' reading: the board after the final move,
// from the POV of the seat that made it, holding its leave. The opponent's
// retained leave and the observation of their last move (board, move, the pool
// unseen to the POV while it was played) come along for the sims.
TEST(GcgPostMoveTest, FinalMoverPovWithOpponentLeaveAndObservation) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBCDE 9D BBB +8 8\n"
    ">Alice: AAAAEFG 10D AAA +6 12\n";

  ParsedGcgPostMove p;
  std::string error;
  ASSERT_TRUE(read_gcg_post_move(gcg, &p, &error)) << error;
  EXPECT_EQ(p.start_player, 0);
  EXPECT_EQ(p.leave.to_string(), "AEFG");
  EXPECT_EQ(p.scores[0], 12);
  EXPECT_EQ(p.scores[1], 8);
  EXPECT_EQ(p.board.num_tiles(), 9);
  // Bob kept BCDE after playing BBB.
  EXPECT_EQ(p.opp_leave.to_string(), "BCDE");
  ASSERT_TRUE(p.opp_observation.has_value());
  // Bob's move was played on the board holding only Alice's AAA ...
  EXPECT_EQ(p.opp_observation->board_before.num_tiles(), 3);
  EXPECT_EQ(p.opp_observation->move.num_glyphs(), 3);
  // ... while Alice held AAAAEFG: the pool unseen to her was 100 - 3 - 7.
  EXPECT_EQ(p.opp_observation->pool.size(), 90);
  EXPECT_EQ(p.opp_observation->pool.count(Tile::from_char('A')), 9 - 3 - 4);
  EXPECT_EQ(p.opp_observation->pool.count(Tile::from_char('B')), 2);
}

TEST(GcgPostMoveTest, OpeningMoveHasNoOpponentEvidence) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n";
  ParsedGcgPostMove p;
  std::string error;
  ASSERT_TRUE(read_gcg_post_move(gcg, &p, &error)) << error;
  EXPECT_EQ(p.leave.to_string(), "AAAA");
  EXPECT_TRUE(p.opp_leave.empty());
  EXPECT_FALSE(p.opp_observation.has_value());
}

TEST(GcgPostMoveTest, RefusesANonPlayFinalMove) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBCDE -  +0 0\n";
  ParsedGcgPostMove p;
  std::string error;
  EXPECT_FALSE(read_gcg_post_move(gcg, &p, &error));
  EXPECT_NE(error.find("not a tile placement"), std::string::npos);
}

TEST(UnseenCountsTest, FullDistributionMinusBoardAndRack) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAA? 8D AAa +6 6\n";
  ParsedGcgPostMove p;
  std::string error;
  ASSERT_TRUE(read_gcg_post_move(gcg, &p, &error)) << error;
  const TileCounts unseen = unseen_counts(p.board, p.leave);
  EXPECT_EQ(unseen.size(), 100 - 3 - 4);
  EXPECT_EQ(unseen.count(Tile::from_char('A')), 9 - 2 - 4);
  EXPECT_EQ(unseen.blanks(), 1);  // the designated blank on the board counts as a blank
}

}  // namespace
}  // namespace scribblez
