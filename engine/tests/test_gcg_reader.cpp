// The .gcg readers: rack pragmas, the decision-point reading
// (read_gcg_position), and the post-move reading the position-evaluation
// datasets use (read_gcg_post_move).

#include "data/gcg_post_move.h"
#include "data/gcg_reader.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "training/lane_analysis.h"

#include <gtest/gtest.h>

#include <string>

namespace scribblez {
namespace {

ParsedGcgGame parse_or_fail(const std::string& gcg) {
  ParsedGcgGame game;
  std::string error;
  EXPECT_TRUE(read_gcg_text(gcg, &game, &error)) << error;
  return game;
}

// A top-of-file #Rack1 pragma is player 1's rack in the final position. The
// reader must put it in the final snapshot, where otherwise the rack would be
// empty after that player's last move.
TEST(GcgReaderTest, InitialRackPragmaRestoresFinalRack) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    "#Rack1 ADEIMRZ\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_FALSE(game.snapshots.empty());
  EXPECT_EQ(game.snapshots.back().racks[0].to_string(), "ADEIMRZ");
}

TEST(GcgReaderTest, InitialRackPragmaRestoresSecondPlayerRack) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    "#Rack2 QUARTZY\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_FALSE(game.snapshots.empty());
  EXPECT_EQ(game.snapshots.back().racks[1].to_string(), "AQRTUYZ");
}

// A #Rack1 pragma after an event line is that player's rack just after the
// event, and updates both that turn's racks and its snapshot.
TEST(GcgReaderTest, PostEventRackPragmaUpdatesThatTurn) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    "#Rack1 EEIORST\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_EQ(game.turns.size(), 2u);
  EXPECT_EQ(game.turns[0].racks_after_turn[0].to_string(), "EEIORST");
  ASSERT_GE(game.snapshots.size(), 2u);
  EXPECT_EQ(game.snapshots[1].racks[0].to_string(), "EEIORST");
}

// The baseline the pragma tests correct: with no pragma, the final snapshot
// holds an empty rack for the mover.
TEST(GcgReaderTest, NoRackPragmaLeavesFinalRackCleared) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  const ParsedGcgGame game = parse_or_fail(gcg);
  ASSERT_FALSE(game.snapshots.empty());
  EXPECT_EQ(game.snapshots.back().racks[0].to_string(), "");
}

// A position-set .gcg is read at its final recorded state, with the side to
// move holding its #RackN pragma rack. With open leaves the opponent's retained
// leave (last rack minus the tiles played) is exposed too.
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
  EXPECT_EQ(p.opp_leave.to_string(), "BBBB");
  // 100 tiles - 6 on the board - the mover's 7 = 87 unseen, less the
  // opponent's rack, assumed full.
  EXPECT_EQ(p.bag_size, 87 - 7);

  ParsedGcgPosition hidden;
  ASSERT_TRUE(read_gcg_position(gcg, /*open_leaves=*/false, &hidden, &error)) << error;
  EXPECT_TRUE(hidden.opp_leave.empty());
}

TEST(GcgPositionTest, AnyRecordedTurnOfACompleteGame) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n"
    ">Alice: AAAACCC 10D CCC +10 16\n";

  ParsedGcgPosition p;
  std::string error;
  ASSERT_TRUE(read_gcg_position_at(gcg, /*turn_index=*/2, /*open_leaves=*/true, &p, &error))
    << error;
  EXPECT_EQ(p.mover, 0);
  EXPECT_EQ(p.rack.to_string(), "AAAACCC");
  EXPECT_EQ(p.scores[0], 6);
  EXPECT_EQ(p.scores[1], 8);
  EXPECT_EQ(p.turns, 2);
  EXPECT_EQ(p.game.turns.size(), 2u);
  EXPECT_EQ(p.game.snapshots.size(), 3u);
  EXPECT_EQ(p.game.game_log.turns.size(), 2u);
  EXPECT_EQ(p.board.num_tiles(), 6);
  EXPECT_EQ(p.opp_leave.to_string(), "BBBB");
  EXPECT_EQ(p.bag_size, 87 - 7);

  ASSERT_TRUE(read_gcg_position_at(gcg, 0, true, &p, &error)) << error;
  EXPECT_EQ(p.board.num_tiles(), 0);
  EXPECT_EQ(p.rack.to_string(), "AAAAAAA");
  EXPECT_TRUE(p.opp_leave.empty());
  EXPECT_EQ(p.turns, 0);

  EXPECT_FALSE(read_gcg_position_at(gcg, 3, true, &p, &error));
  EXPECT_NE(error.find("out of range"), std::string::npos);
}

TEST(GcgReaderTest, CrlfLineEndingsLeaveNoCarriageReturnInTokens) {
  const std::string gcg =
    "#player1 Alice Alice Smith\r\n"
    "#player2 Bob Bob\r\n"
    "#Rack1 CCCDEEE\r\n"
    ">Alice: AAAAAAA 8D AAA +6 6\r\n"
    ">Bob: BBBBBBB 9D BBB +8 8\r\n";
  ParsedGcgPosition p;
  std::string error;
  ASSERT_TRUE(read_gcg_position(gcg, true, &p, &error)) << error;
  EXPECT_EQ(p.game.player_names[0], "Alice Smith");
  EXPECT_EQ(p.game.player_names[1], "Bob");
  EXPECT_EQ(p.rack.to_string(), "CCCDEEE");
  EXPECT_EQ(p.scores[1], 8);
  EXPECT_EQ(p.opp_leave.to_string(), "BBBB");
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

// Every reader takes the side to move's rack from the same header pragma. The
// two-move game below leaves Alice (player 1) to move.
std::string two_move_game(const std::string& header_pragma, const std::string& post_event_pragma) {
  return "#player1 Alice Alice\n"
         "#player2 Bob Bob\n" +
         header_pragma + ">Alice: AAAAAAA 8D AAA +6 6\n" + post_event_pragma +
         ">Bob: BBBBBBB 9D BBB +8 8\n";
}

// The mover's rack as read_gcg_position and the lane-analysis reader take it,
// or "<none>" when a reader refuses the file. The two must always agree.
std::string mover_rack(const std::string& gcg) {
  ParsedGcgPosition p;
  std::string error;
  const bool position_ok = read_gcg_position(gcg, /*open_leaves=*/false, &p, &error);
  GcgAnalysisPosition a;
  const bool analysis_ok = parse_gcg_analysis_position(gcg, &a, &error);
  EXPECT_EQ(position_ok, analysis_ok);
  if (!position_ok || !analysis_ok) return "<none>";
  EXPECT_EQ(p.rack.to_string(), a.rack.to_string());
  return p.rack.to_string();
}

// '_' marks a slot the writer does not reveal: it holds no tile, and is not a
// blank. '?' and a lowercase letter are blanks, as in a turn line's rack.
TEST(GcgRackPragmaTest, UnknownSlotsHoldNoTile) {
  EXPECT_EQ(mover_rack(two_move_game("#Rack1 _CE__MR\n", "")), "CEMR");
  EXPECT_EQ(mover_rack(two_move_game("#Rack1 CE?m\n", "")), "CE??");
}

// Only the capitalized form gcg_writer.h emits is a rack pragma.
TEST(GcgRackPragmaTest, TheLowercaseFormIsNotAPragma) {
  EXPECT_EQ(mover_rack(two_move_game("#rack1 CCCDEEE\n", "")), "<none>");
  EXPECT_EQ(mover_rack(two_move_game("#RACK1 CCCDEEE\n", "")), "<none>");
  EXPECT_EQ(mover_rack(two_move_game("#Rack1x CCCDEEE\n", "")), "<none>");
}

// Only a header pragma gives the final position's rack. A post-event pragma is
// the rack just after that event, which the final position may have moved past.
TEST(GcgRackPragmaTest, OnlyTheHeaderPragmaGivesTheFinalRack) {
  EXPECT_EQ(mover_rack(two_move_game("", "#Rack1 EEIORST\n")), "<none>");
  EXPECT_EQ(mover_rack(two_move_game("#Rack1 CCCDEEE\n", "#Rack1 EEIORST\n")), "CCCDEEE");
}

// The position-evaluation datasets' reading: the board after the final move,
// from the POV of the seat that made it, holding its leave. It also carries the
// opponent's retained leave and an observation of their last move (the board
// before it, the move, and the pool unseen to the POV at the time), which the
// sims use for rack inference.
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
  EXPECT_EQ(p.opp_leave.to_string(), "BCDE");
  ASSERT_TRUE(p.opp_observation.has_value());
  // Bob played onto a board holding only Alice's AAA, while Alice held
  // AAAAEFG: 100 - 3 - 7 tiles were unseen to her.
  EXPECT_EQ(p.opp_observation->board_before.num_tiles(), 3);
  EXPECT_EQ(p.opp_observation->move.num_glyphs(), 3);
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
