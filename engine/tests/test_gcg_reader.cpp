#include "data/gcg_reader.h"
#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "game/tile.h"
#include "sim/gcg_decision.h"

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

// The decision point a .gcg encodes for simulation: its final recorded
// state, the side to move holding the rack its #RackN pragma records. Under
// open leaves the opponent's retained leave (their last rack minus what they
// played) is set.
TEST(GcgDecisionTest, DecisionIsTheFinalStateWithThePragmaRack) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    "#Rack1 CCCDEEE\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";

  GcgDecision d;
  std::string error;
  ASSERT_TRUE(gcg_decision(gcg, /*open_leaves=*/true, &d, &error)) << error;
  EXPECT_EQ(d.pos.mover, 0);
  EXPECT_EQ(d.pos.rack.to_string(), "CCCDEEE");
  EXPECT_EQ(d.pos.scores[0], 6);
  EXPECT_EQ(d.pos.scores[1], 8);
  EXPECT_EQ(d.turn_index, 2);
  // Bob kept BBBB after playing BBB.
  EXPECT_EQ(d.pos.opp_leave.to_string(), "BBBB");
  // AAA and BBB are on the board; the unseen pool is 100 - 6 - 7 = 87, minus
  // the opponent's assumed-full rack.
  EXPECT_EQ(d.bag_size, 87 - 7);

  GcgDecision hidden;
  ASSERT_TRUE(gcg_decision(gcg, /*open_leaves=*/false, &hidden, &error)) << error;
  EXPECT_TRUE(hidden.pos.opp_leave.empty());
}

TEST(GcgDecisionTest, ReplayLeavesTheEncoderAtTheDecision) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    "#Rack1 CCCDEEE\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";
  GcgDecision d;
  std::string error;
  ASSERT_TRUE(gcg_decision(gcg, false, &d, &error)) << error;
  const InputEncodingSpec spec{nullptr, false, false};
  GameStateEncoder enc(spec);
  replay_to_decision(d, &enc);
  EXPECT_EQ(enc.active_player(), 0);
  EXPECT_EQ(enc.turn_index(), 2);
  EXPECT_EQ(enc.score(0), 6);
  EXPECT_EQ(enc.score(1), 8);
  EXPECT_EQ(enc.board().num_tiles(), 6);
}

TEST(GcgDecisionTest, RefusesAMissingRackPragma) {
  const std::string gcg =
    "#player1 Alice Alice\n"
    "#player2 Bob Bob\n"
    ">Alice: AAAAAAA 8D AAA +6 6\n"
    ">Bob: BBBBBBB 9D BBB +8 8\n";
  GcgDecision d;
  std::string error;
  EXPECT_FALSE(gcg_decision(gcg, false, &d, &error));
  EXPECT_NE(error.find("#Rack1"), std::string::npos);
}

}  // namespace
}  // namespace scribblez
