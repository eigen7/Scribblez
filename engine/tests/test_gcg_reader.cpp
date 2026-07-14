#include "data/gcg_reader.h"
#include "game/tile.h"

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

}  // namespace
}  // namespace scribblez
