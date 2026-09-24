// WeirdBotAgent, the diagnostic bot that forces its highest-value retained
// leave tile onto its best cross-check square. Covers the forcing path, the
// HastyBot fallback when no square fits, tracking of its own leave across
// moves, and the leave-less first move.
//
// Several cases seed the tracked leave with a pass: a rack that can neither
// play nor exchange on an empty board is passed and retained whole.
//
// Every case needs the Macondo-bundled NWL23 leave table (both the forcing
// ranking and the fallback price leaves with it) and skips without it.

#include "agent/hasty_bot.h"
#include "agent/weird_bot.h"
#include "data/gcg_reader.h"
#include "endgame_positions.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "util/io.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

// Loads the NWL23 equity tables; false when the leaves file is absent.
bool ensure_equity() {
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(leaves).good()) return false;
  HastyEquity::ensure_initialized("NWL23");
  return true;
}

struct Placed {
  int r;
  int c;
  Glyph g;
};

// The play's newly placed tiles, in lane order.
std::vector<Placed> placed_of(const Move& m) {
  std::vector<Placed> out;
  if (m.type() != MoveType::PLAY) return out;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  int gi = 0;
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    out.push_back({r, c, m.glyph(gi++)});
  }
  return out;
}

// Only newly placed, non-blank tiles match.
bool places_letter_at(const Move& m, int r, int c, char letter) {
  const Tile want = Tile::from_char(letter);
  for (const Placed& p : placed_of(m)) {
    if (p.r == r && p.c == c) return p.g.has_letter() && !p.g.is_blank() && p.g.letter() == want;
  }
  return false;
}

bool places_letter(const Move& m, char letter) {
  const Tile want = Tile::from_char(letter);
  for (const Placed& p : placed_of(m)) {
    if (p.g.has_letter() && !p.g.is_blank() && p.g.letter() == want) return true;
  }
  return false;
}

void set_letter(Board& b, int r, int c, char letter) {
  b.set(r, c, Glyph::of(Tile::from_char(letter)));
}

}  // namespace

// With J as the tracked leave and a board where J's best cross-check square is
// the triple-word corner (7,0) (forming JO down and JA across), WeirdBot forces
// J onto (7,0).
TEST(WeirdBot, ForcesHighestLeaveTileOntoBestCrossCheck) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = Dictionary::build_from_words({"JA", "JO"});
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});
  const Rack opp;

  // Seed the leave with {J}.
  Board b1;
  const Rack r1 = Rack::from_string("J");
  const MoveRequest req1{b1, dict, r1, opp, 0, 0, /*bag_size=*/0};
  const Move m1 = wb.make_move(req1).move;
  ASSERT_EQ(m1.type(), MoveType::PASS);

  Board b2;
  set_letter(b2, 8, 0, 'O');
  set_letter(b2, 7, 1, 'A');
  const Rack r2 = Rack::from_string("J");
  const MoveRequest req2{b2, dict, r2, opp, 0, 0, /*bag_size=*/5};
  const Move m2 = wb.make_move(req2).move;

  ASSERT_EQ(m2.type(), MoveType::PLAY);
  EXPECT_TRUE(places_letter_at(m2, 7, 0, 'J')) << "expected J forced onto (7,0)";
}

// With a non-empty leave (J, Q, Z) but an empty board, no square offers a
// cross-word, so WeirdBot plays HastyBot's static-equity argmax. The leave
// being non-empty is what separates this from the first-move fallback.
TEST(WeirdBot, FallsBackWhenNoCrossCheckSquare) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = tiny_dict();
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});
  const Rack opp;

  // Seed the leave with {J,Q,Z}.
  Board b1;
  const Rack r1 = Rack::from_string("JQZ");
  const MoveRequest req1{b1, dict, r1, opp, 0, 0, /*bag_size=*/0};
  ASSERT_EQ(wb.make_move(req1).move.type(), MoveType::PASS);

  Board b2;
  const Rack r2 = Rack::from_string("JQZCAT");
  const MoveRequest req2{b2, dict, r2, opp, 0, 0, /*bag_size=*/90};
  const Move m2 = wb.make_move(req2).move;

  // HastyBot's argmax here may be an exchange rather than an opening play; the
  // test only requires that WeirdBot matches it.
  EXPECT_EQ(m2, hasty_best_move_wmp(req2));
}

// The forced tile comes from the leave retained after the last move, not from
// the current rack. The first move must play J (only JO is a word), leaving
// {A}. The second rack holds A and a higher-value J, both with forcing squares;
// WeirdBot must force A.
TEST(WeirdBot, TracksOwnLeaveAcrossMoves) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = Dictionary::build_from_words({"AT", "JO"});
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});
  const Rack opp;

  Board b1;
  set_letter(b1, 7, 7, 'O');
  const Rack r1 = Rack::from_string("JA");
  const MoveRequest req1{b1, dict, r1, opp, 0, 0, /*bag_size=*/3};
  const Move m1 = wb.make_move(req1).move;
  ASSERT_EQ(m1.type(), MoveType::PLAY);
  EXPECT_TRUE(places_letter(m1, 'J')) << "first move should consume J";
  EXPECT_FALSE(places_letter(m1, 'A')) << "first move should retain A";

  Board b2;
  set_letter(b2, 7, 1, 'T');  // AT horizontal at (7,0)
  set_letter(b2, 8, 0, 'T');  // AT vertical at (7,0)
  set_letter(b2, 7, 6, 'O');  // JO horizontal at (7,5)
  set_letter(b2, 8, 5, 'O');  // JO vertical at (7,5)
  const Rack r2 = Rack::from_string("AJ");
  const MoveRequest req2{b2, dict, r2, opp, 0, 0, /*bag_size=*/5};
  const Move m2 = wb.make_move(req2).move;

  ASSERT_EQ(m2.type(), MoveType::PLAY);
  EXPECT_TRUE(places_letter(m2, 'A')) << "should force the retained-leave tile A";
  EXPECT_FALSE(places_letter(m2, 'J')) << "must not force the fresh-rack tile J";
}

// With no tracked leave yet, WeirdBot plays HastyBot's move.
TEST(WeirdBot, FirstMoveFallsBackToHasty) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = tiny_dict();
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});

  Board b;
  const Rack my = Rack::from_string("CAT");
  const Rack opp;
  const MoveRequest req{b, dict, my, opp, 0, 0, /*bag_size=*/90};
  const Move m = wb.make_move(req).move;

  ASSERT_EQ(m.type(), MoveType::PLAY);
  EXPECT_EQ(m, hasty_best_move_wmp(req));
}

TEST(WeirdBot, FromSpecParsing) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  EXPECT_NE(WeirdBotAgent::from_spec({}, 0, "WeirdBot"), nullptr);
  EXPECT_THROW(WeirdBotAgent::from_spec({"--bogus-option=1"}, 0, "X"), std::runtime_error);
}

// pos-09 is the position the WeirdBot experiment was built around: the opponent
// holds G, and G at M7 (row 6, col 12) forms GNU with the NU to its right. The
// experiment needs WeirdBot to put real placement mass on M7, so this checks
// the forcing end to end on the real NWL23 board. It would catch a cross-check
// orientation bug that the hand-built boards above cannot.
TEST(WeirdBot, ForcesGAtM7OnPos09) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(kwg).good()) GTEST_SKIP() << "no NWL23 kwg at " << kwg;
  const Dictionary dict = Dictionary::load_kwg(kwg);

  ParsedGcgPosition pos;
  std::string error;
  const std::string gcg = util::read_file(std::string(SCRIBBLEZ_TEST_DATA_DIR) + "/pos09-gnu.gcg");
  ASSERT_TRUE(read_gcg_position(gcg, /*open_leaves=*/true, &pos, &error)) << error;

  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});
  const Rack opp;

  // Seed the leave with {G}.
  Board seed;
  const Rack g = Rack::from_string("G");
  const MoveRequest seed_req{seed, dict, g, opp, 0, 0, /*bag_size=*/0};
  ASSERT_EQ(wb.make_move(seed_req).move.type(), MoveType::PASS);

  const MoveRequest req{pos.board, dict, g, opp, 359, 312, /*bag_size=*/6};
  const Move m = wb.make_move(req).move;

  EXPECT_EQ(m.type(), MoveType::PLAY);
  EXPECT_TRUE(places_letter_at(m, 6, 12, 'G')) << "expected G forced onto M7 (6,12)";
}
