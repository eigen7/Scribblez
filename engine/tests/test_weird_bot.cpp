// GoogleTest suite for WeirdBotAgent, the diagnostic leave-forcing self-play
// bot. Covers the four behaviours the rule is specified by:
//   * the forcing path -- the highest-value leave tile lands on its best
//     cross-check square;
//   * the fallback path -- when the leave tile fits no real cross-check square,
//     the plain HastyBot argmax is played instead;
//   * own-leave tracking -- the leave carried into the next turn is rack minus
//     the tiles the chosen move consumed, so the next forced tile comes from the
//     retained leave, not from freshly-drawn rack tiles;
//   * the first move of a game (no leave) -- fallback.
//
// Every case reads the NWL23 leave table (the forcing-play ranking and the
// fallback both price leaves through it) and skips gracefully when the
// Macondo-bundled file is absent.

#include "agent/macondo_bot.h"
#include "agent/weird_bot.h"
#include "endgame_positions.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

// Load HastyBot's default NWL23 equity tables; false (so the caller skips) when
// the Macondo-bundled leaves file is absent. Idempotent.
bool ensure_equity() {
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(leaves).good()) return false;
  HastyEquity::ensure_initialized("NWL23");
  return true;
}

// One newly-placed tile of a play.
struct Placed {
  int r;
  int c;
  Glyph g;
};

// The play's placed tiles, in lane order (mirrors Board::apply()'s walk).
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

// Whether the play newly places a non-blank tile equal to `letter` on (r, c).
bool places_letter_at(const Move& m, int r, int c, char letter) {
  const Tile want = Tile::from_char(letter);
  for (const Placed& p : placed_of(m)) {
    if (p.r == r && p.c == c) return p.g.has_letter() && !p.g.is_blank() && p.g.letter() == want;
  }
  return false;
}

// Whether the play newly places any non-blank tile equal to `letter`.
bool places_letter(const Move& m, char letter) {
  const Tile want = Tile::from_char(letter);
  for (const Placed& p : placed_of(m)) {
    if (p.g.has_letter() && !p.g.is_blank() && p.g.letter() == want) return true;
  }
  return false;
}

Rack rack_of(const std::string& letters) {
  Rack r;
  for (char ch : letters) r.add(Tile::from_char(ch));
  return r;
}

void set_letter(Board& b, int r, int c, char letter) {
  b.set(r, c, Glyph::of(Tile::from_char(letter)));
}

}  // namespace

// Forcing path: with a tracked leave whose highest tile is J, and a board where
// J's best-scoring cross-check square is the triple-word corner (7,0) -- placing
// J there forms the vertical cross-word JO and the horizontal main word JA --
// WeirdBot forces J onto (7,0). The leave is populated honestly by a first move:
// a rack of a single J on an empty board has no opening word and no exchange, so
// WeirdBot passes and retains {J}.
TEST(WeirdBot, ForcesHighestLeaveTileOntoBestCrossCheck) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = Dictionary::build_from_words({"JA", "JO"});
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});
  const Rack opp;

  // Move 1: empty board, rack {J} -> no play, no exchange (bag < 7) -> pass,
  // retaining {J} as the leave.
  Board b1;
  const Rack r1 = rack_of("J");
  const MoveRequest req1{b1, dict, r1, opp, 0, 0, /*bag_size=*/0};
  const Move m1 = wb.make_move(req1).move;
  ASSERT_EQ(m1.type(), MoveType::PASS);

  // Move 2: JO vertical (O below (7,0)) and JA horizontal (A right of (7,0)).
  Board b2;
  set_letter(b2, 8, 0, 'O');
  set_letter(b2, 7, 1, 'A');
  const Rack r2 = rack_of("J");
  const MoveRequest req2{b2, dict, r2, opp, 0, 0, /*bag_size=*/5};
  const Move m2 = wb.make_move(req2).move;

  ASSERT_EQ(m2.type(), MoveType::PLAY);
  EXPECT_TRUE(places_letter_at(m2, 7, 0, 'J')) << "expected J forced onto (7,0)";
}

// Fallback path: the leave holds high tiles (J, Q, Z), but the board is empty,
// so no square offers a real perpendicular cross-word for the forcing tile.
// WeirdBot returns HastyBot's static-equity argmax for the actual rack.
TEST(WeirdBot, FallsBackWhenNoCrossCheckSquare) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = tiny_dict();
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});
  const Rack opp;

  // Move 1: empty board, rack {J,Q,Z} -> no word, no exchange -> pass; leave is
  // {J,Q,Z}.
  Board b1;
  const Rack r1 = rack_of("JQZ");
  const MoveRequest req1{b1, dict, r1, opp, 0, 0, /*bag_size=*/0};
  ASSERT_EQ(wb.make_move(req1).move.type(), MoveType::PASS);

  // Move 2: still an empty board, but a rack that can open. The tracked leave is
  // non-empty, so this exercises the "T fits no cross-check square" branch, not
  // the empty-leave branch.
  Board b2;
  const Rack r2 = rack_of("JQZCAT");
  const MoveRequest req2{b2, dict, r2, opp, 0, 0, /*bag_size=*/90};
  const Move m2 = wb.make_move(req2).move;

  // No forcing square exists, so WeirdBot's answer is exactly HastyBot's static
  // argmax (whatever type it is -- here HastyBot prefers dumping J/Q/Z to an
  // opening on the tiny dictionary).
  EXPECT_EQ(m2, hasty_best_move_wmp(req2));
}

// Leave tracking across two moves: the first move plays J (every legal play on
// the board consumes J), so the retained leave is {A} -- rack minus the placed
// tile. On the second move WeirdBot forces that leave's A, even though the fresh
// rack also holds a higher-value J that could itself be forced. Forcing A (and
// never J) shows the tile source is the retained leave, not the current rack.
TEST(WeirdBot, TracksOwnLeaveAcrossMoves) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = Dictionary::build_from_words({"AT", "JO"});
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});
  const Rack opp;

  // Move 1: O on the board, rack {J,A}. Only J forms a word (JO); A cannot, so
  // whatever play is chosen consumes J and retains {A}.
  Board b1;
  set_letter(b1, 7, 7, 'O');
  const Rack r1 = rack_of("JA");
  const MoveRequest req1{b1, dict, r1, opp, 0, 0, /*bag_size=*/3};
  const Move m1 = wb.make_move(req1).move;
  ASSERT_EQ(m1.type(), MoveType::PLAY);
  EXPECT_TRUE(places_letter(m1, 'J')) << "first move should consume J";
  EXPECT_FALSE(places_letter(m1, 'A')) << "first move should retain A";

  // Move 2: a board offering a forcing square for A (AT, corner (7,0)) AND one
  // for J (JO, around (7,5)); the rack re-draws a J. A correct leave keys off
  // {A}, forcing A; a rack-driven bug would force the higher J.
  Board b2;
  set_letter(b2, 7, 1, 'T');  // AT horizontal at (7,0)
  set_letter(b2, 8, 0, 'T');  // AT vertical at (7,0)
  set_letter(b2, 7, 6, 'O');  // JO horizontal at (7,5)
  set_letter(b2, 8, 5, 'O');  // JO vertical at (7,5)
  const Rack r2 = rack_of("AJ");
  const MoveRequest req2{b2, dict, r2, opp, 0, 0, /*bag_size=*/5};
  const Move m2 = wb.make_move(req2).move;

  ASSERT_EQ(m2.type(), MoveType::PLAY);
  EXPECT_TRUE(places_letter(m2, 'A')) << "should force the retained-leave tile A";
  EXPECT_FALSE(places_letter(m2, 'J')) << "must not force the fresh-rack tile J";
}

// First move of a game: with no tracked leave, WeirdBot plays HastyBot's move.
TEST(WeirdBot, FirstMoveFallsBackToHasty) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  const Dictionary dict = tiny_dict();
  WeirdBotAgent wb(0, "WeirdBot");
  wb.begin_game({});

  Board b;
  const Rack my = rack_of("CAT");
  const Rack opp;
  const MoveRequest req{b, dict, my, opp, 0, 0, /*bag_size=*/90};
  const Move m = wb.make_move(req).move;

  ASSERT_EQ(m.type(), MoveType::PLAY);
  EXPECT_EQ(m, hasty_best_move_wmp(req));
}

// from_spec: no options yields an agent; an unknown option is rejected.
TEST(WeirdBot, FromSpecParsing) {
  if (!ensure_equity()) GTEST_SKIP() << "no NWL23 leaves";
  EXPECT_NE(WeirdBotAgent::from_spec({}, 0, "WeirdBot"), nullptr);
  EXPECT_THROW(WeirdBotAgent::from_spec({"--bogus-option=1"}, 0, "X"), std::runtime_error);
}
