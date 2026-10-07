// sim/reply_blocking on a hand-built position: CAT across the centre row and
// three candidates, each with one probe whose opponent holds the same rack.
//
//   candidate 0  COT down from the C     opponent replies CATS (S at 8K)
//   candidate 1  CATE (E at 8K)          opponent replies TO (O below the T)
//   candidate 2  AX under the TK         opponent exchanges
//
// CATE takes the square of CATS's S. AX leaves it empty but puts an X under
// it, where SX is no word. AX also takes the square of TO's O; COT blocks
// neither of the others' replies.

#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/lexicon.h"
#include "sim/reply_blocking.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

constexpr int kStride = 4;

// `word`'s letters on consecutive squares from (row, col).
Move play(int row, int col, bool horizontal, const std::string& word) {
  std::array<Glyph, RACK_SIZE> glyphs{};
  uint16_t mask = 0;
  const int lane0 = horizontal ? col : row;
  for (size_t i = 0; i < word.size(); ++i) {
    glyphs[i] = Glyph::of(Tile::letter_from_char(word[i]));
    mask |= uint16_t(1u << (lane0 + int(i)));
  }
  return Move::play(horizontal, horizontal ? row : col, mask, /*score=*/0, glyphs.data(),
                    int(word.size()));
}

Board root() {
  Board b;
  b.apply(play(7, 7, /*horizontal=*/true, "CAT"));
  return b;
}

// The position's arrays, which ProbeReader::Position points into.
struct Probes {
  ProbePositionHeader header{};
  std::vector<ProbeCandidate> candidates;
  std::vector<ProbeRecord> records;
  std::vector<binlog::TurnBlob> turns;

  ProbeReader::Position view() const {
    return {&header, candidates.data(), records.data(), turns.data()};
  }
};

void add(Probes* p, const Move& candidate, const Move& reply) {
  ProbeCandidate c{};
  c.move = candidate;
  p->candidates.push_back(c);
  ProbeRecord r{};
  r.candidate = uint16_t(p->records.size());
  r.opp_rack = Rack::from_string("AEIOSTX");
  r.num_turns = 1;
  p->records.push_back(r);
  p->turns.push_back({reply, Rack()});
}

void seal(Probes* p) {
  p->header.num_candidates = uint32_t(p->candidates.size());
  p->header.num_turns = uint32_t(p->turns.size());
}

Probes position() {
  Probes p;
  add(&p, play(7, 7, /*horizontal=*/false, "COT"), play(7, 10, /*horizontal=*/true, "S"));
  add(&p, play(7, 10, /*horizontal=*/true, "E"), play(8, 9, /*horizontal=*/false, "O"));
  add(&p, play(8, 9, /*horizontal=*/true, "AX"), Move::exchange(TileCounts::from_string("X")));
  seal(&p);
  return p;
}

const Dictionary* nwl23() {
  if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) return nullptr;
  static const Dictionary dict = Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG);
  return &dict;
}

}  // namespace

TEST(ReplyBlocking, MarksTheCandidatesThatTakeOrSpoilAReplysSquares) {
  const Dictionary* dict = nwl23();
  if (!dict) GTEST_SKIP() << "no NWL23 kwg";
  const Probes p = position();
  std::vector<uint8_t> out(p.records.size() * kStride, 0xff);
  reply_blocking(*dict, root(), p.view(), /*probes=*/1, kStride, out.data());

  // Row r is candidate r's reply; column b, whether candidate b blocks it.
  const std::vector<uint8_t> expected = {
    0, 1, 1, 0,  // CATS: CATE takes the S's square, AX spoils its cross-check
    0, 0, 1, 0,  // TO: AX takes the O's square
    0, 0, 0, 0,  // an exchange blocks nothing
  };
  EXPECT_EQ(out, expected);
}

// A single tile's direction is movegen's bookkeeping: alone under the T, the
// O of TO makes only a vertical word and is filed as a vertical play; once a
// B sits beside it (AB down), it also makes BO and is filed as a horizontal
// one. The B leaves the O's square and cross-checks open, so it blocks nothing.
TEST(ReplyBlocking, ASingleTileIsTheSamePlacementInEitherDirection) {
  const Dictionary* dict = nwl23();
  if (!dict) GTEST_SKIP() << "no NWL23 kwg";
  Probes p;
  add(&p, play(7, 7, /*horizontal=*/false, "COT"), play(8, 9, /*horizontal=*/false, "O"));
  add(&p, play(8, 8, /*horizontal=*/false, "B"), Move::exchange(TileCounts::from_string("X")));
  seal(&p);
  std::vector<uint8_t> out(p.records.size() * kStride, 0xff);
  reply_blocking(*dict, root(), p.view(), /*probes=*/1, kStride, out.data());
  EXPECT_EQ(out[0 * kStride + 1], 0);
}

// A candidate that is not a play leaves the root board as it is, so every
// reply open there stays open after it.
TEST(ReplyBlocking, AnExchangeBlocksNothing) {
  const Dictionary* dict = nwl23();
  if (!dict) GTEST_SKIP() << "no NWL23 kwg";
  Probes p;
  add(&p, play(7, 7, /*horizontal=*/false, "COT"), play(7, 10, /*horizontal=*/true, "S"));
  add(&p, Move::exchange(TileCounts::from_string("X")),
      Move::exchange(TileCounts::from_string("X")));
  seal(&p);
  std::vector<uint8_t> out(p.records.size() * kStride, 0xff);
  reply_blocking(*dict, root(), p.view(), /*probes=*/1, kStride, out.data());
  EXPECT_EQ(out[0 * kStride + 1], 0);
}
