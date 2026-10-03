// The .sprobe sidecar (data/probe_log.h): a written position reads back field
// for field, records candidate-major with their turns in record order.

#include "data/probe_log.h"
#include "game/glyph.h"
#include "game/rack.h"
#include "game/tile.h"
#include "temp_dir.h"
#include "util/exception.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace scribblez;

namespace {

Move play(int col, const std::string& word) {
  std::array<Glyph, RACK_SIZE> glyphs{};
  uint16_t mask = 0;
  for (size_t i = 0; i < word.size(); ++i) {
    glyphs[i] = Glyph::of(Tile::letter_from_char(word[i]));
    mask |= uint16_t(1u << (col + int(i)));
  }
  return Move::play(/*horizontal=*/true, 7, mask, /*score=*/10, glyphs.data(), int(word.size()));
}

TurnRecord turn(int player, const Move& m, const std::string& drawn) {
  TurnRecord t{};
  t.player = player;
  t.move = m;
  t.drawn = Rack::from_string(drawn);
  return t;
}

// Candidate c's probe i: a reply and, for odd i, a follow-up; truncated when i
// is even. Its outcome encodes (c, i) so a mixed-up record would show.
void add_probe(ProbePosition* p, int c, int i) {
  Rollout r;
  r.p_win = 0.1 * c + 0.01 * i;
  r.p_draw = 0.05;
  r.p_loss = 1 - r.p_win - r.p_draw;
  r.delta = 10 * c + i;
  r.delta_sq = r.delta * r.delta + 4;
  RolloutTrace t;
  t.initial_racks = {Rack::from_string("AEIRST?"),
                     Rack::from_string(i % 2 ? "QUIZETH" : "QUIZATH")};
  t.turns.push_back(turn(1, play(3, "ZIT"), "AB"));
  if (i % 2) t.turns.push_back(turn(0, play(2, "SIT"), "C"));
  t.truncated = i % 2 == 0;
  p->rollouts[size_t(c)].push_back(r);
  p->traces[size_t(c)].push_back(t);
}

ProbePosition position(int candidates, int probes) {
  ProbePosition p;
  p.at = {3, 17};
  p.mover = 0;
  p.base_seed = 1234;
  p.num_legal_moves = 321;
  p.rollouts.resize(size_t(candidates));
  p.traces.resize(size_t(candidates));
  for (int c = 0; c < candidates; ++c) {
    p.moves.push_back(play(c, "AT"));
    p.equities.push_back(5.5f - float(c));
    p.equity_ranks.push_back(c * 10);
    p.strata.push_back(uint8_t(c % 4));
    for (int i = 0; i < probes; ++i) add_probe(&p, c, i);
  }
  return p;
}

}  // namespace

TEST(ProbeLog, RoundTrips) {
  const std::filesystem::path dir = scribblez::testing::make_temp_dir("scribblez_test_probe_log");
  const std::string path = (dir / "x.sprobe").string();
  const int candidates = 3, probes = 4;
  const ProbePosition written = position(candidates, probes);
  {
    ProbeWriter w(path, kProbeFlagFaceUpLeaves, "abc123", "NWL23", 3, probes);
    w.add_position(written);
    w.add_position(written);
  }
  const ProbeReader r(path);
  EXPECT_EQ(r.header().flags, kProbeFlagFaceUpLeaves);
  EXPECT_EQ(r.header().horizon_plies, 3);
  EXPECT_EQ(std::string(r.header().leaf_model_hash), "abc123");
  EXPECT_EQ(std::string(r.header().lexicon), "NWL23");
  ASSERT_EQ(r.num_positions(), 2);
  ASSERT_EQ(r.probes(), probes);

  const ProbeReader::Position pos = r.position(1);
  EXPECT_EQ(pos.header->game_index, 3u);
  EXPECT_EQ(pos.header->turn_index, 17u);
  EXPECT_EQ(pos.header->base_seed, 1234u);
  EXPECT_EQ(pos.header->num_legal_moves, 321u);
  ASSERT_EQ(pos.header->num_candidates, uint32_t(candidates));
  for (int c = 0; c < candidates; ++c) {
    EXPECT_TRUE(pos.candidates[c].move == written.moves[size_t(c)]);
    EXPECT_EQ(pos.candidates[c].equity, written.equities[size_t(c)]);
    EXPECT_EQ(pos.candidates[c].equity_rank, written.equity_ranks[size_t(c)]);
    EXPECT_EQ(pos.candidates[c].stratum, written.strata[size_t(c)]);
  }
  size_t turn = 0;
  for (int c = 0; c < candidates; ++c) {
    for (int i = 0; i < probes; ++i) {
      const ProbeRecord& rec = pos.records[c * probes + i];
      const Rollout& ro = written.rollouts[size_t(c)][size_t(i)];
      const RolloutTrace& t = written.traces[size_t(c)][size_t(i)];
      EXPECT_EQ(rec.candidate, c);
      EXPECT_EQ(rec.probe, i);
      EXPECT_FLOAT_EQ(rec.p_win, float(ro.p_win));
      EXPECT_FLOAT_EQ(rec.delta, float(ro.delta));
      EXPECT_FLOAT_EQ(rec.delta_sq, float(ro.delta_sq));
      EXPECT_TRUE(rec.mover_rack == t.initial_racks[0]);
      EXPECT_TRUE(rec.opp_rack == t.initial_racks[1]);
      EXPECT_EQ(bool(rec.truncated), t.truncated);
      ASSERT_EQ(rec.num_turns, t.turns.size());
      for (const TurnRecord& tr : t.turns) {
        EXPECT_TRUE(pos.turns[turn].move == tr.move);
        EXPECT_TRUE(pos.turns[turn].drawn == tr.drawn);
        ++turn;
      }
    }
  }
  EXPECT_EQ(turn, pos.header->num_turns);
  std::filesystem::remove_all(dir);
}

TEST(ProbeLog, RejectsAForeignFile) {
  const std::filesystem::path dir = scribblez::testing::make_temp_dir("scribblez_test_probe_log2");
  const std::string path = (dir / "x.sprobe").string();
  std::vector<char> junk(200, 'x');
  {
    std::ofstream(path, std::ios::binary).write(junk.data(), std::streamsize(junk.size()));
  }
  EXPECT_THROW(ProbeReader r(path), util::Exception);
  std::filesystem::remove_all(dir);
}
