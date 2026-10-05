// The .sprobe sidecar (data/probe_log.h): a written position reads back field
// for field, records candidate-major with their turns in record order; and its
// replay (data/probe_replay.h).

#include "data/probe_log.h"
#include "data/probe_replay.h"
#include "game/game_log.h"
#include "game/glyph.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "temp_dir.h"
#include "util/exception.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

// A writer destroyed by an exception leaves no file: a short one would pass for
// a finished one, and a resumed run would skip its .slog.
TEST(ProbeLog, AnUnwindingWriterLeavesNoFile) {
  const std::filesystem::path dir = scribblez::testing::make_temp_dir("scribblez_test_probe_log3");
  const std::string path = (dir / "x.sprobe").string();
  try {
    ProbeWriter w(path, kProbeFlagFaceUpLeaves, "abc123", "NWL23", 3, 2);
    w.add_position(position(2, 2));
    throw std::runtime_error("a leaf readout was not finite");
  } catch (const std::runtime_error&) {
  }
  EXPECT_FALSE(std::filesystem::exists(path));
  std::filesystem::remove_all(dir);
}

TEST(ProbeLog, RejectsATruncatedFile) {
  const std::filesystem::path dir = scribblez::testing::make_temp_dir("scribblez_test_probe_log4");
  const std::string path = (dir / "x.sprobe").string();
  {
    ProbeWriter w(path, kProbeFlagFaceUpLeaves, "abc123", "NWL23", 3, 2);
    w.add_position(position(2, 2));
  }
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  EXPECT_THROW(ProbeReader r(path), util::Exception);
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

// The replay (data/probe_replay.h) of a hand-built position: after turn 0 of
// a game, player 1 plays ZIT; one probe has the opponent play XY and the root
// mover exchange.
namespace {

Move play_scored(int col, const std::string& word, int score) {
  std::array<Glyph, RACK_SIZE> glyphs{};
  uint16_t mask = 0;
  for (size_t i = 0; i < word.size(); ++i) {
    glyphs[i] = Glyph::of(Tile::letter_from_char(word[i]));
    mask |= uint16_t(1u << (col + int(i)));
  }
  return Move::play(/*horizontal=*/true, 7, mask, uint16_t(score), glyphs.data(), int(word.size()));
}

Rack rack_of(const char* codes) {
  Rack r;
  for (int i = 0; i < RACK_SIZE; ++i)
    if (uint8_t(codes[i]) != Tile::empty().index()) r.add(Tile::of(uint8_t(codes[i])));
  return r;
}

GameLogStorage one_turn_game() {
  GameLogStorage g;
  g.initial_racks = {Rack::from_string("AEINRST"), Rack::from_string("DIOTUZS")};
  g.turns.resize(1);
  g.turns[0].move = play_scored(7, "AT", 4);
  g.turns[0].drawn = Rack::from_string("BC");
  return g;
}

ProbePosition zit_position() {
  ProbePosition p;
  p.at = {0, 1};
  p.mover = 1;
  p.moves = {play_scored(3, "ZIT", 24)};
  p.equities = {20.0f};
  p.equity_ranks = {0};
  p.strata = {0};
  RolloutTrace t;
  t.initial_racks = {Rack::from_string("EINRSXY"), Rack::from_string("DOUSAEE")};
  t.turns.push_back(turn(0, play_scored(1, "XY", 10), "LM"));
  t.turns.push_back(turn(1, Move::exchange(TileCounts::from_string("AEE")), "FGH"));
  t.truncated = true;
  p.rollouts = {{Rollout{}}};
  p.traces = {{t}};
  return p;
}

}  // namespace

TEST(ProbeReplay, FollowsTheRecordFromThePosition) {
  const std::filesystem::path dir =
    scribblez::testing::make_temp_dir("scribblez_test_probe_replay");
  const std::string path = (dir / "x.sprobe").string();
  {
    ProbeWriter w(path, kProbeFlagFaceUpLeaves, "abc123", "NWL23", 3, 1);
    w.add_position(zit_position());
  }
  const ProbeReader r(path);
  const GameLogStorage game = one_turn_game();
  ProbeReplay out;
  replay_probe_position(game.view(), r.position(0), r.probes(), &out);

  ASSERT_EQ(out.roots.size(), 1u);
  const ProbeRootState& root = out.roots[0];
  EXPECT_EQ(root.mover, 1);
  EXPECT_EQ(root.score_diff, -4);
  EXPECT_EQ(root.bag_size, 84);
  EXPECT_TRUE(rack_of(root.rack) == Rack::from_string("DIOTUZS"));
  EXPECT_TRUE(rack_of(root.opp_leave) == Rack::from_string("EINRS"));

  ASSERT_EQ(out.candidates.size(), 1u);
  EXPECT_TRUE(rack_of(out.candidates[0].leave) == Rack::from_string("DOUS"));
  EXPECT_EQ(out.candidates[0].score_diff, 20);
  EXPECT_EQ(out.candidates[0].bag_size, 81);

  ASSERT_EQ(out.starts.size(), 1u);
  EXPECT_TRUE(rack_of(out.starts[0].mover_drawn) == Rack::from_string("AEE"));
  EXPECT_TRUE(rack_of(out.starts[0].opp_drawn) == Rack::from_string("XY"));
  EXPECT_TRUE(rack_of(out.starts[0].opp_rack) == Rack::from_string("EINRSXY"));

  ASSERT_EQ(out.turns.size(), 2u);
  const ProbeTurnState& reply = out.turns[0];
  EXPECT_EQ(reply.root_mover, 0);
  EXPECT_EQ(reply.ply, 1);
  EXPECT_EQ(reply.bag_size, 81);
  EXPECT_EQ(reply.score_diff, 20);
  EXPECT_TRUE(rack_of(reply.leave) == Rack::from_string("EINRS"));
  EXPECT_TRUE(rack_of(reply.rack_after) == Rack::from_string("EINRSLM"));
  const ProbeTurnState& exchange = out.turns[1];
  EXPECT_EQ(exchange.root_mover, 1);
  EXPECT_EQ(exchange.ply, 2);
  EXPECT_EQ(exchange.bag_size, 79);
  EXPECT_EQ(exchange.score_diff, 10);
  EXPECT_TRUE(rack_of(exchange.leave) == Rack::from_string("DOUS"));
  EXPECT_TRUE(rack_of(exchange.drawn) == Rack::from_string("FGH"));
  EXPECT_TRUE(rack_of(exchange.rack_after) == Rack::from_string("DOUSFGH"));
  std::filesystem::remove_all(dir);
}

TEST(ProbeReplay, RejectsARecordThatDoesNotFollowFromThePosition) {
  const std::filesystem::path dir =
    scribblez::testing::make_temp_dir("scribblez_test_probe_replay2");
  const std::string path = (dir / "x.sprobe").string();
  ProbePosition p = zit_position();
  p.traces[0][0].initial_racks[1] = Rack::from_string("AEEIOUY");  // lacks the leave DOUS
  {
    ProbeWriter w(path, kProbeFlagFaceUpLeaves, "abc123", "NWL23", 3, 1);
    w.add_position(p);
  }
  const ProbeReader r(path);
  ProbeReplay out;
  EXPECT_THROW(replay_probe_position(one_turn_game().view(), r.position(0), r.probes(), &out),
               util::Exception);
  std::filesystem::remove_all(dir);
}
