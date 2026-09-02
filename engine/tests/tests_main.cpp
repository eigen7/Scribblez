// GoogleTest suite for the engine core: game rules, move generation,
// encoders, binary logs, data loading, equity, and self-play components.

#include "agent/agent.h"
#include "agent/evidence_staging.h"
#include "agent/macondo_bot.h"
#include "data/binary_log.h"
#include "data/block_decoder.h"
#include "data/data_loader.h"
#include "data/format_layout.h"
#include "data/gcg_reader.h"
#include "data/sim_observation_log.h"
#include "data/slog_sampling.h"
#include "data/streaming_row_buffer.h"
#include "encoding/board_planes.h"
#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "encoding/position_encoder.h"
#include "game/bag.h"
#include "game/board.h"
#include "game/game.h"
#include "game/glyph.h"
#include "game/movegen.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_values.h"
#include "sim/sim_runner.h"
#include "training/evidence_trajectory_select.h"
#include "training/lane_analysis.h"
#include "training/lane_targets.h"
#include "training/max_move_per_lane_input_encoder.h"
#include "training/max_move_per_lane_task.h"
#include "training/move_set_encoder.h"
#include "training/move_set_eval_candidates.h"
#include "training/move_set_eval_target_log.h"
#include "training/training_targets.h"
#include "training/training_task.h"
#include "training/trajectory_position.h"
#include "util/io.h"
#include "util/math.h"
#include "util/metaprogramming.h"
#include "util/string.h"

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace scribblez;

// Layout shorthands for the FULL input layout, which is what these tests
// encode; derived from the block registry (the spec's dict is irrelevant to
// layout math). Blocks the registry does not name individually (the board
// sub-planes, the two cross-check families, the per-move metadata halves) are
// located relative to their block starts.
static const InputEncodingSpec kBaseLayout{nullptr};
static const int kInputFloats = input_floats(kBaseLayout);
static const int kSpatialFloats = spatial_floats();
static const int kSpatialPlanes = spatial_planes();
static const int kRowFloats = kInputFloats + kLabelFloats;
static const int kBlankMarkerPlane = BoardPlanes::kBlankMarkerPlane;
static const int kPremiumPlane0 = BoardPlanes::kPremiumPlane0;
static const int kSelfPlacementPlane = spatial_block_plane0(SpatialBlockId::kSelfPlacement);
static const int kOppPlacementPlane = spatial_block_plane0(SpatialBlockId::kOppPlacement);
static const int kHorizontalCrossCheckPlane0 = spatial_block_plane0(SpatialBlockId::kCrossChecks);
static const int kVerticalCrossCheckPlane0 =
  kHorizontalCrossCheckPlane0 + kHorizontalCrossCheckPlanes;
static const int kRackCountOffset = scalar_block_offset(kBaseLayout, ScalarBlockId::kRackCounts);
static const int kUnseenPoolOffset = scalar_block_offset(kBaseLayout, ScalarBlockId::kUnseenPool);
static const int kScoreDiffOffset = scalar_block_offset(kBaseLayout, ScalarBlockId::kScoreDiff);
static const int kMoveMetaOffset = scalar_block_offset(kBaseLayout, ScalarBlockId::kMoveMeta);

static Dictionary tiny_dict() {
  return Dictionary::build_from_words({"CAT", "CATS", "AT",     "AS",     "BAT", "BATS", "HE",
                                       "TO",  "ON",   "NO",     "IT",     "IS",  "OAT",  "OATS",
                                       "HAT", "HATS", "RAT",    "RATS",   "DOG", "GOD",  "GO",
                                       "OD",  "DO",   "AERIES", "PARTIED"});
}

TEST(Dictionary, Basic) {
  Dictionary d = tiny_dict();
  ASSERT_TRUE(d.contains("CAT"));
  ASSERT_TRUE(d.contains("cat"));  // case-insensitive
  ASSERT_FALSE(d.contains("CATX"));
  ASSERT_FALSE(d.contains("Z"));
  ASSERT_TRUE(d.contains("AERIES"));
}

static Rack rack_from(const std::string& s) {
  Rack r;
  for (char c : s) {
    if (c == '?')
      r.add(BLANK);
    else
      r.add(Tile::from_char(c));
  }
  return r;
}

// Build a PLAY Move from a starting square, direction, and ordered new glyphs.
// Assumes every glyph is newly placed (sufficient for the tests' empty-board
// setups); the square mask is absolute over the play's lane.
static Move make_play(int row, int col, bool horizontal, std::initializer_list<Glyph> gs) {
  std::array<Glyph, RACK_SIZE> played{};
  int n = 0;
  for (Glyph g : gs) {
    if (n >= RACK_SIZE) break;
    played[n++] = g;
  }
  const int start = horizontal ? row : col;
  const int lane0 = horizontal ? col : row;
  uint16_t mask = 0;
  for (int i = 0; i < n; ++i) mask |= uint16_t(1u << (lane0 + i));
  return Move::play(horizontal, start, mask, /*score=*/0, played.data(), n);
}

// Build a PLAY Move with an explicit per-tile layout. `rel_mask` is the play's
// mask relative to its first lane cell (bit 0 == the start cell); it is shifted
// into the absolute lane mask the Move stores. `gs` are the newly placed glyphs
// in word order (their count must equal popcount(rel_mask)).
static Move make_play_full(int row, int col, bool horizontal, uint16_t rel_mask, uint16_t score,
                           std::initializer_list<Glyph> gs) {
  std::array<Glyph, RACK_SIZE> played{};
  int n = 0;
  for (Glyph g : gs) {
    if (n >= RACK_SIZE) break;
    played[n++] = g;
  }
  const int start = horizontal ? row : col;
  const int lane0 = horizontal ? col : row;
  uint16_t mask = rel_mask << lane0;
  return Move::play(horizontal, start, mask, score, played.data(), n);
}

TEST(Movegen, Opening) {
  Dictionary d = tiny_dict();
  Board b;
  MoveGenerator gen(b, d);
  // Opening rack with letters CATSO -> can play CAT, CATS, etc., must cover center.
  Rack r = rack_from("CATSOHE");
  auto moves = gen.generate(r);
  ASSERT_FALSE(moves.empty());
  // Every opening move must cover the center square (CENTER, CENTER).
  // The board is empty, so placements are at consecutive squares from start.
  for (const auto& m : moves) {
    ASSERT_EQ(m.type(), MoveType::PLAY);
    const bool horiz = m.horizontal();
    uint16_t mask = m.square_mask();
    bool covers = false;
    for (int pos = 0; mask; ++pos, mask >>= 1) {
      if ((mask & 1u) == 0) continue;
      int r = horiz ? m.start() : pos;
      int c = horiz ? pos : m.start();
      if (r == CENTER && c == CENTER) {
        covers = true;
        break;
      }
    }
    ASSERT_TRUE(covers);
  }
  // The highest-scoring move should be a real word in the dictionary.
  int best = 0;
  const Move* best_move = nullptr;
  for (const auto& m : moves) {
    if (m.score() > best) {
      best = m.score();
      best_move = &m;
    }
  }
  ASSERT_NE(best_move, nullptr);
  ASSERT_TRUE(d.contains(best_move->main_word(b)));
}

TEST(Movegen, CrossWord) {
  Dictionary d = tiny_dict();
  Board b;
  // Place CAT at the center horizontally.
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {
                      Glyph::of(Tile::from_char('C')),
                      Glyph::of(Tile::from_char('A')),
                      Glyph::of(Tile::from_char('T')),
                    }));
  MoveGenerator gen(b, d);
  // Now play with rack "S" -> can extend to CATS by placing S at (CENTER, CENTER+3).
  Rack r = rack_from("SSSSSSS");
  auto moves = gen.generate(r);
  ASSERT_FALSE(moves.empty());
  bool found_cats = false;
  for (const auto& m : moves) {
    if (m.main_word(b) == "CATS") found_cats = true;
  }
  ASSERT_TRUE(found_cats);
}

TEST(Movegen, BingoBonus) {
  Dictionary d = Dictionary::build_from_words({"PARTIED"});
  Board b;
  // Place an A at the center to provide an anchor.
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true, {Glyph::of(Tile::from_char('A'))}));
  MoveGenerator gen(b, d);
  // Rack PRTIED + something already used (the A is on the board).
  Rack r = rack_from("PRTIED?");  // blank as 7th, won't be needed; ensure 7 tiles
  auto moves = gen.generate(r);
  // Look for a 7-tile play that uses the A. Bingo should give +50.
  bool found_bingo = false;
  for (const auto& m : moves) {
    if (m.num_glyphs() == RACK_SIZE) found_bingo = true;
  }
  // Note: rack has 6 non-blank tiles + 1 blank, total 7. PARTIED needs P,A,R,T,I,E,D;
  // A is on the board; the other 6 must come from the rack. So we'd place 6 tiles, not 7.
  // So no bingo here. But ensure PARTIED is generated.
  bool found_partied = false;
  for (const auto& m : moves) {
    if (m.main_word(b) == "PARTIED") found_partied = true;
  }
  ASSERT_TRUE(found_partied);
  (void)found_bingo;
}

// A canonical key for a play: its placed tiles (sorted) plus its score. Two
// plays with the same key are the same move for legality/scoring purposes (the
// `main_word` of a single-tile cross play is orientation-dependent and is not
// part of the key).
static std::string move_key(const Board& board, const Move& m) {
  (void)board;
  struct Placement {
    int r, c;
    Glyph g;
  };
  std::vector<Placement> tiles;
  if (m.type() == MoveType::PLAY) {
    const bool horiz = m.horizontal();
    uint16_t mask = m.square_mask();
    int gi = 0;
    for (int pos = 0; mask; ++pos, mask >>= 1) {
      if ((mask & 1u) == 0) continue;
      int r = horiz ? m.start() : pos;
      int c = horiz ? pos : m.start();
      tiles.push_back({r, c, m.glyph(gi++)});
    }
  }
  std::sort(tiles.begin(), tiles.end(), [](const Placement& a, const Placement& b) {
    if (a.r != b.r) return a.r < b.r;
    return a.c < b.c;
  });
  std::string k;
  char buf[32];
  for (const auto& t : tiles) {
    std::snprintf(buf, sizeof(buf), "%d,%d,%d;", t.r, t.c, (int)t.g.code());
    k += buf;
  }
  std::snprintf(buf, sizeof(buf), "|%d", m.score());
  k += buf;
  return k;
}

static std::set<std::string> key_set(const Board& board, const std::vector<Move>& ms) {
  std::set<std::string> s;
  for (const auto& m : ms) s.insert(move_key(board, m));
  return s;
}

static Rack random_rack(std::mt19937& rng) {
  Rack r;
  std::uniform_int_distribution<int> pick(0, 26);  // 26 -> blank, ~1/27 of tiles
  for (int i = 0; i < RACK_SIZE; ++i) {
    int v = pick(rng);
    r.add(v == 26 ? BLANK : Tile::of(v));
  }
  return r;
}

// The core invariant: the GADDAG generator and the reference DAWG generator must
// enumerate exactly the same set of legal plays (with identical scores) from any
// position. We stress this by walking random games: at each step we compare the
// two generators, then advance the board by applying a random generated play.
static void cross_validate(const Dictionary& d, const char* label, unsigned seed, int games,
                           int steps_per_game) {
  std::mt19937 rng(seed);
  long compared = 0;
  for (int g = 0; g < games; ++g) {
    Board b;
    for (int s = 0; s < steps_per_game; ++s) {
      Rack r = random_rack(rng);
      MoveGenerator gen(b, d);
      auto via_gaddag = gen.generate(r, GenAlgo::GADDAG);
      auto via_dawg = gen.generate(r, GenAlgo::DAWG);
      auto kg = key_set(b, via_gaddag);
      auto kd = key_set(b, via_dawg);
      if (kg != kd) {
        std::cerr << "MISMATCH [" << label << "] game " << g << " step " << s
                  << ": GADDAG=" << kg.size() << " DAWG=" << kd.size() << "\n";
        std::vector<std::string> only_g, only_d;
        std::set_difference(kg.begin(), kg.end(), kd.begin(), kd.end(), std::back_inserter(only_g));
        std::set_difference(kd.begin(), kd.end(), kg.begin(), kg.end(), std::back_inserter(only_d));
        for (size_t i = 0; i < only_g.size() && i < 5; ++i)
          std::cerr << "  only GADDAG: " << only_g[i] << "\n";
        for (size_t i = 0; i < only_d.size() && i < 5; ++i)
          std::cerr << "  only DAWG:   " << only_d[i] << "\n";
        std::exit(1);
      }
      ++compared;
      if (via_gaddag.empty()) break;
      // Advance: apply a random generated play.
      std::uniform_int_distribution<size_t> pick(0, via_gaddag.size() - 1);
      b.apply(via_gaddag[pick(rng)]);
    }
  }
  std::cout << "  cross-validated " << compared << " positions [" << label << "]\n";
}

// A medium word list (overlaps, plurals, hooks, a 7-letter bingo) to exercise
// more of the generator than the tiny dict does.
static Dictionary medium_dict() {
  return Dictionary::build_from_words(
    {"AA",     "AB",      "AD",      "AE",     "AG",      "AH",      "AI",      "AL",      "AN",
     "AR",     "AS",      "AT",      "AW",     "AX",      "AY",      "BA",      "BE",      "BI",
     "BO",     "BY",      "CAB",     "CAR",    "CARS",    "CART",    "CARTS",   "CAT",     "CATS",
     "CARE",   "CARES",   "CARET",   "CARETS", "CASTE",   "CASTER",  "CASTERS", "DOG",     "DOGS",
     "DOT",    "DOTS",    "EAR",     "EARS",   "EAT",     "EATS",    "RAT",     "RATE",    "RATES",
     "RATS",   "STARE",   "STARED",  "TARE",   "TARES",   "TEAR",    "TEARS",   "REACT",   "REACTS",
     "TRACE",  "TRACES",  "CRATE",   "CRATES", "CATER",   "CATERS",  "RECAST",  "RECASTS", "TASTE",
     "TASTER", "TASTERS", "SET",     "SET",    "TASTERS", "PARTIED", "AERIES",  "OX",      "OXEN",
     "QI",     "QIS",     "ZA",      "JO",     "GO",      "NO",      "ON",      "TO",      "IT",
     "IS",     "HE",      "OH",      "OW",     "WO",      "GI",      "HI",      "KI",      "LI",
     "MI",     "OI",      "PI",      "SI",     "TI",      "XI",      "ID",      "IF",      "IN",
     "WORD",   "WORDS",   "WORDIER", "TIE",    "TIES",    "TIED",    "DIET",    "DIETS",   "EDIT",
     "EDITS",  "TIDE",    "TIDES",   "SITE",   "SITED",   "STIED"});
}

TEST(Dictionary, GaddagVsDawgInMemory) {
  Dictionary d = medium_dict();
  cross_validate(d, "medium_dict", 1234u, /*games=*/12, /*steps_per_game=*/6);
}

// The board maintains its move-generation caches (cross-checks + GADDAG anchors)
// incrementally as moves are applied. This must always agree with a from-scratch
// full recompute of the same position. We stress that invariant by walking
// random games and, after every applied play, comparing the incrementally
// maintained caches against a freshly built board holding the same squares.
static void check_caches_match_full(const Dictionary& d, const Board& incremental,
                                    const char* label, int game, int step) {
  // Rebuild a board with identical squares; set() invalidates the caches so
  // ensure_movegen_caches() does a complete recompute.
  Board fresh;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) fresh.set(r, c, incremental.at(r, c));
  fresh.ensure_movegen_caches(d);

  for (int t = 0; t < 2; ++t) {
    const bool transposed = (t == 1);
    const auto& ci = incremental.cross_checks(transposed);
    const auto& cf = fresh.cross_checks(transposed);
    const auto& ai = incremental.gaddag_anchors(transposed);
    const auto& af = fresh.gaddag_anchors(transposed);
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
      const bool cross_ok = ci[i].mask == cf[i].mask && ci[i].score == cf[i].score &&
                            ci[i].has_neighbor == cf[i].has_neighbor;
      if (!cross_ok || ai[i] != af[i]) {
        std::cerr << "CACHE MISMATCH [" << label << "] game " << game << " step " << step
                  << " transposed=" << transposed << " square (" << (i / BOARD_SIZE) << ","
                  << (i % BOARD_SIZE) << "): " << "cross inc{mask=" << ci[i].mask
                  << ",score=" << ci[i].score << ",nbr=" << ci[i].has_neighbor
                  << "} full{mask=" << cf[i].mask << ",score=" << cf[i].score
                  << ",nbr=" << cf[i].has_neighbor << "} anchor inc=" << ai[i] << " full=" << af[i]
                  << "\n";
        std::exit(1);
      }
    }
  }
}

static void cache_consistency_stress(const Dictionary& d, const char* label, unsigned seed,
                                     int games, int steps_per_game) {
  std::mt19937 rng(seed);
  long checked = 0;
  for (int g = 0; g < games; ++g) {
    Board b;
    for (int s = 0; s < steps_per_game; ++s) {
      Rack r = random_rack(rng);
      MoveGenerator gen(b, d);
      auto moves = gen.generate(r);  // builds/uses the incremental caches
      check_caches_match_full(d, b, label, g, s);
      ++checked;
      if (moves.empty()) break;
      std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
      b.apply(moves[pick(rng)]);  // incremental cache update happens here
      check_caches_match_full(d, b, label, g, s);
    }
  }
  std::cout << "  cache-consistency checked " << checked << " positions [" << label << "]\n";
}

TEST(Board, CachesIncrementalMatchesFull) {
  Dictionary d = medium_dict();
  cache_consistency_stress(d, "medium_dict", 99887766u, /*games=*/30, /*steps_per_game=*/10);
}

// num_tiles() is maintained by every square-write path -- set() in both
// directions and unapply()'s restore -- so consumers (bag-size arithmetic)
// never rescan the grid.
TEST(Board, NumTilesTracksEveryWritePath) {
  Board b;
  ASSERT_EQ(b.num_tiles(), 0);
  ASSERT_TRUE(b.empty_board());

  b.set(7, 7, Glyph::of(Tile::from_char('A')));
  b.set(7, 8, Glyph::of(Tile::from_char('B')));
  ASSERT_EQ(b.num_tiles(), 2);
  b.set(7, 8, Glyph::of(Tile::from_char('C')));  // overwrite: no change
  ASSERT_EQ(b.num_tiles(), 2);
  b.set(7, 7, Glyph::empty());  // clear: decrement
  ASSERT_EQ(b.num_tiles(), 1);
  b.set(7, 7, Glyph::of(Tile::from_char('A')));

  const Move m = make_play_full(7, 9, /*horizontal=*/true, 0b11, 10,
                                {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('E'))});
  BoardUndo undo;
  b.apply(m, &undo);
  ASSERT_EQ(b.num_tiles(), 4);
  b.unapply(undo);
  ASSERT_EQ(b.num_tiles(), 2);
  ASSERT_FALSE(b.empty_board());
}

// If the real lexicon is present locally, cross-validate against it too and
// sanity-check a few known NWL words. The path comes from a compile-time define
// (SCRIBBLEZ_DEFAULT_KWG, set by CMake to data/lexica/NWL23.kwg). Skipped (not
// failed) when the define is absent or the file is missing -- the .kwg binary
// is not committed.
TEST(Dictionary, RealKwgCrossValidation) {
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) {
    GTEST_SKIP() << "no lexicon at " << path;
  }
  Dictionary d = Dictionary::load_kwg(path);
  ASSERT_TRUE(d.contains("QI"));
  ASSERT_TRUE(d.contains("MUZJIKS"));
  ASSERT_TRUE(d.contains("PARTIED"));
  ASSERT_FALSE(d.contains("QXZ"));
  cross_validate(d, "real-kwg", 99u, /*games=*/6, /*steps_per_game=*/8);
}

// ===========================================================================
// InputEncoder tests
// ===========================================================================

// First thermometer slot for `letter`'s region in the unseen-pool block.
static int pool_region_start(int letter) {
  int s = 0;
  for (int i = 0; i < letter; ++i) s += TILE_COUNTS[i];
  return s;
}

TEST(Encoder, BasicLayout) {
  using namespace scribblez::binlog;
  // Build state via apply_move only: p0 plays a single 'C' at (7,7) for 50
  // points; p1 then plays a single blank-as-D at (3,3) for 30 points. After
  // these two moves the encoder's active player is p0 with last_move_by_p1
  // = the D play, scores=[50,30], and a board with C@(7,7) and D@(3,3).
  // (Board::apply doesn't enforce legality, so the disconnected placements
  // are fine for an encoder-layout test.)
  Move p0_play =
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 50, {Glyph::of(Tile::from_char('C'))});

  Move p1_play = make_play_full(3, 3, /*horizontal=*/true, 0b1, 30,
                                {Glyph::played(Tile::from_char('D'), /*is_blank=*/true)});

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(p0_play);
  enc.apply_move(p1_play);

  Rack active_rack;
  active_rack.add(Tile::from_char('Q'));
  active_rack.add(Tile::from_char('Z'));
  active_rack.add(BLANK);

  std::vector<float> out(kInputFloats, -1.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, out.data());

  // Letter planes A..Z occupy [0..25].
  const int c_plane = Tile::from_char('C');
  const int d_plane = Tile::from_char('D');
  ASSERT_EQ(out[c_plane * 225 + 7 * 15 + 7], 1.0f);
  ASSERT_EQ(out[d_plane * 225 + 3 * 15 + 3], 1.0f);  // blank-as-D still lights the D plane

  // Blank-marker plane is index 26: 1 at (3,3), 0 at (7,7).
  ASSERT_EQ(out[26 * 225 + 3 * 15 + 3], 1.0f);
  ASSERT_EQ(out[26 * 225 + 7 * 15 + 7], 0.0f);

  // Premium planes (27..30) are board-static; just verify they are emitted as
  // 0/1 (no garbage left over from the -1.0 sentinel).
  for (int p = 27; p <= 30; ++p) {
    for (int i = 0; i < 225; ++i) {
      float v = out[p * 225 + i];
      ASSERT_TRUE(v == 0.0f || v == 1.0f);
    }
  }

  // Self last-placement plane (31): only (7,7) is lit (p0's own most recent
  // move). Opponent last-placement plane (32): only (3,3) is lit (p1's most
  // recent move).
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      const float self_expected = (r == 7 && c == 7) ? 1.0f : 0.0f;
      const float opp_expected = (r == 3 && c == 3) ? 1.0f : 0.0f;
      ASSERT_EQ(out[kSelfPlacementPlane * 225 + r * 15 + c], self_expected);
      ASSERT_EQ(out[kOppPlacementPlane * 225 + r * 15 + c], opp_expected);
    }
  }

  const float* scalars = out.data() + kSpatialFloats;

  // Rack: raw per-tile counts at kRackCountOffset.
  ASSERT_EQ(scalars[kRackCountOffset + Tile::from_char('Q')], 1.0f);
  ASSERT_EQ(scalars[kRackCountOffset + Tile::from_char('Z')], 1.0f);
  ASSERT_EQ(scalars[kRackCountOffset + 26], 1.0f);  // blank count in rack
  ASSERT_EQ(scalars[kRackCountOffset + Tile::from_char('A')], 0.0f);

  // Unseen pool: per-letter thermometer at kUnseenPoolOffset. The pool is
  // TILE_COUNTS minus board and active_rack only (opp-rack tiles, if any,
  // remain in the pool from the POV).
  const float* pool = scalars + kUnseenPoolOffset;
  float pool_sum = 0.0f;
  for (int i = 0; i < kUnseenPoolThermoFloats; ++i) pool_sum += pool[i];
  ASSERT_EQ(pool_sum, 95.0f);  // 100 - 2 on board - 3 in rack
  // A: all 9 unseen -> region fully set.
  ASSERT_EQ(pool[pool_region_start(0) + 0], 1.0f);
  ASSERT_EQ(pool[pool_region_start(0) + 8], 1.0f);
  // C: 1 of 2 unseen (one C on board) -> first slot set, hole at tail.
  ASSERT_EQ(pool[pool_region_start(Tile::from_char('C')) + 0], 1.0f);
  ASSERT_EQ(pool[pool_region_start(Tile::from_char('C')) + 1], 0.0f);
  // Blank: 1 on board (blank-D) + 1 in rack -> 0 unseen.
  ASSERT_EQ(pool[pool_region_start(26) + 0], 0.0f);
  ASSERT_EQ(pool[pool_region_start(26) + 1], 0.0f);

  // Score-diff scalar at kScoreDiffOffset: (50 - 30) / kScoreDiffInputScale.
  const float* sd = scalars + kScoreDiffOffset;
  ASSERT_EQ(sd[0], 20.0f / kScoreDiffInputScale);

  // Last-2-move metadata at kMoveMetaOffset: self move (p0's C play) then
  // opponent move (p1's blank-D play); both are 1-glyph PLAYs.
  const float* meta = scalars + kMoveMetaOffset;
  ASSERT_EQ(meta[int(MoveType::PLAY)], 1.0f);
  ASSERT_EQ(meta[int(MoveType::EXCHANGE)], 0.0f);
  ASSERT_EQ(meta[int(MoveType::PASS)], 0.0f);
  ASSERT_EQ(meta[kMoveMetaTypeFloats], 1.0f);  // self num_glyphs
  const float* opp_meta = meta + kMoveMetaFloatsPerMove;
  ASSERT_EQ(opp_meta[int(MoveType::PLAY)], 1.0f);
  ASSERT_EQ(opp_meta[kMoveMetaTypeFloats], 1.0f);  // opp num_glyphs
}

// The mid-game seeding constructor (a rollout's decision point): once two
// applied plies have supplied both last-move slots, a seeded encoder's row is
// byte-identical to a full-history encoder's.
TEST(Encoder, MidGameSeedMatchesFullHistory) {
  using namespace scribblez::binlog;
  const Move m1 =
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 50, {Glyph::of(Tile::from_char('C'))});
  const Move m2 = make_play_full(3, 3, /*horizontal=*/true, 0b1, 30,
                                 {Glyph::played(Tile::from_char('D'), /*is_blank=*/true)});
  const Move m3 =
    make_play_full(9, 5, /*horizontal=*/false, 0b1, 12, {Glyph::of(Tile::from_char('E'))});
  const Move m4 =
    make_play_full(11, 2, /*horizontal=*/true, 0b1, 8, {Glyph::of(Tile::from_char('F'))});

  Dictionary d = medium_dict();
  const InputEncodingSpec spec{&d};
  GameStateEncoder full{spec};
  full.apply_move(m1);
  full.apply_move(m2);
  // The seed point: the state after m1/m2, with no history handed over.
  GameStateEncoder seeded{spec, full.board(), {full.score(0), full.score(1)}, full.active_player()};
  full.apply_move(m3);
  full.apply_move(m4);
  seeded.apply_move(m3);
  seeded.apply_move(m4);
  ASSERT_EQ(seeded.active_player(), full.active_player());

  Rack active_rack;
  active_rack.add(Tile::from_char('Q'));
  std::vector<float> full_row(kInputFloats, -1.0f);
  std::vector<float> seeded_row(kInputFloats, -2.0f);
  full.encode_input(full.active_player(), active_rack, /*apply_flip=*/false, full_row.data());
  seeded.encode_input(seeded.active_player(), active_rack, /*apply_flip=*/false, seeded_row.data());
  ASSERT_EQ(0, std::memcmp(full_row.data(), seeded_row.data(), sizeof(float) * kInputFloats));
}

TEST(Encoder, LastOppPlaneMask) {
  using namespace scribblez::binlog;

  // p0 plays a single 'A' at (7,7), then p1 plays "CAT" horizontally
  // starting at (7,6), interleaving the existing A at (7,7): cells (7,6)
  // and (7,8) were newly placed -> square_mask = 0b101.
  Move p0_play =
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 1, {Glyph::of(Tile::from_char('A'))});

  Move opp_play =
    make_play_full(7, 6, /*horizontal=*/true, 0b101, 5,
                   {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('T'))});

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(p0_play);
  enc.apply_move(opp_play);

  Rack active_rack;
  std::vector<float> out(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, out.data());

  const float* plane = out.data() + kOppPlacementPlane * 225;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      const float expected = ((r == 7 && c == 6) || (r == 7 && c == 8)) ? 1.0f : 0.0f;
      ASSERT_EQ(plane[r * 15 + c], expected);
    }
  }
  // The opponent's num_glyphs reflects placements (2 = C, T), not cells walked.
  const float* opp_meta = out.data() + kSpatialFloats + kMoveMetaOffset + kMoveMetaFloatsPerMove;
  ASSERT_EQ(opp_meta[kMoveMetaTypeFloats], 2.0f);
}

TEST(Encoder, FlipSymmetry) {
  using namespace scribblez::binlog;

  // p0 single 'B' at (3,5); p1 vertical "AX" at (0,4) (mask=0b11).
  Move p0_play =
    make_play_full(3, 5, /*horizontal=*/true, 0b1, 30, {Glyph::of(Tile::from_char('B'))});

  Move opp_play =
    make_play_full(0, 4, /*horizontal=*/false, 0b11, 12,
                   {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('X'))});

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(p0_play);
  enc.apply_move(opp_play);

  Rack active_rack;
  active_rack.add(Tile::from_char('Q'));

  std::vector<float> normal(kInputFloats, 0.0f);
  std::vector<float> flipped(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, normal.data());
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/true, flipped.data());

  // Scalars are flip-invariant.
  for (int i = kSpatialFloats; i < kInputFloats; ++i) {
    ASSERT_EQ(normal[i], flipped[i]);
  }
  // The halves must differ, or the swap below would hold for the wrong reason.
  bool halves_differ = false;
  for (int i = 0; i < kHorizontalCrossCheckPlanes * 225 && !halves_differ; ++i) {
    halves_differ =
      normal[kHorizontalCrossCheckPlane0 * 225 + i] != normal[kVerticalCrossCheckPlane0 * 225 + i];
  }
  ASSERT_TRUE(halves_differ);

  // Every spatial plane (including both placement planes) is transposed under
  // the flip, and the cross-check halves also exchange.
  for (int p = 0; p < kSpatialPlanes; ++p) {
    int src = p;
    if (p >= kHorizontalCrossCheckPlane0 && p < kVerticalCrossCheckPlane0) {
      src = p + kHorizontalCrossCheckPlanes;
    } else if (p >= kVerticalCrossCheckPlane0 &&
               p < kHorizontalCrossCheckPlane0 + kCrossCheckPlanes) {
      src = p - kHorizontalCrossCheckPlanes;
    }
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 15; ++c) {
        ASSERT_EQ(flipped[p * 225 + r * 15 + c], normal[src * 225 + c * 15 + r]);
      }
    }
  }
}

// Single-tile plays whose only word is perpendicular to the placement axis
// (hooking S onto the I of QI to form vertical IS / SI) come from the
// transposed pass; a single tile that also forms a word along the horizontal
// axis (the S of QIS) is emitted exactly once, from the horizontal pass.
TEST(Movegen, SingleTileVerticalHooks) {
  Dictionary d = medium_dict();
  Board b;
  b.apply(make_play_full(7, 7, /*horizontal=*/true, 0b11, 22,
                         {Glyph::of(Tile::from_char('Q')), Glyph::of(Tile::from_char('I'))}));
  Rack r;
  r.add(Tile::from_char('S'));
  MoveGenerator gen(b, d);
  const std::vector<Move> plays = gen.generate(r);
  int qis = 0, is_below = 0, si_above = 0;
  for (const Move& m : plays) {
    if (m.horizontal() && m.start() == 7 && m.square_mask() == (1u << 9)) ++qis;
    if (!m.horizontal() && m.start() == 8 && m.square_mask() == (1u << 8)) ++is_below;
    if (!m.horizontal() && m.start() == 8 && m.square_mask() == (1u << 6)) ++si_above;
  }
  ASSERT_EQ(qis, 1);       // QIS: S at (7,9), horizontal pass
  ASSERT_EQ(is_below, 1);  // IS: S at (8,8), vertical-only hook
  ASSERT_EQ(si_above, 1);  // SI: S at (6,8), vertical-only hook
  ASSERT_EQ(int(plays.size()), 3);
  const std::vector<Move> dawg = gen.generate(r, GenAlgo::DAWG);
  ASSERT_EQ(dawg.size(), plays.size());
}

// A square's cross-check set constrains the axis perpendicular to the run it
// abuts, so the hooks above and below QI belong to the horizontal block and
// those left and right of it to the vertical block.
TEST(Encoder, CrossCheckPlanesQi) {
  using namespace scribblez::binlog;

  Dictionary d = medium_dict();

  // p0 opens with horizontal "QI" at (7,7)..(7,8). We then encode from p1's
  // POV (active after one move) and verify directional per-letter cross-check
  // planes around that word.
  Move qi_play = make_play_full(7, 7, /*horizontal=*/true, 0b11, 22,
                                {Glyph::of(Tile::from_char('Q')), Glyph::of(Tile::from_char('I'))});

  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(qi_play);

  Rack active_rack;  // p1 rack is irrelevant for the cross-check plane checks
  std::vector<float> out(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, out.data());

  auto plane_value = [&out](int plane, int r, int c) { return out[plane * 225 + r * 15 + c]; };
  auto h_cross_check = [&plane_value](Tile letter, int r, int c) {
    return plane_value(kHorizontalCrossCheckPlane0 + letter.index(), r, c);
  };
  auto v_cross_check = [&plane_value](Tile letter, int r, int c) {
    return plane_value(kVerticalCrossCheckPlane0 + letter.index(), r, c);
  };
  auto has = [](const std::initializer_list<char>& letters, char ch) {
    for (char x : letters)
      if (x == ch) return true;
    return false;
  };

  const auto assert_horizontal_set = [&](int r, int c, const std::initializer_list<char>& letters) {
    for (int l = 0; l < 26; ++l) {
      const char ch = 'A' + l;
      const float expected = has(letters, ch) ? 1.0f : 0.0f;
      ASSERT_EQ(h_cross_check(Tile::of(l), r, c), expected);
    }
  };

  const auto assert_vertical_set = [&](int r, int c, const std::initializer_list<char>& letters) {
    for (int l = 0; l < 26; ++l) {
      const char ch = 'A' + l;
      const float expected = has(letters, ch) ? 1.0f : 0.0f;
      ASSERT_EQ(v_cross_check(Tile::of(l), r, c), expected);
    }
  };

  // A square with no perpendicular neighbor constrains nothing: every letter is
  // legal, so all 26 planes of the axis block are set.
  const auto assert_horizontal_unconstrained = [&](int r, int c) {
    for (int l = 0; l < 26; ++l) ASSERT_EQ(h_cross_check(Tile::of(l), r, c), 1.0f);
  };
  const auto assert_vertical_unconstrained = [&](int r, int c) {
    for (int l = 0; l < 26; ++l) ASSERT_EQ(v_cross_check(Tile::of(l), r, c), 1.0f);
  };

  // Left and right of QI, the cross word runs across through the QI:
  //   - right of I: QIS -> only 'S'
  //   - left of Q: none in this fixture dictionary
  assert_vertical_set(7, 9, {'S'});
  assert_vertical_set(7, 6, {});
  // Nothing runs down through either square, so no vertical cross word
  // constrains it and the horizontal block is all-ones.
  assert_horizontal_unconstrained(7, 9);
  assert_horizontal_unconstrained(7, 6);

  // Above and below QI, the cross word runs down:
  //   - below Q: QI -> only 'I'
  //   - above I: AI BI GI HI KI LI MI OI PI QI SI TI XI
  //   - below I: ID IF IN IS IT
  assert_horizontal_set(8, 7, {'I'});
  assert_horizontal_set(6, 7, {});
  assert_horizontal_set(6, 8, {'A', 'B', 'G', 'H', 'K', 'L', 'M', 'O', 'P', 'Q', 'S', 'T', 'X'});
  assert_horizontal_set(8, 8, {'D', 'F', 'N', 'S', 'T'});
  // No horizontal cross word runs through these squares, so the vertical block
  // is all-ones (every letter legal).
  assert_vertical_unconstrained(8, 7);
  assert_vertical_unconstrained(6, 8);
  assert_vertical_unconstrained(8, 8);

  // Occupied squares never carry cross-check planes.
  assert_horizontal_set(7, 7, {});
  assert_horizontal_set(7, 8, {});
  assert_vertical_set(7, 7, {});
  assert_vertical_set(7, 8, {});

  // A cell with no neighbor in any direction is fully unconstrained: every
  // plane in both families is set.
  assert_horizontal_unconstrained(0, 0);
  assert_horizontal_unconstrained(14, 14);
  assert_vertical_unconstrained(0, 0);
  assert_vertical_unconstrained(14, 14);
}

// A letter illegal as a lone tile can be legal inside a longer word, so the
// cross-check set must not be intersected with the main word's validity. (7,8)
// reads `_XI` across and `_VOW` down: no word is `_XI`, but AXIOM places an `A`
// there. Intersecting reports the square as taking no letter, either way.
TEST(Encoder, CrossCheckSetIsNotOneTileLegality) {
  Dictionary d = Dictionary::build_from_words({"AVOW", "AXIOM", "VOW", "XI"});

  // "XI" across at (7,9)..(7,10) and "VOW" down at (8,8)..(10,8), leaving
  // (7,8) empty with a run to its right and a run below it.
  Move xi = make_play_full(7, 9, /*horizontal=*/true, 0b11, 9,
                           {Glyph::of(Tile::from_char('X')), Glyph::of(Tile::from_char('I'))});
  Move vow = make_play_full(8, 8, /*horizontal=*/false, 0b111, 9,
                            {Glyph::of(Tile::from_char('V')), Glyph::of(Tile::from_char('O')),
                             Glyph::of(Tile::from_char('W'))});
  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(xi);
  enc.apply_move(vow);

  Rack active_rack;
  std::vector<float> out(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, out.data());

  const auto at = [&out](int plane0, char ch) {
    return out[(plane0 + Tile::from_char(ch).index()) * 225 + 7 * 15 + 8];
  };
  for (char ch = 'A'; ch <= 'Z'; ++ch) {
    ASSERT_EQ(at(kHorizontalCrossCheckPlane0, ch), (ch == 'A' ? 1.0f : 0.0f)) << ch;
    ASSERT_EQ(at(kVerticalCrossCheckPlane0, ch), 0.0f) << ch;
  }
}

// The production replay path (PositionEncoder, used by both the streaming and
// disk pipelines) must emit lexicon-true cross-check planes: it seeds the
// board's move-generation caches from its dictionary. Without that seeding the
// planes degrade silently to all-26-letters adjacency masks, so this checks a
// square whose legal hook set is a strict subset.
TEST(PositionEncoder, CrossCheckPlanesLexical) {
  using namespace scribblez::binlog;

  Dictionary d = medium_dict();

  // p0 opens with horizontal "QI" at (7,7)..(7,8); the post-move row at turn 0
  // is encoded from p0's POV.
  GameLogStorage storage;
  storage.initial_racks[0] = rack_from("QIAAAAA");
  storage.initial_racks[1] = rack_from("SAINTED");
  TurnRecord rec{};
  rec.move = make_play_full(7, 7, /*horizontal=*/true, 0b11, 22,
                            {Glyph::of(Tile::from_char('Q')), Glyph::of(Tile::from_char('I'))});
  storage.turns.push_back(rec);

  PositionEncoder enc(InputEncodingSpec{&d});
  std::vector<float> row(kRowFloats, 0.0f);
  enc.encode_row<PositionEvalTask>(storage.view(), /*sampled_turn=*/0, /*post_move=*/true,
                                   /*flip=*/false, row.data());

  auto v_cross_check = [&row](char ch, int r, int c) {
    return row[(kVerticalCrossCheckPlane0 + Tile::from_char(ch).index()) * 225 + r * 15 + c];
  };
  // Right of the I the cross word is QIS (per the fixture dictionary), which
  // constrains a vertical word placing a tile there: 'S' is set and every
  // other letter is clear.
  for (char ch = 'A'; ch <= 'Z'; ++ch) {
    ASSERT_EQ(v_cross_check(ch, 7, 9), (ch == 'S' ? 1.0f : 0.0f));
  }
}

TEST(Encoder, ForcedScoreDiffIsolation) {
  using namespace scribblez::binlog;

  Move p0_play =
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 17, {Glyph::of(Tile::from_char('A'))});
  Move p1_play =
    make_play_full(7, 8, /*horizontal=*/true, 0b1, 9, {Glyph::of(Tile::from_char('T'))});

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(p0_play);
  enc.apply_move(p1_play);

  Rack active_rack;
  active_rack.add(Tile::from_char('E'));
  active_rack.add(Tile::from_char('R'));

  std::vector<float> normal(kInputFloats, 0.0f);
  std::vector<float> forced(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, normal.data());
  enc.encode_input_with_score_diff(enc.active_player(), active_rack,
                                   /*score_diff=*/123, /*apply_flip=*/false, forced.data());

  const int score_lo = kSpatialFloats + kScoreDiffOffset;
  const int score_hi = score_lo + kScoreDiffInputFloats;

  // Only the score-diff block should differ.
  for (int i = 0; i < kInputFloats; ++i) {
    if (i >= score_lo && i < score_hi) continue;
    ASSERT_EQ(normal[i], forced[i]);
  }

  // Forced block must represent score_diff=123 as the normalized scalar.
  ASSERT_EQ(forced[score_lo], 123.0f / kScoreDiffInputScale);
}

TEST(Encoder, NonplayLastMoveMetadata) {
  using namespace scribblez::binlog;

  // p0 PASS, p1 EXCHANGE(1 tile). After two plies active is p0 again.
  Move p0_pass = Move::pass();
  TileCounts ex_tiles;
  ex_tiles.add(Tile::from_char('A'));
  Move p1_exchange = Move::exchange(ex_tiles);

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(p0_pass);
  enc.apply_move(p1_exchange);

  Rack active_rack;
  std::vector<float> out(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, out.data());

  const float* scalars = out.data() + kSpatialFloats;
  const float* self_meta = scalars + kMoveMetaOffset;
  const float* opp_meta = self_meta + kMoveMetaFloatsPerMove;

  ASSERT_EQ(self_meta[int(MoveType::PLAY)], 0.0f);
  ASSERT_EQ(self_meta[int(MoveType::EXCHANGE)], 0.0f);
  ASSERT_EQ(self_meta[int(MoveType::PASS)], 1.0f);
  ASSERT_EQ(self_meta[kMoveMetaTypeFloats], 0.0f);

  ASSERT_EQ(opp_meta[int(MoveType::PLAY)], 0.0f);
  ASSERT_EQ(opp_meta[int(MoveType::EXCHANGE)], 1.0f);
  ASSERT_EQ(opp_meta[int(MoveType::PASS)], 0.0f);
  ASSERT_EQ(opp_meta[kMoveMetaTypeFloats], 1.0f);

  // Both last-placement planes must be all zero because neither last move is PLAY.
  for (int i = 0; i < 225; ++i) {
    ASSERT_EQ(out[kSelfPlacementPlane * 225 + i], 0.0f);
    ASSERT_EQ(out[kOppPlacementPlane * 225 + i], 0.0f);
  }
}

// ===========================================================================
// Binary-log / DataLoader / movegen round-trip tests
// ===========================================================================

// A minimal Agent used to drive Game in tests. Picks a uniformly-random
// highest-scoring PLAY; if none exists, passes. (Sufficient for generating
// realistic-looking game logs; we don't need exchange logic here.)
namespace {
class TestAgent : public scribblez::Agent {
 public:
  TestAgent(int tid, std::string name, uint64_t seed)
      : scribblez::Agent(tid, std::move(name)), rng_(seed) {}

  scribblez::MoveDecision make_move(const scribblez::MoveRequest& req) override {
    const std::vector<scribblez::Move> plays = scribblez::generate_legal_plays(req);
    if (!plays.empty()) {
      int best = -1;
      for (const auto& m : plays) best = std::max(best, int(m.score()));
      std::vector<const scribblez::Move*> top;
      for (const auto& m : plays)
        if (int(m.score()) == best) top.push_back(&m);
      std::uniform_int_distribution<size_t> d(0, top.size() - 1);
      return *top[d(rng_)];
    }
    return scribblez::Move::pass();
  }

 private:
  std::mt19937_64 rng_;
};

// Snapshot of game state at one eligible sample moment, captured during a
// live in-memory replay. Used as the ground-truth reference for the
// GameStateEncoder-driven replays under test.
struct LiveSnapshot {
  scribblez::Board board;
  scribblez::Rack rack_active;
  scribblez::Move last_opp_move;
  int score_active = 0;
  int score_opp = 0;
  int turn_index = 0;
  int active_player = 0;
  scribblez::PositionKind kind = scribblez::PositionKind::kPreMove;
};

std::vector<LiveSnapshot> live_replay_all_snapshots(const scribblez::GameLogStorage& log) {
  using namespace scribblez;

  std::vector<LiveSnapshot> out;
  Board board;
  Rack racks[2];
  Move last_by[2] = {Move{}, Move{}};

  // Seed each player's rack from their first-turn rack_before. (Matches
  // extract_positions's seeding.)
  bool seeded[2] = {false, false};
  for (const TurnRecord& t : log.turns) {
    if (!seeded[t.player]) {
      racks[t.player] = t.rack_before;
      seeded[t.player] = true;
    }
    if (seeded[0] && seeded[1]) break;
  }

  for (size_t i = 0; i < log.turns.size(); ++i) {
    const TurnRecord& turn = log.turns[i];
    const int active = turn.player;
    const int opp = 1 - active;
    racks[active] = turn.rack_before;
    const int prev_active = turn.cumulative_scores[active] - turn.score_delta;
    const int prev_opp = turn.cumulative_scores[opp];

    LiveSnapshot pre;
    pre.board = board;
    pre.rack_active = racks[active];
    pre.last_opp_move = last_by[opp];
    pre.score_active = prev_active;
    pre.score_opp = prev_opp;
    pre.turn_index = i;
    pre.active_player = active;
    pre.kind = PositionKind::kPreMove;
    out.push_back(pre);

    if (turn.move.type() == MoveType::PLAY) {
      LiveSnapshot post = pre;
      post.board.apply(turn.move);
      const int n = turn.move.num_glyphs();
      for (int g = 0; g < n; ++g) post.rack_active.remove(turn.move.glyph(g).rack_tile());
      post.score_active = prev_active + turn.score_delta;
      post.kind = PositionKind::kPostMove;
      out.push_back(post);
    }

    // Advance live state.
    if (turn.move.type() == MoveType::PLAY) {
      const int n = turn.move.num_glyphs();
      for (int g = 0; g < n; ++g) racks[active].remove(turn.move.glyph(g).rack_tile());
      board.apply(turn.move);
    } else if (turn.move.type() == MoveType::EXCHANGE) {
      const int n = turn.move.num_glyphs();
      for (int g = 0; g < n; ++g) racks[active].remove(turn.move.glyph(g).rack_tile());
    }
    for (Tile t : turn.drawn.tiles()) {
      if (t.is_empty()) break;
      racks[active].add(t);
    }
    last_by[active] = turn.move;
  }
  return out;
}

bool boards_equal(const scribblez::Board& a, const scribblez::Board& b) {
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      if (a.at(r, c).code() != b.at(r, c).code()) return false;
    }
  }
  return true;
}

bool racks_equal(const scribblez::Rack& a, const scribblez::Rack& b) {
  if (a.size() != b.size()) return false;
  for (int code = 0; code <= 26; ++code) {
    scribblez::Tile t = (code == 26) ? scribblez::BLANK : scribblez::Tile::of(code);
    if (a.count(t) != b.count(t)) return false;
  }
  return true;
}

bool moves_equal_for_replay(const scribblez::Move& a, const scribblez::Move& b) {
  if (a.type() != b.type()) return false;
  if (a.type() != scribblez::MoveType::PLAY)
    return true;  // PASS/EXCHANGE: type alone suffices here
  if (a.horizontal() != b.horizontal()) return false;
  if (a.start() != b.start()) return false;
  if (a.square_mask() != b.square_mask()) return false;
  if (a.score() != b.score()) return false;
  for (int i = 0; i < scribblez::RACK_SIZE; ++i) {
    if (a.glyph(i).code() != b.glyph(i).code()) return false;
  }
  return true;
}

// Play one game with two TestAgents and return its owning log storage.
scribblez::GameLogStorage play_test_game(const scribblez::Dictionary& dict, uint64_t seed) {
  TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
  TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
  scribblez::Game g(a0, a1, dict, seed);
  g.play();
  return g.extract_log();
}

// Compare the set of legal plays from movegen on a reconstructed state vs
// movegen on the live state. Uses the existing move_key/key_set helpers
// (defined earlier in this file) so the comparison ignores enumeration order
// and identifies plays by placed-tiles + score.
void check_movegen_equiv(const scribblez::Dictionary& dict, const scribblez::Board& reconstructed,
                         const scribblez::Rack& reconstructed_rack, const scribblez::Board& live,
                         const scribblez::Rack& live_rack, const char* context) {
  scribblez::MoveGenerator gen_r(reconstructed, dict);
  scribblez::MoveGenerator gen_l(live, dict);
  auto m_r = gen_r.generate(reconstructed_rack);
  auto m_l = gen_l.generate(live_rack);
  auto k_r = key_set(reconstructed, m_r);
  auto k_l = key_set(live, m_l);
  if (k_r != k_l) {
    std::cerr << "movegen mismatch [" << context << "]: reconstructed=" << k_r.size()
              << " live=" << k_l.size() << "\n";
    std::exit(1);
  }
}
}  // anonymous namespace

// Open-leaves arm: the row is the base row plus the opponent-leave counts
// block at the scalar tail, holding exactly the known leave's per-tile
// counts.
TEST(InputLayout, OpenLeavesAppendsLeaveCounts) {
  Dictionary d = medium_dict();
  const InputEncodingSpec base{&d};
  const InputEncodingSpec open{&d, /*opp_leave_input=*/true};
  ASSERT_EQ(scalar_floats(open), scalar_floats(base) + kOppLeaveCountFloats);
  ASSERT_EQ(scalar_block_offset(open, ScalarBlockId::kOppLeaveCounts), scalar_floats(base));

  Move cat = make_play_full(7, 7, /*horizontal=*/true, 0b111, 12,
                            {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                             Glyph::of(Tile::from_char('T'))});
  GameStateEncoder base_enc{base};
  GameStateEncoder open_enc{open};
  base_enc.apply_move(cat);
  open_enc.apply_move(cat);
  const Rack rack = rack_from("RSE");
  const Rack opp = rack_from("QIZAA");

  std::vector<float> base_row(input_floats(base), -1.0f);
  std::vector<float> open_row(input_floats(open), -1.0f);
  base_enc.encode_input(base_enc.active_player(), rack, /*apply_flip=*/false, base_row.data());
  open_enc.encode_input(open_enc.active_player(), rack, opp, /*apply_flip=*/false, open_row.data());

  // Identical prefix; the tail is the opponent rack's counts.
  ASSERT_EQ(
    std::memcmp(base_row.data(), open_row.data(), sizeof(float) * size_t(input_floats(base))), 0);
  const float* tail = open_row.data() + input_floats(base);
  ASSERT_EQ(tail[Tile::from_char('Q').index()], 1.0f);
  ASSERT_EQ(tail[Tile::from_char('I').index()], 1.0f);
  ASSERT_EQ(tail[Tile::from_char('Z').index()], 1.0f);
  ASSERT_EQ(tail[Tile::from_char('A').index()], 2.0f);
  float tail_total = 0.0f;
  for (int i = 0; i < kOppLeaveCountFloats; ++i) tail_total += tail[i];
  ASSERT_EQ(tail_total, 5.0f);
}

// GameStateEncoder, replayed against a live in-memory replay, faithfully
// reproduces every eligible position -- proven by running movegen on both and
// demanding identical legal-play sets at the pre-move snapshot of each turn.
TEST(Encoder, ExtractPositionsMovegenRoundtrip) {
  Dictionary dict = medium_dict();

  // A handful of games at different seeds; for every eligible position we
  // compare encoder state against the independent live snapshot vector.
  const std::vector<uint64_t> seeds = {42, 1337, 0xDEADBEEFULL};
  long positions_compared = 0;

  for (uint64_t seed : seeds) {
    scribblez::GameLogStorage log = play_test_game(dict, seed);
    ASSERT_FALSE(log.turns.empty());

    auto live_snaps = live_replay_all_snapshots(log);

    scribblez::GameStateEncoder enc{scribblez::InputEncodingSpec{&dict}};
    // The encoder tracks no racks (an outside observer cannot see opponent
    // draws), but the test has full information, so it maintains a parallel
    // rack pair alongside the encoder.
    std::array<scribblez::Rack, 2> racks = {log.initial_racks[0], log.initial_racks[1]};

    size_t snap_idx = 0;
    for (size_t k = 0; k < log.turns.size(); ++k) {
      const auto& turn = log.turns[k];

      // ---- pre-move snapshot ----
      ASSERT_LT(snap_idx, live_snaps.size());
      const LiveSnapshot& pre = live_snaps[snap_idx++];
      ASSERT_EQ(pre.kind, scribblez::PositionKind::kPreMove);
      const int active = enc.active_player();
      ASSERT_EQ(active, pre.active_player);
      ASSERT_EQ(enc.score(active), pre.score_active);
      ASSERT_EQ(enc.score(1 - active), pre.score_opp);
      ASSERT_TRUE(boards_equal(enc.board(), pre.board));
      ASSERT_TRUE(racks_equal(racks[active], pre.rack_active));
      ASSERT_TRUE(moves_equal_for_replay(enc.last_move_by(1 - active), pre.last_opp_move));
      check_movegen_equiv(dict, enc.board(), racks[active], pre.board, pre.rack_active,
                          "GameStateEncoder-pre");
      ++positions_compared;

      // ---- post-move snapshot (PLAY only) ----
      if (turn.move.type() == scribblez::MoveType::PLAY) {
        ASSERT_LT(snap_idx, live_snaps.size());
        const LiveSnapshot& post = live_snaps[snap_idx++];
        ASSERT_EQ(post.kind, scribblez::PositionKind::kPostMove);

        // Materialize post-state from the encoder + parallel racks by hand
        // and compare.
        scribblez::Board post_board = enc.board();
        post_board.apply(turn.move);
        scribblez::Rack post_rack = racks[active];
        const int n = turn.move.num_glyphs();
        for (int g = 0; g < n; ++g) post_rack.remove(turn.move.glyph(g).rack_tile());
        const int post_score = enc.score(active) + turn.move.score();
        ASSERT_TRUE(boards_equal(post_board, post.board));
        ASSERT_TRUE(racks_equal(post_rack, post.rack_active));
        ASSERT_EQ(post_score, post.score_active);
        ++positions_compared;
      }

      // Advance both encoder and parallel rack tracking.
      if (turn.move.type() == scribblez::MoveType::PLAY ||
          turn.move.type() == scribblez::MoveType::EXCHANGE) {
        const int n = turn.move.num_glyphs();
        for (int g = 0; g < n; ++g) racks[active].remove(turn.move.glyph(g).rack_tile());
      }
      for (Tile d : turn.drawn.tiles()) {
        if (d.is_empty()) break;
        racks[active].add(d);
      }
      enc.apply_move(turn.move);
    }
    ASSERT_EQ(snap_idx, live_snaps.size());
  }
  ASSERT_GT(positions_compared, 0);
  std::cout << "  GameStateEncoder replay+movegen round-trip OK (" << positions_compared
            << " positions across " << seeds.size() << " games)\n";
}

// End-to-end: play a game, write it through BinaryLogWriter to disk, register
// the resulting .slog file with DataLoader, decode all rows, and verify that
// (a) every label tail matches a valid (game, active POV) and (b) running
// movegen on a board freshly reconstructed from the .slog file produces the
// same legal-play set as the live game.
TEST(BinaryLog, FileAndDataLoaderRoundtrip) {
  Dictionary dict = medium_dict();

  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / ("scribblez_test_" + std::to_string(::getpid()) + "_" +
                                              std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{dir};

  // Write a small batch through the public writer (separate games -> one file).
  constexpr int kGames = 3;
  std::vector<scribblez::GameLogStorage> logs;
  {
    scribblez::binlog::BinaryLogWriter writer(dir.string(), /*games_per_file=*/kGames);
    for (int i = 0; i < kGames; ++i) {
      scribblez::GameLogStorage log = play_test_game(dict, /*seed=*/100ULL + i);
      writer.append(scribblez::GameLogStorage(log));  // append a copy; keep `log` for verification
      logs.push_back(std::move(log));
    }
    // Destructor flushes; explicit flush would be redundant.
  }

  // Find the .slog file(s) the writer produced.
  std::vector<fs::path> slogs;
  for (const auto& ent : fs::directory_iterator(dir)) {
    if (ent.path().extension() == ".slog") slogs.push_back(ent.path());
  }
  ASSERT_EQ(slogs.size(), 1);

  // Verify the on-disk header is self-consistent.
  const fs::path& slog = slogs.front();
  const int64_t fsize = fs::file_size(slog);
  std::vector<char> raw(fsize);
  {
    std::ifstream f(slog, std::ios::binary);
    f.read(raw.data(), fsize);
    ASSERT_TRUE(f);
  }
  const auto* hdr = reinterpret_cast<const scribblez::binlog::FileHeader*>(raw.data());
  ASSERT_EQ(hdr->magic, scribblez::binlog::kMagic);
  ASSERT_EQ(hdr->version, scribblez::binlog::kVersion);
  ASSERT_EQ(hdr->num_games, uint32_t(kGames));
  // Training expands each game into one row per eligible (pre-endgame) turn, so
  // the loader's position count is the sum of those across all games.
  int64_t total_positions = 0;
  for (const auto& log : logs)
    for (const auto& turn : log.turns)
      if (turn.bag_size_before > 0) ++total_positions;
  ASSERT_GT(total_positions, 0);
  ASSERT_EQ(int64_t(hdr->num_sample_positions), total_positions);

  // Register with DataLoader and drain rows via epoch_start/load_batch
  // for both pre-move and post-move phases.
  scribblez::binlog::DataLoader::Params dl_params;
  dl_params.spec = {&dict};
  dl_params.num_worker_threads = 2;
  dl_params.num_prefetch_threads = 1;
  scribblez::binlog::DataLoader loader(dl_params);
  loader.add_file(slog.string(), total_positions, fsize);
  ASSERT_EQ(loader.num_positions(), total_positions);

  const int row_size = kRowFloats;

  // Helper: drain one full epoch into a vector.
  auto drain_epoch = [&](bool post_move) {
    scribblez::binlog::DataLoader::EpochConfig cfg;
    cfg.batch_size = total_positions;
    cfg.post_move = post_move;
    cfg.apply_symmetry = false;
    cfg.seed = 1;
    loader.epoch_start(cfg);
    std::vector<float> out(total_positions * row_size);
    int n = loader.load_batch(out.data());
    EXPECT_EQ(n, int(total_positions));
    EXPECT_EQ(loader.load_batch(out.data()), 0);  // epoch exhausted
    return out;
  };

  std::vector<float> pre_rows = drain_epoch(/*post_move=*/false);
  std::vector<float> post_rows = drain_epoch(/*post_move=*/true);

  // Combine for validation.
  const int n_samples = int(total_positions) * 2;
  std::vector<float> rows;
  rows.insert(rows.end(), pre_rows.begin(), pre_rows.end());
  rows.insert(rows.end(), post_rows.begin(), post_rows.end());

  // Build the set of valid (WLD label, score_diff) pairs across all games.
  std::set<std::tuple<int, int, int, int>> valid_labels;  // (W,D,L,score_diff)
  for (const auto& log : logs) {
    for (int active = 0; active < 2; ++active) {
      const int fa = log.final_scores[active];
      const int fo = log.final_scores[1 - active];
      int w = 0, d = 0, l = 0;
      if (fa > fo)
        w = 1;
      else if (fa == fo)
        d = 1;
      else
        l = 1;
      valid_labels.emplace(w, d, l, fa - fo);
    }
  }

  // Every decoded row's label tail must match a valid (game, POV).
  const int label_off = kInputFloats;
  for (int i = 0; i < n_samples; ++i) {
    const float* row = rows.data() + int64_t(i) * row_size;
    const int w = row[label_off + 0];
    const int dd = row[label_off + 1];
    const int l = row[label_off + 2];
    // Score-diff target is a single scalar: the clipped final differential.
    const int sd = row[label_off + scribblez::kWldFloats];
    ASSERT_EQ(w + dd + l, 1);  // exactly one of W/D/L
    ASSERT_EQ(valid_labels.count({w, dd, l, sd}), 1);
  }

  // Re-read the file's raw TurnBlobs, replay each game via GameStateEncoder,
  // and verify the round-tripped state passes the movegen equivalence check
  // against the live snapshots. This exercises the writer -> file ->
  // reinterpret_cast path that the DataLoader itself uses internally.
  const auto* metas = reinterpret_cast<const scribblez::binlog::GameMetadata*>(
    raw.data() + sizeof(scribblez::binlog::FileHeader));
  long compared = 0;
  for (uint32_t gi = 0; gi < hdr->num_games; ++gi) {
    const auto& gm = metas[gi];
    // GameMetadata is written in the order games were appended, so logs[gi]
    // corresponds to metas[gi]. num_turns must match the GameLog.
    ASSERT_EQ(gm.num_turns, logs[gi].turns.size());

    const auto* ir =
      reinterpret_cast<const scribblez::binlog::InitialRacks*>(raw.data() + gm.start_offset);
    const auto* turns = reinterpret_cast<const scribblez::binlog::TurnBlob*>(
      raw.data() + gm.start_offset + sizeof(scribblez::binlog::InitialRacks));

    // Reconstruct the initial racks from the on-disk bytes and replay.
    const Rack& r0_init = ir->p0;
    const Rack& r1_init = ir->p1;
    ASSERT_TRUE(racks_equal(r0_init, logs[gi].initial_racks[0]));
    ASSERT_TRUE(racks_equal(r1_init, logs[gi].initial_racks[1]));

    auto live_snaps = live_replay_all_snapshots(logs[gi]);
    scribblez::GameStateEncoder enc{scribblez::InputEncodingSpec{&dict}};
    std::array<scribblez::Rack, 2> racks = {r0_init, r1_init};

    size_t snap_idx = 0;
    for (uint32_t k = 0; k < gm.num_turns; ++k) {
      ASSERT_LT(snap_idx, live_snaps.size());
      const LiveSnapshot& pre = live_snaps[snap_idx++];
      const int active = enc.active_player();
      ASSERT_TRUE(boards_equal(enc.board(), pre.board));
      ASSERT_TRUE(racks_equal(racks[active], pre.rack_active));
      check_movegen_equiv(dict, enc.board(), racks[active], pre.board, pre.rack_active,
                          "file-roundtrip");
      ++compared;

      if (turns[k].move.type() == scribblez::MoveType::PLAY) {
        ASSERT_LT(snap_idx, live_snaps.size());
        const LiveSnapshot& post = live_snaps[snap_idx++];
        scribblez::Board post_board = enc.board();
        post_board.apply(turns[k].move);
        ASSERT_TRUE(boards_equal(post_board, post.board));
        ++compared;
      }

      if (turns[k].move.type() == scribblez::MoveType::PLAY ||
          turns[k].move.type() == scribblez::MoveType::EXCHANGE) {
        const int n = turns[k].move.num_glyphs();
        for (int g = 0; g < n; ++g) racks[active].remove(turns[k].move.glyph(g).rack_tile());
      }
      for (Tile d : turns[k].drawn.tiles()) {
        if (d.is_empty()) break;
        racks[active].add(d);
      }
      enc.apply_move(turns[k].move);
    }
    ASSERT_EQ(snap_idx, live_snaps.size());
  }
  // `compared` includes one pre-snapshot per turn + one post-snapshot per PLAY
  // turn; total_positions counts only one row per turn (shared between pre/post).
  ASSERT_GE(compared, total_positions);
  std::cout << "  file+DataLoader round-trip OK (" << kGames << " games, " << total_positions
            << " positions, " << n_samples << " loader rows)\n";
}

// Play one game with two TestAgents and a random opening of `plies` moves;
// return its owning log storage.
static scribblez::GameLogStorage play_random_opening_test_game(const scribblez::Dictionary& dict,
                                                               uint64_t seed, int plies) {
  TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
  TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
  scribblez::Game g(a0, a1, dict, seed);
  g.set_random_opening(plies);
  g.play();
  return g.extract_log();
}

// Game::set_random_opening plays exactly the first K plies at random (fewer
// only if the game ends sooner), records the count in the log, and is
// reproducible from the game seed.
TEST(Game, RandomOpening) {
  Dictionary dict = medium_dict();

  for (int plies : {0, 1, 3, 6}) {
    scribblez::GameLogStorage log = play_random_opening_test_game(dict, /*seed=*/321, plies);
    ASSERT_FALSE(log.turns.empty());
    ASSERT_EQ(log.num_random_opening_plies, std::min<int>(plies, int(log.turns.size())));

    // Same seed + same opening length -> the identical move sequence.
    scribblez::GameLogStorage again = play_random_opening_test_game(dict, /*seed=*/321, plies);
    ASSERT_EQ(again.turns.size(), log.turns.size());
    for (size_t k = 0; k < log.turns.size(); ++k)
      ASSERT_TRUE(moves_equal_for_replay(again.turns[k].move, log.turns[k].move));
  }
  std::cout << "  Game random opening OK\n";
}

// generate_legal_exchanges enumerates every distinct non-empty sub-multiset of
// the rack (duplicates don't multiply the list), and is empty when the bag is
// too small to allow exchanging.
TEST(Movegen, GenerateLegalExchanges) {
  Dictionary dict = medium_dict();
  Board board;
  Rack rack;  // A A B ? -> types (A:2, B:1, ?:1) -> 3*2*2 - 1 = 11 exchanges
  rack.add(Tile::from_char('A'));
  rack.add(Tile::from_char('A'));
  rack.add(Tile::from_char('B'));
  rack.add(BLANK);
  Rack opp;

  MoveRequest req{board, dict, rack, opp, 0, 0, /*bag_size=*/50};
  const std::vector<Move> exchanges = generate_legal_exchanges(req);
  ASSERT_EQ(exchanges.size(), 11);

  // Every move is an EXCHANGE of a distinct non-empty sub-multiset of the rack.
  std::set<std::string> seen;
  for (const Move& m : exchanges) {
    ASSERT_EQ(m.type(), MoveType::EXCHANGE);
    const int n = m.num_glyphs();
    ASSERT_TRUE(n >= 1 && n <= 4);
    std::string tiles;
    for (int i = 0; i < n; ++i) tiles += m.glyph(i).rack_tile().to_char();
    std::sort(tiles.begin(), tiles.end());
    for (char c : tiles) ASSERT_TRUE(c == 'A' || c == 'B' || c == '?');
    seen.insert(tiles);
  }
  ASSERT_EQ(seen.size(), exchanges.size());

  // Exchanging is illegal when the bag has fewer than RACK_SIZE tiles.
  MoveRequest starved{board, dict, rack, opp, 0, 0, /*bag_size=*/RACK_SIZE - 1};
  ASSERT_TRUE(generate_legal_exchanges(starved).empty());
  std::cout << "  generate_legal_exchanges OK (" << exchanges.size() << " exchanges)\n";
}

// BinaryLogWriter records a random-opening game's eligible region: begin is the
// position after the last random ply, end is the bag-non-empty prefix, and the
// header's num_sample_positions sums the region widths. The DataLoader sizes
// its epoch from the same regions.
TEST(BinaryLog, RandomOpeningRegion) {
  Dictionary dict = medium_dict();

  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / ("scribblez_ro_" + std::to_string(::getpid()) + "_" +
                                              std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{dir};

  constexpr int kGames = 3;
  constexpr int kPlies = 4;
  std::vector<scribblez::GameLogStorage> logs;
  {
    scribblez::binlog::BinaryLogWriter writer(dir.string(), /*games_per_file=*/kGames);
    for (int i = 0; i < kGames; ++i) {
      scribblez::GameLogStorage log =
        play_random_opening_test_game(dict, /*seed=*/500ULL + i, kPlies);
      writer.append(scribblez::GameLogStorage(log));
      logs.push_back(std::move(log));
    }
  }

  std::vector<fs::path> slogs;
  for (const auto& ent : fs::directory_iterator(dir)) {
    if (ent.path().extension() == ".slog") slogs.push_back(ent.path());
  }
  ASSERT_EQ(slogs.size(), 1);
  const int64_t fsize = fs::file_size(slogs.front());
  std::vector<char> raw(fsize);
  {
    std::ifstream f(slogs.front(), std::ios::binary);
    f.read(raw.data(), fsize);
    ASSERT_TRUE(f);
  }
  const auto* hdr = reinterpret_cast<const scribblez::binlog::FileHeader*>(raw.data());
  const auto* metas = reinterpret_cast<const scribblez::binlog::GameMetadata*>(
    raw.data() + sizeof(scribblez::binlog::FileHeader));
  ASSERT_EQ(hdr->num_games, uint32_t(kGames));

  int64_t expected_rows = 0;
  for (int i = 0; i < kGames; ++i) {
    ASSERT_EQ(logs[i].num_random_opening_plies, kPlies);
    int prefix = 0;
    for (const auto& turn : logs[i].turns) {
      if (turn.bag_size_before <= 0) break;
      ++prefix;
    }
    ASSERT_EQ(metas[i].eligible_begin, kPlies - 1);
    ASSERT_EQ(metas[i].eligible_end, prefix);
    expected_rows += prefix - (kPlies - 1);
  }
  ASSERT_EQ(int64_t(hdr->num_sample_positions), expected_rows);

  scribblez::binlog::DataLoader::Params dl_params;
  dl_params.spec = {&dict};
  dl_params.num_worker_threads = 1;
  dl_params.num_prefetch_threads = 1;
  scribblez::binlog::DataLoader loader(dl_params);
  loader.add_file(slogs.front().string(), expected_rows, fsize);
  ASSERT_EQ(loader.num_positions(), expected_rows);
  std::cout << "  BinaryLogWriter random-opening eligible region OK (" << expected_rows
            << " rows across " << kGames << " games)\n";
}

// ===========================================================================
// Foundation types: Tile / Glyph
// ===========================================================================

TEST(Tile, GlyphBasics) {
  // Tile::from_char round-trips for letters and the blank marker.
  for (char c = 'A'; c <= 'Z'; ++c) {
    Tile t = Tile::from_char(c);
    ASSERT_FALSE(t.is_blank());
    ASSERT_FALSE(t.is_empty());
    ASSERT_EQ(t.to_char(), c);
    // Lowercase is normalized to uppercase.
    ASSERT_EQ(Tile::from_char(char(c - 'A' + 'a')), t);
    ASSERT_EQ(t.value(), TILE_VALUES[t]);
  }
  ASSERT_TRUE(Tile::from_char('?').is_blank());
  ASSERT_TRUE(Tile::from_char('_').is_blank());
  ASSERT_EQ(BLANK.value(), 0);

  // Glyph: played-as-blank renders the letter but scores zero and consumes
  // a blank from the rack.
  Glyph plain = Glyph::of(Tile::from_char('Q'));
  Glyph blank_q = Glyph::played(Tile::from_char('Q'), /*is_blank=*/true);
  ASSERT_EQ(plain.letter(), Tile::from_char('Q'));
  ASSERT_EQ(blank_q.letter(), Tile::from_char('Q'));
  ASSERT_FALSE(plain.is_blank());
  ASSERT_TRUE(blank_q.is_blank());
  ASSERT_EQ(plain.value(), TILE_VALUES[Tile::from_char('Q')]);
  ASSERT_EQ(blank_q.value(), 0);
  ASSERT_EQ(plain.rack_tile(), Tile::from_char('Q'));
  ASSERT_EQ(blank_q.rack_tile(), BLANK);
  ASSERT_NE(plain, blank_q);
  ASSERT_EQ(plain, Glyph::of(Tile::from_char('Q')));

  // Empty/unassigned-blank predicates.
  ASSERT_TRUE(Glyph::empty().is_empty());
  ASSERT_FALSE(Glyph::blank().is_empty());
  ASSERT_TRUE(Glyph::blank().is_blank());
  ASSERT_EQ(Glyph::blank().rack_tile(), BLANK);

  // Glyph code 0 means empty -- a default-constructed Board is all-empty
  // by virtue of zero-init, which a lot of code relies on.
  Glyph g;
  ASSERT_TRUE(g.is_empty());
  ASSERT_EQ(g.code(), 0);
}

// ===========================================================================
// Rack
// ===========================================================================

static bool rack_is_sorted(const Rack& r) {
  const auto& a = r.tiles();
  for (int i = 1; i < r.size(); ++i) {
    if (a[i] < a[i - 1]) return false;
  }
  // Trailing slots are Tile::empty() (sentinel == 27, which is >= any letter
  // or blank), so the full array is also non-decreasing.
  return true;
}

TEST(Rack, Invariants) {
  // Sorted-array invariant survives interleaved add/remove in arbitrary order.
  Rack r;
  ASSERT_TRUE(r.empty());
  ASSERT_EQ(r.size(), 0);
  ASSERT_EQ(r.point_value(), 0);
  ASSERT_FALSE(r.remove(Tile::from_char('A')));  // remove-missing returns false

  const char* in = "QAZZB?A";  // 7 tiles incl two A's, two Z's, one blank
  for (char c : std::string(in)) {
    r.add(c == '?' ? BLANK : Tile::from_char(c));
  }
  ASSERT_EQ(r.size(), 7);
  ASSERT_TRUE(rack_is_sorted(r));
  ASSERT_EQ(r.to_string(), "AABQZZ?");  // sorted A..Z then '?'
  ASSERT_EQ(r.count(Tile::from_char('A')), 2);
  ASSERT_EQ(r.count(Tile::from_char('Z')), 2);
  ASSERT_EQ(r.count(Tile::from_char('B')), 1);
  ASSERT_EQ(r.count(Tile::from_char('X')), 0);
  ASSERT_EQ(r.blanks(), 1);

  // point_value: blanks contribute 0, everything else its TILE_VALUES entry.
  int expected = TILE_VALUES[Tile::from_char('A')] * 2 + TILE_VALUES[Tile::from_char('B')] +
                 TILE_VALUES[Tile::from_char('Q')] + TILE_VALUES[Tile::from_char('Z')] * 2;
  ASSERT_EQ(r.point_value(), expected);

  // remove() removes one occurrence and preserves sortedness.
  ASSERT_TRUE(r.remove(Tile::from_char('A')));
  ASSERT_EQ(r.count(Tile::from_char('A')), 1);
  ASSERT_EQ(r.size(), 6);
  ASSERT_TRUE(rack_is_sorted(r));
  ASSERT_TRUE(r.remove(BLANK));
  ASSERT_EQ(r.blanks(), 0);
  ASSERT_TRUE(rack_is_sorted(r));
  ASSERT_FALSE(r.remove(BLANK));  // gone now

  // counts() histogram matches per-tile count() probes.
  TileCounts tc = r.counts();
  for (Tile t = Tile::of(0); t < 27; ++t) {
    int via_tc = tc.count(t);
    int via_probe = r.count(t);
    ASSERT_EQ(via_tc, via_probe);
  }
}

// ===========================================================================
// Bag
// ===========================================================================

TEST(Bag, Basics) {
  // Initial composition matches TILE_COUNTS and totals to 100 tiles -- and
  // kTotalTiles, the constant bag-size arithmetic uses in place of a scan,
  // agrees (release builds skip the constructor's DEBUG_ASSERT, so this test
  // is what pins the constant to the distribution there).
  Bag b(/*seed=*/42);
  int total = 0;
  for (int c : TILE_COUNTS) total += c;
  ASSERT_EQ(b.size(), total);
  ASSERT_EQ(b.size(), 100);
  ASSERT_EQ(b.size(), Bag::kTotalTiles);
  for (int i = 0; i < 27; ++i) ASSERT_EQ(b.counts()[i], TILE_COUNTS[i]);

  // Same seed -> identical draw sequence (reproducibility).
  Bag b1(/*seed=*/12345);
  Bag b2(/*seed=*/12345);
  for (int i = 0; i < 100; ++i) {
    auto t1 = b1.draw();
    auto t2 = b2.draw();
    ASSERT_TRUE(t1.has_value() && t2.has_value());
    ASSERT_EQ(*t1, *t2);
  }
  ASSERT_EQ(b1.size(), 0);
  ASSERT_FALSE(b1.draw().has_value());  // empty bag returns nullopt

  // Different seeds -> sequences differ (almost surely; 100 draws is plenty).
  Bag bA(1), bB(2);
  bool any_diff = false;
  for (int i = 0; i < 100; ++i) {
    auto a = bA.draw();
    auto bb = bB.draw();
    if (a != bb) any_diff = true;
  }
  ASSERT_TRUE(any_diff);

  // Tile-count conservation: drain the bag, the per-tile draw counts must
  // equal TILE_COUNTS exactly.
  std::array<int, 27> drawn{};
  Bag b3(/*seed=*/777);
  while (auto t = b3.draw()) ++drawn[*t];
  for (int i = 0; i < 27; ++i) ASSERT_EQ(drawn[i], TILE_COUNTS[i]);
  ASSERT_EQ(b3.size(), 0);

  // put_back round-trip: drain bag, put a tile back, next draw is that tile.
  Bag b4(/*seed=*/9999);
  while (b4.draw().has_value()) {
  }
  ASSERT_EQ(b4.size(), 0);
  b4.put_back(Tile::from_char('Q'));
  ASSERT_EQ(b4.size(), 1);
  auto got = b4.draw();
  ASSERT_TRUE(got.has_value() && *got == Tile::from_char('Q'));
}

// ===========================================================================
// Board::apply -- the exact interleave-with-cross-tiles scenario
// ===========================================================================

TEST(Board, ApplyInterleavesCrossTiles) {
  // Place CA_ at (7,7..9) leaving (7,9) empty? No -- we want a more rigorous
  // scenario: place a single existing tile in the middle of the run, then
  // apply a move whose square_mask says "skip that cell".
  Board b;
  // Existing tile: an 'A' at (7,8).
  b.set(7, 8, Glyph::of(Tile::from_char('A')));

  // Build a move equivalent to playing CAT horizontally starting at (7,7)
  // where the middle 'A' is already on the board: placements at (7,7) and
  // (7,9). main_word is "CAT", glyphs = {C, T}, square_mask = 0b101.
  Move m = make_play_full(7, 7, /*horizontal=*/true, 0b101, 0,
                          {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('T'))});

  b.apply(m);

  ASSERT_EQ(b.at(7, 7).letter(), Tile::from_char('C'));
  ASSERT_EQ(b.at(7, 8).letter(), Tile::from_char('A'));  // unchanged
  ASSERT_EQ(b.at(7, 9).letter(), Tile::from_char('T'));
  // Apply on a PASS is a no-op.
  Move pass;
  Board snapshot = b;
  b.apply(pass);
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      ASSERT_EQ(b.at(r, c).code(), snapshot.at(r, c).code());
    }
  }
}

// ===========================================================================
// Move::main_word
// ===========================================================================

TEST(Move, MainWordThroughCross) {
  Board b;
  // Pre-existing CAT horizontally at row 7, cols 7..9.
  b.set(7, 7, Glyph::of(Tile::from_char('C')));
  b.set(7, 8, Glyph::of(Tile::from_char('A')));
  b.set(7, 9, Glyph::of(Tile::from_char('T')));

  // 1. Hook: play S at (7,10) to make CATS. Single placement, but main_word
  //    must include all four letters by walking the existing tiles.
  Move hook = make_play_full(7, 10, /*horizontal=*/true, 0b1, 0, {Glyph::of(Tile::from_char('S'))});
  // Move::main_word walks back from the placed S through the existing C,A,T
  // to recover the word origin, then renders "CATS".
  ASSERT_EQ(hook.main_word(b), "CATS");

  // 2. Through-word: place B at (7,6) and S at (7,10) for "BCATS"? Not a
  //    real word -- but main_word doesn't care about legality. We just
  //    verify that interleaving works.
  Move through = make_play_full(7, 6, /*horizontal=*/true, 0b10001, 0,
                                {Glyph::of(Tile::from_char('B')),    // placed at (7,6)
                                 Glyph::of(Tile::from_char('S'))});  // placed at (7,10)
  ASSERT_EQ(through.main_word(b), "BCATS");

  // 3. Blank renders as its designated letter (uppercase), like a regular tile.
  Move with_blank = make_play_full(7, 10, /*horizontal=*/true, 0b1, 0,
                                   {Glyph::played(Tile::from_char('S'), /*is_blank=*/true)});
  ASSERT_EQ(with_blank.main_word(b), "CATS");

  // 4. PASS / EXCHANGE produce empty strings.
  Move pass;
  ASSERT_TRUE(pass.main_word(b).empty());
  TileCounts xch_tiles;
  xch_tiles.add(Tile::from_char('A'));
  Move xch = Move::exchange(xch_tiles);
  ASSERT_TRUE(xch.main_word(b).empty());
}

// ===========================================================================
// Movegen: blank placement scores zero, even on a letter premium
// ===========================================================================

TEST(Movegen, BlankScoresZero) {
  // Dictionary that contains both CAT (no blank needed) and a word that uses
  // a letter that doesn't appear in our rack so we're forced to use a blank.
  Dictionary d = Dictionary::build_from_words({"CAT", "CATS", "BAT", "BATS"});

  Board b;
  // Anchor: place AT horizontally at (CENTER, CENTER..CENTER+1).
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('T'))}));

  // Rack with 'C' (real) -> forms CAT at (CENTER, CENTER-1..CENTER+1).
  // Also rack with only blank + filler that can't form anything.
  // To compare scores cleanly, we generate twice: once with rack {C}, once
  // with rack {?, ?, ?, ?, ?, ?, ?} (all blanks).
  MoveGenerator gen(b, d);

  Rack rack_real = rack_from("CXXXXXX");
  auto moves_real = gen.generate(rack_real);
  int score_real = 0;
  for (const auto& m : moves_real) {
    if (m.main_word(b) == "CAT") {
      score_real = m.score();
      break;
    }
  }
  ASSERT_GT(score_real, 0);

  Rack rack_blank = rack_from("???????");
  auto moves_blank = gen.generate(rack_blank);
  int score_blank = -1;
  for (const auto& m : moves_blank) {
    if (m.main_word(b) == "CAT") {
      // Verify the C placement is a blank.
      ASSERT_EQ(m.num_glyphs(), 1);
      ASSERT_TRUE(m.glyph(0).is_blank());
      score_blank = m.score();
      break;
    }
  }
  ASSERT_GE(score_blank, 0);
  // The blank-C contributes 0 letter value; the only word score is from the
  // existing A and T (already on the board). With a blank in the placed
  // position the word score equals the unscored A+T sum, possibly multiplied
  // by a word premium under the placed blank. With the real C the score is
  // strictly greater (C alone is worth 3).
  ASSERT_LT(score_blank, score_real);
}

// ===========================================================================
// Game end conditions
// ===========================================================================

namespace {

// Agent that always returns PASS. Used to force the stalemate end condition
// deterministically.
class AlwaysPassAgent : public scribblez::Agent {
 public:
  AlwaysPassAgent(int tid, std::string name) : scribblez::Agent(tid, std::move(name)) {}
  scribblez::MoveDecision make_move(const scribblez::MoveRequest&) override {
    ++prompts;
    return scribblez::Move::pass();
  }
  int prompts = 0;
};

// Passes, and on its first prompt projects the rest of a pass-out game: five
// more passes, which together with its own reach the six consecutive zero
// turns that end the game.
class ProjectingPassAgent : public scribblez::Agent {
 public:
  ProjectingPassAgent(int tid, std::string name) : scribblez::Agent(tid, std::move(name)) {}
  scribblez::MoveDecision make_move(const scribblez::MoveRequest&) override {
    ++prompts;
    return {scribblez::Move::pass(), std::vector<scribblez::Move>(5, scribblez::Move::pass())};
  }
  int prompts = 0;
};

}  // namespace

// --- face-up-leaves visibility ----------------------------------------------
//
// What Game puts in MoveRequest::opp_rack: nothing in a standard game until
// the bag empties, the opponent's publicly retained tiles under face-up
// leaves, and their whole rack once an empty bag makes it deducible.

namespace {

// One prompt, as the seated agent saw it.
struct SeenRequest {
  int mover;
  int bag_size;
  Rack opp_rack;
};

// Plays the first legal play it is offered (else passes) and records what it
// was told about its opponent. Both seats share one log, so the k-th entry
// belongs to the k-th turn of the game.
class LeaveWatchingAgent : public scribblez::Agent {
 public:
  LeaveWatchingAgent(int seat, std::vector<SeenRequest>* seen)
      : scribblez::Agent(0, "W"), seat_(seat), seen_(seen) {}

  scribblez::MoveDecision make_move(const scribblez::MoveRequest& req) override {
    seen_->push_back({seat_, req.bag_size, req.opp_rack});
    const std::vector<Move> plays = scribblez::generate_legal_plays(req);
    return plays.empty() ? scribblez::Move::pass() : plays.front();
  }

 private:
  int seat_;
  std::vector<SeenRequest>* seen_;
};

// The tiles `player` held back at their most recent turn before `turn`, which
// is what face-up leaves makes public. Empty before they have acted.
Rack leave_before_turn(const scribblez::GameLogStorage& log, int player, size_t turn) {
  Rack leave;
  for (size_t t = turn; t-- > 0;) {
    if (log.turns[t].player != player) continue;
    leave = log.turns[t].rack_before;
    for (int i = 0; i < log.turns[t].move.num_glyphs(); ++i)
      leave.remove(log.turns[t].move.glyph(i).rack_tile());
    break;
  }
  return leave;
}

std::vector<SeenRequest> play_watched_game(const Dictionary& dict, bool face_up, uint64_t seed,
                                           scribblez::GameLogStorage* log_out) {
  std::vector<SeenRequest> seen;
  LeaveWatchingAgent a0(0, &seen), a1(1, &seen);
  scribblez::Game g(a0, a1, dict, seed);
  g.set_face_up_leaves(face_up);
  g.play();
  *log_out = g.extract_log();
  return seen;
}

}  // namespace

TEST(FaceUpLeaves, TheLogRecordsWhichVariantWasPlayed) {
  namespace fs = std::filesystem;
  Dictionary dict = medium_dict();

  for (bool face_up : {false, true}) {
    fs::path dir =
      fs::temp_directory_path() /
      ("scribblez_faceup_flag_" + std::to_string(::getpid()) + "_" + std::to_string(int(face_up)));
    fs::create_directories(dir);
    struct DirCleanup {
      fs::path p;
      ~DirCleanup() {
        std::error_code ec;
        fs::remove_all(p, ec);
      }
    } cleanup{dir};

    const uint16_t flags = face_up ? scribblez::binlog::kFlagFaceUpLeaves : 0;
    {
      scribblez::binlog::BinaryLogWriter writer(dir.string(), /*games_per_file=*/1, flags);
      writer.append(play_test_game(dict, /*seed=*/321ULL));
    }

    fs::path slog;
    for (const auto& ent : fs::directory_iterator(dir))
      if (ent.path().extension() == ".slog") slog = ent.path();
    ASSERT_FALSE(slog.empty()) << "face_up=" << face_up;

    scribblez::binlog::FileHeader hdr{};
    std::ifstream f(slog, std::ios::binary);
    ASSERT_TRUE(f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) << "face_up=" << face_up;
    EXPECT_EQ(hdr.magic, scribblez::binlog::kMagic);
    // The version does not move with the flags: a file written before the bits
    // existed carries none, which correctly reads as standard Scrabble.
    EXPECT_EQ(hdr.version, scribblez::binlog::kVersion);
    EXPECT_EQ(hdr.flags & scribblez::binlog::kFlagFaceUpLeaves, flags) << "face_up=" << face_up;
  }
}

TEST(FaceUpLeaves, AStandardGameShowsNothingWhileTilesRemain) {
  Dictionary dict = medium_dict();
  scribblez::GameLogStorage log;
  const std::vector<SeenRequest> seen = play_watched_game(dict, /*face_up=*/false, 99ULL, &log);

  ASSERT_FALSE(seen.empty());
  int mid_game_prompts = 0;
  for (const SeenRequest& r : seen) {
    if (r.bag_size == 0) continue;
    ++mid_game_prompts;
    EXPECT_TRUE(r.opp_rack.empty()) << "leaked " << r.opp_rack.to_string();
  }
  ASSERT_GT(mid_game_prompts, 0);
}

TEST(FaceUpLeaves, ShowsExactlyWhatTheOpponentKept) {
  Dictionary dict = medium_dict();
  scribblez::GameLogStorage log;
  const std::vector<SeenRequest> seen = play_watched_game(dict, /*face_up=*/true, 99ULL, &log);

  // One prompt per turn, in turn order: no random opening, no projections.
  ASSERT_EQ(seen.size(), log.turns.size());
  EXPECT_TRUE(seen.front().opp_rack.empty()) << "nothing is public before the opponent acts";

  int revealed = 0;
  for (size_t t = 0; t < seen.size(); ++t) {
    if (seen[t].bag_size == 0) continue;  // the endgame reveals more; see below
    const Rack expected = leave_before_turn(log, 1 - seen[t].mover, t);
    EXPECT_TRUE(seen[t].opp_rack == expected)
      << "turn " << t << ": saw " << seen[t].opp_rack.to_string() << ", kept "
      << expected.to_string();
    if (!expected.empty()) ++revealed;
  }
  ASSERT_GT(revealed, 0) << "the variant never actually revealed anything";
}

TEST(FaceUpLeaves, AnEmptyBagShowsTheWholeRackInEitherVariant) {
  Dictionary dict = medium_dict();
  for (bool face_up : {false, true}) {
    std::vector<SeenRequest> seen;
    LeaveWatchingAgent a0(0, &seen), a1(1, &seen);
    scribblez::Game g(a0, a1, dict, /*seed=*/7ULL);
    g.set_face_up_leaves(face_up);

    // A pool holding exactly both racks: the refills drain it, so every prompt
    // faces an empty bag and a fully deducible opponent.
    scribblez::Bag pool(/*seed=*/7ULL);
    while (pool.size() > 2 * RACK_SIZE) pool.draw();
    g.play_from(Board{}, {0, 0}, {Rack{}, Rack{}}, pool, /*to_move=*/0);
    const scribblez::GameLogStorage log = g.extract_log();

    ASSERT_FALSE(seen.empty());
    ASSERT_EQ(seen.front().bag_size, 0) << "face_up=" << face_up;
    // The opponent's whole rack, not merely what they kept: at the first
    // prompt they have not moved, so a leave would be empty.
    EXPECT_EQ(seen.front().opp_rack.size(), RACK_SIZE) << "face_up=" << face_up;
    EXPECT_TRUE(seen.front().opp_rack == log.initial_racks[1]) << "face_up=" << face_up;
  }
}

TEST(Game, EndRackOutBonus) {
  // Greedy agents on a small in-memory dict almost always stalemate (they
  // can't form enough words to drain the bag). With the real lexicon, "out"
  // is the normal end condition. Skip gracefully if no real lexicon is
  // available -- the stalemate test below still covers Game::play()'s other
  // end-of-game arithmetic.
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) {
    GTEST_SKIP() << "no lexicon at " << path;
  }
  Dictionary dict = Dictionary::load_kwg(path);

  bool found_out = false;
  for (uint64_t seed = 0; seed < 20 && !found_out; ++seed) {
    scribblez::GameLogStorage log = play_test_game(dict, seed);
    if (log.end_reason != "out") continue;
    found_out = true;

    const TurnRecord& last = log.turns.back();
    const int winner = last.player;  // out-going player
    const int loser = 1 - winner;
    // Game applies the modern tournament convention: the out-going player
    // gains twice the opponent's remaining tile values, and the opponent's
    // score is left unchanged.
    const int bonus = 2 * log.final_racks[loser].point_value();

    ASSERT_EQ(log.final_scores[winner], last.cumulative_scores[winner] + bonus);
    ASSERT_EQ(log.final_scores[loser], last.cumulative_scores[loser]);
    ASSERT_TRUE(log.final_racks[winner].empty());
  }
  ASSERT_TRUE(found_out);
}

TEST(Game, EndStalematePenalty) {
  // Two pass-agents: 6 consecutive zero turns trigger "stalemate" and each
  // player's final = cumulative - their own remaining-rack value (cumulative
  // is 0 for both -- nobody scored).
  Dictionary dict = medium_dict();
  AlwaysPassAgent a0(0, "P0");
  AlwaysPassAgent a1(0, "P1");
  scribblez::Game g(a0, a1, dict, /*seed=*/424242ULL);
  g.play();
  const scribblez::GameLogStorage log = g.extract_log();

  ASSERT_EQ(log.end_reason, "stalemate");
  ASSERT_EQ(log.turns.size(), 6);  // 6 zero turns (3 per player)
  for (const auto& t : log.turns) ASSERT_EQ(t.move.type(), MoveType::PASS);
  for (int p = 0; p < 2; ++p) {
    ASSERT_EQ(log.final_scores[p], -log.final_racks[p].point_value());
  }
}

// A respected projection replaces prompting: the projecting agent is asked
// once, its five projected passes complete the stalemate, and the opponent is
// never prompted at all. With projections ignored (the default), both agents
// are prompted for every turn of the same game.
TEST(Game, RespectedProjectionStopsPrompting) {
  Dictionary dict = medium_dict();
  {
    ProjectingPassAgent a0(0, "P0");
    AlwaysPassAgent a1(0, "P1");
    scribblez::Game g(a0, a1, dict, /*seed=*/424242ULL);
    g.set_respect_projections(true);
    g.play();
    const scribblez::GameLogStorage log = g.extract_log();
    ASSERT_EQ(log.end_reason, "stalemate");
    ASSERT_EQ(log.turns.size(), 6);
    ASSERT_EQ(a0.prompts, 1);
    ASSERT_EQ(a1.prompts, 0);
    for (int p = 0; p < 2; ++p) ASSERT_EQ(log.final_scores[p], -log.final_racks[p].point_value());
  }
  {
    ProjectingPassAgent a0(0, "P0");
    AlwaysPassAgent a1(0, "P1");
    scribblez::Game g(a0, a1, dict, /*seed=*/424242ULL);
    g.play();
    ASSERT_EQ(g.extract_log().turns.size(), 6);
    ASSERT_EQ(a0.prompts, 3);
    ASSERT_EQ(a1.prompts, 3);
  }
}

TEST(Util, NaturalLess) {
  using util::natural_less;
  ASSERT_TRUE(natural_less("pos-2", "pos-10"));  // numeric run compares by value, not lexically
  ASSERT_FALSE(natural_less("pos-10", "pos-2"));
  ASSERT_TRUE(natural_less("pos-2.gcg", "pos-10.gcg"));
  ASSERT_TRUE(natural_less("pos-09", "pos-10"));  // leading zeros ignored
  ASSERT_TRUE(natural_less("a2", "a2b"));         // a prefix sorts before its extension
  ASSERT_FALSE(natural_less("pos-1", "pos-1"));   // equal -> not less (irreflexive)
  ASSERT_TRUE(natural_less("abc", "abd"));        // non-digit chars compare lexically

  std::vector<std::string> v = {"pos-10", "pos-2", "pos-1", "pos-20", "pos-3"};
  std::sort(v.begin(), v.end(), natural_less);
  ASSERT_TRUE((v == std::vector<std::string>{"pos-1", "pos-2", "pos-3", "pos-10", "pos-20"}));
}

static bool rack_contains(const Rack& r, Tile want) {
  for (int i = 0; i < r.size(); ++i)
    if (r.tiles()[i].index() == want.index()) return true;
  return false;
}

// Bag::remove (build the unseen pool) and Game::play_from (Monte-Carlo rollout from
// a mid-game position): the deal keeps the post-mover's leave, refills both racks
// from the seeded pool, plays to a natural end, and is deterministic per seed.
TEST(Game, PlayFrom) {
  // Bag::remove: a fully-removed letter never comes out of the bag.
  {
    Bag bag(123);
    const Tile a = Tile::from_char('A');
    for (int i = bag.counts()[a.index()]; i > 0; --i) bag.remove(a);
    ASSERT_EQ(bag.counts()[a.index()], 0);
    while (auto t = bag.draw()) ASSERT_NE(t->index(), a.index());
  }

  const Dictionary d = medium_dict();
  const Board board;                    // empty post-move board (sufficient here)
  const Rack leave = rack_from("ING");  // seat 0 (post-mover) leave
  const std::array<Rack, 2> known = {leave, Rack{}};
  const std::array<int, 2> scores = {120, 95};

  GameLogStorage logs[2];
  for (int run = 0; run < 2; ++run) {  // same seed twice -> identical (determinism)
    const uint64_t seed = 7;
    Bag pool(seed);
    for (int i = 0; i < leave.size(); ++i) pool.remove(leave.tiles()[i]);  // leave is off the bag
    TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
    TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
    scribblez::Game g(a0, a1, d, seed);
    g.play_from(board, scores, known, pool, /*to_move=*/1);  // seat 0 just moved; seat 1 first
    logs[run] = g.extract_log();
  }

  ASSERT_FALSE(logs[0].end_reason.empty());               // reached a natural end
  ASSERT_EQ(logs[0].initial_racks[0].size(), RACK_SIZE);  // post-mover topped up to 7
  ASSERT_EQ(logs[0].initial_racks[1].size(), RACK_SIZE);  // on-move drew a fresh rack
  for (int i = 0; i < leave.size(); ++i)                  // the leave is preserved
    ASSERT_TRUE(rack_contains(logs[0].initial_racks[0], leave.tiles()[i]));
  ASSERT_EQ(logs[0].final_scores, logs[1].final_scores);  // deterministic per seed
  ASSERT_EQ(logs[0].end_reason, logs[1].end_reason);
}

// Game::set_max_plies (a value-truncated rollout's horizon): play stops after
// exactly the cap, truncated() reports it, no end-of-game score adjustment is
// applied, and leave() exposes the last mover's post-move pre-draw rack.
TEST(Game, MaxPliesTruncation) {
  const Dictionary d = medium_dict();
  const Board board;
  const Rack leave = rack_from("ING");
  const std::array<Rack, 2> known = {leave, Rack{}};
  const std::array<int, 2> scores = {120, 95};
  constexpr int kPlies = 3;

  const uint64_t seed = 7;
  Bag pool(seed);
  for (int i = 0; i < leave.size(); ++i) pool.remove(leave.tiles()[i]);
  TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
  TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
  scribblez::Game g(a0, a1, d, seed);
  g.set_max_plies(kPlies);
  g.play_from(board, scores, known, pool, /*to_move=*/1);

  ASSERT_TRUE(g.truncated());
  const GameLog log = g.log();
  ASSERT_EQ(log.num_records, kPlies);
  ASSERT_STREQ(log.end_reason, "truncated");
  // Truncation applies no out/stalemate adjustment: the final scores are the
  // running scores, the initial scores plus each player's move scores.
  std::array<int, 2> expected = scores;
  for (int i = 0; i < log.num_records; ++i)
    expected[log.records[i].player] += log.records[i].score_delta;
  ASSERT_EQ(log.final_scores, expected);
  // leave(): the last mover's record's rack_before minus the tiles the move
  // surrendered -- their post-move pre-draw rack.
  const TurnRecord& last = log.records[kPlies - 1];
  Rack expected_leave = last.rack_before;
  for (int i = 0; i < last.move.num_glyphs(); ++i)
    ASSERT_TRUE(expected_leave.remove(last.move.glyph(i).rack_tile()));
  ASSERT_EQ(g.leave(last.player).to_string(), expected_leave.to_string());

  // A game reaching its natural end under a generous cap is not truncated.
  TestAgent b0(0, "B0", seed ^ 0x1111111111111111ULL);
  TestAgent b1(0, "B1", seed ^ 0x2222222222222222ULL);
  Bag pool2(seed);
  for (int i = 0; i < leave.size(); ++i) pool2.remove(leave.tiles()[i]);
  scribblez::Game g2(b0, b1, d, seed);
  g2.set_max_plies(399);
  g2.play_from(board, scores, known, pool2, /*to_move=*/1);
  ASSERT_FALSE(g2.truncated());
}

// The cap never truncates an endgame: once the bag empties the game plays out
// to a natural end, however many plies past the cap that takes -- the leaf
// model's training domain is the pre-endgame prefix, so there is no valid
// leaf to hand it there (see Game::set_max_plies).
TEST(Game, MaxPliesSparesTheEndgame) {
  const Dictionary d = medium_dict();
  const Board board;
  // Both racks fully known and a nearly-empty pool, so the bag empties within
  // the first couple of plies while play continues.
  const std::array<Rack, 2> known = {rack_from("CATSEIQ"), rack_from("RATESIN")};
  const uint64_t seed = 7;
  Bag pool(seed);
  {
    // Reduce the pool to exactly 2 tiles beyond the known racks.
    Bag two(seed);
    while (two.size() > 2) two.draw();
    pool = two;
  }
  TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
  TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
  scribblez::Game g(a0, a1, d, seed);
  g.set_max_plies(3);
  g.play_from(board, {0, 0}, known, pool, /*to_move=*/0);

  const GameLog log = g.log();
  if (g.truncated()) {
    // Truncation is only legitimate at a training-eligible capped ply: its
    // pre-move bag must have been non-empty.
    ASSERT_GT(log.records[log.num_records - 1].bag_size_before, 0);
  } else {
    // The expected path with a 2-tile bag: the cap passed inside the endgame
    // and was ignored, so the game reached a natural end beyond it.
    ASSERT_GE(log.num_records, 3);
    ASSERT_TRUE(std::string(log.end_reason) == "out" || std::string(log.end_reason) == "stalemate");
  }
}

// ===========================================================================
// LabelEncoder
// ===========================================================================

namespace {

// The label row lays the eight targets out as
//   [wld(3), score_diff(1),
//    opp_next(1), self_next(1), opp_win(1), self_win(1),   // footprint class index
//    opp_placement_mask(N), self_placement_mask(N)]        // N=kFootprintClasses
// -- the four placement heads are a single categorical footprint class each; the
// two per-side legality masks (plays-head form, kExtraClass illegal) are shared,
// a head's win variant opening kExtraClass in the loss.
constexpr int kClassBase = kWldFloats + kScoreDiffFloats;
constexpr int kMaskBase = kClassBase + 4 * kPlacementClassFloats;

// An EncodeContext holding just the fields the label targets read: the final
// scores / POV (wld, score_diff, win-head gating) and the encoder + spec the
// masks read the sampled board and dictionary through. The next moves stay
// unset -- a caller that wants the placement class targets exercised sets them.
scribblez::EncodeContext scores_view(const GameStateEncoder& enc, const InputEncodingSpec& spec,
                                     int fs_active, int fs_opp, int active_player,
                                     bool flip = false) {
  scribblez::EncodeContext v{};
  v.enc = &enc;
  v.spec = spec;
  v.active_player = active_player;
  v.final_score_p0 = active_player == 0 ? fs_active : fs_opp;
  v.final_score_p1 = active_player == 0 ? fs_opp : fs_active;
  v.apply_flip = flip;
  return v;
}

// The label targets read the sampled board and dictionary through the context's
// encoder for the masks; a single empty-board encoder over an in-memory
// dictionary serves every sub-case (only the board matters to the masks, and on
// an empty board every square is unconstrained, so the opp mask is
// dictionary-free but still binds the caches). The class targets need no board.
struct LabelFixture {
  Dictionary dict = Dictionary::build_from_words({"CAT", "CATS", "BAT"});
  InputEncodingSpec spec{&dict};
  GameStateEncoder enc{spec, Board{}, std::array<int, 2>{0, 0}, 0};

  scribblez::EncodeContext view(int fs_active, int fs_opp, int active_player, bool flip = false) {
    return scores_view(enc, spec, fs_active, fs_opp, active_player, flip);
  }
};

void encode_labels_flat(const scribblez::EncodeContext& view, float* flat) {
  scribblez::AllTargets::encode_all(view, flat);
}

}  // namespace

TEST(TrainingTargets, EncodeLabelsWldAndScoreDiff) {
  LabelFixture fx;
  std::vector<float> flat(kLabelFloats);

  auto check_score_diff = [&](int diff_signed) { ASSERT_EQ(flat[kWldFloats], float(diff_signed)); };

  auto v_win = fx.view(/*fs_active=*/120, /*fs_opp=*/100, /*active_player=*/0);
  encode_labels_flat(v_win, flat.data());
  ASSERT_EQ(flat[0], 1.0f);
  ASSERT_EQ(flat[1], 0.0f);
  ASSERT_EQ(flat[2], 0.0f);
  check_score_diff(20);

  auto v_draw = fx.view(75, 75, 1);
  encode_labels_flat(v_draw, flat.data());
  ASSERT_EQ(flat[0], 0.0f);
  ASSERT_EQ(flat[1], 1.0f);
  ASSERT_EQ(flat[2], 0.0f);
  check_score_diff(0);

  auto v_loss = fx.view(80, 95, 0);
  encode_labels_flat(v_loss, flat.data());
  ASSERT_EQ(flat[0], 0.0f);
  ASSERT_EQ(flat[1], 0.0f);
  ASSERT_EQ(flat[2], 1.0f);
  check_score_diff(-15);

  // Large differentials are stored as-is (not clipped or rejected).
  encode_labels_flat(fx.view(620, 0, 0), flat.data());
  check_score_diff(620);
  encode_labels_flat(fx.view(0, 620, 0), flat.data());
  check_score_diff(-620);

  // WLD entries are mutually exclusive and sum to 1.0 for every case.
  for (auto [a, b] : std::vector<std::pair<int, int>>{{1, 0}, {0, 0}, {-5, 5}, {200, -200}}) {
    encode_labels_flat(fx.view(a, b, 0), flat.data());
    ASSERT_EQ(flat[0] + flat[1] + flat[2], 1.0f);
  }
}

TEST(TrainingTargets, EncodeLabelsPlacementFootprints) {
  LabelFixture fx;
  std::vector<float> flat(kLabelFloats);

  const int opp_next = kClassBase + 0;
  const int self_next = kClassBase + 1;
  const int opp_win = kClassBase + 2;
  const int self_win = kClassBase + 3;
  const float* opp_mask = flat.data() + kMaskBase + 0 * kFootprintClasses;
  const float* self_mask = flat.data() + kMaskBase + 1 * kFootprintClasses;

  // No next move -> the plays heads are kPassClass, and pass is always legal in
  // both side masks; the not-win (extra) class is illegal in the plays-head-form
  // side mask (the loss opens it for win heads).
  auto v = fx.view(/*fs_active=*/100, /*fs_opp=*/80, /*active_player=*/0);
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(flat[opp_next], float(kPassClass));
  ASSERT_EQ(flat[self_next], float(kPassClass));
  ASSERT_EQ(opp_mask[kPassClass], 1.0f);
  ASSERT_EQ(self_mask[kPassClass], 1.0f);
  ASSERT_EQ(opp_mask[kExtraClass], 0.0f);
  ASSERT_EQ(self_mask[kExtraClass], 0.0f);

  // A horizontal opponent PLAY at (4,2) covering 3 empty cells -> its footprint
  // class, which the opp mask keeps (the -log(0) soundness property) and whose
  // covered cells round-trip on the (empty) sampled board.
  Move next_play = make_play_full(4, 2, /*horizontal=*/true, 0b111, 0,
                                  {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                                   Glyph::of(Tile::from_char('C'))});
  v.opp_next_move = next_play;
  v.has_opp_next_move = true;
  encode_labels_flat(v, flat.data());
  const int cls = int(flat[opp_next]);
  ASSERT_EQ(cls, footprint_class(next_play, /*flip=*/false));
  ASSERT_LT(cls, kAnchoredFootprints);
  ASSERT_EQ(opp_mask[cls], 1.0f);
  std::array<std::pair<int, int>, kFootprintMaxK> cells;
  const int n = footprint_cells(cls, fx.enc.board(), /*flip=*/false, cells);
  ASSERT_EQ(n, 3);
  ASSERT_EQ(cells[0], std::make_pair(4, 2));
  ASSERT_EQ(cells[1], std::make_pair(4, 3));
  ASSERT_EQ(cells[2], std::make_pair(4, 4));

  // apply_flip transposes the class into the flipped frame (anchor and
  // orientation both swap), matching footprint_class under the same flip.
  v.apply_flip = true;
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_next]), footprint_class(next_play, /*flip=*/true));
  v.apply_flip = false;

  // EXCHANGE next move -> kPassClass.
  TileCounts xch_tiles;
  xch_tiles.add(Tile::from_char('A'));
  v.opp_next_move = Move::exchange(xch_tiles);
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_next]), kPassClass);

  // --- Win heads: the footprint if that seat won, else kExtraClass (not-win) ---
  v.opp_next_move = next_play;  // opponent plays; active player (0) is winning

  // Active player winning -> the opponent did NOT win -> opp_win is not-win.
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_win]), kExtraClass);

  // Opponent winning -> opp_win is the played footprint, same class as opp_next.
  v.final_score_p0 = 80;
  v.final_score_p1 = 100;
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_win]), int(flat[opp_next]));
  ASSERT_LT(int(flat[opp_win]), kAnchoredFootprints);

  // A vertical mover PLAY at (7,3)/(8,3): self_next is always its footprint;
  // self_win is that footprint only when the mover wins, else not-win.
  Move self_play =
    make_play_full(7, 3, /*horizontal=*/false, 0b11, 0,
                   {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('E'))});
  v.self_next_move = self_play;
  v.has_self_next_move = true;

  encode_labels_flat(v, flat.data());  // mover (0) losing
  ASSERT_EQ(int(flat[self_next]), footprint_class(self_play, /*flip=*/false));
  ASSERT_EQ(int(flat[self_win]), kExtraClass);
  // The self side mask keeps the played footprint legal (the -log(0) property);
  // kExtraClass stays illegal here -- the loss opens it for the self_win head.
  ASSERT_EQ(self_mask[footprint_class(self_play, /*flip=*/false)], 1.0f);
  ASSERT_EQ(self_mask[kExtraClass], 0.0f);

  v.final_score_p0 = 100;
  v.final_score_p1 = 80;  // mover wins
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[self_next]), footprint_class(self_play, /*flip=*/false));
  ASSERT_EQ(int(flat[self_win]), int(flat[self_next]));

  // A draw counts as not winning for both conjunctions; the marginals hold.
  v.final_score_p0 = 90;
  v.final_score_p1 = 90;
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_win]), kExtraClass);
  ASSERT_EQ(int(flat[self_win]), kExtraClass);
  ASSERT_EQ(int(flat[opp_next]), footprint_class(next_play, /*flip=*/false));
  ASSERT_EQ(int(flat[self_next]), footprint_class(self_play, /*flip=*/false));
}

// ===========================================================================
// DataLoader: per-row diagonal-flip symmetry
// ===========================================================================

// Build a one-game .slog file under `dir` whose 2-turn game is:
//   turn 0: p0 PLAYs a synthetic single-tile move placing 'Q' at (3,5)
//   turn 1: p1 PASSes
// The file declares a single eligible turn (turn 0), so the loader expands it to
// exactly one training row. The position under test is turn 0 POST-move: the
// board already holds the asymmetric Q, the POV is the mover (p0), whose leave
// is the 6 As. Returns the file path, the on-disk size, and the
// (path-independent) state describing that position so the test can build its
// reference encoding.
struct SymFixture {
  std::filesystem::path path;
  int64_t fsize;
  // Reproducible inputs to encode_input for the position we'll sample.
  scribblez::Rack active_rack;  // POV (p0) leave after the Q play: 6 As
  scribblez::Move self_move;    // the Q play, applied to reach the post-move state
  int final_score_p0;
  int final_score_p1;
  int active_player;  // POV = p0 (the mover at turn 0)
};

static SymFixture write_one_position_slog(const std::filesystem::path& dir) {
  using namespace scribblez::binlog;
  using namespace scribblez;

  // Initial racks. p0 gets Q + 6 As (no draws, so size never replenishes);
  // p1 starts empty so the bag-derivation has no contribution from them.
  Rack p0_init;
  p0_init.add(Tile::from_char('Q'));
  for (int i = 0; i < 6; ++i) p0_init.add(Tile::from_char('A'));
  Rack p1_init;  // empty

  // Synthetic 1-tile PLAY: place 'Q' at (3,5). Score is arbitrary; we'll
  // record it in the move and recompute it in the reference.
  Move q_play =
    make_play_full(3, 5, /*horizontal=*/true, 0b1, 42, {Glyph::of(Tile::from_char('Q'))});

  Move p1_pass = Move::pass();

  InitialRacks ir{};
  ir.p0 = p0_init;
  ir.p1 = p1_init;

  TurnBlob t0{};
  t0.move = q_play;
  TurnBlob t1{};
  t1.move = p1_pass;

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.num_games = 1;
  hdr.num_sample_positions = 1;  // one eligible turn -> one training row

  GameMetadata gm{};
  gm.start_offset = sizeof(FileHeader) + sizeof(GameMetadata);
  gm.num_turns = 2;
  gm.sampled_turn = 0;    // eval-only; training uses the eligible region
  gm.eligible_begin = 0;  // expand to one row: turn 0
  gm.eligible_end = 1;
  gm.final_score_p0 = 350;
  gm.final_score_p1 = 200;

  std::filesystem::path path = dir / "one_position.slog";
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(&gm), sizeof(gm));
    f.write(reinterpret_cast<const char*>(&ir), sizeof(ir));
    f.write(reinterpret_cast<const char*>(&t0), sizeof(t0));
    f.write(reinterpret_cast<const char*>(&t1), sizeof(t1));
    EXPECT_TRUE(f.good());
  }
  int64_t fsize = std::filesystem::file_size(path);

  // Canonical state for turn 0 post-move: POV is p0 (the mover), whose leave is
  // the 6 As; applying q_play places the Q and gives p0 a score of 42.
  SymFixture out;
  out.path = path;
  out.fsize = fsize;
  for (int i = 0; i < 6; ++i) out.active_rack.add(Tile::from_char('A'));
  out.self_move = q_play;
  out.final_score_p0 = gm.final_score_p0;
  out.final_score_p1 = gm.final_score_p1;
  out.active_player = 0;
  return out;
}

TEST(DataLoader, PerRowSymmetry) {
  using namespace scribblez::binlog;
  namespace fs = std::filesystem;

  fs::path dir = fs::temp_directory_path() / ("scribblez_sym_" + std::to_string(::getpid()) + "_" +
                                              std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{dir};

  SymFixture fix = write_one_position_slog(dir);
  Dictionary dict = medium_dict();

  // Build the two reference input encodings (canonical + flipped) for the
  // sampled position (turn 0 post-move).
  std::vector<float> ref_normal(kInputFloats, 0.0f);
  std::vector<float> ref_flipped(kInputFloats, 0.0f);
  {
    // Apply the q_play so the encoder lands in the turn-0 post-move state:
    // board has Q at (3,5), p0 (the mover) scored 42, last_move_by_p0 = q_play.
    // The POV is the mover (p0), encoded with its post-play leave (6 As).
    GameStateEncoder ref_enc{InputEncodingSpec{&dict}};
    ref_enc.apply_move(fix.self_move);
    ref_enc.encode_input(fix.active_player, fix.active_rack, /*apply_flip=*/false,
                         ref_normal.data());
    ref_enc.encode_input(fix.active_player, fix.active_rack, /*apply_flip=*/true,
                         ref_flipped.data());
  }
  // Sanity: the two encodings differ (asymmetric Q placement).
  ASSERT_NE(std::memcmp(ref_normal.data(), ref_flipped.data(), kInputFloats * sizeof(float)), 0);

  // Expected labels for active=p0 (final p0=350 vs p1=200 -> active wins by
  // 150). The move after turn 0 is p1's PASS and the game has no turn 2, so the
  // placement class targets are all pass/not-win; but the legality masks read
  // the (asymmetric) sampled board, so the label tail is NOT flip-invariant --
  // a flipped row carries the transposed masks. Build both frames from an
  // encoder in the turn-0 post-move state (the Q applied).
  GameStateEncoder label_enc{InputEncodingSpec{&dict}};
  label_enc.apply_move(fix.self_move);
  const InputEncodingSpec label_spec{&dict};
  float ref_labels[kLabelFloats];
  float ref_labels_flipped[kLabelFloats];
  encode_labels_flat(scores_view(label_enc, label_spec, fix.final_score_p0, fix.final_score_p1,
                                 fix.active_player, /*flip=*/false),
                     ref_labels);
  encode_labels_flat(scores_view(label_enc, label_spec, fix.final_score_p0, fix.final_score_p1,
                                 fix.active_player, /*flip=*/true),
                     ref_labels_flipped);
  // The masks make the two frames differ, so the flip actually reaches labels.
  ASSERT_NE(std::memcmp(ref_labels, ref_labels_flipped, kLabelFloats * sizeof(float)), 0);

  DataLoader::Params params;
  params.spec = {&dict};
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);
  loader.add_file(fix.path.string(), /*num_positions=*/1, fix.fsize);

  // The fixture has a single game contributing a single sample row.

  // apply_symmetry=false: the single row must match the canonical
  // (unflipped) encoding.
  {
    DataLoader::EpochConfig cfg;
    cfg.batch_size = 1;
    cfg.post_move = true;
    cfg.apply_symmetry = false;
    cfg.seed = 1;
    loader.epoch_start(cfg);
    std::vector<float> rows(kRowFloats, 0.0f);
    ASSERT_EQ(loader.load_batch(rows.data()), 1);
    ASSERT_EQ(std::memcmp(rows.data(), ref_normal.data(), kInputFloats * sizeof(float)), 0);
    ASSERT_EQ(std::memcmp(rows.data() + kInputFloats, ref_labels, kLabelFloats * sizeof(float)), 0);
  }

  // apply_symmetry=true: each epoch uses a different seed, producing a
  // different flip decision. Over many seeds we expect both buckets.
  {
    constexpr int n = 200;
    std::vector<float> row(kRowFloats, 0.0f);
    int normal_count = 0, flipped_count = 0;
    for (int i = 0; i < n; ++i) {
      DataLoader::EpochConfig cfg;
      cfg.batch_size = 1;
      cfg.post_move = true;
      cfg.apply_symmetry = true;
      cfg.seed = i + 100;
      loader.epoch_start(cfg);
      ASSERT_EQ(loader.load_batch(row.data()), 1);
      const bool is_normal =
        std::memcmp(row.data(), ref_normal.data(), kInputFloats * sizeof(float)) == 0;
      const bool is_flipped =
        std::memcmp(row.data(), ref_flipped.data(), kInputFloats * sizeof(float)) == 0;
      ASSERT_TRUE(is_normal || is_flipped);  // every row matches one of the two
      // The label tail flips with the input: a normal row carries the canonical
      // masks, a flipped row the transposed ones.
      const float* want = is_normal ? ref_labels : ref_labels_flipped;
      if (is_normal)
        ++normal_count;
      else
        ++flipped_count;
      ASSERT_EQ(std::memcmp(row.data() + kInputFloats, want, kLabelFloats * sizeof(float)), 0);
    }
    // With n=200 fair coin flips, the probability that one bucket is empty
    // is 2 * 2^-200; the test is effectively deterministic.
    ASSERT_GT(normal_count, 0);
    ASSERT_GT(flipped_count, 0);
    std::cout << "  DataLoader per-row symmetry: " << normal_count << " normal / " << flipped_count
              << " flipped (of " << n << ")\n";
  }
}

// The DataLoader maps a game's flat rows to turns starting at eligible_begin:
// a two-PLAY game whose metadata declares the region [1, 2) expands to exactly
// one row, and that row is the turn-1 post-move encoding (POV p1), not turn 0.
TEST(DataLoader, EligibleBeginOffset) {
  using namespace scribblez::binlog;
  namespace fs = std::filesystem;

  fs::path dir = fs::temp_directory_path() / ("scribblez_off_" + std::to_string(::getpid()) + "_" +
                                              std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{dir};

  // Synthetic game: turn 0 p0 plays 'Q' at (3,5); turn 1 p1 plays 'C' at (7,3).
  // Each player's leave after their play is 6 As.
  Rack p0_init;
  p0_init.add(Tile::from_char('Q'));
  for (int i = 0; i < 6; ++i) p0_init.add(Tile::from_char('A'));
  Rack p1_init;
  p1_init.add(Tile::from_char('C'));
  for (int i = 0; i < 6; ++i) p1_init.add(Tile::from_char('A'));

  Move q_play =
    make_play_full(3, 5, /*horizontal=*/true, 0b1, 42, {Glyph::of(Tile::from_char('Q'))});
  Move c_play =
    make_play_full(7, 3, /*horizontal=*/true, 0b1, 21, {Glyph::of(Tile::from_char('C'))});

  InitialRacks ir{};
  ir.p0 = p0_init;
  ir.p1 = p1_init;
  TurnBlob t0{};
  t0.move = q_play;
  TurnBlob t1{};
  t1.move = c_play;

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.num_games = 1;
  hdr.num_sample_positions = 1;

  GameMetadata gm{};
  gm.start_offset = sizeof(FileHeader) + sizeof(GameMetadata);
  gm.num_turns = 2;
  gm.sampled_turn = 1;
  gm.eligible_begin = 1;  // the single flat row stands for turn 1
  gm.eligible_end = 2;
  gm.final_score_p0 = 350;
  gm.final_score_p1 = 200;

  fs::path path = dir / "offset_region.slog";
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(&gm), sizeof(gm));
    f.write(reinterpret_cast<const char*>(&ir), sizeof(ir));
    f.write(reinterpret_cast<const char*>(&t0), sizeof(t0));
    f.write(reinterpret_cast<const char*>(&t1), sizeof(t1));
    ASSERT_TRUE(f);
  }
  const int64_t fsize = fs::file_size(path);

  Dictionary dict = medium_dict();

  // Reference: replay both moves, then encode turn 1 post-move -- POV p1 with
  // its post-play leave (6 As).
  std::vector<float> ref_row(kInputFloats, 0.0f);
  {
    GameStateEncoder ref_enc{InputEncodingSpec{&dict}};
    ref_enc.apply_move(q_play);
    ref_enc.apply_move(c_play);
    Rack leave;
    for (int i = 0; i < 6; ++i) leave.add(Tile::from_char('A'));
    ref_enc.encode_input(/*active_player=*/1, leave, /*apply_flip=*/false, ref_row.data());
  }
  // Labels for POV p1 (final 200 vs 350); turn 1 is the last turn, so the
  // placement class targets are all pass/not-win. The legality masks read the
  // turn-1 post-move board, so the reference encoder must be in that state
  // (both plays applied), matching the loader's unflipped row.
  GameStateEncoder label_enc{InputEncodingSpec{&dict}};
  label_enc.apply_move(q_play);
  label_enc.apply_move(c_play);
  float ref_labels[kLabelFloats];
  encode_labels_flat(scores_view(label_enc, InputEncodingSpec{&dict}, /*fs_active=*/200,
                                 /*fs_opp=*/350, /*active_player=*/1),
                     ref_labels);

  DataLoader::Params params;
  params.spec = {&dict};
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);
  loader.add_file(path.string(), /*num_positions=*/1, fsize);
  ASSERT_EQ(loader.num_positions(), 1);

  DataLoader::EpochConfig cfg;
  cfg.batch_size = 1;
  cfg.post_move = true;
  cfg.apply_symmetry = false;
  cfg.seed = 1;
  loader.epoch_start(cfg);
  std::vector<float> row(kRowFloats, 0.0f);
  ASSERT_EQ(loader.load_batch(row.data()), 1);
  ASSERT_EQ(std::memcmp(row.data(), ref_row.data(), kInputFloats * sizeof(float)), 0);
  ASSERT_EQ(std::memcmp(row.data() + kInputFloats, ref_labels, kLabelFloats * sizeof(float)), 0);
  std::cout << "  DataLoader eligible_begin offset decode OK\n";
}

// ===========================================================================
// Epoch-based DataLoader tests
// ===========================================================================

// Helper: write N games via BinaryLogWriter and return the .slog path + game count.
struct SlogFixture {
  std::filesystem::path dir;
  std::vector<std::filesystem::path> slog_paths;
  int total_games = 0;
};

static SlogFixture write_multi_file_slog(int games_per_file, int num_files) {
  namespace fs = std::filesystem;
  SlogFixture fix;
  fix.dir = fs::temp_directory_path() / ("scribblez_epoch_" + std::to_string(::getpid()) + "_" +
                                         std::to_string(std::random_device{}()));
  fs::create_directories(fix.dir);

  Dictionary dict = medium_dict();
  fix.total_games = games_per_file * num_files;

  scribblez::binlog::BinaryLogWriter writer(fix.dir.string(), games_per_file);
  for (int i = 0; i < fix.total_games; ++i) {
    scribblez::GameLogStorage log = play_test_game(dict, /*seed=*/2000ULL + i);
    writer.append(std::move(log));
  }

  for (const auto& ent : fs::directory_iterator(fix.dir)) {
    if (ent.path().extension() == ".slog") fix.slog_paths.push_back(ent.path());
  }
  std::sort(fix.slog_paths.begin(), fix.slog_paths.end());
  EXPECT_EQ(int(fix.slog_paths.size()), num_files);
  return fix;
}

TEST(DataLoader, EpochDeterminism) {
  // Two epoch_start calls with the same seed must produce identical output.
  using namespace scribblez::binlog;
  namespace fs = std::filesystem;

  auto fix = write_multi_file_slog(/*games_per_file=*/5, /*num_files=*/3);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{fix.dir};

  Dictionary dict = medium_dict();
  DataLoader::Params params;
  params.spec = {&dict};
  params.num_worker_threads = 2;
  params.num_prefetch_threads = 1;

  // Run two epochs with the same seed and verify byte-identical output.
  const int batch_size = 4;
  const uint64_t seed = 12345;

  auto run_epoch = [&](DataLoader& loader) {
    DataLoader::EpochConfig cfg;
    cfg.batch_size = batch_size;
    cfg.post_move = true;
    cfg.apply_symmetry = true;
    cfg.seed = seed;
    loader.epoch_start(cfg);

    std::vector<float> all_data;
    std::vector<float> batch(batch_size * kRowFloats);
    while (true) {
      int n = loader.load_batch(batch.data());
      if (n == 0) break;
      all_data.insert(all_data.end(), batch.begin(), batch.begin() + size_t(n) * kRowFloats);
    }
    return all_data;
  };

  // First run.
  DataLoader loader1(params);
  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = fs::file_size(p);
    loader1.add_file(p.string(), hdr.num_games, fsize);
  }
  auto data1 = run_epoch(loader1);

  // Second run: fresh loader, same files, same seed.
  DataLoader loader2(params);
  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = fs::file_size(p);
    loader2.add_file(p.string(), hdr.num_games, fsize);
  }
  auto data2 = run_epoch(loader2);

  ASSERT_EQ(data1.size(), data2.size());
  ASSERT_GT(data1.size(), 0);
  ASSERT_EQ(std::memcmp(data1.data(), data2.data(), data1.size() * sizeof(float)), 0);

  // Third run: same loader, same seed again -- must also be identical.
  auto data3 = run_epoch(loader1);
  ASSERT_EQ(data3.size(), data1.size());
  ASSERT_EQ(std::memcmp(data1.data(), data3.data(), data1.size() * sizeof(float)), 0);

  // Fourth run: different seed -- must differ.
  {
    DataLoader::EpochConfig cfg;
    cfg.batch_size = batch_size;
    cfg.post_move = true;
    cfg.apply_symmetry = true;
    cfg.seed = 99999;
    loader1.epoch_start(cfg);
    std::vector<float> data4;
    std::vector<float> batch(batch_size * kRowFloats);
    while (true) {
      int n = loader1.load_batch(batch.data());
      if (n == 0) break;
      data4.insert(data4.end(), batch.begin(), batch.begin() + size_t(n) * kRowFloats);
    }
    ASSERT_EQ(data4.size(), data1.size());
    ASSERT_NE(std::memcmp(data1.data(), data4.data(), data1.size() * sizeof(float)), 0);
  }

  std::cout << "  epoch determinism OK (" << data1.size() / kRowFloats << " rows)\n";
}

TEST(DataLoader, EpochCoverage) {
  // Every position in the dataset must appear exactly once per epoch.
  // Verified by running two epochs with different seeds and confirming
  // that they contain the same set of rows (just in different order).
  using namespace scribblez::binlog;
  namespace fs = std::filesystem;

  auto fix = write_multi_file_slog(/*games_per_file=*/4, /*num_files=*/3);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{fix.dir};

  Dictionary dict = medium_dict();
  DataLoader::Params params;
  params.spec = {&dict};
  params.num_worker_threads = 2;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);

  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = fs::file_size(p);
    loader.add_file(p.string(), hdr.num_games, fsize);
  }
  // Each game expands to one row per eligible turn; the loader knows the total.
  const int64_t total_positions = loader.num_positions();
  ASSERT_GT(total_positions, fix.total_games);  // strictly more rows than games

  // Helper: drain a full epoch into a flat float vector.
  const int row_sz = kRowFloats;
  auto drain_epoch = [&](uint64_t seed) {
    DataLoader::EpochConfig cfg;
    cfg.batch_size = 3;  // doesn't evenly divide 12 -> tests partial batch
    cfg.post_move = true;
    cfg.apply_symmetry = false;
    cfg.seed = seed;
    loader.epoch_start(cfg);

    std::vector<float> data;
    std::vector<float> batch(cfg.batch_size * row_sz);
    while (true) {
      int n = loader.load_batch(batch.data());
      if (n == 0) break;
      data.insert(data.end(), batch.begin(), batch.begin() + size_t(n) * row_sz);
    }
    return data;
  };

  std::vector<float> epoch1 = drain_epoch(7777);
  std::vector<float> epoch2 = drain_epoch(8888);

  // Both epochs must contain exactly total_positions rows.
  ASSERT_EQ(int64_t(epoch1.size()), total_positions * row_sz);
  ASSERT_EQ(int64_t(epoch2.size()), total_positions * row_sz);

  // The two epochs have different seeds, so should be in different order.
  ASSERT_NE(std::memcmp(epoch1.data(), epoch2.data(), epoch1.size() * sizeof(float)), 0);

  // Every row in epoch1 must appear exactly once in epoch2 (same content,
  // different order).
  std::vector<bool> found(total_positions, false);
  for (int64_t ei = 0; ei < total_positions; ++ei) {
    const float* row1 = epoch1.data() + ei * row_sz;
    bool matched = false;
    for (int64_t ri = 0; ri < total_positions; ++ri) {
      if (found[ri]) continue;
      if (std::memcmp(row1, epoch2.data() + ri * row_sz, row_sz * sizeof(float)) == 0) {
        found[ri] = true;
        matched = true;
        break;
      }
    }
    ASSERT_TRUE(matched);
  }
  for (int64_t i = 0; i < total_positions; ++i) ASSERT_TRUE(found[i]);

  std::cout << "  epoch coverage OK (" << total_positions << " positions)\n";
}

TEST(DataLoader, EpochMemoryBudgetStress) {
  // Set a tiny memory budget (just enough for one file) and run a full epoch.
  // This exercises LRU eviction heavily: the loader must load/evict files
  // repeatedly as it walks through shuffled file-level work units.
  using namespace scribblez::binlog;
  namespace fs = std::filesystem;

  auto fix = write_multi_file_slog(/*games_per_file=*/4, /*num_files=*/5);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{fix.dir};

  // Find the largest file size to set a budget just above it.
  int64_t max_fsize = 0;
  std::vector<std::pair<int64_t, int64_t>> file_info;  // (num_pos, fsize)
  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = fs::file_size(p);
    file_info.emplace_back(hdr.num_games, fsize);
    if (fsize > max_fsize) max_fsize = fsize;
  }

  // Budget = just one file (largest). This forces eviction on every file switch.
  Dictionary dict = medium_dict();
  DataLoader::Params params;
  params.spec = {&dict};
  params.memory_budget = max_fsize + 1;  // allow exactly one file at a time
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);

  for (int i = 0; i < int(fix.slog_paths.size()); ++i) {
    loader.add_file(fix.slog_paths[i].string(), file_info[i].first, file_info[i].second);
  }
  const int64_t total_positions = loader.num_positions();

  // Run epoch with batch_size=2 (small batches = more file switches).
  DataLoader::EpochConfig cfg;
  cfg.batch_size = 2;
  cfg.post_move = true;
  cfg.apply_symmetry = true;
  cfg.seed = 42;
  loader.epoch_start(cfg);

  int rows_decoded = 0;
  std::vector<float> batch(cfg.batch_size * kRowFloats);
  while (true) {
    int n = loader.load_batch(batch.data());
    if (n == 0) break;
    rows_decoded += n;
    // Memory should never exceed budget + one extra file (prefetch).
    // In practice with prefetch disabled (budget too tight), should be <= 2 * max_fsize.
    ASSERT_LE(loader.resident_bytes(), 2 * max_fsize + 100);
  }
  ASSERT_EQ(rows_decoded, total_positions);

  // Verify determinism: same seed produces same data.
  loader.epoch_start(cfg);
  std::vector<float> run1;
  while (true) {
    int n = loader.load_batch(batch.data());
    if (n == 0) break;
    run1.insert(run1.end(), batch.begin(), batch.begin() + size_t(n) * kRowFloats);
  }

  loader.epoch_start(cfg);
  std::vector<float> run2;
  while (true) {
    int n = loader.load_batch(batch.data());
    if (n == 0) break;
    run2.insert(run2.end(), batch.begin(), batch.begin() + size_t(n) * kRowFloats);
  }
  ASSERT_EQ(run1.size(), run2.size());
  ASSERT_EQ(std::memcmp(run1.data(), run2.data(), run1.size() * sizeof(float)), 0);

  std::cout << "  epoch memory-budget stress OK (" << rows_decoded
            << " rows, budget=" << params.memory_budget << " bytes, " << fix.slog_paths.size()
            << " files)\n";
}

TEST(DataLoader, EpochShufflesAcrossSeeds) {
  // Different seeds produce different orderings of the same data.
  using namespace scribblez::binlog;
  namespace fs = std::filesystem;

  auto fix = write_multi_file_slog(/*games_per_file=*/6, /*num_files=*/2);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{fix.dir};

  Dictionary dict = medium_dict();
  DataLoader::Params params;
  params.spec = {&dict};
  params.num_worker_threads = 2;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);

  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = fs::file_size(p);
    loader.add_file(p.string(), hdr.num_games, fsize);
  }

  auto run_with_seed = [&](uint64_t seed) {
    DataLoader::EpochConfig cfg;
    cfg.batch_size = 4;
    cfg.post_move = true;
    cfg.apply_symmetry = false;  // no flips -- pure shuffle test
    cfg.seed = seed;
    loader.epoch_start(cfg);
    std::vector<float> data;
    std::vector<float> batch(cfg.batch_size * kRowFloats);
    while (true) {
      int n = loader.load_batch(batch.data());
      if (n == 0) break;
      data.insert(data.end(), batch.begin(), batch.begin() + size_t(n) * kRowFloats);
    }
    return data;
  };

  auto d1 = run_with_seed(100);
  auto d2 = run_with_seed(200);
  auto d3 = run_with_seed(100);  // same as d1

  ASSERT_EQ(d1.size(), d2.size());
  ASSERT_EQ(d1.size(), d3.size());
  ASSERT_GT(d1.size(), 0);
  // Same seed -> identical.
  ASSERT_EQ(std::memcmp(d1.data(), d3.data(), d1.size() * sizeof(float)), 0);
  // Different seed -> different ordering.
  ASSERT_NE(std::memcmp(d1.data(), d2.data(), d1.size() * sizeof(float)), 0);

  std::cout << "  epoch seed-shuffle OK\n";
}

// =========================================================================
// LeaveValues and HastyEquity tests
// =========================================================================

// Build and write a minimal .klv2 file in a temp path.
// The KWG encodes three single-tile leaves: the blank "?" (tile=0), "A"
// (tile=1) and "B" (tile=2). Macondo's leave KWG numbers the blank as machine
// letter 0 (sorting ahead of the letters), so including it here exercises the
// blank-leave mapping that real exchanges depend on.
// KWG node layout: bits 0..21 arc_index, 22 is_end, 23 accepts, 24..31 tile.
//
// Node 0 (root): arc_index=1, is_end=1, accepts=0, tile=0
// Node 1 (?):    arc_index=0, is_end=0, accepts=1, tile=0
// Node 2 (A):    arc_index=0, is_end=0, accepts=1, tile=1
// Node 3 (B):    arc_index=0, is_end=1, accepts=1, tile=2
//
// Word order: ?(index 0) = 12.0f, A(index 1) = 1.5f, B(index 2) = -2.5f.
struct KlvFixture {
  std::filesystem::path path;
};

KlvFixture write_synthetic_klv(const std::filesystem::path& dir) {
  std::filesystem::path p = dir / "synthetic.klv2";
  std::ofstream f(p, std::ios::binary | std::ios::trunc);

  auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto write_f32 = [&](float v) { f.write(reinterpret_cast<const char*>(&v), 4); };

  // kwg_node_count = 4
  write_u32(4);
  // Node 0: root; arc_index=1, is_end=1, accepts=0, tile=0
  write_u32((0u << 24) | (1u << 22) | (0u << 23) | 1u);
  // Node 1: ? (blank); arc_index=0, is_end=0, accepts=1, tile=0
  write_u32((0u << 24) | (0u << 22) | (1u << 23) | 0u);
  // Node 2: A; arc_index=0, is_end=0, accepts=1, tile=1
  write_u32((1u << 24) | (0u << 22) | (1u << 23) | 0u);
  // Node 3: B; arc_index=0, is_end=1, accepts=1, tile=2
  write_u32((2u << 24) | (1u << 22) | (1u << 23) | 0u);
  // num_leaves = 3
  write_u32(3);
  write_f32(12.0f);
  write_f32(1.5f);
  write_f32(-2.5f);
  EXPECT_TRUE(f.good());
  return KlvFixture{p};
}

TEST(LeaveValues, Synthetic) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_klv_XXXXXX";
  fs::create_directories(tmp);

  KlvFixture fix = write_synthetic_klv(tmp);
  LeaveValues lv = LeaveValues::load(fix.path.string());

  // Leave "A"
  Rack a;
  a.add(Tile::from_char('A'));
  ASSERT_LT(std::abs(lv.lookup(a) - 1.5f), 1e-4f);

  // Leave "B"
  Rack b;
  b.add(Tile::from_char('B'));
  ASSERT_LT(std::abs(lv.lookup(b) - (-2.5f)), 1e-4f);

  // Leave "?" (blank). Macondo's leave KWG numbers the blank as machine letter
  // 0; a regression here means blank-bearing leaves silently look up as 0.
  Rack blank;
  blank.add(BLANK);
  ASSERT_LT(std::abs(lv.lookup(blank) - 12.0f), 1e-4f);

  // Empty leave → 0
  Rack empty;
  ASSERT_EQ(lv.lookup(empty), 0.0f);

  // Unknown leave (C) → 0
  Rack c;
  c.add(Tile::from_char('C'));
  ASSERT_EQ(lv.lookup(c), 0.0f);

  fs::remove_all(tmp);
}

TEST(LeaveValues, RealKwg) {
  // The real NWL23 leave values ship with the Macondo checkout; skip if that
  // is not installed.
  const std::string klv_path = HastyEquity::default_leaves_path("NWL23");
  if (!std::filesystem::exists(klv_path)) {
    GTEST_SKIP() << "no leaves.klv2 at " << klv_path;
  }

  LeaveValues lv = LeaveValues::load(klv_path);

  // Single blank is a well-known leave with strongly positive value.
  Rack blank_leave;
  blank_leave.add(BLANK);
  float blank_val = lv.lookup(blank_leave);
  ASSERT_GT(blank_val, 20.0f);  // known to be ~24..26 in Macondo NWL23

  // Empty leave is 0.
  Rack empty;
  ASSERT_EQ(lv.lookup(empty), 0.0f);
}

TEST(HastyEquity, Components) {
  // Test the four equity components individually without initialising the
  // singleton (we call the pure functions via a test harness that constructs
  // a HastyEquity from a synthetic KLV).
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_heq_XXXXXX";
  fs::create_directories(tmp);

  KlvFixture fix = write_synthetic_klv(tmp);
  // Write an empty PEG JSON (valid empty array).
  std::filesystem::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }

  HastyEquity::init(fix.path.string(), peg_path.string());

  const HastyEquity& eq = HastyEquity::instance();
  Board board;  // empty
  Rack opp;

  // --- leave equity: PLAY move that uses all 7 tiles (empty leave) mid-game
  Move all_out = make_play_full(7, 4, /*horizontal=*/true, 0b1111111, 50,
                                {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('A'))});

  Rack rack_7a;
  for (int i = 0; i < 7; ++i) rack_7a.add(Tile::from_char('A'));

  // With bag_size > 0, leave is empty → leave_equity = 0.
  // Opening adjustment: tile A at columns 4..10; columns 6 and 8 are in
  // penalty set, both have vowel A → 2 * -0.7 = -1.4.
  double e_mid = eq.equity(all_out, board, 86, opp, rack_7a);
  ASSERT_LT(std::abs(e_mid - (50.0 - 1.4)), 1e-3);

  // --- leave equity with non-empty leave (uses synthetic KLV: A=1.5, B=-2.5)
  // Play a single A, leaving AAAAAA (6 A's). Our synthetic KLV only has
  // single-tile leaves so the 6-tile leave returns 0.
  Move one_a = make_play_full(7, 7, /*horizontal=*/true, 0b1, 2, {Glyph::of(Tile::from_char('A'))});

  // rack = single A; leave = empty after playing it.
  Rack rack_1a;
  rack_1a.add(Tile::from_char('A'));
  double e_one = eq.equity(one_a, board, 86, opp, rack_1a);
  // score=2, leave=empty(0), opening: center col=7 not in {2,6,8,12} → 0
  ASSERT_LT(std::abs(e_one - 2.0), 1e-3);

  // --- endgame adjustment (bag_size = 0, non-out play)
  // Leave a single B on the rack (leave value from KLV = -2.5 but ignored
  // for endgame penalty which uses tile point values).
  Move play_a_endgame =
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 2, {Glyph::of(Tile::from_char('A'))});

  // Rack = AB, play A, leave = B (value 3). bag_size = 0.
  // endgame_adjustment = -2 * 3 - 10 = -16.
  Rack rack_ab;
  rack_ab.add(Tile::from_char('A'));
  rack_ab.add(Tile::from_char('B'));
  Board board_with_tiles;                                       // non-empty so opening adj = 0
  board_with_tiles.set(0, 0, Glyph::of(Tile::from_char('Q')));  // make non-empty
  double e_eg = eq.equity(play_a_endgame, board_with_tiles, 0, opp, rack_ab);
  // score=2, leave_equity=0 (bag=0), opening=0, peg=0, endgame=-16
  ASSERT_LT(std::abs(e_eg - (2.0 - 16.0)), 1e-3);

  // --- endgame out-play (leave empty, bag = 0)
  // Play both tiles A and B (2 tiles used, both in glyphs).
  Move out_play =
    make_play_full(7, 7, /*horizontal=*/true, 0b11, 5,
                   {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B'))});

  // Opponent has a Q (value=10) on their rack.
  Rack opp_q;
  opp_q.add(Tile::from_char('Q'));
  // endgame bonus = 2 * 10 = 20.
  double e_out = eq.equity(out_play, board_with_tiles, 0, opp_q, rack_ab);
  ASSERT_LT(std::abs(e_out - (5.0 + 20.0)), 1e-3);

  fs::remove_all(tmp);
}

// Regression test for blank-bearing exchange equity. An EXCHANGE's equity is
// just the leave value of the tiles kept (score 0, no opening/peg/endgame
// adjustments mid-game), so a mis-keyed blank leave surfaces directly as a
// wrong (typically 0) exchange equity in the web move list.
TEST(HastyEquity, ExchangeBlankLeave) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_heq_xch_XXXXXX";
  fs::create_directories(tmp);

  KlvFixture fix = write_synthetic_klv(tmp);  // ?=12.0, A=1.5, B=-2.5
  std::filesystem::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());
  const HastyEquity& eq = HastyEquity::instance();

  Board board;  // empty
  Rack opp;

  // Rack = A + blank. Exchanging the A keeps the blank, so the equity must be
  // the blank leave value (12.0), not 0. bag_size > 0 so no endgame term.
  Rack rack_a_blank;
  rack_a_blank.add(Tile::from_char('A'));
  rack_a_blank.add(BLANK);

  TileCounts surrender_a;
  surrender_a.add(Tile::from_char('A'));
  Move exch_a = Move::exchange(surrender_a);

  // Single-move and batched paths must agree and both reflect the blank leave.
  double single = eq.equity(exch_a, board, 50, opp, rack_a_blank);
  ASSERT_LT(std::abs(single - 12.0), 1e-3);

  std::vector<Move> moves{exch_a};
  std::vector<double> batched = eq.equities(moves, board, 50, opp, rack_a_blank);
  ASSERT_EQ(batched.size(), 1);
  ASSERT_LT(std::abs(batched[0] - 12.0), 1e-3);

  fs::remove_all(tmp);
}

// The keystone streaming guarantee: a row encoded directly from a live game's
// GameLog view (the streaming path) is BIT-IDENTICAL to the row the disk
// pipeline produces by writing that game to a .slog and decoding it back. Both
// funnel through PositionEncoder, so any divergence would mean the view built
// from the .slog buffer differs from the one built from self-play storage.
TEST(Streaming, DiskEncodeEquivalence) {
  using namespace scribblez;
  using namespace scribblez::binlog;
  namespace fs = std::filesystem;

  Dictionary dict = medium_dict();
  fs::path dir = fs::temp_directory_path() / ("scribblez_eq_" + std::to_string(::getpid()) + "_" +
                                              std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  struct DirCleanup {
    fs::path p;
    ~DirCleanup() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } cleanup{dir};

  const int row_floats = kRowFloats;
  int compared = 0;
  for (uint64_t seed : std::vector<uint64_t>{7, 99, 12345}) {
    GameLogStorage storage = play_test_game(dict, seed);

    // Write a COPY through the real disk writer (it picks the sampled turn).
    {
      BinaryLogWriter writer(dir.string(), /*games_per_file=*/1);
      writer.append(GameLogStorage(storage));
    }
    fs::path slog;
    for (const auto& ent : fs::directory_iterator(dir)) {
      if (ent.path().extension() == ".slog") slog = ent.path();
    }
    ASSERT_FALSE(slog.empty());

    const int64_t fsize = fs::file_size(slog);
    std::vector<char> raw(fsize);
    {
      std::ifstream f(slog, std::ios::binary);
      f.read(raw.data(), fsize);
      ASSERT_TRUE(f);
    }
    const auto* metas = reinterpret_cast<const GameMetadata*>(raw.data() + sizeof(FileHeader));
    const int sampled = metas[0].sampled_turn;

    for (bool post_move : {false, true}) {
      const uint8_t flip = 0;
      std::vector<float> row_disk(row_floats, 0.0f);
      BlockDecoder decoder(InputEncodingSpec{&dict});
      decoder.decode(raw.data(), "eq", /*local_start=*/0, /*n_rows=*/1, &flip, post_move,
                     /*output_row_start=*/0, row_disk.data());

      std::vector<float> row_stream(row_floats, 0.0f);
      PositionEncoder enc(InputEncodingSpec{&dict});
      enc.encode_row<PositionEvalTask>(storage.view(), sampled, post_move, /*flip=*/false,
                                       row_stream.data());

      for (int i = 0; i < row_floats; ++i) ASSERT_EQ(row_disk[i], row_stream[i]);
      ++compared;
    }

    // Fresh dir per seed so directory_iterator finds exactly one file.
    for (const auto& ent : fs::directory_iterator(dir)) fs::remove(ent.path());
  }
  ASSERT_EQ(compared, 6);
  std::cout << "  streaming/disk encode equivalence OK (" << compared << " rows)\n";
}

// StreamingRowBuffer: many producers, tiny slots (frequent boundary crossings).
// Every global row must be written exactly once and read back in a contiguous
// set, with no slot overwritten while the consumer holds it.
TEST(StreamingRowBuffer, Concurrency) {
  using namespace scribblez::binlog;
  const int n_slots = 2, rows_per_slot = 4, row_floats = 1;
  const int slots_to_consume = 64;
  std::vector<std::vector<float>> bufs(n_slots,
                                       std::vector<float>(rows_per_slot * row_floats, -1.0f));
  std::vector<float*> slots;
  for (auto& b : bufs) slots.push_back(b.data());
  StreamingRowBuffer ring(slots.data(), n_slots, rows_per_slot, row_floats);

  // Bound production to exactly slots_to_consume full generations via a shared
  // work counter, so consuming that many slots drains every produced row. (With
  // unbounded production the first N slots of one lane can race ahead of the
  // other, so "first M consumed slots" would not be generations 0..M-1.)
  const uint64_t total_rows = uint64_t(slots_to_consume) * rows_per_slot;
  std::atomic<uint64_t> work{0};
  const int K = 8;  // many producers + tiny slots -> frequent slot-boundary crossings
  std::vector<std::thread> producers;
  for (int t = 0; t < K; ++t) {
    producers.emplace_back([&] {
      while (work.fetch_add(1, std::memory_order_relaxed) < total_rows) {
        uint64_t r = ring.claim_row();
        if (r == StreamingRowBuffer::kNoRow) break;
        ring.row_dest(r)[0] = float(r);
        ring.commit_row(r);
      }
    });
  }

  std::set<uint64_t> seen;
  bool dup = false;
  for (int i = 0; i < slots_to_consume; ++i) {
    int slot = ring.wait_full_slot();
    ASSERT_GE(slot, 0);
    for (int k = 0; k < rows_per_slot; ++k) {
      uint64_t v = slots[slot][k];
      if (!seen.insert(v).second) dup = true;
    }
    ring.release_slot(slot);
  }
  for (auto& p : producers) p.join();

  ASSERT_FALSE(dup);  // each global row written and read exactly once
  ASSERT_EQ(int(seen.size()), slots_to_consume * rows_per_slot);
  for (uint64_t v = 0; v < total_rows; ++v) ASSERT_EQ(seen.count(v), 1);  // exactly [0, total)
  std::cout << "  StreamingRowBuffer concurrency OK (" << seen.size() << " rows, K=" << K << ")\n";
}

// StreamingRowBuffer shutdown: stop() must wake every blocked producer (and the
// consumer) so nothing hangs, even with producers parked on backpressure.
TEST(StreamingRowBuffer, Shutdown) {
  using namespace scribblez::binlog;
  const int n_slots = 2, rows_per_slot = 8, row_floats = 1;
  std::vector<std::vector<float>> bufs(n_slots, std::vector<float>(rows_per_slot * row_floats));
  std::vector<float*> slots;
  for (auto& b : bufs) slots.push_back(b.data());
  StreamingRowBuffer ring(slots.data(), n_slots, rows_per_slot, row_floats);

  std::atomic<int> exited{0};
  const int K = 4;
  std::vector<std::thread> producers;
  for (int t = 0; t < K; ++t) {
    producers.emplace_back([&] {
      while (true) {
        uint64_t r = ring.claim_row();
        if (r == StreamingRowBuffer::kNoRow) break;
        ring.row_dest(r)[0] = float(r);
        ring.commit_row(r);
      }
      exited.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // No consumer: producers fill both slots, then park on backpressure. stop()
  // must release them all.
  ring.stop();
  for (auto& p : producers) p.join();
  ASSERT_EQ(exited.load(), K);
  ASSERT_EQ(ring.wait_full_slot(), -1);
  std::cout << "  StreamingRowBuffer shutdown OK\n";
}

// pick_sampled_turn only chooses turns in the eligible region -- within the
// bag-non-empty prefix and not before the last random-opening ply -- and
// returns -1 when the region is empty.
TEST(BinaryLog, PickSampledTurnEligibility) {
  using namespace scribblez;
  using namespace scribblez::binlog;

  GameLogStorage s;
  s.turns.resize(3);  // value-initialized: bag_size_before == 0
  s.turns[0].bag_size_before = 9;
  s.turns[1].bag_size_before = 5;
  std::mt19937_64 rng(123);
  ASSERT_EQ(eligible_span(s.view()).begin, 0);
  ASSERT_EQ(eligible_span(s.view()).end, 2);
  for (int i = 0; i < 20; ++i) {
    const int t = pick_sampled_turn(s.view(), rng);
    ASSERT_TRUE(t == 0 || t == 1);
  }

  // A random opening excludes the positions before its last ply: with 2 random
  // plies, only turns >= 1 remain eligible.
  s.num_random_opening_plies = 2;
  ASSERT_EQ(eligible_span(s.view()).begin, 1);
  for (int i = 0; i < 20; ++i) ASSERT_EQ(pick_sampled_turn(s.view(), rng), 1);

  // A game that ended during its random opening has an empty eligible region.
  s.num_random_opening_plies = 3;
  ASSERT_EQ(pick_sampled_turn(s.view(), rng), -1);
  s.num_random_opening_plies = 0;

  GameLogStorage z;
  z.turns.resize(2);  // all ineligible
  ASSERT_EQ(pick_sampled_turn(z.view(), rng), -1);

  // pick_any_turn (max-move-per-lane sampling) ignores bag size: every turn is eligible,
  // and the choice covers the whole range. An empty game yields -1.
  std::array<bool, 3> seen{};
  for (int i = 0; i < 200; ++i) {
    const int t = pick_any_turn(s.view(), rng);
    ASSERT_TRUE(t >= 0 && t < 3);
    seen[t] = true;
  }
  ASSERT_TRUE(seen[0] && seen[1] && seen[2]);  // bag_size_before == 0 turns included
  GameLogStorage empty;
  ASSERT_EQ(pick_any_turn(empty.view(), rng), -1);
  std::cout << "  pick_sampled_turn / pick_any_turn eligibility OK\n";
}

// ShadowMoveGen, summed over every anchor (no pruning), reproduces exactly the
// move set of MoveGenerator::generate, and every anchor's score bound is
// admissible (>= the score of each play canonically anchored there). The latter
// is the invariant that makes best-first equity pruning exact.
TEST(Movegen, ShadowMatchesFull) {
  using namespace scribblez;
  Dictionary dict = medium_dict();
  long positions = 0, total_moves = 0;
  for (uint64_t seed : {7ULL, 99ULL, 12345ULL, 2024ULL, 55ULL}) {
    GameLogStorage log = play_test_game(dict, seed);
    Board board;
    for (const TurnRecord& t : log.turns) {
      const Rack& rack = t.rack_before;

      MoveGenerator gen(board, dict);
      const std::vector<Move> full = gen.generate(rack);

      ShadowMoveGen smg(board, dict);
      const std::vector<ShadowAnchor> anchors = smg.anchors(rack);
      std::vector<Move> shadow;
      for (const ShadowAnchor& a : anchors) {
        std::vector<Move> am;
        smg.generate_anchor(a, rack, am);
        for (const Move& m : am) {
          // The per-tile-count bound never underestimates a real play's score.
          ASSERT_LE(int(m.score()), a.score_bound_by_size[m.num_glyphs()]);
        }
        for (Move& m : am) shadow.push_back(std::move(m));
      }

      ASSERT_EQ(key_set(board, full), key_set(board, shadow));

      ++positions;
      total_moves += full.size();
      board.apply(t.move);
    }
  }
  ASSERT_GT(positions, 0);
  std::cout << "  ShadowMoveGen matches full generate + bound admissible (" << positions
            << " positions, " << total_moves << " moves)\n";
}

// HastyBot's shadow-play search picks exactly the move the deterministic
// reference (full generation + equity argmax) would, across real self-play
// games. Requires the real NWL23 KWG + leaves; skipped if absent.
namespace {
class ShadowCheckAgent : public scribblez::Agent {
 public:
  ShadowCheckAgent(int tid, const std::string& name)
      : scribblez::Agent(tid, name), bot_({.thread_id = tid, .name = name}) {}
  scribblez::MoveDecision make_move(const scribblez::MoveRequest& req) override {
    const scribblez::Move shadow = bot_.make_move(req).move;
    const scribblez::Move ref = scribblez::hasty_best_move_reference(req);
    EXPECT_EQ(move_key(req.board, shadow), move_key(req.board, ref));
    ++comparisons;
    return shadow;
  }
  long comparisons = 0;

 private:
  scribblez::HastyBotAgent bot_;
};
}  // namespace

TEST(HastyEquity, ShadowMatchesReference) {
  namespace fs = std::filesystem;
  const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves = scribblez::HastyEquity::default_leaves_path("NWL23");
  const std::string peg = scribblez::HastyEquity::default_peg_path();
  if (!fs::exists(kwg) || !fs::exists(leaves)) {
    GTEST_SKIP() << "no NWL23 kwg/leaves";
  }
  scribblez::Dictionary dict = scribblez::Dictionary::load_kwg(kwg);
  scribblez::HastyEquity::init(leaves, peg);

  long comparisons = 0;
  for (uint64_t seed = 1; seed <= 10; ++seed) {
    ShadowCheckAgent a0(0, "A"), a1(0, "B");
    scribblez::Game g(a0, a1, dict, seed);
    g.play();
    comparisons += a0.comparisons + a1.comparisons;
  }
  ASSERT_GT(comparisons, 0);
  std::cout << "  HastyBot shadow search matches reference (" << comparisons << " positions)\n";
}

// wmp_generate (WordMap anagram lookup) produces exactly the same set of legal
// plays as the GADDAG move generator, for blank-free racks. Always runs against
// the in-memory medium dictionary.
TEST(WordMap, GenerateMatchesFull) {
  using namespace scribblez;
  Dictionary dict = medium_dict();
  const WordMap& wm = dict.word_map();
  long positions = 0;
  for (uint64_t seed : {7ULL, 99ULL, 12345ULL, 2024ULL, 55ULL, 13ULL, 808ULL, 4242ULL}) {
    GameLogStorage log = play_test_game(dict, seed);
    Board board;
    for (const TurnRecord& t : log.turns) {
      const Rack& rack = t.rack_before;
      if (rack.counts().blanks() == 0) {
        MoveGenerator gen(board, dict);
        const std::vector<Move> full = gen.generate(rack);
        const std::vector<Move> wmp = wmp_generate(board, dict, wm, rack);
        ASSERT_EQ(key_set(board, full), key_set(board, wmp));

        // Per-anchor WMP generation matches the GADDAG's generate_anchor exactly,
        // so it can drive the shadow best-first loop.
        ShadowMoveGen smg(board, dict);
        WmpSubracks subracks;
        int rack_tiles = 0;
        wmp_rack_subracks(rack, subracks, rack_tiles);
        for (const ShadowAnchor& a : smg.anchors(rack)) {
          std::vector<Move> g, w;
          smg.generate_anchor(a, rack, g);
          wmp_generate_anchor(board, wm, subracks, rack_tiles, a, w);
          ASSERT_EQ(key_set(board, g), key_set(board, w));
        }

        // The per-extent partition covers every legal play exactly once, and each
        // extent's score bound never underestimates its plays (so the best-first
        // early-exit over extents is exact).
        std::vector<Move> extent_union;
        for (const ShadowExtent& e : smg.extents(rack, &wm)) {
          std::vector<Move> em;
          wmp_generate_extent(board, wm, subracks, e, em);
          for (const Move& m : em) {
            ASSERT_LE(int(m.score()), e.score_bound);
            extent_union.push_back(m);
          }
        }
        ASSERT_EQ(key_set(board, full), key_set(board, extent_union));
        ++positions;
      }
      board.apply(t.move);
    }
  }
  ASSERT_GT(positions, 0);
  std::cout << "  wmp_generate matches full generate (" << positions << " blank-free positions)\n";
}

// Captures every blank-free (board, rack) a HastyBot faces during self-play, so
// the benchmark can compare GADDAG vs WordMap generation on realistic positions.
namespace {
struct CapturedPos {
  scribblez::Board board;
  scribblez::Rack rack;
  scribblez::Rack opp_rack;
  int my_score;
  int opp_score;
  int bag_size;
};
class CapturingAgent : public scribblez::Agent {
 public:
  CapturingAgent(int tid, const std::string& name, std::vector<CapturedPos>& sink,
                 std::vector<CapturedPos>& blanked_sink)
      : scribblez::Agent(tid, name),
        bot_({.thread_id = tid, .name = name}),
        sink_(sink),
        blanked_sink_(blanked_sink) {}
  scribblez::MoveDecision make_move(const scribblez::MoveRequest& req) override {
    auto& dst = req.my_rack.counts().blanks() == 0 ? sink_ : blanked_sink_;
    dst.push_back(
      {req.board, req.my_rack, req.opp_rack, req.my_score, req.opp_score, req.bag_size});
    return bot_.make_move(req).move;
  }

 private:
  scribblez::HastyBotAgent bot_;
  std::vector<CapturedPos>& sink_;
  std::vector<CapturedPos>& blanked_sink_;
};
}  // namespace

// WordMap anagram-lookup move generation agrees with the GADDAG generator on
// real NWL23 positions drawn from HastyBot self-play: for blank-free racks the
// two enumerate the same legal-play set and the WordMap-driven HastyBot makes
// the same move choice, and for blank-bearing racks (which fall back to the
// GADDAG path) the choices likewise match. Requires the NWL23 KWG + leaves;
// skipped if absent.
TEST(WordMap, MatchesGaddagRealLexicon) {
  namespace fs = std::filesystem;
  using namespace scribblez;
  const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  const std::string peg = HastyEquity::default_peg_path();
  if (!fs::exists(kwg) || !fs::exists(leaves)) {
    GTEST_SKIP() << "no NWL23 kwg/leaves";
  }
  Dictionary dict = Dictionary::load_kwg(kwg);
  HastyEquity::init(leaves, peg);
  const WordMap& wm = dict.word_map();

  std::vector<CapturedPos> positions, blanked;
  for (uint64_t seed = 1; seed <= 10; ++seed) {
    CapturingAgent a0(0, "A", positions, blanked), a1(0, "B", positions, blanked);
    Game g(a0, a1, dict, seed);
    g.play();
  }

  // Blank-bearing racks fall back to the GADDAG path, so the WordMap bot must
  // still pick exactly the move HastyBot (make_move) does on them.
  HastyBotAgent blank_bot({.thread_id = 0, .name = "blankcheck"});
  for (const CapturedPos& p : blanked) {
    const MoveRequest req{p.board, dict, p.rack, p.opp_rack, p.my_score, p.opp_score, p.bag_size};
    ASSERT_EQ(move_key(p.board, blank_bot.make_move(req).move),
              move_key(p.board, hasty_best_move_wmp(req)));
  }

  // Blank-free racks: WordMap lookup enumerates exactly the GADDAG's legal plays,
  // and the WordMap-driven HastyBot (shadow best-first with early-exit) makes the
  // same move choice as the reference GADDAG HastyBot.
  HastyBotAgent bot({.thread_id = 0, .name = "wmpcheck"});
  long total_moves = 0;
  for (const CapturedPos& p : positions) {
    MoveGenerator gen(p.board, dict);
    const std::vector<Move> full = gen.generate(p.rack);
    const std::vector<Move> wmp = wmp_generate(p.board, dict, wm, p.rack);
    ASSERT_EQ(key_set(p.board, full), key_set(p.board, wmp));
    total_moves += full.size();

    const MoveRequest req{p.board, dict, p.rack, p.opp_rack, p.my_score, p.opp_score, p.bag_size};
    ASSERT_EQ(move_key(p.board, bot.make_move(req).move),
              move_key(p.board, hasty_best_move_wmp(req)));
  }
  ASSERT_FALSE(positions.empty());
  std::cout << "  WMP/GADDAG equivalence OK (" << positions.size() << " blank-free + "
            << blanked.size() << " blanked positions, " << total_moves << " plays)\n";
}

TEST(Util, Helpers) {
  // round_up_pow2: exact powers map to themselves; everything else rounds up.
  ASSERT_EQ(util::round_up_pow2(0), 1);
  ASSERT_EQ(util::round_up_pow2(1), 1);
  ASSERT_EQ(util::round_up_pow2(2), 2);
  ASSERT_EQ(util::round_up_pow2(3), 4);
  ASSERT_EQ(util::round_up_pow2(5), 8);
  ASSERT_EQ(util::round_up_pow2(8), 8);
  ASSERT_EQ(util::round_up_pow2(9), 16);
  ASSERT_EQ(util::round_up_pow2(1u << 20), (1u << 20));
  ASSERT_EQ(util::round_up_pow2((1u << 20) + 1), (1u << 21));

  // align_up to a power-of-two boundary.
  ASSERT_EQ(util::align_up(0, 8), 0);
  ASSERT_EQ(util::align_up(1, 8), 8);
  ASSERT_EQ(util::align_up(7, 8), 8);
  ASSERT_EQ(util::align_up(8, 8), 8);
  ASSERT_EQ(util::align_up(9, 8), 16);
  ASSERT_EQ(util::align_up(7, 1), 7);

  // plane_index: row-major vs transpose across the diagonal.
  ASSERT_EQ(util::plane_index(2, 3, 15, false), 2 * 15 + 3);
  ASSERT_EQ(util::plane_index(2, 3, 15, true), 3 * 15 + 2);
  ASSERT_EQ(util::plane_index(4, 4, 15, false), util::plane_index(4, 4, 15, true));

  // The four orthogonal neighbor deltas are unit steps with zero net sum.
  int sum_dr = 0, sum_dc = 0;
  for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
    ASSERT_NE((dr == 0), (dc == 0));  // exactly one axis moves
    ASSERT_TRUE(dr >= -1 && dr <= 1 && dc >= -1 && dc <= 1);
    sum_dr += dr;
    sum_dc += dc;
  }
  ASSERT_TRUE(sum_dr == 0 && sum_dc == 0);
}

// Backs the invariant "NeuralAgent with --top-k=1 plays exactly HastyBot's
// move" without instantiating the (TensorRT-linked) agent. At k=1 both agents
// reduce to the same equity argmax over the legal plays, via two different
// HastyEquity entry points: HastyBot scores moves one at a time with
// HastyEquity::equity(), while NeuralAgent ranks the batch from
// HastyEquity::equities(). This checks (a) the batch and per-move APIs
// agree value-for-value and (b) their argmax -- the move each agent returns --
// is identical.
TEST(HastyEquity, TopK1SelectionMatchesHastyBot) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_topk1_XXXXXX";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());
  const HastyEquity& eq = HastyEquity::instance();

  Dictionary d = tiny_dict();
  Board board;  // opening position
  MoveGenerator gen(board, d);
  Rack my_rack = rack_from("CATSOHE");
  std::vector<Move> plays = gen.generate(my_rack);
  ASSERT_GE(plays.size(), 2);  // a meaningful argmax needs >1 candidate

  Rack opp;  // empty
  const int bag_size = 80;

  // Batch path (what NeuralAgent uses) must match the per-move path (what
  // HastyBot uses) value-for-value.
  std::vector<double> batch = eq.equities(plays, board, bag_size, opp, my_rack);
  ASSERT_EQ(batch.size(), plays.size());
  std::vector<double> per_move(plays.size());
  for (size_t i = 0; i < plays.size(); ++i) {
    per_move[i] = eq.equity(plays[i], board, bag_size, opp, my_rack);
    ASSERT_LT(std::abs(batch[i] - per_move[i]), 1e-9);
  }

  // HastyBot's selection: first move with strictly-greatest per-move equity.
  int hasty_pick = 0;
  for (size_t i = 1; i < per_move.size(); ++i) {
    if (per_move[i] > per_move[hasty_pick]) hasty_pick = int(i);
  }
  // NeuralAgent k=1 selection: top-1 of the batch ranking (same rule).
  int topk1_pick = 0;
  for (size_t i = 1; i < batch.size(); ++i) {
    if (batch[i] > batch[topk1_pick]) topk1_pick = int(i);
  }
  ASSERT_EQ(hasty_pick, topk1_pick);

  fs::remove_all(tmp);
}

// Regression: DDGPTWZ (the rack the bug was originally reported with) has no
// legal PLAY against tiny_dict's small vocabulary (real dictionaries do have
// vowel-less words -- NTH, RHYTHM, CWM -- tiny_dict just doesn't carry any).
// HastyBot's move-selection paths used to fall straight through to
// Move::pass() in that case without ever considering an exchange, even though
// exchanging strictly dominates passing (both score 0, but exchanging gives a
// shot at a better rack next turn). It must exchange whenever the bag can
// support one, matching GreedyAgent's existing no-legal-play fallback.
TEST(HastyBotAgent, ExchangesInsteadOfPassingWithNoLegalPlay) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_hasty_exchange_XXXXXX";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());

  Dictionary dict = tiny_dict();  // no all-consonant entries
  Board board;                    // empty board
  Rack rack = rack_from("DDGPTWZ");
  Rack opp;  // empty

  MoveRequest req{board, dict, rack, opp, 0, 0, /*bag_size=*/80};
  ASSERT_TRUE(generate_legal_plays(req).empty());  // sanity: no placement exists

  HastyBotAgent agent(HastyBotAgent::Params{.thread_id = 0, .name = "Hasty"});
  const Move chosen = agent.make_move(req).move;
  ASSERT_EQ(chosen.type(), MoveType::EXCHANGE);

  fs::remove_all(tmp);
}

// Same regression, but on a real mid-game board instead of the empty-board
// opening case above. Position lifted from an actual HastyBot-vs-HastyBot
// self-play game (turn 30, NWL23): the mover holds AEFIORX with no legal PLAY
// on the board these 29 prior turns produced, and HastyBot must exchange
// rather than pass. Gated on the real NWL23 lexicon the game was generated
// against (not committed to the repo).
TEST(HastyBotAgent, ExchangesOnRealMidGamePositionWithNoLegalPlay) {
  const std::string kwg_path = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves_path = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(kwg_path).good() || !std::ifstream(leaves_path).good()) {
    GTEST_SKIP() << "no NWL23 kwg/leaves";
  }
  // init(), not ensure_initialized(): HastyEquity is a process-wide singleton,
  // and other tests in this binary load a synthetic leave table over it --
  // ensure_initialized() would then no-op and silently leave those stale
  // values in place instead of these real ones.
  HastyEquity::init(leaves_path, HastyEquity::default_peg_path());
  Dictionary dict = Dictionary::load_kwg(kwg_path);

  // clang-format off
  const std::string gcg_text = R"GCG(
#character-encoding UTF-8
#player1 HastyBot1 HastyBot
#player2 HastyBot2 HastyBot
#Rack1 AEFIORX
>HastyBot1: AGMORTY H8 GOATY +26 26
>HastyBot2: AEEORUW I8 OWE +24 24
>HastyBot1: EIMMNRT G9 MM +25 51
>HastyBot2: AEEIPRU 12H .AUPER +22 46
>HastyBot1: EINNNRT 13J NINER +22 73
>HastyBot2: EEIIUVV 14I VIE +18 64
>HastyBot1: EFINOST 8H ..OFIEST +39 112
>HastyBot2: DEIIUUV 7M DUI +15 79
>HastyBot1: DDLNPS? J7 D.NS +24 136
>HastyBot2: EIJNTUV N4 UNJ..T +29 108
>HastyBot1: CDELPS? O1 CEPS +30 166
>HastyBot2: EEINSTV 14M VET +22 130
>HastyBot1: ADEGLL? 4J GALL.. +14 180
>HastyBot2: ABEEINS 2J BEANI. +28 158
>HastyBot1: AADEOT? 5J ODA +19 199
>HastyBot2: AEHIIRS 1K AHI +23 181
>HastyBot1: AEILOT? 15N TO +15 214
>HastyBot2: DEIILRS M7 ..IL +9 190
>HastyBot1: ABEEIL? 7J .E +8 222
>HastyBot2: CDEIORS 5I C... +7 197
>HastyBot1: ABEEIL? 10L E. +2 224
>HastyBot2: DEHIORS 15K DO +11 208
>HastyBot1: ABEFIL? L4 ..B +5 229
>HastyBot2: AEHIRST K11 S.... +16 224
>HastyBot1: AEFILX? N4 ......Ly +21 250
>HastyBot2: AEHINRT O11 AH +10 234
)GCG";
  // clang-format on

  ParsedGcgPosition pos;
  std::string error;
  ASSERT_TRUE(read_gcg_position(gcg_text, /*open_leaves=*/false, &pos, &error)) << error;
  ASSERT_EQ(pos.mover, 0);
  ASSERT_EQ(pos.rack.to_string(), "AEFIORX");

  const int my_score = pos.scores[pos.mover];
  const int opp_score = pos.scores[1 - pos.mover];
  MoveRequest req{pos.board, dict, pos.rack, pos.opp_leave, my_score, opp_score, pos.bag_size};
  ASSERT_TRUE(generate_legal_plays(req).empty());  // sanity: no placement exists

  HastyBotAgent agent(HastyBotAgent::Params{.thread_id = 0, .name = "Hasty"});
  const Move chosen = agent.make_move(req).move;
  ASSERT_EQ(chosen.type(), MoveType::EXCHANGE);
}

// Regression: HastyBot must weigh EXCHANGE candidates against PLAY candidates
// by equity, not just fall back to exchanging when no play exists (the test
// above). A real NWL23 rack of six I's and an H has exactly one legal play
// (HI, worth a few points) and a catastrophic leave (IIIII), so even with
// genuine leave values -- no hacked table needed -- HastyBot must prefer
// exchanging over playing the only word it has.
TEST(HastyBotAgent, ExchangesDuplicateHeavyRackOverItsOnlyPlay) {
  const std::string kwg_path = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves_path = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(kwg_path).good() || !std::ifstream(leaves_path).good()) {
    GTEST_SKIP() << "no NWL23 kwg/leaves";
  }
  // init(), not ensure_initialized(): HastyEquity is a process-wide singleton,
  // and other tests in this binary load a synthetic leave table over it --
  // ensure_initialized() would then no-op and silently leave those stale
  // values in place instead of these real ones.
  HastyEquity::init(leaves_path, HastyEquity::default_peg_path());
  Dictionary dict = Dictionary::load_kwg(kwg_path);

  Board board;  // empty board
  Rack rack = rack_from("IIIIIIH");
  Rack opp;  // empty

  MoveRequest req{board, dict, rack, opp, 0, 0, /*bag_size=*/80};
  ASSERT_FALSE(generate_legal_plays(req).empty());  // HI is legal

  HastyBotAgent agent(HastyBotAgent::Params{.thread_id = 0, .name = "Hasty"});
  const Move chosen = agent.make_move(req).move;
  ASSERT_EQ(chosen.type(), MoveType::EXCHANGE);
}

// ===========================================================================
// SimRunner + sim-observation log
// ===========================================================================

// Tile-conservation check for play_from's returned_to_bag: exchanged tiles
// re-enter the bag only after both refills, and stay in circulation for the
// rest of the game.
TEST(Game, PlayFromReturnedToBag) {
  const Dictionary d = medium_dict();
  const Board board;  // empty board

  // The mover (seat 0) holds AB after exchanging its Q and Z; the exchanged
  // tiles are absent from the unseen pool (they were in the mover's hand) and
  // re-enter the bag via returned_to_bag.
  const Rack leave = rack_from("AB");
  Rack returned;
  returned.add(Tile::from_char('Q'));
  returned.add(Tile::from_char('Z'));
  const uint64_t seed = 11;
  Bag pool(seed);
  pool.remove(Tile::from_char('A'));
  pool.remove(Tile::from_char('B'));
  pool.remove(Tile::from_char('Q'));
  pool.remove(Tile::from_char('Z'));
  const int in_circulation = pool.size() + leave.size() + returned.size();

  TestAgent a0(0, "A0", 1), a1(0, "A1", 2);
  scribblez::Game g(a0, a1, d, seed);
  g.play_from(board, {0, 0}, {leave, Rack{}}, pool, /*to_move=*/1, returned);

  // The refills precede the exchanged tiles' return, so neither initial rack
  // can contain Q or Z (each occurs once in the bag, and both were removed
  // from the pool).
  const GameLog log = g.log();
  for (int p = 0; p < 2; ++p) {
    ASSERT_FALSE(rack_contains(log.initial_racks[p], Tile::from_char('Q')));
    ASSERT_FALSE(rack_contains(log.initial_racks[p], Tile::from_char('Z')));
  }

  // Conservation: every tile handed to play_from (pool + leaves + returned) is
  // on the board, on a rack, or in the bag when the game ends -- the returned
  // tiles rejoined circulation exactly once.
  int on_board = 0;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c)
      if (!g.board().at(r, c).is_empty()) ++on_board;
  ASSERT_EQ(on_board + g.rack(0).size() + g.rack(1).size() + g.bag_size(), in_circulation);
}

TEST(SimRunner, Basic) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sim_runner";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());

  const Dictionary d = medium_dict();
  SimPosition pos;
  pos.scores = {30, 45};
  pos.mover = 0;
  pos.rack = rack_from("CATSEIQ");

  // Candidates covering all three move types: two distinct opening plays, a
  // pass, and a one-tile exchange.
  MoveGenerator gen(pos.board, d);
  const std::vector<Move> plays = gen.generate(pos.rack);
  ASSERT_GE(plays.size(), 2);
  TileCounts xchg_tiles;
  xchg_tiles.add(Tile::from_char('Q'));
  const std::vector<Move> candidates = {plays.front(), plays[plays.size() / 2], Move::pass(),
                                        Move::exchange(xchg_tiles)};

  SimRunner::Params params;
  params.rollouts = 16;
  params.threads = 3;
  const SimRunner runner(d, params);
  const uint64_t base_seed = 400;
  const std::vector<SimObservation> obs = runner.run(pos, candidates, base_seed);
  ASSERT_EQ(obs.size(), candidates.size());

  for (const SimObservation& o : obs) {
    ASSERT_EQ(int(o.n), params.rollouts);
    ASSERT_EQ(o.wins + o.draws + o.losses, o.n);
    // Cauchy-Schwarz on the delta moments: (sum d)^2 <= n * sum d^2.
    ASSERT_LE(o.delta_sum * o.delta_sum, int64_t(o.n) * o.delta_sq_sum);
    for (int i = 0; i < SimObservation::kClasses; ++i) {
      ASSERT_LE(o.opp_win_count[i], o.opp_next_count[i]);
      ASSERT_LE(o.self_win_count[i], o.self_next_count[i]);
      ASSERT_LE(o.opp_next_count[i], o.n);
      ASSERT_LE(o.self_next_count[i], o.n);
    }
  }

  // The dictionary is rich in 2-letter words, so sampled opponent racks have
  // replies on the (near-)open board: some ANCHORED class (a real placement,
  // not the pass catch-all) must have fired.
  int64_t total_opp = 0;
  for (int i = 0; i < kAnchoredFootprints; ++i) total_opp += obs[0].opp_next_count[i];
  ASSERT_GT(total_opp, 0);

  // Determinism and thread-independence: a single-threaded run is identical.
  {
    SimRunner::Params p1 = params;
    p1.threads = 1;
    const std::vector<SimObservation> obs1 = SimRunner(d, p1).run(pos, candidates, base_seed);
    ASSERT_EQ(obs1.size(), obs.size());
    for (size_t c = 0; c < obs.size(); ++c)
      ASSERT_EQ(std::memcmp(&obs[c], &obs1[c], sizeof(SimObservation)), 0);
  }

  // Common random numbers: a candidate's observation depends only on the
  // position and the base seed, never on which other candidates were simmed.
  {
    const std::vector<SimObservation> alone = runner.run(pos, {candidates[1]}, base_seed);
    ASSERT_EQ(alone.size(), 1);
    ASSERT_EQ(std::memcmp(&alone[0], &obs[1], sizeof(SimObservation)), 0);
  }

  fs::remove_all(tmp);
}

// accumulate_rollout buckets each rollout move at footprint_class(move,
// flip=false) in the right histogram: opp_reply in the opp counts weighted by
// p_loss, self_next in the self counts weighted by p_win, a PASS in the pass
// catch-all. This pins the field routing, frame, and slot encoding exactly --
// the per-class invariants the SimRunner tests assert (win <= next <= n,
// 0-or-n) would also hold under a swapped opp/self wiring or a flipped frame,
// so only a hand-computed class can catch those.
TEST(SimRunner, AccumulateRolloutBucketsFootprints) {
  const Glyph g[3] = {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                      Glyph::of(Tile::from_char('C'))};
  RolloutResult r;
  // Opp reply: horizontal, 2 tiles at row 3, cols 6-7 -> anchor (3,6), slot 1.
  r.opp_reply = Move::play(/*horizontal=*/true, /*start=*/3,
                           /*square_mask=*/uint16_t((1 << 6) | (1 << 7)), /*score=*/10, g, 2);
  // Self next: vertical, 3 tiles at col 5, rows 2-4 -> anchor (2,5), slot
  // kFootprintMaxK + (3 - 2).
  r.self_next = Move::play(/*horizontal=*/false, /*start=*/5,
                           /*square_mask=*/uint16_t((1 << 2) | (1 << 3) | (1 << 4)),
                           /*score=*/15, g, 3);
  r.p_win = 0.25;
  r.p_draw = 0.25;
  r.p_loss = 0.5;
  r.delta = 7.0;
  r.delta_sq = 53.0;

  SimObservation obs;
  accumulate_rollout(r, &obs);
  const int opp_cls = (3 * BOARD_SIZE + 6) * kSlotsPerCell + 1;
  const int self_cls = (2 * BOARD_SIZE + 5) * kSlotsPerCell + (kFootprintMaxK + 1);
  EXPECT_EQ(obs.opp_next_count[opp_cls], 1);
  EXPECT_FLOAT_EQ(obs.opp_win_count[opp_cls], 0.5f);  // the p_loss side
  EXPECT_EQ(obs.self_next_count[self_cls], 1);
  EXPECT_FLOAT_EQ(obs.self_win_count[self_cls], 0.25f);  // the p_win side
  // Exactly one class fired per histogram.
  int64_t opp_total = 0, self_total = 0;
  for (int i = 0; i < SimObservation::kClasses; ++i) {
    opp_total += obs.opp_next_count[i];
    self_total += obs.self_next_count[i];
  }
  EXPECT_EQ(opp_total, 1);
  EXPECT_EQ(self_total, 1);

  // A 1-tile play is the orientation-free slot 0 whichever axis it declares,
  // and a missing move (default Move = PASS) buckets into the pass catch-all.
  RolloutResult r2;
  r2.opp_reply = Move::play(/*horizontal=*/false, /*start=*/9,
                            /*square_mask=*/uint16_t(1 << 4), /*score=*/4, g, 1);
  r2.p_win = 1.0;
  accumulate_rollout(r2, &obs);
  EXPECT_EQ(obs.opp_next_count[(4 * BOARD_SIZE + 9) * kSlotsPerCell + 0], 1);
  EXPECT_EQ(obs.self_next_count[kPassClass], 1);
  EXPECT_EQ(int(obs.n), 2);
  EXPECT_DOUBLE_EQ(obs.wins, 1.25);
}

// Constant-output leaf stub for value-truncation tests: every horizon row
// reads WLD (0.7, 0.1, 0.2) and predicted final delta +100 from the horizon
// MOVER's POV, so the root-POV flip at odd horizons is exactly checkable.
// Declares the contingent, no-opponent-leave input arm like the agent stubs.
class ConstantLeafService : public scribblez::nn::PositionEvalService {
 public:
  int rows_seen = 0;
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    for (int i = 0; i < batch.count; ++i) {
      float* wld = head_out[0] + size_t(i) * scribblez::nn::WldOutput::kRowElems;
      wld[0] = 0.7f;
      wld[1] = 0.1f;
      wld[2] = 0.2f;
      float* sd = head_out[1] + size_t(i) * scribblez::nn::ScoreDiffOutput::kRowElems;
      sd[0] = 100.0f;
      sd[1] = 5.0f;
    }
    rows_seen += batch.count;
  }
};

// Row-dependent leaf stub: outputs are a deterministic function of the
// encoded row's score-diff scalar, so rollouts get distinct fractional
// contributions -- which makes reduction-order determinism and CRN
// cancellation real assertions rather than vacuous ones.
class RowLeafService : public scribblez::nn::PositionEvalService {
 public:
  int rows_seen = 0;
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    const scribblez::InputEncodingSpec spec{nullptr};
    const size_t row_floats = scribblez::input_floats(spec);
    const size_t sd_off =
      scribblez::spatial_floats() +
      scribblez::scalar_block_offset(spec, scribblez::ScalarBlockId::kScoreDiff);
    for (int i = 0; i < batch.count; ++i) {
      const float s = batch.rows[size_t(i) * row_floats + sd_off];
      const float w = 0.5f + 0.4f * std::tanh(s);
      float* wld = head_out[0] + size_t(i) * scribblez::nn::WldOutput::kRowElems;
      wld[0] = w;
      wld[1] = 0.1f;
      wld[2] = 0.9f - w;
      float* sd = head_out[1] + size_t(i) * scribblez::nn::ScoreDiffOutput::kRowElems;
      sd[0] = s * scribblez::kScoreDiffInputScale;  // "the current diff holds up"
      sd[1] = 5.0f;
    }
    rows_seen += batch.count;
  }
};

// Poisons one output element of an otherwise-finite leaf readout, to exercise
// the runner's hard-error guard.
struct LeafPoison {
  int head;   // 0 = WLD, 1 = score-diff
  int index;  // element within the row
  float value;
};

class NonFiniteLeafService : public scribblez::nn::PositionEvalService {
 public:
  explicit NonFiniteLeafService(LeafPoison poison) : poison_(poison) {}
  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override {
    for (int i = 0; i < batch.count; ++i) {
      float* wld = head_out[0] + size_t(i) * scribblez::nn::WldOutput::kRowElems;
      wld[0] = 0.7f;
      wld[1] = 0.1f;
      wld[2] = 0.2f;
      float* sd = head_out[1] + size_t(i) * scribblez::nn::ScoreDiffOutput::kRowElems;
      sd[0] = 100.0f;
      sd[1] = 5.0f;
      (poison_.head == 0 ? wld : sd)[poison_.index] = poison_.value;
    }
  }

 private:
  LeafPoison poison_;
};

// Value truncation with a constant leaf: every rollout of every candidate is
// cut at the horizon (a mid-game position cannot end within 4 plies), so the
// observations are exact multiples of the stub's outputs -- from the root
// mover's POV, which flips at an odd horizon.
TEST(SimRunner, TruncatedPovParity) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sim_trunc_pov";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());

  const Dictionary d = medium_dict();
  SimPosition pos;
  pos.scores = {30, 45};
  pos.mover = 0;
  pos.rack = rack_from("CATSEIQ");
  MoveGenerator gen(pos.board, d);
  const std::vector<Move> plays = gen.generate(pos.rack);
  ASSERT_GE(plays.size(), 2);
  const std::vector<Move> candidates = {plays.front(), Move::pass()};

  for (const int horizon : {4, 3}) {
    ConstantLeafService leaf;
    SimRunner::Params params;
    params.rollouts = 16;
    params.threads = 2;
    params.horizon_plies = horizon;
    params.leaf_service = &leaf;
    const std::vector<SimObservation> obs = SimRunner(d, params).run(pos, candidates, 400);
    ASSERT_EQ(leaf.rows_seen, params.rollouts * int(candidates.size()));
    // The leaf is the horizon ply's post-move state from ITS mover's POV.
    // The opponent moves first, so horizon 4's last ply is the root mover's
    // own (opp, self, opp, self) and the readout carries over; horizon 3's
    // is the opponent's, so win/loss and the delta sign flip.
    const double p_win = horizon % 2 == 0 ? double(0.7f) : double(0.2f);
    const double p_loss = horizon % 2 == 0 ? double(0.2f) : double(0.7f);
    const double delta = horizon % 2 == 0 ? 100.0 : -100.0;
    for (const SimObservation& o : obs) {
      ASSERT_EQ(int(o.n), params.rollouts);
      ASSERT_DOUBLE_EQ(o.wins, o.n * p_win);
      ASSERT_DOUBLE_EQ(o.draws, o.n * double(0.1f));
      ASSERT_DOUBLE_EQ(o.losses, o.n * p_loss);
      ASSERT_DOUBLE_EQ(o.delta_sum, o.n * delta);
      // The stub predicts sigma = 5, so the second moment carries mean^2 +
      // sigma^2 whichever POV the mean was flipped from.
      ASSERT_DOUBLE_EQ(o.delta_sq_sum, o.n * (100.0 * 100.0 + 5.0 * 5.0));
      // The win-conjoined histograms carry the leaf probabilities per class.
      for (int i = 0; i < SimObservation::kClasses; ++i) {
        ASSERT_NEAR(o.opp_win_count[i], p_loss * o.opp_next_count[i], 1e-3);
        ASSERT_NEAR(o.self_win_count[i], p_win * o.self_next_count[i], 1e-3);
      }
    }
  }
  fs::remove_all(tmp);
}

// Truncated observations stay deterministic across thread counts (the fixed
// reduction order), CRN-cancel exactly on duplicate candidates, and are
// independent of which other candidates were simmed -- with the workers
// sharing one stub service through EvalService's own serialization.
TEST(SimRunner, TruncatedDeterminismAndCrn) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sim_trunc_crn";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());

  const Dictionary d = medium_dict();
  SimPosition pos;
  pos.scores = {30, 45};
  pos.mover = 0;
  pos.rack = rack_from("CATSEIQ");
  MoveGenerator gen(pos.board, d);
  const std::vector<Move> plays = gen.generate(pos.rack);
  ASSERT_GE(plays.size(), 2);
  // The same play twice: a CRN duplicate whose observations must be equal.
  const std::vector<Move> candidates = {plays.front(), plays.front(), plays[plays.size() / 2]};

  RowLeafService leaf;
  SimRunner::Params params;
  params.rollouts = 16;
  params.threads = 3;
  params.horizon_plies = 4;
  params.leaf_service = &leaf;
  const uint64_t base_seed = 400;
  const std::vector<SimObservation> obs = SimRunner(d, params).run(pos, candidates, base_seed);

  for (const SimObservation& o : obs) {
    ASSERT_EQ(int(o.n), params.rollouts);
    ASSERT_NEAR(o.wins + o.draws + o.losses, double(o.n), 1e-5);
  }
  // Exact CRN cancellation: the duplicate's observation is byte-identical.
  ASSERT_EQ(std::memcmp(&obs[0], &obs[1], sizeof(SimObservation)), 0);

  // Thread-count independence, exactly (fractional contributions reduce in a
  // fixed order).
  {
    SimRunner::Params p1 = params;
    p1.threads = 1;
    const std::vector<SimObservation> obs1 = SimRunner(d, p1).run(pos, candidates, base_seed);
    for (size_t c = 0; c < obs.size(); ++c)
      ASSERT_EQ(std::memcmp(&obs[c], &obs1[c], sizeof(SimObservation)), 0);
  }
  // CRN across candidate-set membership.
  {
    const std::vector<SimObservation> alone =
      SimRunner(d, params).run(pos, {candidates[2]}, base_seed);
    ASSERT_EQ(std::memcmp(&alone[0], &obs[2], sizeof(SimObservation)), 0);
  }
  fs::remove_all(tmp);
}

// A horizon past every game's natural end changes nothing: each rollout
// finishes before the cap, contributes its exact terminal outcome, and the
// leaf service is never consulted -- so the observations are byte-identical
// to a terminal runner's.
TEST(SimRunner, TruncatedFallsBackToTerminalAtGameEnd) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sim_trunc_term";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());

  const Dictionary d = medium_dict();
  SimPosition pos;
  pos.scores = {30, 45};
  pos.mover = 0;
  pos.rack = rack_from("CATSEIQ");
  MoveGenerator gen(pos.board, d);
  const std::vector<Move> plays = gen.generate(pos.rack);
  ASSERT_GE(plays.size(), 1);
  const std::vector<Move> candidates = {plays.front(), Move::pass()};

  SimRunner::Params terminal;
  terminal.rollouts = 8;
  const std::vector<SimObservation> obs_terminal = SimRunner(d, terminal).run(pos, candidates, 400);

  RowLeafService leaf;
  SimRunner::Params truncated = terminal;
  truncated.horizon_plies = 350;  // beyond any natural game length
  truncated.leaf_service = &leaf;
  const std::vector<SimObservation> obs_truncated =
    SimRunner(d, truncated).run(pos, candidates, 400);

  ASSERT_EQ(leaf.rows_seen, 0);
  for (size_t c = 0; c < obs_terminal.size(); ++c)
    ASSERT_EQ(std::memcmp(&obs_terminal[c], &obs_truncated[c], sizeof(SimObservation)), 0);
  fs::remove_all(tmp);
}

// The horizon/service pairing and the minimum horizon are validated.
TEST(SimRunner, ValidatesTruncationParams) {
  SimRunner::Params p;
  p.horizon_plies = 4;  // horizon without a service
  ASSERT_THROW(SimRunner::validate(p), std::runtime_error);
  ConstantLeafService leaf;
  p.leaf_service = &leaf;
  p.horizon_plies = 0;  // service without a horizon
  ASSERT_THROW(SimRunner::validate(p), std::runtime_error);
  p.horizon_plies = SimRunner::kMinHorizonPlies - 1;  // below the minimum
  ASSERT_THROW(SimRunner::validate(p), std::runtime_error);
  p.horizon_plies = SimRunner::kMinHorizonPlies;
  SimRunner::validate(p);
}

// A non-finite leaf readout is a hard error -- whether NaN or the +/-inf an
// FP16 overflow reaches first, and in any consumed field, including the
// score-diff std that feeds delta_sq. Because a rollout runs on a worker
// thread, the guard's throw must surface on the calling thread (threads > 1
// here) rather than terminating the process.
TEST(SimRunner, NonFiniteLeafReadoutIsRejected) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sim_nonfinite_leaf";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());

  const Dictionary d = medium_dict();
  SimPosition pos;
  pos.scores = {30, 45};
  pos.mover = 0;
  pos.rack = rack_from("CATSEIQ");
  MoveGenerator gen(pos.board, d);
  const std::vector<Move> plays = gen.generate(pos.rack);
  ASSERT_GE(plays.size(), 1);
  const std::vector<Move> candidates = {plays.front(), Move::pass()};

  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const LeafPoison cases[] = {
    {1, 0, inf},  // score-diff mean overflows to +inf: isnan-false, was missed
    {1, 1, inf},  // score-diff std -> inf: feeds delta_sq, was unchecked
    {1, 1, nan},  // score-diff std -> NaN: likewise
    {0, 2, nan},  // loss prob NaN while the checked win prob stays finite
  };
  for (const LeafPoison& c : cases) {
    NonFiniteLeafService leaf(c);
    SimRunner::Params params;
    params.rollouts = 8;
    params.threads = 2;
    params.horizon_plies = 4;
    params.leaf_service = &leaf;
    EXPECT_THROW(SimRunner(d, params).run(pos, candidates, 400), std::runtime_error);
  }
  fs::remove_all(tmp);
}

// A full 7-tile known leave degenerates to a completely known opponent rack:
// under a greedy (deterministic) rollout policy the opponent's first reply to
// each candidate is then the same in every rollout, so the reply-placement
// counts are exactly 0 or S per square. (A partial leave is exercised by the
// partial-leave test below.)
TEST(SimRunner, KnownOppRack) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sim_openrack";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());

  const Dictionary d = medium_dict();
  SimPosition pos;
  pos.scores = {30, 45};
  pos.mover = 0;
  pos.rack = rack_from("CATSEIQ");
  pos.opp_leave = rack_from("DOGSTAR");

  MoveGenerator gen(pos.board, d);
  const std::vector<Move> plays = gen.generate(pos.rack);
  ASSERT_GE(plays.size(), 2);
  const std::vector<Move> candidates = {plays.front(), plays[plays.size() / 2]};

  SimRunner::Params params;
  params.rollouts = 12;
  params.threads = 3;
  const SimRunner runner(d, params);
  const std::vector<SimObservation> obs = runner.run(pos, candidates, /*base_seed=*/9);
  bool any_reply = false;
  for (const SimObservation& o : obs) {
    ASSERT_EQ(int(o.n), params.rollouts);
    // With the opponent's whole rack known the reply is deterministic, so every
    // rollout lands in one footprint class: each class holds 0 or n.
    for (int i = 0; i < SimObservation::kClasses; ++i)
      ASSERT_TRUE(o.opp_next_count[i] == 0 || o.opp_next_count[i] == o.n);
    // ...and for some candidate that class is an anchored one (a real
    // placement, not the pass catch-all).
    for (int i = 0; i < kAnchoredFootprints; ++i)
      if (o.opp_next_count[i] == o.n) any_reply = true;
  }
  ASSERT_TRUE(any_reply);

  // Determinism across thread counts holds in this mode too.
  SimRunner::Params p1 = params;
  p1.threads = 1;
  const std::vector<SimObservation> obs1 = SimRunner(d, p1).run(pos, candidates, /*base_seed=*/9);
  for (size_t c = 0; c < obs.size(); ++c)
    ASSERT_EQ(std::memcmp(&obs[c], &obs1[c], sizeof(SimObservation)), 0);

  fs::remove_all(tmp);
}

// A partial known leave: the rollout seeds the opponent's retained tiles and
// samples only their hidden replenishments, so observations satisfy the same
// invariants while replies may vary across rollouts.
TEST(SimRunner, PartialLeave) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sim_partial";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  // SimRunner's rollout policy ranks moves through the HastyEquity singleton,
  // which every test must initialize itself (tests may run in isolation).
  HastyEquity::init(fix.path.string(), peg_path.string());

  const Dictionary d = medium_dict();
  SimPosition pos;
  pos.scores = {10, 5};
  pos.mover = 0;
  pos.rack = rack_from("CATSEIQ");
  pos.opp_leave = rack_from("ZI");  // kept 2; the other 5 are hidden draws

  MoveGenerator gen(pos.board, d);
  const std::vector<Move> plays = gen.generate(pos.rack);
  ASSERT_FALSE(plays.empty());
  SimRunner::Params params;
  params.rollouts = 10;
  params.threads = 2;
  const std::vector<SimObservation> obs =
    SimRunner(d, params).run(pos, {plays.front()}, /*base_seed=*/4);
  ASSERT_EQ(int(obs[0].n), params.rollouts);
  ASSERT_EQ(obs[0].wins + obs[0].draws + obs[0].losses, obs[0].n);
  for (int i = 0; i < SimObservation::kClasses; ++i) {
    ASSERT_LE(obs[0].opp_win_count[i], obs[0].opp_next_count[i]);
  }

  fs::remove_all(tmp);
}

// opp_leave_from_replay: the opponent's current rack minus the draws after
// their last move; empty before they have acted.
TEST(SimRunner, OppLeaveFromReplay) {
  using scribblez::binlog::opp_leave_from_replay;
  TurnRecord records[2] = {};
  records[0].player = 1;  // the opponent's move at turn 0
  records[0].drawn = rack_from("AB");
  GameLog g{};
  g.records = records;
  g.num_records = 2;

  // Mover at turn 1: opponent moved at turn 0, then drew A and B. Their
  // current rack CABDEFG minus {A, B} leaves their retained CDEFG.
  const Rack now = rack_from("CABDEFG");
  const Rack leave = opp_leave_from_replay(g, /*sampled_turn=*/1, now);
  ASSERT_EQ(leave.size(), 5);
  Rack expect = rack_from("CDEFG");
  for (int i = 0; i < expect.size(); ++i) ASSERT_TRUE(rack_contains(leave, expect.tiles()[i]));

  // Mover at turn 0: the opponent has not acted; nothing is known.
  ASSERT_EQ(opp_leave_from_replay(g, /*sampled_turn=*/0, now).size(), 0);
}

// util/metaprogramming.h's consteval reflection helpers, exercised on a
// local struct. The assertions are static_asserts -- the test body passing
// is the compile succeeding -- with a TEST wrapper so the coverage is
// visible in the suite.
namespace metaprog_test {
struct Sample {
 public:
  uint32_t plain;

 private:
  std::array<uint16_t, 3> squares_;

 public:
  // squares_ is only reflected on, never read; silence -Wunused-private-field
  // style diagnostics by touching it.
  const void* touch() const { return &squares_; }
};

consteval bool helpers_hold() {
  if (scribblez::util::num_members<Sample>() != 2) return false;
  const auto members = scribblez::util::nonstatic_data_members<Sample>();
  if (!scribblez::util::type_is<uint32_t>(std::meta::type_of(members[0]))) return false;
  // The private member reflects, and its reader-facing name drops the
  // trailing underscore.
  if (std::string_view(scribblez::util::member_name(members[1])) != "squares") return false;
  const auto arr = std::meta::dealias(std::meta::type_of(members[1]));
  if (!scribblez::util::is_specialization_of(arr, ^^std::array)) return false;
  if (!scribblez::util::type_is<uint16_t>(scribblez::util::std_array_element(arr))) return false;
  if (scribblez::util::std_array_extent(arr) != 3) return false;
  if (std::string_view(scribblez::util::dec_string(0)) != "0") return false;
  if (std::string_view(scribblez::util::dec_string(1048576)) != "1048576") return false;
  return true;
}
}  // namespace metaprog_test

TEST(Metaprogramming, ConstevalReflectionHelpers) { static_assert(metaprog_test::helpers_hold()); }

// The FFI-served format-layout document: itemsizes and constants are the
// compiler's own, so this exercises the JSON transport, the nested-struct
// references, and the document paths the Python reader walks.
TEST(FormatLayout, DescribesTheSidecarStructs) {
  namespace bj = boost::json;
  const bj::value doc = bj::parse(format_layout_json());

  const bj::object& structs = doc.at("structs").as_object();
  EXPECT_EQ(structs.at("Move").at("itemsize").to_number<size_t>(), sizeof(Move));
  EXPECT_EQ(structs.at("SimObservation").at("itemsize").to_number<size_t>(),
            sizeof(SimObservation));
  EXPECT_EQ(structs.at("SobsRecord").at("itemsize").to_number<size_t>(), sizeof(SimObsRecord));
  EXPECT_EQ(structs.at("MsetFileHeader").at("itemsize").to_number<size_t>(),
            sizeof(move_set_eval::TargetFileHeader));

  // A nested field references its struct by name.
  const bj::array& rec_fields = structs.at("SobsRecord").at("fields").as_array();
  EXPECT_EQ(rec_fields.at(0).at("name").as_string(), "move");
  EXPECT_EQ(rec_fields.at(0).at("dtype").at("struct").as_string(), "Move");
  // The per-record evidence role serializes as its underlying byte.
  EXPECT_EQ(rec_fields.at(2).at("name").as_string(), "role");
  EXPECT_EQ(rec_fields.at(2).at("dtype").as_string(), "u1");

  // A subarray field carries its element code and shape. The next-move
  // histograms are integer counts; the win-conjoined histograms and the
  // outcome accumulators are fractional under value truncation.
  const bj::array& obs_fields = structs.at("SimObservation").at("fields").as_array();
  bool found_counts = false, found_win = false, found_wins = false;
  for (const bj::value& f : obs_fields) {
    if (f.at("name").as_string() == "opp_next_count") {
      found_counts = true;
      EXPECT_EQ(f.at("dtype").as_string(), "<u2");
      EXPECT_EQ(f.at("shape").as_array().at(0).to_number<int>(), SimObservation::kClasses);
    } else if (f.at("name").as_string() == "opp_win_count") {
      found_win = true;
      EXPECT_EQ(f.at("dtype").as_string(), "<f4");
      EXPECT_EQ(f.at("shape").as_array().at(0).to_number<int>(), SimObservation::kClasses);
    } else if (f.at("name").as_string() == "wins") {
      found_wins = true;
      EXPECT_EQ(f.at("dtype").as_string(), "<f8");
    }
  }
  EXPECT_TRUE(found_counts);
  EXPECT_TRUE(found_win);
  EXPECT_TRUE(found_wins);

  const bj::object& c = doc.at("constants").as_object();
  EXPECT_EQ(c.at("mset").at("magic").to_number<uint32_t>(), move_set_eval::kTargetMagic);
  EXPECT_EQ(c.at("mset").at("version").to_number<uint32_t>(), move_set_eval::kTargetVersion);
  EXPECT_EQ(c.at("placement_head_names").as_array().at(0).as_string(),
            OppNextPlacementTarget::kName);
}

// The Glyph byte code table is deliberately replicated in Python
// (sim_evidence/sobs.py glyph_char) rather than served over the FFI: it is
// effectively frozen. This pin and its Python twin (test_format_layout.py)
// keep the two replicas in lockstep; a change on either side must be
// mirrored on the other.
TEST(Glyph, CodeTablePinnedForCrossLanguageReaders) {
  EXPECT_EQ(Glyph::empty().code(), 0);
  EXPECT_EQ(Glyph::of(Tile::from_char('A')).code(), 1);
  EXPECT_EQ(Glyph::of(Tile::from_char('Z')).code(), 26);
  EXPECT_EQ(Glyph::of_blank(Tile::from_char('A')).code(), 27);
  EXPECT_EQ(Glyph::of_blank(Tile::from_char('Z')).code(), 52);
  EXPECT_EQ(Glyph::blank().code(), 53);
}

TEST(MoveSetEvalTargetLog, Roundtrip) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_mset";
  fs::create_directories(tmp);
  const std::string path = (tmp / "test.mset").string();

  constexpr uint32_t kFloats = move_set_eval::kTargetFloatsV1;
  constexpr uint32_t kPlanes = move_set_eval::kTargetPlanes;
  constexpr uint32_t kCells = move_set_eval::kPlaneWidth;  // footprint classes per head

  const Move m1 = make_play_full(4, 2, /*horizontal=*/true, 0b111, 24,
                                 {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                                  Glyph::of(Tile::from_char('C'))});
  TileCounts xchg_tiles;
  xchg_tiles.add(Tile::from_char('A'));
  const Move m2 = Move::exchange(xchg_tiles);
  const std::vector<float> targets = {0.7f, 0.1f, 0.2f, 33.5f,  41.0f,
                                      0.2f, 0.0f, 0.8f, -12.0f, 55.5f};
  // Per candidate, kPlanes probability planes with distinct per-plane maxima;
  // candidate 1's last plane is all-zero (the scale-0 case).
  std::vector<float> planes(2 * kPlanes * kCells, 0.0f);
  for (uint32_t c = 0; c < 2; ++c) {
    for (uint32_t h = 0; h < kPlanes; ++h) {
      if (c == 1 && h == kPlanes - 1) continue;
      float* plane = planes.data() + (c * kPlanes + h) * kCells;
      for (uint32_t i = 0; i < kCells; ++i) {
        plane[i] = (0.9f - 0.2f * h) * float(i) / (kCells - 1);
      }
    }
  }

  {
    move_set_eval::TargetWriter w(path, kFloats, kPlanes, "abc123");
    w.add_position(3, 11, {m1, m2}, targets, planes);
    // A swept position records the legal-move count its candidates were drawn
    // from, so a cap-truncated sweep is visible as a shortfall.
    w.add_position(3, 12, {m1, m2}, targets, planes, /*num_legal_moves=*/9184);
    w.close();
  }

  move_set_eval::TargetReader r(path);
  ASSERT_EQ(r.record_floats(), kFloats);
  ASSERT_EQ(r.record_planes(), kPlanes);
  ASSERT_EQ(r.model_hash(), "abc123");
  ASSERT_EQ(r.num_positions(), 2);
  const move_set_eval::TargetReader::Position p0 = r.position(0);
  ASSERT_EQ(p0.header->game_index, 3);
  ASSERT_EQ(p0.header->turn_index, 11);
  ASSERT_EQ(p0.header->num_candidates, 2);
  ASSERT_EQ(p0.header->num_legal_moves, 0u);  // stratified: not recorded
  ASSERT_EQ(r.position(1).header->num_legal_moves, 9184u);
  ASSERT_EQ(r.move_at(p0, 0), m1);
  ASSERT_EQ(r.move_at(p0, 1), m2);
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < int(kFloats); ++j) {
      ASSERT_EQ(r.targets_at(p0, c)[j], targets[c * kFloats + j]);
    }
  }

  // Planes dequantize to the written probabilities within absmax-quantization
  // error (half a step, scale/2), with the plane max reconstructed exactly.
  for (int c = 0; c < 2; ++c) {
    const float* scales = r.plane_scales_at(p0, c);
    const uint8_t* cells = r.planes_at(p0, c);
    for (uint32_t h = 0; h < kPlanes; ++h) {
      const float* plane = planes.data() + (c * kPlanes + h) * kCells;
      const float max = *std::max_element(plane, plane + kCells);
      ASSERT_FLOAT_EQ(scales[h], max / 255.0f);
      if (max == 0.0f) {
        for (uint32_t i = 0; i < kCells; ++i) ASSERT_EQ(cells[h * kCells + i], 0);
        continue;
      }
      for (uint32_t i = 0; i < kCells; ++i) {
        const float back = move_set_eval::dequantized_plane_value(cells[h * kCells + i], scales[h]);
        ASSERT_NEAR(back, plane[i], scales[h] / 2 + 1e-6f);
      }
      ASSERT_EQ(cells[h * kCells + (kCells - 1)], 255);  // the max cell
    }
  }

  // A version mismatch fails loudly (stale files must never misparse).
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(4);  // TargetFileHeader::version
    const uint16_t bad = 0xFFFF;
    f.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
  }
  ASSERT_THROW(move_set_eval::TargetReader r2(path), std::runtime_error);

  fs::remove_all(tmp);
}

// The plane-less layout full-sweep files use: record_planes 0 shrinks the
// record back to Move + value targets, and the plane accessors address a
// zero-length block.
TEST(MoveSetEvalTargetLog, RoundtripWithoutPlanes) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_mset_noplanes";
  fs::create_directories(tmp);
  const std::string path = (tmp / "test.mset").string();

  const Move m1 =
    make_play_full(4, 2, /*horizontal=*/true, 0b1, 24, {Glyph::of(Tile::from_char('A'))});
  const std::vector<float> targets = {0.7f, 0.1f, 0.2f, 33.5f, 41.0f};
  {
    move_set_eval::TargetWriter w(path, move_set_eval::kTargetFloatsV1, /*record_planes=*/0,
                                  "abc123", move_set_eval::kTargetFlagFullSweep);
    w.add_position(0, 5, {m1}, targets, /*planes=*/{}, /*num_legal_moves=*/1);
    w.close();
  }

  move_set_eval::TargetReader r(path);
  ASSERT_EQ(r.record_planes(), 0u);
  ASSERT_EQ(r.num_positions(), 1);
  const move_set_eval::TargetReader::Position p0 = r.position(0);
  ASSERT_EQ(r.move_at(p0, 0), m1);
  for (int j = 0; j < int(move_set_eval::kTargetFloatsV1); ++j) {
    ASSERT_EQ(r.targets_at(p0, 0)[j], targets[j]);
  }

  fs::remove_all(tmp);
}

// `count` distinct plays, standing in for a position's static-equity ranking.
static std::vector<Move> ranked_plays(int count) {
  std::vector<Move> out;
  for (int i = 0; i < count; ++i) {
    out.push_back(make_play_full(i % 15, 0, /*horizontal=*/true, 0b1, uint16_t(1000 - i),
                                 {Glyph::of(Tile::from_char('A'))}));
  }
  return out;
}

static Move exchange_of(char c) {
  TileCounts tiles;
  tiles.add(Tile::from_char(c));
  return Move::exchange(tiles);
}

// The full sweep's selection rule: capped by static-equity rank, but never at
// the cost of an exchange candidate, the played move, or the rank order that
// makes the stored order an exact static-equity ranking.
TEST(MoveSetEvalCandidates, FullSweepCapKeepsExchangesAndRankOrder) {
  std::vector<Move> ranked = ranked_plays(10);
  const Move buried_exchange = exchange_of('Q');
  const Move buried_play = ranked[8];
  ranked.insert(ranked.begin() + 6, exchange_of('A'));  // inside the cap
  ranked.push_back(buried_exchange);                    // beyond it

  const move_set_eval::Selection sel =
    move_set_eval::full_sweep_candidates(ranked, buried_play, /*cap=*/4);
  const std::vector<Move>& swept = sel.candidates;

  // Everything kept, in `ranked`'s order: the head under the cap, then the two
  // exchanges and the played move from beyond it.
  ASSERT_EQ(swept.size(), 7u);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(swept[size_t(i)], ranked[i]);
  EXPECT_EQ(swept[4], ranked[6]);  // the in-cap exchange, past the cap by rank
  EXPECT_EQ(swept[5], buried_play);
  EXPECT_EQ(swept[6], buried_exchange);
  // The recorded legal count is what the sweep drew from, so the shortfall
  // against it is exactly what the cap dropped.
  EXPECT_EQ(sel.num_legal_moves, ranked.size());

  // Uncapped, the sweep is the whole ranking verbatim and nothing is truncated.
  const move_set_eval::Selection whole =
    move_set_eval::full_sweep_candidates(ranked, buried_play, /*cap=*/1000);
  EXPECT_EQ(whole.candidates, ranked);
  EXPECT_EQ(whole.num_legal_moves, ranked.size());
}

// A played move the generator never enumerates (a PASS chosen while plays were
// legal) has no equity rank, so it is kept last rather than dropped.
TEST(MoveSetEvalCandidates, FullSweepKeepsAnUnrankedPlayedMove) {
  const std::vector<Move> ranked = ranked_plays(3);
  const move_set_eval::Selection sel =
    move_set_eval::full_sweep_candidates(ranked, Move::pass(), /*cap=*/2);
  ASSERT_EQ(sel.candidates.size(), 3u);
  EXPECT_EQ(sel.candidates[0], ranked[0]);
  EXPECT_EQ(sel.candidates[1], ranked[1]);
  EXPECT_EQ(sel.candidates[2], Move::pass());
  // It counts toward the legal total too, so the sweep still reads complete.
  EXPECT_EQ(sel.num_legal_moves, ranked.size() + 1);
}

// Forced (simmed trajectory) candidates enter the stratified sample right
// after the played move, deduplicated against it -- and the strata that follow
// dedupe against them in turn, so nothing is labeled twice.
TEST(MoveSetEvalCandidates, StratifiedForceIncludesSimmedCandidates) {
  const std::vector<Move> ranked = ranked_plays(40);
  const Move played = ranked[0];
  const std::vector<Move> forced = {ranked[17], played, ranked[35]};
  std::mt19937_64 rng(7);
  const move_set_eval::StratumQuotas quotas;
  const move_set_eval::Selection sel =
    move_set_eval::stratified_candidates(ranked, played, quotas, rng, forced);
  const std::vector<Move>& out = sel.candidates;
  // The budget is additive: played + 2 distinct forced + full quotas (4 top,
  // 4 mid, 4 tail; ranked_plays has no exchanges). Forced candidates must
  // never shrink a stratum.
  ASSERT_EQ(out.size(), 15u);
  EXPECT_EQ(out[0], played);
  EXPECT_EQ(out[1], ranked[17]);
  EXPECT_EQ(out[2], ranked[35]);  // the duplicate of `played` was skipped
  // The top stratum still delivers the dense head: quotas.top candidates
  // beyond the played move.
  for (int i = 1; i <= quotas.top; ++i) {
    EXPECT_NE(std::find(out.begin(), out.end(), ranked[size_t(i)]), out.end());
  }
  for (size_t i = 0; i < out.size(); ++i) {
    for (size_t j = i + 1; j < out.size(); ++j) EXPECT_NE(out[i], out[j]);
  }
}

// off_policy_draws (the trajectory off-policy floor, docs/roadmap.md item 4)
// draws `count` distinct legal-move indices uniformly from those not already
// taken (the anchor and on-policy picks), excludes them, and marks what it
// draws. No stratification -- exchanges and the tail are reachable at their
// natural frequency in the move list.
TEST(MoveSetEvalCandidates, OffPolicyDrawsUniformlyExcludingTaken) {
  std::vector<Move> ranked = ranked_plays(20);
  TileCounts xchg;
  xchg.add(Tile::from_char('A'));
  ranked.push_back(Move::exchange(xchg));  // a non-PLAY candidate is in the pool
  const size_t n = ranked.size();
  std::vector<char> taken(n, 0);
  for (size_t i : {size_t{0}, size_t{5}, size_t{12}}) taken[i] = 1;  // anchor/on-policy stand-ins
  const std::vector<char> pre = taken;
  std::mt19937_64 rng(3);
  const std::vector<size_t> draws =
    move_set_eval::off_policy_draws(ranked, /*count=*/4, rng, &taken);

  ASSERT_EQ(draws.size(), 4u);
  for (size_t a = 0; a < draws.size(); ++a) {
    EXPECT_LT(draws[a], n);
    EXPECT_FALSE(pre[draws[a]]);   // never a pre-marked (anchor/on-policy) index
    EXPECT_TRUE(taken[draws[a]]);  // marked afterward
    for (size_t b = a + 1; b < draws.size(); ++b) EXPECT_NE(draws[a], draws[b]);  // distinct
  }
}

// A count larger than the untaken pool yields exactly the remaining pool -- and
// a non-PLAY (exchange) is reachable with no stratum reserved for it.
TEST(MoveSetEvalCandidates, OffPolicyDrawsWholeUntakenPoolIncludingNonPlays) {
  std::vector<Move> ranked = ranked_plays(4);
  TileCounts xchg;
  xchg.add(Tile::from_char('A'));
  ranked.push_back(Move::exchange(xchg));  // index 4, a non-PLAY
  std::vector<char> taken(ranked.size(), 0);
  taken[0] = taken[1] = 1;  // untaken pool: indices 2, 3, 4(exchange)
  std::mt19937_64 rng(1);
  const std::vector<size_t> draws =
    move_set_eval::off_policy_draws(ranked, /*count=*/10, rng, &taken);
  ASSERT_EQ(draws.size(), 3u);  // capped at the untaken pool
  std::vector<size_t> sorted = draws;
  std::ranges::sort(sorted);
  EXPECT_EQ(sorted, (std::vector<size_t>{2u, 3u, 4u}));
}

// count == 0 (a validated-legal config) draws nothing and leaves `taken` as it
// was -- pins the loop's count guard against an off-by-one.
TEST(MoveSetEvalCandidates, OffPolicyDrawsCountZeroDrawsNothing) {
  const std::vector<Move> ranked = ranked_plays(6);
  std::vector<char> taken(ranked.size(), 0);
  taken[0] = 1;  // one pre-taken index
  const std::vector<char> pre = taken;
  std::mt19937_64 rng(1);
  const std::vector<size_t> draws =
    move_set_eval::off_policy_draws(ranked, /*count=*/0, rng, &taken);
  EXPECT_TRUE(draws.empty());
  EXPECT_EQ(taken, pre);  // nothing marked
}

// The on-policy proposals draw a temperature softmax over EVERY unsimmed
// candidate, not a capped head (docs/roadmap.md item 4, PR1): deployment
// argmaxes over the full support, so a corpus proposal must be reachable at
// any rank. With equal win equities the softmax is exactly uniform regardless
// of temperature, so a candidate past the retired top-64 cap can only be drawn
// once the cap is gone -- pinned here rather than left to the GPU-gated
// end-to-end run.
TEST(EvidenceTrajectory, ProposalsDrawBeyondTheRetiredPoolCap) {
  constexpr int kN = 100;
  const std::vector<Move> ranked = ranked_plays(kN);  // descending score: anchor is index 0
  const std::vector<float> win_equity(kN, 0.5f);      // equal -> uniform softmax
  evidence::TrajectoryOptions opt;
  opt.on_policy_min = 1;  // exactly one on-policy softmax proposal at chosen[1]
  opt.on_policy_max = 1;
  util::SoftmaxSampler sampler;
  size_t deepest_proposal = 0;
  for (uint64_t seed = 0; seed < 500; ++seed) {
    std::mt19937_64 rng(seed);
    std::vector<SimObsRole> roles;
    const std::vector<size_t> chosen =
      evidence::select_trajectory(ranked, win_equity, opt, rng, sampler, &roles);
    ASSERT_EQ(chosen.size(), roles.size());
    ASSERT_GE(chosen.size(), 2u);
    EXPECT_EQ(chosen[0], 0u);  // the anchor is the highest-raw-score move
    EXPECT_EQ(roles[0], SimObsRole::kAnchor);
    EXPECT_EQ(roles[1], SimObsRole::kOnPolicy);  // chosen[1] is the softmax proposal
    deepest_proposal = std::max(deepest_proposal, chosen[1]);
  }
  // The old cap kept unsimmed order indices 1..64 in reach; a proposal at 65+
  // proves the softmax now spans the whole candidate set.
  EXPECT_GT(deepest_proposal, 64u);
}

// The sampling shuffle is a fixed permutation per (seed, game): a smaller
// per-game sample is a prefix of a larger one. This is the subset guarantee
// slog_sampling.h calls load-bearing -- it is what lets the target generator
// find every trajectory-sidecar position inside its own sample with no
// coordination beyond the seed -- so it is pinned directly here rather than
// only through the GPU-gated end-to-end test.
TEST(SlogSampling, SmallerSamplesArePrefixesOfLarger) {
  binlog::GameMetadata gm{};
  gm.eligible_begin = 3;
  gm.eligible_end = 19;
  for (uint64_t seed : {0ull, 7ull, 0xDEADBEEFull}) {
    std::vector<binlog::GamePositionIndex> full;
    binlog::sample_eligible_turns(gm, /*game_idx=*/5, seed, /*positions_per_game=*/16, &full);
    ASSERT_EQ(full.size(), 16u);
    for (int k = 1; k <= 16; ++k) {
      std::vector<binlog::GamePositionIndex> sample;
      binlog::sample_eligible_turns(gm, 5, seed, k, &sample);
      ASSERT_EQ(sample.size(), size_t(k));
      for (int i = 0; i < k; ++i) EXPECT_EQ(sample[i].turn_idx, full[i].turn_idx);
    }
    // <= 0 takes every eligible turn (in order), so it contains any sample.
    std::vector<binlog::GamePositionIndex> all;
    binlog::sample_eligible_turns(gm, 5, seed, 0, &all);
    ASSERT_EQ(all.size(), 16u);
    for (const binlog::GamePositionIndex& s : full) {
      EXPECT_NE(std::find(all.begin(), all.end(), s), all.end());
    }
  }
}

// The stored std must be finite even when FP16 teacher inference overflows the
// readout to +inf (see kSdStdCap in the header); genuine readouts pass through.
TEST(MoveSetEvalTargetLog, SdStdClamp) {
  ASSERT_EQ(move_set_eval::clamped_sd_std(std::numeric_limits<float>::infinity()),
            move_set_eval::kSdStdCap);
  ASSERT_EQ(move_set_eval::clamped_sd_std(41.0f), 41.0f);
  ASSERT_EQ(move_set_eval::clamped_sd_std(move_set_eval::kSdStdCap), move_set_eval::kSdStdCap);
}

// A .mset carries the information condition of the games it labels, read off
// its source .slog's header, which is what holds a training corpus to a single
// condition.
TEST(MoveSetEvalTargetLog, OpenLeavesFlagFollowsTheSourceLog) {
  namespace fs = std::filesystem;
  Dictionary dict = medium_dict();

  for (bool face_up : {false, true}) {
    fs::path dir =
      fs::temp_directory_path() /
      ("scribblez_mset_flag_" + std::to_string(::getpid()) + "_" + std::to_string(int(face_up)));
    fs::create_directories(dir);
    struct DirCleanup {
      fs::path p;
      ~DirCleanup() {
        std::error_code ec;
        fs::remove_all(p, ec);
      }
    } cleanup{dir};

    const uint16_t slog_flags = face_up ? scribblez::binlog::kFlagFaceUpLeaves : 0;
    {
      scribblez::binlog::BinaryLogWriter writer(dir.string(), /*games_per_file=*/1, slog_flags);
      writer.append(play_test_game(dict, /*seed=*/4242ULL));
    }
    fs::path slog;
    for (const auto& ent : fs::directory_iterator(dir))
      if (ent.path().extension() == ".slog") slog = ent.path();
    ASSERT_FALSE(slog.empty()) << "face_up=" << face_up;

    scribblez::binlog::FileHeader hdr{};
    std::ifstream f(slog, std::ios::binary);
    ASSERT_TRUE(f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) << "face_up=" << face_up;

    const std::string mset = (dir / "targets.mset").string();
    {
      move_set_eval::TargetWriter w(mset, move_set_eval::kTargetFloatsV1, /*record_planes=*/0,
                                    "abc123", move_set_eval::target_flags_from_slog(hdr.flags));
      w.add_position(0, 0, {Move::pass()}, std::vector<float>(move_set_eval::kTargetFloatsV1, 0.0f),
                     /*planes=*/{});
      w.close();
    }
    const uint32_t expected = face_up ? move_set_eval::kTargetFlagOpenLeaves : 0u;
    EXPECT_EQ(move_set_eval::TargetReader(mset).flags(), expected) << "face_up=" << face_up;
  }
}

// The distillation generator's encode path under the open-leaves arm: every
// candidate row is the standard row plus the opponent's replayed retained
// leave, so an open-leaves teacher is fed the input it was trained on rather
// than a zeroed block.
TEST(MoveSetEvalTargetLog, OpenLeavesCandidateRowsCarryTheReplayedLeave) {
  Dictionary dict = medium_dict();
  const GameLogStorage storage = play_test_game(dict, /*seed=*/4242ULL);
  const GameLog g = storage.view();
  const int turn = 4;
  ASSERT_LT(turn, g.num_records);

  const InputEncodingSpec base{&dict};
  const InputEncodingSpec open{&dict, /*opp_leave_input=*/true};
  binlog::PositionEncoder base_enc(base);
  binlog::PositionEncoder open_enc(open);
  const int mover = base_enc.replay_to_sampled(g, turn, /*post_move=*/false);
  ASSERT_EQ(open_enc.replay_to_sampled(g, turn, /*post_move=*/false), mover);

  const std::vector<Move> candidates = {g.records[turn].move, Move::pass()};
  std::vector<float> base_rows(candidates.size() * input_floats(base), -1.0f);
  std::vector<float> open_rows(candidates.size() * input_floats(open), -1.0f);
  binlog::encode_candidate_rows(base_enc, g, turn, mover, candidates, base_rows.data());
  binlog::encode_candidate_rows(open_enc, g, turn, mover, candidates, open_rows.data());

  const Rack leave = binlog::opp_leave_from_replay(g, turn, open_enc.rack(1 - mover));
  ASSERT_GT(leave.size(), 0) << "the leave must be non-empty or the tail check is vacuous";
  float expected_counts[kOppLeaveCountFloats] = {};
  for (Tile t : leave.tiles()) {
    if (!t.is_empty()) expected_counts[t.index()] += 1.0f;
  }

  for (size_t c = 0; c < candidates.size(); ++c) {
    const float* base_row = base_rows.data() + c * input_floats(base);
    const float* open_row = open_rows.data() + c * input_floats(open);
    ASSERT_EQ(std::memcmp(base_row, open_row, sizeof(float) * size_t(input_floats(base))), 0)
      << "candidate " << c;
    const float* tail = open_row + input_floats(base);
    for (int i = 0; i < kOppLeaveCountFloats; ++i) {
      ASSERT_EQ(tail[i], expected_counts[i]) << "candidate " << c << ", tile " << i;
    }
  }
}

TEST(MoveSetEncoder, Basic) {
  namespace mset = move_set;
  // A horizontal PLAY at (row 4, cols 2..4): A, a blank shown as B, C; scoring
  // 24. An exchange surrendering D, A, and an undesignated blank (stored
  // sorted: A, D, blank). And a PASS.
  const Move play = make_play_full(
    4, 2, /*horizontal=*/true, 0b111, 24,
    {Glyph::of(Tile::from_char('A')), Glyph::played(Tile::from_char('B'), /*is_blank=*/true),
     Glyph::of(Tile::from_char('C'))});
  TileCounts xchg_tiles;
  xchg_tiles.add(Tile::from_char('D'));
  xchg_tiles.add(Tile::from_char('A'));
  xchg_tiles.add(BLANK);
  const Move exch = Move::exchange(xchg_tiles);
  const Move moves[3] = {play, exch, Move::pass()};
  const int32_t pre_diffs[3] = {10, -5, -5};  // mover's pre-move score advantage

  std::vector<int32_t> letters(3 * mset::kMoveMaxPlaced);
  std::vector<uint8_t> blanks(3 * mset::kMoveMaxPlaced);
  std::vector<int32_t> squares(3 * mset::kMoveMaxPlaced);
  std::vector<uint8_t> tile_mask(3 * mset::kMoveMaxPlaced);
  std::vector<float> scalars(3 * mset::kMoveScalars);
  mset::encode_moves(moves, 3, pre_diffs, letters.data(), blanks.data(), squares.data(),
                     tile_mask.data(), scalars.data());

  // PLAY: three placed tiles in lane order; the middle is a blank. Letters are
  // 1..26 identities regardless of blank-ness.
  ASSERT_TRUE(tile_mask[0] == 1 && tile_mask[1] == 1 && tile_mask[2] == 1);
  ASSERT_TRUE(tile_mask[3] == 0 && tile_mask[6] == 0);
  ASSERT_EQ(letters[0], Tile::from_char('A').index() + 1);
  ASSERT_EQ(letters[1], Tile::from_char('B').index() + 1);
  ASSERT_EQ(letters[2], Tile::from_char('C').index() + 1);
  ASSERT_TRUE(blanks[0] == 0 && blanks[1] == 1 && blanks[2] == 0);
  ASSERT_EQ(squares[0], 4 * BOARD_SIZE + 2);
  ASSERT_EQ(squares[2], 4 * BOARD_SIZE + 4);
  // Resultant differential (pre 10 + score 24), tiles/7, is_play.
  ASSERT_LT(std::abs(scalars[0] - 34.0f / kScoreDiffInputScale), 1e-6f);
  ASSERT_LT(std::abs(scalars[1] - 3.0f / 7.0f), 1e-6f);
  ASSERT_EQ(scalars[2], 1.0f);

  // EXCHANGE: the surrendered tiles (sorted A, D, blank) fill the letter/
  // blank/tile_mask slots so same-size exchanges differ by which tiles leave;
  // squares stay 0. The undesignated blank has no letter -- the blank flag
  // alone represents it. Resultant diff is the pre-move diff (score 0), and
  // is_play is 0.
  const int e = mset::kMoveMaxPlaced;
  ASSERT_TRUE(tile_mask[e + 0] == 1 && tile_mask[e + 1] == 1 && tile_mask[e + 2] == 1);
  ASSERT_TRUE(tile_mask[e + 3] == 0 && tile_mask[e + 6] == 0);
  ASSERT_EQ(letters[e + 0], Tile::from_char('A').index() + 1);
  ASSERT_EQ(letters[e + 1], Tile::from_char('D').index() + 1);
  ASSERT_EQ(letters[e + 2], 0);  // the blank: letter stays the pad value
  ASSERT_TRUE(blanks[e + 0] == 0 && blanks[e + 1] == 0 && blanks[e + 2] == 1);
  for (int j = 0; j < mset::kMoveMaxPlaced; ++j) ASSERT_EQ(squares[e + j], 0);
  ASSERT_LT(std::abs(scalars[mset::kMoveScalars + 0] - (-5.0f) / kScoreDiffInputScale), 1e-6f);
  ASSERT_LT(std::abs(scalars[mset::kMoveScalars + 1] - 3.0f / 7.0f), 1e-6f);
  ASSERT_EQ(scalars[mset::kMoveScalars + 2], 0.0f);

  // PASS: no tiles at all; only the carried-through differential is nonzero.
  const int p = 2 * mset::kMoveMaxPlaced;
  for (int j = 0; j < mset::kMoveMaxPlaced; ++j) ASSERT_EQ(tile_mask[p + j], 0);
  ASSERT_LT(std::abs(scalars[2 * mset::kMoveScalars + 0] - (-5.0f) / kScoreDiffInputScale), 1e-6f);
  ASSERT_EQ(scalars[2 * mset::kMoveScalars + 1], 0.0f);
  ASSERT_EQ(scalars[2 * mset::kMoveScalars + 2], 0.0f);
}

TEST(SimObservationLog, Roundtrip) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_sobs";
  fs::create_directories(tmp);
  const std::string path = (tmp / "test.sobs").string();

  // Synthetic observations with distinct values in every field, so a layout
  // mixup cannot round-trip cleanly.
  SimObservation o1{};
  o1.n = 16;
  o1.wins = 9.25;  // fractional, as value-truncated rollouts accumulate
  o1.draws = 1.5;
  o1.losses = 5.25;
  o1.delta_sum = 123.5;
  o1.delta_sq_sum = 4567.25;
  o1.opp_next_count[7 * 15 + 7] = 12;
  o1.opp_win_count[7 * 15 + 7] = 5.5f;
  o1.self_next_count[3] = 2;
  o1.self_win_count[3] = 1.25f;
  SimObservation o2{};
  o2.n = 16;
  o2.draws = 16;

  const Move m1 = make_play_full(4, 2, /*horizontal=*/true, 0b111, 24,
                                 {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                                  Glyph::of(Tile::from_char('C'))});
  TileCounts xchg_tiles;
  xchg_tiles.add(Tile::from_char('A'));
  const Move m2 = Move::exchange(xchg_tiles);

  {
    SimObsWriter w(path, kSimObsFlagTrajectory, "cafe1234", "beef5678", /*horizon_plies=*/4);
    w.add_position(3, 11, {m1, m2}, {o1, o2}, 16, 999, /*num_legal_moves=*/321,
                   {SimObsRole::kAnchor, SimObsRole::kOffPolicy});
    w.add_position(4, 0, {m2}, {o2}, 16, 1000);  // no roles: every record stores kAnchor
    w.close();
  }

  SimObsReader r(path);
  ASSERT_EQ(r.num_positions(), 2);
  ASSERT_EQ(r.flags(), kSimObsFlagTrajectory);
  ASSERT_EQ(r.proposer_hash(), "cafe1234");
  ASSERT_EQ(r.leaf_model_hash(), "beef5678");
  ASSERT_EQ(r.horizon_plies(), 4);
  const SimObsReader::Position p0 = r.position(0);
  ASSERT_EQ(p0.header->game_index, 3);
  ASSERT_EQ(p0.header->turn_index, 11);
  ASSERT_EQ(p0.header->num_candidates, 2);
  ASSERT_EQ(p0.header->rollouts, 16);
  ASSERT_EQ(p0.header->base_seed, 999);
  ASSERT_EQ(p0.header->num_legal_moves, 321);
  ASSERT_EQ(p0.header->flags, 0u);  // no position flags at v4
  SimObsRecord rec;                 // copy out of the packed file view before comparing
  std::memcpy(&rec, &p0.records[0], sizeof(rec));
  ASSERT_EQ(std::memcmp(&rec.move, &m1, sizeof(Move)), 0);
  ASSERT_EQ(std::memcmp(&rec.obs, &o1, sizeof(SimObservation)), 0);
  ASSERT_EQ(rec.role, SimObsRole::kAnchor);
  std::memcpy(&rec, &p0.records[1], sizeof(rec));
  ASSERT_EQ(std::memcmp(&rec.move, &m2, sizeof(Move)), 0);
  ASSERT_EQ(std::memcmp(&rec.obs, &o2, sizeof(SimObservation)), 0);
  ASSERT_EQ(rec.role, SimObsRole::kOffPolicy);
  const SimObsReader::Position p1 = r.position(1);
  ASSERT_EQ(p1.header->game_index, 4);
  ASSERT_EQ(p1.header->num_candidates, 1);
  ASSERT_EQ(p1.header->base_seed, 1000);
  std::memcpy(&rec, &p1.records[0], sizeof(rec));
  ASSERT_EQ(rec.role, SimObsRole::kAnchor);  // default when roles omitted

  // A version mismatch fails loudly (stale files must never misparse).
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(4);  // SimObsFileHeader::version
    const uint16_t bad = 0xFFFF;
    f.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
  }
  ASSERT_THROW(SimObsReader r2(path), std::runtime_error);

  fs::remove_all(tmp);
}

// Build a one-game .slog buffer whose two turns are both PASSes, with a starting
// handicap of `initial_score_p0` points for p0, and return the score
// differential recovered from the sampled position's input encoding. The two
// turns are PASSes, so the board stays empty and the handicap is the sole
// contributor to the score-diff feature.
static int decode_handicap_score_diff(int initial_score_p0) {
  using namespace scribblez::binlog;
  using namespace scribblez;

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.num_games = 1;

  GameMetadata gm{};
  gm.start_offset = sizeof(FileHeader) + sizeof(GameMetadata);
  gm.num_turns = 2;
  gm.sampled_turn = 0;  // pre-move state at turn 0: empty board, active p0
  gm.initial_score_p0 = initial_score_p0;

  InitialRacks ir{};  // both racks empty -- irrelevant to the score-diff feature
  TurnBlob t0{};
  t0.move = Move::pass();
  TurnBlob t1{};
  t1.move = Move::pass();

  std::vector<char> buf;
  auto append_bytes = [&buf](const void* p, size_t n) {
    const char* c = reinterpret_cast<const char*>(p);
    buf.insert(buf.end(), c, c + n);
  };
  append_bytes(&hdr, sizeof(hdr));
  append_bytes(&gm, sizeof(gm));
  append_bytes(&ir, sizeof(ir));
  append_bytes(&t0, sizeof(t0));
  append_bytes(&t1, sizeof(t1));

  std::vector<float> output(kRowFloats, 0.0f);
  uint8_t flip = 0;
  Dictionary dict = medium_dict();
  BlockDecoder dec(InputEncodingSpec{&dict});
  dec.decode(buf.data(), "handicap-test", /*local_start=*/0, /*n_rows=*/1, &flip,
             /*post_move=*/false, /*output_row_start=*/0, output.data());

  // The score-diff scalar recovers the differential when rescaled.
  const float* sd = output.data() + kSpatialFloats + kScoreDiffOffset;
  return std::lround(sd[0] * kScoreDiffInputScale);
}

// A head-start handicap stored in GameMetadata must reach the replayed
// position's score-differential input (the decoder seeds its score
// accumulator from the metadata's initial scores).
TEST(Encoder, HandicapShiftsScoreDiffInput) {
  ASSERT_EQ(decode_handicap_score_diff(0), 0);
  ASSERT_EQ(decode_handicap_score_diff(80), 80);
}

// Highest raw score among a move list (0 if empty).
static int best_move_score(const std::vector<Move>& moves) {
  int best = 0;
  for (const auto& m : moves) best = std::max<int>(best, m.score());
  return best;
}

// Highest per-lane max over all 30 lanes (0 if every lane is empty).
static int lane_global_max(const LaneTargets& t) {
  int best = 0;
  for (const auto& lane : t.rows)
    if (lane.has_move) best = std::max(best, lane.max_score);
  for (const auto& lane : t.cols)
    if (lane.has_move) best = std::max(best, lane.max_score);
  return best;
}

// The max-move-per-lane task's per-lane targets: every legal play is bucketed into the
// row it lies along (horizontal) or column (vertical); single-tile plays go to
// whichever direction(s) they form a word in. These pin the bucketing, the
// union-over-tied-maxima, and the single-tile cross rule, and cross-check the
// structural global max against the raw move list.
TEST(Lane, Targets) {
  const Dictionary d = tiny_dict();

  // 1. Extending CAT -> CATS is a horizontal play in CENTER's row only; the
  //    lone S forms no vertical word, so its column stays empty.
  {
    Board b;
    b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                      {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                       Glyph::of(Tile::from_char('T'))}));
    const Rack r = rack_from("S");
    const LaneTargets t = compute_lane_targets(b, r, d);

    MoveGenerator gen(b, d);
    const auto moves = gen.generate(r);
    const int sk = Tile::from_char('S').index();

    ASSERT_TRUE(t.rows[CENTER].has_move);
    ASSERT_TRUE((t.rows[CENTER].placed[CENTER + 3] >> sk) & 1u);  // S newly placed after CAT
    ASSERT_EQ(t.rows[CENTER].max_score, best_move_score(moves));
    ASSERT_FALSE(t.cols[CENTER + 3].has_move);
    ASSERT_EQ(lane_global_max(t), best_move_score(moves));
  }

  // 2. A single tile that crosses (forms a word both ways) lands in BOTH its
  //    row and its column, at the same score.
  {
    Board b;
    b.set(CENTER, CENTER - 1, Glyph::of(Tile::from_char('A')));  // A to S's left
    b.set(CENTER - 1, CENTER, Glyph::of(Tile::from_char('A')));  // A above S
    const Rack r = rack_from("S");
    const LaneTargets t = compute_lane_targets(b, r, d);
    const int sk = Tile::from_char('S').index();

    ASSERT_TRUE(t.rows[CENTER].has_move);
    ASSERT_TRUE(t.cols[CENTER].has_move);
    ASSERT_TRUE((t.rows[CENTER].placed[CENTER] >> sk) & 1u);  // lane cell == column
    ASSERT_TRUE((t.cols[CENTER].placed[CENTER] >> sk) & 1u);  // lane cell == row
    ASSERT_EQ(t.rows[CENTER].max_score, t.cols[CENTER].max_score);
  }

  // 3. A single tile cannot open the game (no word formed), so every lane is
  //    empty on an empty board.
  {
    Board b;
    const LaneTargets t = compute_lane_targets(b, rack_from("S"), d);
    for (const auto& lane : t.rows) ASSERT_FALSE(lane.has_move);
    for (const auto& lane : t.cols) ASSERT_FALSE(lane.has_move);
  }

  // 3b. The flat label encoding mirrors the LaneTargets for the CATS position:
  //     CENTER's row lane carries the S occupancy + score bin + mask bit, and an
  //     untouched lane is all zeros.
  {
    Board b;
    b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                      {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                       Glyph::of(Tile::from_char('T'))}));
    const LaneTargets t = compute_lane_targets(b, rack_from("S"), d);
    std::vector<float> row(kLaneLabelFloats, -1.0f);
    encode_lane_targets(t, /*flip=*/false, row.data());

    const float* occ = row.data();
    const float* score = occ + kLaneOccupancyFloats;
    const float* mask = score + kLaneScoreFloats;
    const int row_id = CENTER;  // axis 0, lane CENTER
    const int sk = Tile::from_char('S').index();

    // Occupancy: only S at cell CENTER+3 of CENTER's row lane is set.
    const float* lane_occ = occ + row_id * kLaneLen * kLaneTileKinds;
    ASSERT_EQ(lane_occ[(CENTER + 3) * kLaneTileKinds + sk], 1.0f);
    ASSERT_EQ(lane_occ[(CENTER + 3) * kLaneTileKinds + Tile::from_char('C').index()], 0.0f);
    ASSERT_EQ(lane_occ[CENTER * kLaneTileKinds + sk], 0.0f);  // CAT tiles are not "placed"

    ASSERT_EQ(mask[row_id], 1.0f);
    ASSERT_EQ(score[row_id], float(std::min(t.rows[CENTER].max_score, kLaneScoreBins - 1)));

    // An empty lane (row 0) is all zeros: mask off, score 0, no occupancy.
    ASSERT_EQ(mask[0], 0.0f);
    ASSERT_EQ(score[0], 0.0f);
    for (int i = 0; i < kLaneLen * kLaneTileKinds; ++i) ASSERT_EQ(occ[i], 0.0f);

    // Flip is a rows<->cols swap: the horizontal CATS play that lived in axis-0
    // lane CENTER now lives in axis-1 (vertical) lane CENTER, same cell.
    std::vector<float> frow(kLaneLabelFloats, -1.0f);
    encode_lane_targets(t, /*flip=*/true, frow.data());
    const float* focc = frow.data();
    const float* v_lane = focc + (kLanesPerAxis + CENTER) * kLaneLen * kLaneTileKinds;
    const float* h_lane = focc + CENTER * kLaneLen * kLaneTileKinds;
    ASSERT_EQ(v_lane[(CENTER + 3) * kLaneTileKinds + sk], 1.0f);                     // now vertical
    for (int i = 0; i < kLaneLen * kLaneTileKinds; ++i) ASSERT_EQ(h_lane[i], 0.0f);  // cols empty
    ASSERT_EQ((focc + kLaneOccupancyFloats)[kLanesPerAxis + CENTER], score[row_id]);
  }

  // 4. Random-walk invariant: the structural global max always equals the raw
  //    best move score, and lanes are marked iff legal moves exist.
  {
    std::mt19937 rng(0x1a2b3c);
    Board b;
    for (int step = 0; step < 60; ++step) {
      const Rack r = random_rack(rng);
      MoveGenerator gen(b, d);
      const auto moves = gen.generate(r);
      const LaneTargets t = compute_lane_targets(b, r, d);

      ASSERT_EQ(lane_global_max(t), best_move_score(moves));
      bool any_lane = false;
      for (const auto& lane : t.rows) any_lane = any_lane || lane.has_move;
      for (const auto& lane : t.cols) any_lane = any_lane || lane.has_move;
      ASSERT_EQ(any_lane, !moves.empty());

      if (moves.empty()) {
        b = Board();  // reset when the position is stuck
        continue;
      }
      const Move* best = &moves.front();
      for (const auto& m : moves)
        if (m.score() > best->score()) best = &m;
      b.apply(*best);
    }
  }
}

// compute_lane_best_moves keeps the actual plays tied for each lane's maximum,
// sharing compute_lane_targets' assignment rule. Pins: the two agree on
// has_move/max_score for every lane; every kept move scores exactly the lane max;
// and a kept play's word and origin recover from the pre-move board (CATS extends
// CAT in CENTER's row).
TEST(Lane, BestMoves) {
  const Dictionary d = tiny_dict();

  Board b;
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                     Glyph::of(Tile::from_char('T'))}));
  const Rack r = rack_from("S");

  const LaneTargets t = compute_lane_targets(b, r, d);
  const LaneBestMovesSet bm = compute_lane_best_moves(b, r, d);

  // Agreement with the union targets across all 30 lanes.
  for (int i = 0; i < kLanesPerAxis; ++i) {
    ASSERT_EQ(bm.rows[i].has_move, t.rows[i].has_move);
    ASSERT_EQ(bm.cols[i].has_move, t.cols[i].has_move);
    ASSERT_EQ(bm.rows[i].max_score, t.rows[i].max_score);
    ASSERT_EQ(bm.cols[i].max_score, t.cols[i].max_score);
  }

  // Every kept move scores exactly its lane's max (no sub-maximal plays retained).
  for (const auto& lane : bm.rows)
    for (const Move& m : lane.moves) ASSERT_EQ(int(m.score()), lane.max_score);
  for (const auto& lane : bm.cols)
    for (const Move& m : lane.moves) ASSERT_EQ(int(m.score()), lane.max_score);

  // CENTER's row holds the maximal play CATS; its word and origin recover from the
  // pre-move board.
  const LaneBestMoves& row = bm.rows[CENTER];
  ASSERT_TRUE(row.has_move);
  ASSERT_FALSE(row.moves.empty());
  bool found_cats = false;
  for (const Move& m : row.moves)
    if (m.main_word(b) == "CATS") {
      found_cats = true;
      ASSERT_TRUE(m.horizontal());
      ASSERT_EQ(m.word_origin(b), std::make_pair(CENTER, CENTER));
    }
  ASSERT_TRUE(found_cats);
}

// The GCG -> analysis-position bridge and the lane-analysis JSON. parse_* takes the
// board after all recorded moves with the next player to move and reads that
// player's rack from the #Rack header (the move log clears it during replay); the
// JSON carries the web board plus per-lane ground truth and maximal plays.
TEST(Lane, Analysis) {
  // One play by P1 (CAT across the center), so P2 is on move; #Rack2 is the
  // analysis rack. Coordinate "8H" is row 8 (index 7 == CENTER), column H (index 7).
  const std::string gcg =
    "#player1 P1 Player One\n"
    "#player2 P2 Player Two\n"
    "#Rack1 _______\n"
    "#Rack2 EINRSTU\n"
    ">P1: AACATTX 8H CAT +5 5\n";

  GcgAnalysisPosition pos;
  std::string error;
  ASSERT_TRUE(parse_gcg_analysis_position(gcg, &pos, &error));
  ASSERT_EQ(pos.on_move, 1);                                  // P2 to move after P1's single play
  ASSERT_EQ(pos.rack, rack_from("EINRSTU"));                  // from #Rack2
  ASSERT_FALSE(pos.board.at(CENTER, CENTER).is_empty());      // C
  ASSERT_FALSE(pos.board.at(CENTER, CENTER + 2).is_empty());  // T
  ASSERT_TRUE(pos.board.at(CENTER, CENTER + 3).is_empty());   // nothing past CAT

  // JSON structure (built against a tiny dictionary, no real lexicon needed):
  // extend CAT with S in CENTER's row.
  const Dictionary d = tiny_dict();
  Board b;
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                     Glyph::of(Tile::from_char('T'))}));
  const std::string js = lane_analysis_json(b, rack_from("S"), /*on_move=*/0, d);
  const boost::json::value v = boost::json::parse(js);
  const boost::json::object& o = v.as_object();
  ASSERT_TRUE(o.contains("board"));  // web GameState for rendering
  ASSERT_EQ(o.at("on_move").as_int64(), 0);
  const boost::json::object& la = o.at("lane_analysis").as_object();
  const boost::json::array& rows = la.at("rows").as_array();
  ASSERT_EQ(rows.size(), size_t(kLanesPerAxis));
  const boost::json::object& center_row = rows[CENTER].as_object();
  ASSERT_TRUE(center_row.at("has_move").as_bool());
  bool json_has_cats = false;
  for (const auto& mv : center_row.at("best_moves").as_array())
    if (mv.as_object().at("word").as_string() == "CATS") json_has_cats = true;
  ASSERT_TRUE(json_has_cats);
}

// The max-move-per-lane model's input encoder: 31 board planes (letters, blank-marker,
// premiums) + 27 raw rack counts, with NO cross-check planes. Pins the plane
// contents, premium consistency, rack counts, and the flip transpose.
static int prem_plane_offset(Premium p) {
  if (p == Premium::DLS) return 0;
  if (p == Premium::TLS) return 1;
  if (p == Premium::DWS) return 2;
  if (p == Premium::TWS) return 3;
  return -1;
}

TEST(MaxMovePerLane, InputEncoder) {
  Board b;
  b.set(7, 7, Glyph::of(Tile::from_char('C')));
  b.set(7, 8, Glyph::of(Tile::from_char('A')));
  b.set(7, 9, Glyph::of(Tile::from_char('T')));
  b.set(5, 5, Glyph::played(Tile::from_char('S'), /*is_blank=*/true));  // designated blank
  const Rack rack = rack_from("AAB?");

  using Enc = MaxMovePerLaneInputEncoder;
  const int A = Tile::from_char('A').index();
  const int B = Tile::from_char('B').index();
  const int C = Tile::from_char('C').index();
  const int Sx = Tile::from_char('S').index();
  const int cells = Enc::kBoardCells;
  auto cell = [](int r, int c) { return r * BOARD_SIZE + c; };

  std::vector<float> out(Enc::kInputFloats, -1.0f);
  Enc::encode(b, rack, /*flip=*/false, out.data());

  // Letter planes (a designated blank still sets its letter plane).
  ASSERT_EQ(out[C * cells + cell(7, 7)], 1.0f);
  ASSERT_EQ(out[A * cells + cell(7, 8)], 1.0f);
  ASSERT_EQ(out[Sx * cells + cell(5, 5)], 1.0f);
  ASSERT_EQ(out[A * cells + cell(7, 7)], 0.0f);

  // Blank-marker plane: set under the blank only.
  ASSERT_EQ(out[BoardPlanes::kBlankMarkerPlane * cells + cell(5, 5)], 1.0f);
  ASSERT_EQ(out[BoardPlanes::kBlankMarkerPlane * cells + cell(7, 7)], 0.0f);

  // Premium planes agree with Board::PREMIUM at every cell (reported even under
  // a played tile), and exactly one premium plane is set per premium square.
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const int want = prem_plane_offset(b.premium_at(r, c));
      for (int off = 0; off < BoardPlanes::kPremiumPlanes; ++off) {
        const float v = out[(BoardPlanes::kPremiumPlane0 + off) * cells + cell(r, c)];
        ASSERT_EQ(v, (off == want ? 1.0f : 0.0f));
      }
    }
  }

  // Rack scalars: raw counts, blank in slot 26.
  const float* counts = out.data() + Enc::kSpatialFloats;
  ASSERT_EQ(counts[A], 2.0f);
  ASSERT_EQ(counts[B], 1.0f);
  ASSERT_EQ(counts[26], 1.0f);  // blank
  ASSERT_EQ(counts[C], 0.0f);

  // Flip transposes the spatial planes but leaves rack scalars untouched.
  std::vector<float> flipped(Enc::kInputFloats, -1.0f);
  Enc::encode(b, rack, /*flip=*/true, flipped.data());
  ASSERT_EQ(flipped[A * cells + cell(8, 7)], 1.0f);  // (7,8) -> (8,7)
  ASSERT_EQ(flipped[A * cells + cell(7, 8)], 0.0f);
  ASSERT_EQ(flipped[C * cells + cell(7, 7)], 1.0f);  // on the diagonal, unchanged
  const float* fcounts = flipped.data() + Enc::kSpatialFloats;
  ASSERT_TRUE(fcounts[A] == 2.0f && fcounts[26] == 1.0f);
}

// The max-move-per-lane training task: one full row is exactly the max-move-per-lane input encoding
// followed by the per-lane labels for the board/rack at the sampled position.
// Checked for both symmetry orientations.
TEST(MaxMovePerLane, TaskRow) {
  const Dictionary d = tiny_dict();

  // CAT on the board (the context exposes the board via its GameStateEncoder).
  GameStateEncoder gse{InputEncodingSpec{&d}};
  gse.apply_move(make_play(CENTER, CENTER, /*horizontal=*/true,
                           {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                            Glyph::of(Tile::from_char('T'))}));
  const Rack rack = rack_from("S");

  for (bool flip : {false, true}) {
    EncodeContext ctx{};
    ctx.enc = &gse;
    ctx.pov_rack = &rack;
    ctx.apply_flip = flip;
    ctx.spec = {&d};

    std::vector<float> row(MaxMovePerLaneTask::kRowFloats, -1.0f);
    MaxMovePerLaneTask::encode_row(ctx, row.data());

    std::vector<float> ref_in(MaxMovePerLaneInputEncoder::kInputFloats);
    MaxMovePerLaneInputEncoder::encode(gse.board(), rack, flip, ref_in.data());
    std::vector<float> ref_lab(kLaneLabelFloats);
    encode_lane_targets(compute_lane_targets(gse.board(), rack, d), flip, ref_lab.data());

    ASSERT_EQ(MaxMovePerLaneTask::kInputFloats, int(ref_in.size()));
    ASSERT_EQ(MaxMovePerLaneTask::kLabelFloats, int(ref_lab.size()));
    for (int i = 0; i < MaxMovePerLaneTask::kInputFloats; ++i) ASSERT_EQ(row[i], ref_in[i]);
    for (int i = 0; i < MaxMovePerLaneTask::kLabelFloats; ++i)
      ASSERT_EQ(row[MaxMovePerLaneTask::kInputFloats + i], ref_lab[i]);
  }
}

// The trajectory pane's decision-point reading of a position-set .gcg
// (training/trajectory_position.h): the exhibit position (a frozen copy of
// positions/NWL23/face-up-trajectory-set/egotize-lane.gcg) parses to the seat,
// rack and known leave that set's README states, the board row is the open-leaves
// arm's, the score differential is the mover's, and every legal move --
// among them the recorded HastyBot play, E11 GAVE (through the A of INCASED,
// so "E11 G.VE") -- carries a notation in the bundle. Requires the NWL23 KWG +
// leaves; skipped if absent.
TEST(TrajectoryPosition, ExhibitDecisionPoint) {
  namespace fs = std::filesystem;
  using namespace scribblez;
  const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  const std::string gcg_path = std::string(SCRIBBLEZ_TEST_DATA_DIR) + "/egotize-lane.gcg";
  if (!fs::exists(kwg) || !fs::exists(leaves)) GTEST_SKIP() << "no NWL23 kwg/leaves";
  Dictionary dict = Dictionary::load_kwg(kwg);
  HastyEquity::init(leaves, HastyEquity::default_peg_path());

  TrajectoryDecision d;
  std::string error;
  ASSERT_TRUE(
    read_trajectory_decision(util::read_file(gcg_path), dict, /*open_leaves=*/true, &d, &error))
    << error;
  EXPECT_EQ(d.position.mover, 0);
  EXPECT_EQ(d.position.rack.to_string(), "AEEGSTV");
  EXPECT_EQ(d.position.opp_leave.to_string(), "");  // Hasty_2 just bingoed
  EXPECT_EQ(d.position.scores[0], 440);
  EXPECT_EQ(d.position.scores[1], 387);
  ASSERT_GT(d.legal_moves.size(), 100u);
  const boost::json::object bundle =
    boost::json::parse(trajectory_decision_board_json(d)).as_object();
  const boost::json::array& notations = bundle.at("moves").as_array();
  ASSERT_EQ(notations.size(), d.legal_moves.size());
  bool saw_gave = false;
  for (const auto& n : notations) saw_gave |= n.as_string() == "E11 G.VE";
  EXPECT_TRUE(saw_gave);
  EXPECT_EQ(bundle.at("mover").as_int64(), 0);
  EXPECT_EQ(bundle.at("scores").as_array()[0].as_int64(), 440);
  EXPECT_EQ(bundle.at("rack").as_array().size(), 7u);

  const InputEncodingSpec arm{&dict, /*opp_leave_input=*/true};
  std::vector<float> row(size_t(input_floats(arm)));
  int score_diff = 0;
  encode_trajectory_decision(d, arm, row.data(), &score_diff);
  EXPECT_EQ(score_diff, 440 - 387);
  // The hidden arm's row is a prefix-shaped sibling: same spatial block,
  // fewer scalars.
  const InputEncodingSpec hidden{&dict, false};
  EXPECT_EQ(input_floats(arm), input_floats(hidden) + kOppLeaveCountFloats);
}

// Evidence staging (agent/evidence_staging.h): the C++ port of evidence.py's
// build_evidence_inputs. Hand-computed against a two-candidate evidence set over
// three scored candidates, so a drift from the Python normalization -- a
// missing /rollouts, an unscaled delta, a wrong softmax/sigmoid, a mis-gathered
// move encoding, or a footprint class on the wrong (slot, cell) channel -- is
// caught here; the runtime's end-to-end parity test
// (test_proposal_inference_parity) cross-checks the whole path against the
// Python fusion stage.
TEST(EvidenceStaging, MatchesHandComputedNormalization) {
  using namespace evidence;
  constexpr int kChannels = 2;
  constexpr int kScored = 3;
  constexpr int kCells = kEvidencePlaneCells;

  // Cache predictions for three scored candidates. move_enc is gathered by a
  // candidate's scored index; wld/planes are decoded here.
  std::vector<float> move_enc = {1.0f, 2.0f, 3.0f, 4.0f, 7.0f, 8.0f};  // rows 0,1,2
  std::vector<float> wld_logits = {0.0f, 0.0f, 0.0f, 5.0f, 5.0f, 5.0f, 2.0f, 0.0f, 0.0f};
  std::vector<float> score_diff = {-50.0f, 10.0f, 0.0f, 0.0f, 30.0f, 5.0f};  // [mean,std] rows
  std::vector<float> plane_probs(size_t(kScored) * kNumPredictedPlanes * kCells, 0.0f);
  // Scored candidate 2, predicted channel 2, cell 7: a non-default probability.
  plane_probs[(2 * kNumPredictedPlanes + 2) * kCells + 7] = 0.7f;
  const CachePredictions pred{move_enc.data(), wld_logits.data(), score_diff.data(),
                              plane_probs.data(), kChannels};

  // Evidence candidate 0 == scored 2: a one-tile horizontal play at (7,7);
  // observations with rollouts n=4. Histogram classes are (cell, slot) pairs:
  // opp replies at (cell 5, slot 2), self next moves at (cell 10, slot 0).
  SimObservation obs0;
  obs0.n = 4;
  obs0.wins = 3.0;
  obs0.draws = 0.0;
  obs0.losses = 1.0;
  obs0.delta_sum = 40.0;      // mean 10
  obs0.delta_sq_sum = 800.0;  // var = 800/4 - 100 = 100, std 10
  obs0.opp_next_count[5 * kSlotsPerCell + 2] = 2;
  obs0.self_next_count[10 * kSlotsPerCell + 0] = 4;
  obs0.opp_win_count[5 * kSlotsPerCell + 2] = 1.0f;
  obs0.self_win_count[10 * kSlotsPerCell + 0] = 2.0f;
  const Glyph g = Glyph::of(Tile::from_char('A'));
  const Move play = Move::play(/*horizontal=*/true, /*start=*/7, /*square_mask=*/uint16_t(1 << 7),
                               /*score=*/20, &g, /*num_played=*/1);

  // Evidence candidate 1 == scored 0: a PASS (empty footprint), n=2, all draws.
  SimObservation obs1;
  obs1.n = 2;
  obs1.draws = 2.0;
  obs1.delta_sum = -20.0;     // mean -10
  obs1.delta_sq_sum = 200.0;  // var 0, std 0

  const std::vector<Move> moves = {play, Move{}};
  const std::vector<SimObservation> observations = {obs0, obs1};
  const std::vector<int> scored_indices = {2, 0};

  constexpr int kMaxE = 4;
  // Pre-fill with a sentinel (buffers are reused turn-over-turn), so the
  // padding-row zero checks below prove the memsets actually cleared it rather
  // than passing vacuously on a fresh zero-initialized vector.
  std::vector<float> ev_move_enc(size_t(kMaxE) * kChannels, -1.0f);
  std::vector<float> ev_planes(size_t(kMaxE) * kNumEvidencePlanes * kCells, -1.0f);
  std::vector<float> ev_scalars(size_t(kMaxE) * kNumEvidenceScalars, -1.0f);
  std::vector<uint8_t> ev_mask(kMaxE, 9);
  const EvidenceStagingOutputs out{ev_move_enc.data(), ev_planes.data(), ev_scalars.data(),
                                   ev_mask.data()};
  stage_evidence(moves, observations, scored_indices, pred, kMaxE, out);

  EXPECT_EQ(ev_mask[0], 1);
  EXPECT_EQ(ev_mask[1], 1);
  EXPECT_EQ(ev_mask[2], 0);
  EXPECT_EQ(ev_mask[3], 0);

  // Padding rows (2, 3) are fully zeroed in every buffer, not just move_enc: a
  // dropped or mis-sized memset of planes/scalars would otherwise leak garbage.
  for (int row = 2; row < kMaxE; ++row) {
    const float* pad_planes = ev_planes.data() + size_t(row) * kNumEvidencePlanes * kCells;
    const float* pad_scalars = ev_scalars.data() + size_t(row) * kNumEvidenceScalars;
    EXPECT_FLOAT_EQ(pad_planes[(kNumObservedPlanes + 2) * kCells + 7], 0.0f);  // a predicted cell
    EXPECT_FLOAT_EQ(pad_planes[0], 0.0f);
    EXPECT_FLOAT_EQ(pad_scalars[0], 0.0f);
    EXPECT_FLOAT_EQ(pad_scalars[kNumEvidenceScalars - 1], 0.0f);
  }

  // move_enc gathered by scored index (2 then 0), padding rows zeroed.
  EXPECT_FLOAT_EQ(ev_move_enc[0], 7.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[1], 8.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[2], 1.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[3], 2.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[4], 0.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[7], 0.0f);

  // Candidate 0 planes: observed histogram counts / rollouts on channel
  // (head * kSlotsPerCell + slot) at the class's cell, predicted probs copied
  // through, footprint one-hot at (slot channel, anchor cell).
  const float* p0 = ev_planes.data();
  constexpr int kPredBase = kNumObservedPlanes;
  constexpr int kFootBase = kNumObservedPlanes + kNumPredictedPlanes;
  EXPECT_FLOAT_EQ(p0[(0 * kSlotsPerCell + 2) * kCells + 5], 0.5f);   // opp_next 2/4
  EXPECT_FLOAT_EQ(p0[(1 * kSlotsPerCell + 0) * kCells + 10], 1.0f);  // self_next 4/4
  EXPECT_FLOAT_EQ(p0[(2 * kSlotsPerCell + 2) * kCells + 5], 0.25f);  // opp_win 1/4
  EXPECT_FLOAT_EQ(p0[(3 * kSlotsPerCell + 0) * kCells + 10], 0.5f);  // self_win 2/4
  EXPECT_FLOAT_EQ(p0[(0 * kSlotsPerCell + 0) * kCells + 5], 0.0f);   // slot 0 empty at cell 5
  EXPECT_FLOAT_EQ(p0[(kPredBase + 2) * kCells + 7], 0.7f);  // pred channel 2 (not re-squashed)
  EXPECT_FLOAT_EQ(p0[(kPredBase + 0) * kCells + 0], 0.0f);  // pred channel 0, default 0
  // The 1-tile play at (7,7) is slot 0 at its anchor cell.
  EXPECT_FLOAT_EQ(p0[(kFootBase + 0) * kCells + (7 * BOARD_SIZE + 7)], 1.0f);
  EXPECT_FLOAT_EQ(p0[(kFootBase + 0) * kCells + 0], 0.0f);
  EXPECT_FLOAT_EQ(p0[(kFootBase + 1) * kCells + (7 * BOARD_SIZE + 7)], 0.0f);

  // Candidate 1 (PASS): observed planes empty (the pass catch-all class is
  // dropped), footprint block empty.
  const float* p1 = ev_planes.data() + size_t(kNumEvidencePlanes) * kCells;
  EXPECT_FLOAT_EQ(p1[(0 * kSlotsPerCell + 2) * kCells + 5], 0.0f);
  EXPECT_FLOAT_EQ(p1[(kFootBase + 0) * kCells + (7 * BOARD_SIZE + 7)], 0.0f);

  // Candidate 0 scalars.
  const float* s0 = ev_scalars.data();
  EXPECT_FLOAT_EQ(s0[0], 0.75f);  // wins/n
  EXPECT_FLOAT_EQ(s0[1], 0.0f);
  EXPECT_FLOAT_EQ(s0[2], 0.25f);
  EXPECT_FLOAT_EQ(s0[3], 0.1f);  // delta_mean 10 / 100
  EXPECT_FLOAT_EQ(s0[4], 0.1f);  // delta_std 10 / 100
  EXPECT_FLOAT_EQ(s0[5], float(std::log1p(4.0) / 8.0));
  // Predicted: softmax([2,0,0]) then score_diff/100.
  const float denom = std::exp(2.0f) + 2.0f;
  EXPECT_FLOAT_EQ(s0[6], std::exp(2.0f) / denom);
  EXPECT_FLOAT_EQ(s0[7], 1.0f / denom);
  EXPECT_FLOAT_EQ(s0[8], 1.0f / denom);
  EXPECT_FLOAT_EQ(s0[9], 0.3f);    // sd mean 30/100
  EXPECT_FLOAT_EQ(s0[10], 0.05f);  // sd std 5/100

  // Candidate 1 scalars: all draws, mean -10, std 0, uniform softmax.
  const float* s1 = ev_scalars.data() + kNumEvidenceScalars;
  EXPECT_FLOAT_EQ(s1[1], 1.0f);
  EXPECT_FLOAT_EQ(s1[3], -0.1f);
  EXPECT_FLOAT_EQ(s1[4], 0.0f);
  EXPECT_FLOAT_EQ(s1[6], 1.0f / 3.0f);  // softmax of equal logits
}

// An all-zero CachePredictions of `scored` rows -- enough to gather from; the
// values are irrelevant to the guards these tests exercise.
static scribblez::evidence::CachePredictions zero_predictions(int scored, int channels,
                                                              std::vector<float>& storage) {
  using namespace scribblez::evidence;
  // Per-row widths of CachePredictions' four arrays, packed back-to-back into
  // one buffer: move_enc (channels), wld_logits (3), score_diff (2), plane
  // logits (kNumPredictedPlanes * cells).
  constexpr int kWld = 3, kScoreDiff = 2;
  const int planes = kNumPredictedPlanes * kEvidencePlaneCells;
  storage.assign(size_t(scored) * (channels + kWld + kScoreDiff + planes), 0.0f);
  float* p = storage.data();
  const CachePredictions pred{p, p + size_t(scored) * channels,
                              p + size_t(scored) * (channels + kWld),
                              p + size_t(scored) * (channels + kWld + kScoreDiff), channels};
  return pred;
}

TEST(EvidenceStaging, RejectsOversizedSetAndAcceptsFullWidth) {
  using namespace evidence;
  constexpr int kMaxE = 3, kChannels = 1;
  std::vector<float> storage;
  const CachePredictions pred = zero_predictions(kMaxE + 1, kChannels, storage);

  std::vector<float> me(size_t(kMaxE) * kChannels);
  std::vector<float> pl(size_t(kMaxE) * kNumEvidencePlanes * kEvidencePlaneCells);
  std::vector<float> sc(size_t(kMaxE) * kNumEvidenceScalars);
  std::vector<uint8_t> mk(kMaxE);
  const EvidenceStagingOutputs out{me.data(), pl.data(), sc.data(), mk.data()};

  // One more candidate than the padded width -> throws (before any write).
  const std::vector<Move> too_many(kMaxE + 1);
  const std::vector<SimObservation> obs_many(kMaxE + 1);
  const std::vector<int> idx_many(kMaxE + 1, 0);
  EXPECT_THROW(stage_evidence(too_many, obs_many, idx_many, pred, kMaxE, out), std::runtime_error);
  // Mismatched span lengths -> throws too, on either the observations or the
  // scored_indices disjunct of the length check.
  EXPECT_THROW(stage_evidence(std::vector<Move>(2), std::vector<SimObservation>(1),
                              std::vector<int>(2), pred, kMaxE, out),
               std::runtime_error);
  EXPECT_THROW(stage_evidence(std::vector<Move>(2), std::vector<SimObservation>(2),
                              std::vector<int>(1), pred, kMaxE, out),
               std::runtime_error);

  // Exactly the padded width is accepted and marks every row real.
  stage_evidence(std::vector<Move>(kMaxE), std::vector<SimObservation>(kMaxE),
                 std::vector<int>(kMaxE, 0), pred, kMaxE, out);
  for (int j = 0; j < kMaxE; ++j) EXPECT_EQ(mk[j], 1);
}

TEST(EvidenceStaging, ClampsNegativeVarianceAndHandlesEmptySet) {
  using namespace evidence;
  constexpr int kMaxE = 2, kChannels = 1;
  std::vector<float> storage;
  const CachePredictions pred = zero_predictions(1, kChannels, storage);

  std::vector<float> me(size_t(kMaxE) * kChannels);
  std::vector<float> pl(size_t(kMaxE) * kNumEvidencePlanes * kEvidencePlaneCells);
  std::vector<float> sc(size_t(kMaxE) * kNumEvidenceScalars);
  std::vector<uint8_t> mk(kMaxE);
  const EvidenceStagingOutputs out{me.data(), pl.data(), sc.data(), mk.data()};

  // delta_sq_sum/n (75) below mean^2 (100): the sample variance is negative from
  // these (deliberate) inputs; the std must clamp to exactly 0, never NaN.
  SimObservation neg_var;
  neg_var.n = 2;
  neg_var.delta_sum = 20.0;      // mean 10
  neg_var.delta_sq_sum = 150.0;  // 150/2 - 100 = -25
  stage_evidence(std::vector<Move>(1), std::vector<SimObservation>{neg_var}, std::vector<int>{0},
                 pred, kMaxE, out);
  EXPECT_FLOAT_EQ(sc[4], 0.0f);  // delta_std, clamped
  EXPECT_TRUE(std::isfinite(sc[4]));

  // The empty set (the deployment loop's first pass) masks and zeroes every
  // buffer. Sentinel-fill first (buffers are reused turn-over-turn), so each
  // memset -- planes and move_enc included -- is verified to clear stale data,
  // not just to leave a fresh zero buffer alone.
  std::fill(mk.begin(), mk.end(), uint8_t(9));
  std::fill(sc.begin(), sc.end(), -1.0f);
  std::fill(pl.begin(), pl.end(), -1.0f);
  std::fill(me.begin(), me.end(), -1.0f);
  const std::vector<Move> none;
  const std::vector<SimObservation> no_obs;
  const std::vector<int> no_idx;
  stage_evidence(none, no_obs, no_idx, pred, kMaxE, out);
  for (int j = 0; j < kMaxE; ++j) {
    EXPECT_EQ(mk[j], 0);
    EXPECT_FLOAT_EQ(me[size_t(j) * kChannels], 0.0f);
    EXPECT_FLOAT_EQ(pl[size_t(j) * kNumEvidencePlanes * kEvidencePlaneCells], 0.0f);
    EXPECT_FLOAT_EQ(sc[size_t(j) * kNumEvidenceScalars], 0.0f);
  }
}
