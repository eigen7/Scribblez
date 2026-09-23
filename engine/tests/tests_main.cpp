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
#include "data/gcg_writer.h"
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
#include "lexicon/lexicon.h"
#include "move_key.h"
#include "sim/rollout_summary.h"
#include "sim/setup_plays.h"
#include "sim/sim_runner.h"
#include "sim/slog_position_simmer.h"
#include "training/cross_check_delta.h"
#include "training/evidence_trajectory_select.h"
#include "training/footprint_mask.h"
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
#include "util/assert.h"
#include "util/io.h"
#include "util/math.h"
#include "util/metaprogramming.h"
#include "util/string.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>
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
#include <map>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace scribblez;
using scribblez::testing::key_set;
using scribblez::testing::move_key;

// Offsets into the full input layout, which is what these tests encode. They
// come from the block registry; a null dictionary is fine because the layout
// does not depend on it.
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
static const int kOppReachPlane = spatial_block_plane0(SpatialBlockId::kOppReach);
static const int kSelfReachPlane = spatial_block_plane0(SpatialBlockId::kSelfReach);
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

// A PLAY of consecutive newly placed glyphs starting at (row, col). Score is 0.
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

// A PLAY with gaps for tiles already on the board. `rel_mask` bit 0 is the cell
// at (row, col); `gs` are the newly placed glyphs in word order, one per set bit.
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
  Rack r = rack_from("CATSOHE");
  auto moves = gen.generate(r);
  ASSERT_FALSE(moves.empty());
  // Every opening move must cover the center square.
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
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {
                      Glyph::of(Tile::from_char('C')),
                      Glyph::of(Tile::from_char('A')),
                      Glyph::of(Tile::from_char('T')),
                    }));
  MoveGenerator gen(b, d);
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
  MoveGenerator gen(b, d);
  auto moves = gen.generate(rack_from("PARTIED"));
  // The opening PARTIED on row 7 from the center star rightward: P on the DWS
  // center, I on the DLS at column 11.
  const uint16_t cols_7_to_13 = uint16_t(0x7Fu << CENTER);
  const Move* bingo = nullptr;
  for (const auto& m : moves) {
    if (m.horizontal() && m.start() == CENTER && m.square_mask() == cols_7_to_13) bingo = &m;
  }
  ASSERT_NE(bingo, nullptr);
  ASSERT_EQ(bingo->num_glyphs(), RACK_SIZE);
  const int word_score = 2 * (3 + 1 + 1 + 1 + 2 * 1 + 1 + 2);
  EXPECT_EQ(bingo->score(), word_score + 50);
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

// The GADDAG generator and the reference DAWG generator must enumerate the same
// plays with the same scores from any position. Walks random games, comparing
// the two at every step.
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
      auto kg = key_set(via_gaddag);
      auto kd = key_set(via_dawg);
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
      std::uniform_int_distribution<size_t> pick(0, via_gaddag.size() - 1);
      b.apply(via_gaddag[pick(rng)]);
    }
  }
  std::cout << "  cross-validated " << compared << " positions [" << label << "]\n";
}

// Overlaps, plurals, hooks and 7-letter words, to exercise more of the
// generator than tiny_dict() does.
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

// The board updates its move-generation caches (cross-checks and GADDAG
// anchors) incrementally as moves are applied. After every applied play in a
// random game, they must match a from-scratch recompute of the same squares.
static void check_caches_match_full(const Dictionary& d, const Board& incremental,
                                    const char* label, int game, int step) {
  // set() invalidates the caches, so ensure_movegen_caches() recomputes fully.
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
      auto moves = gen.generate(r);
      check_caches_match_full(d, b, label, g, s);
      ++checked;
      if (moves.empty()) break;
      std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
      b.apply(moves[pick(rng)]);
      check_caches_match_full(d, b, label, g, s);
    }
  }
  std::cout << "  cache-consistency checked " << checked << " positions [" << label << "]\n";
}

TEST(Board, CachesIncrementalMatchesFull) {
  Dictionary d = medium_dict();
  cache_consistency_stress(d, "medium_dict", 99887766u, /*games=*/30, /*steps_per_game=*/10);
}

// num_tiles() is a maintained counter, not a scan, so every square-write path
// must update it: set() in both directions, apply(), and unapply().
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

// GADDAG/DAWG cross-validation on the real NWL23 lexicon. Skipped when the
// .kwg file (installed under /workspace/mount/lexica, never committed) is
// missing.
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

// --lexicon parsed after dict() has loaded would change name() without
// reloading, so it must throw like set_params() does.
TEST(Lexicon, OptionsAfterLoadThrow) {
  Lexicon& lex = Lexicon::instance();
  if (!std::ifstream(lex.kwg_path()).good()) GTEST_SKIP() << "no lexicon at " << lex.kwg_path();
  lex.dict();

  namespace po = boost::program_options;
  po::options_description desc;
  lex.add_options(desc);
  const char* argv[] = {"test", "--lexicon=CSW21"};
  po::variables_map vm;
  po::store(po::parse_command_line(2, argv, desc), vm);
  EXPECT_THROW(po::notify(vm), util::Exception);
  EXPECT_EQ(lex.name(), "NWL23");
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
  // p0 plays C at (7,7) for 50, then p1 plays blank-as-D at (3,3) for 30. The
  // placements are disconnected, which is fine: apply does not check legality.
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
  enc.encode_input(enc.active_player(), active_rack, out.data());

  // Letter planes A..Z are planes 0..25.
  const int c_plane = Tile::from_char('C');
  const int d_plane = Tile::from_char('D');
  ASSERT_EQ(out[c_plane * 225 + 7 * 15 + 7], 1.0f);
  ASSERT_EQ(out[d_plane * 225 + 3 * 15 + 3], 1.0f);  // blank-as-D still lights the D plane

  // The blank-marker plane lights the blank-as-D, not the real C.
  ASSERT_EQ(out[BoardPlanes::kBlankMarkerPlane * 225 + 3 * 15 + 3], 1.0f);
  ASSERT_EQ(out[BoardPlanes::kBlankMarkerPlane * 225 + 7 * 15 + 7], 0.0f);

  // Premium planes: only check that every cell was written as 0 or 1,
  // overwriting the -1.0 fill.
  for (int p = BoardPlanes::kPremiumPlane0;
       p < BoardPlanes::kPremiumPlane0 + BoardPlanes::kPremiumPlanes; ++p) {
    for (int i = 0; i < 225; ++i) {
      float v = out[p * 225 + i];
      ASSERT_TRUE(v == 0.0f || v == 1.0f);
    }
  }

  // The self and opponent last-placement planes each light only that player's
  // most recent move.
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      const float self_expected = (r == 7 && c == 7) ? 1.0f : 0.0f;
      const float opp_expected = (r == 3 && c == 3) ? 1.0f : 0.0f;
      ASSERT_EQ(out[kSelfPlacementPlane * 225 + r * 15 + c], self_expected);
      ASSERT_EQ(out[kOppPlacementPlane * 225 + r * 15 + c], opp_expected);
    }
  }

  const float* scalars = out.data() + kSpatialFloats;

  // The rack block holds raw per-tile counts.
  ASSERT_EQ(scalars[kRackCountOffset + Tile::from_char('Q')], 1.0f);
  ASSERT_EQ(scalars[kRackCountOffset + Tile::from_char('Z')], 1.0f);
  ASSERT_EQ(scalars[kRackCountOffset + 26], 1.0f);  // blank count in rack
  ASSERT_EQ(scalars[kRackCountOffset + Tile::from_char('A')], 0.0f);

  // The unseen pool is a per-letter thermometer over TILE_COUNTS minus the
  // board and the active rack; the opponent's rack stays in the pool.
  const float* pool = scalars + kUnseenPoolOffset;
  float pool_sum = 0.0f;
  for (int i = 0; i < kUnseenPoolThermoFloats; ++i) pool_sum += pool[i];
  ASSERT_EQ(pool_sum, 95.0f);  // 100 - 2 on board - 3 in rack
  // A: all 9 unseen, so the region is full.
  ASSERT_EQ(pool[pool_region_start(0) + 0], 1.0f);
  ASSERT_EQ(pool[pool_region_start(0) + 8], 1.0f);
  // C: 1 of 2 unseen (one is on the board).
  ASSERT_EQ(pool[pool_region_start(Tile::from_char('C')) + 0], 1.0f);
  ASSERT_EQ(pool[pool_region_start(Tile::from_char('C')) + 1], 0.0f);
  // Blank: one on the board (as D) and one in the rack, so none unseen.
  ASSERT_EQ(pool[pool_region_start(26) + 0], 0.0f);
  ASSERT_EQ(pool[pool_region_start(26) + 1], 0.0f);

  const float* sd = scalars + kScoreDiffOffset;
  ASSERT_EQ(sd[0], 20.0f / kScoreDiffInputScale);

  // Last-move metadata: the self move first, then the opponent's.
  const float* meta = scalars + kMoveMetaOffset;
  ASSERT_EQ(meta[int(MoveType::PLAY)], 1.0f);
  ASSERT_EQ(meta[int(MoveType::EXCHANGE)], 0.0f);
  ASSERT_EQ(meta[int(MoveType::PASS)], 0.0f);
  ASSERT_EQ(meta[kMoveMetaTypeFloats], 1.0f);  // self num_glyphs
  const float* opp_meta = meta + kMoveMetaFloatsPerMove;
  ASSERT_EQ(opp_meta[int(MoveType::PLAY)], 1.0f);
  ASSERT_EQ(opp_meta[kMoveMetaTypeFloats], 1.0f);  // opp num_glyphs
}

// An encoder seeded mid-game (as a rollout starts) has no move history, but once
// two applied plies fill both last-move slots its rows are byte-identical to a
// full-history encoder's.
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
  full.encode_input(full.active_player(), active_rack, full_row.data());
  seeded.encode_input(seeded.active_player(), active_rack, seeded_row.data());
  ASSERT_EQ(0, std::memcmp(full_row.data(), seeded_row.data(), sizeof(float) * kInputFloats));
}

TEST(Encoder, LastOppPlaneMask) {
  using namespace scribblez::binlog;

  // p1's CAT plays through p0's A, so only (7,6) and (7,8) are newly placed.
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
  enc.encode_input(enc.active_player(), active_rack, out.data());

  const float* plane = out.data() + kOppPlacementPlane * 225;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      const float expected = ((r == 7 && c == 6) || (r == 7 && c == 8)) ? 1.0f : 0.0f;
      ASSERT_EQ(plane[r * 15 + c], expected);
    }
  }
  // num_glyphs counts placed tiles (C, T), not the word's length.
  const float* opp_meta = out.data() + kSpatialFloats + kMoveMetaOffset + kMoveMetaFloatsPerMove;
  ASSERT_EQ(opp_meta[kMoveMetaTypeFloats], 2.0f);
}

// Two glyphs denote the same square content: both empty, or the same letter
// with the same blank designation.
static bool same_glyph(Glyph a, Glyph b) {
  if (a.is_empty() || b.is_empty()) return a.is_empty() == b.is_empty();
  return a.letter().index() == b.letter().index() && a.is_blank() == b.is_blank();
}

// Board::transpose reflects the squares across the diagonal and toggles the
// frame bit. Its carried-over move-generation caches must equal a rebuild on
// the transposed squares.
TEST(Board, Transpose) {
  Dictionary d = medium_dict();
  Board b;
  const Glyph ax[2] = {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('X'))};
  b.apply(Move::play(/*horizontal=*/true, 3, (1u << 5) | (1u << 6), 0, ax, 2));
  b.ensure_movegen_caches(d);
  ASSERT_FALSE(b.transposed());

  const Board t = b.transpose();
  ASSERT_TRUE(t.transposed());
  ASSERT_EQ(t.num_tiles(), b.num_tiles());
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) ASSERT_TRUE(same_glyph(t.at(r, c), b.at(c, r)));

  Board rebuilt;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) rebuilt.set(c, r, b.at(r, c));
  rebuilt.ensure_movegen_caches(d);
  for (int view = 0; view < 2; ++view) {
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
      const CrossCheck& x = t.cross_checks(view)[i];
      const CrossCheck& y = rebuilt.cross_checks(view)[i];
      ASSERT_TRUE(x.mask == y.mask && x.score == y.score && x.has_neighbor == y.has_neighbor);
      ASSERT_EQ(t.gaddag_anchors(view)[i], rebuilt.gaddag_anchors(view)[i]);
    }
  }

  const Board back = t.transpose();
  ASSERT_FALSE(back.transposed());
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) ASSERT_TRUE(same_glyph(back.at(r, c), b.at(r, c)));
}

// Move::transpose swaps a PLAY's orientation but keeps its start and mask, and
// toggles the frame bit for every move type. In debug builds a board rejects a
// move from the other frame.
TEST(Move, Transpose) {
  const Glyph ax[2] = {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('X'))};
  const Move m = Move::play(/*horizontal=*/true, 3, (1u << 5) | (1u << 6), 30, ax, 2);
  ASSERT_FALSE(m.transposed());
  const Move t = m.transpose();
  ASSERT_TRUE(t.transposed());
  ASSERT_FALSE(t.horizontal());
  ASSERT_EQ(t.start(), m.start());
  ASSERT_EQ(t.square_mask(), m.square_mask());
  ASSERT_EQ(t.score(), m.score());
  ASSERT_EQ(t.num_glyphs(), m.num_glyphs());
  ASSERT_TRUE(t != m);  // the frame bit is part of the value
  ASSERT_TRUE(t.transpose() == m);

  const Move p = Move::pass().transpose();
  ASSERT_EQ(p.type(), MoveType::PASS);
  ASSERT_TRUE(p.transposed());

  if (scribblez::util::kDebugBuild) {
    Board b;
    ASSERT_THROW(b.apply(t), scribblez::util::AssertionError);
  }
}

// Encoding the transposed state equals transposing the encoding: every spatial
// plane transposes, the horizontal and vertical cross-check blocks swap, and the
// scalars are unchanged.
TEST(Encoder, TransposeSymmetry) {
  using namespace scribblez::binlog;

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
  enc.encode_input(enc.active_player(), active_rack, normal.data());
  enc.transpose().encode_input(enc.active_player(), active_rack, flipped.data());

  // Scalars are transpose-invariant.
  for (int i = kSpatialFloats; i < kInputFloats; ++i) {
    ASSERT_EQ(normal[i], flipped[i]);
  }
  // The two cross-check blocks must differ here, or the swap check below would
  // pass vacuously.
  bool halves_differ = false;
  for (int i = 0; i < kHorizontalCrossCheckPlanes * 225 && !halves_differ; ++i) {
    halves_differ =
      normal[kHorizontalCrossCheckPlane0 * 225 + i] != normal[kVerticalCrossCheckPlane0 * 225 + i];
  }
  ASSERT_TRUE(halves_differ);

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

// The two reachability planes equal footprint_reachable_cells over their pools.
// The opponent plane uses the tiles unseen by the mover. The self plane uses
// every unplayed tile minus any the opponent is known to hold; here the
// opponent's rack is hidden, so that is every unplayed tile. The self pool is a
// superset of the opponent pool, so the self plane reaches at least as many
// cells.
TEST(Encoder, ReachabilityPlanes) {
  using namespace scribblez::binlog;
  Move p0_play = make_play_full(7, 7, /*horizontal=*/true, 0b111, 20,
                                {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('T'))});
  Move p1_play =
    make_play_full(6, 8, /*horizontal=*/false, 0b1, 5, {Glyph::of(Tile::from_char('S'))});

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(p0_play);
  enc.apply_move(p1_play);

  Rack active_rack = rack_from("QESTUV");
  std::vector<float> out(kInputFloats, -1.0f);
  enc.encode_input(enc.active_player(), active_rack, out.data());

  Board board = enc.board();
  board.ensure_movegen_caches(d);
  uint8_t opp_pool[27], self_pool[27];
  compute_unseen_pool(opp_pool, board, active_rack);
  Rack empty;
  compute_unseen_pool(self_pool, board, empty);

  std::vector<float> opp_expected(225), self_expected(225);
  footprint_reachable_cells(board, opp_pool, kMaskTileBudget, opp_expected.data());
  footprint_reachable_cells(board, self_pool, kMaskTileBudget, self_expected.data());

  float opp_sum = 0.0f, self_sum = 0.0f;
  for (int i = 0; i < 225; ++i) {
    ASSERT_EQ(out[kOppReachPlane * 225 + i], opp_expected[i]) << "opp cell " << i;
    ASSERT_EQ(out[kSelfReachPlane * 225 + i], self_expected[i]) << "self cell " << i;
    // Every cell is overwritten with 0 or 1; none keeps the -1.0 fill.
    ASSERT_TRUE(out[kOppReachPlane * 225 + i] == 0.0f || out[kOppReachPlane * 225 + i] == 1.0f);
    opp_sum += opp_expected[i];
    self_sum += self_expected[i];
  }
  ASSERT_GE(self_sum, opp_sum);
}

// A single-tile play is generated exactly once. Hooks that form only a vertical
// word (IS, SI through the I of QI) come from the transposed pass; the S of QIS
// comes from the horizontal pass and is not duplicated by the vertical one.
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

// A square's cross-check set constrains plays perpendicular to the run it
// abuts. So the hooks above and below a horizontal QI land in the horizontal
// block (a horizontal play there forms a vertical cross word), and the squares
// left and right of it land in the vertical block.
TEST(Encoder, CrossCheckPlanesQi) {
  using namespace scribblez::binlog;

  Dictionary d = medium_dict();

  Move qi_play = make_play_full(7, 7, /*horizontal=*/true, 0b11, 22,
                                {Glyph::of(Tile::from_char('Q')), Glyph::of(Tile::from_char('I'))});

  GameStateEncoder enc{InputEncodingSpec{&d}};
  enc.apply_move(qi_play);

  Rack active_rack;
  std::vector<float> out(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, out.data());

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

  // A square with no perpendicular neighbor allows every letter.
  const auto assert_horizontal_unconstrained = [&](int r, int c) {
    for (int l = 0; l < 26; ++l) ASSERT_EQ(h_cross_check(Tile::of(l), r, c), 1.0f);
  };
  const auto assert_vertical_unconstrained = [&](int r, int c) {
    for (int l = 0; l < 26; ++l) ASSERT_EQ(v_cross_check(Tile::of(l), r, c), 1.0f);
  };

  // Left and right of QI, the cross word runs across through QI: QIS allows
  // only S to the right, and medium_dict() has no ?QI word.
  assert_vertical_set(7, 9, {'S'});
  assert_vertical_set(7, 6, {});
  assert_horizontal_unconstrained(7, 9);
  assert_horizontal_unconstrained(7, 6);

  // Above and below QI, the cross word runs down. From medium_dict():
  //   below Q: QI; above Q: none
  //   above I: AI BI GI HI KI LI MI OI PI QI SI TI XI
  //   below I: ID IF IN IS IT
  assert_horizontal_set(8, 7, {'I'});
  assert_horizontal_set(6, 7, {});
  assert_horizontal_set(6, 8, {'A', 'B', 'G', 'H', 'K', 'L', 'M', 'O', 'P', 'Q', 'S', 'T', 'X'});
  assert_horizontal_set(8, 8, {'D', 'F', 'N', 'S', 'T'});
  assert_vertical_unconstrained(8, 7);
  assert_vertical_unconstrained(6, 8);
  assert_vertical_unconstrained(8, 8);

  // Occupied squares never carry cross-check planes.
  assert_horizontal_set(7, 7, {});
  assert_horizontal_set(7, 8, {});
  assert_vertical_set(7, 7, {});
  assert_vertical_set(7, 8, {});

  assert_horizontal_unconstrained(0, 0);
  assert_horizontal_unconstrained(14, 14);
  assert_vertical_unconstrained(0, 0);
  assert_vertical_unconstrained(14, 14);
}

// Each cross-check block depends only on the cross word in its own direction,
// not on whether a single tile there forms legal words both ways. At (7,8), the
// run below reads _VOW (AVOW allows A) and the run to the right reads _XI (no
// such word). A lone A there would be illegal, yet AXIOM places one, so the
// horizontal block must still allow A.
TEST(Encoder, CrossCheckSetIsNotOneTileLegality) {
  Dictionary d = Dictionary::build_from_words({"AVOW", "AXIOM", "VOW", "XI"});

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
  enc.encode_input(enc.active_player(), active_rack, out.data());

  const auto at = [&out](int plane0, char ch) {
    return out[(plane0 + Tile::from_char(ch).index()) * 225 + 7 * 15 + 8];
  };
  for (char ch = 'A'; ch <= 'Z'; ++ch) {
    ASSERT_EQ(at(kHorizontalCrossCheckPlane0, ch), (ch == 'A' ? 1.0f : 0.0f)) << ch;
    ASSERT_EQ(at(kVerticalCrossCheckPlane0, ch), 0.0f) << ch;
  }
}

// PositionEncoder, the replay path that produces training rows, must seed the
// board's move-generation caches from its dictionary. Without that, the
// cross-check planes silently degrade to all-letters adjacency masks, so this
// checks a square whose legal hook set is a strict subset.
TEST(PositionEncoder, CrossCheckPlanesLexical) {
  using namespace scribblez::binlog;

  Dictionary d = medium_dict();

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
                                   /*transpose=*/false, row.data());

  auto v_cross_check = [&row](char ch, int r, int c) {
    return row[(kVerticalCrossCheckPlane0 + Tile::from_char(ch).index()) * 225 + r * 15 + c];
  };
  // Right of QI only S hooks (QIS).
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
  enc.encode_input(enc.active_player(), active_rack, normal.data());
  enc.encode_input_with_score_diff(enc.active_player(), active_rack,
                                   /*score_diff=*/123, forced.data());

  const int score_lo = kSpatialFloats + kScoreDiffOffset;
  const int score_hi = score_lo + kScoreDiffInputFloats;

  for (int i = 0; i < kInputFloats; ++i) {
    if (i >= score_lo && i < score_hi) continue;
    ASSERT_EQ(normal[i], forced[i]);
  }

  ASSERT_EQ(forced[score_lo], 123.0f / kScoreDiffInputScale);
}

TEST(Encoder, NonplayLastMoveMetadata) {
  using namespace scribblez::binlog;

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
  enc.encode_input(enc.active_player(), active_rack, out.data());

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

  // Neither last move is a PLAY, so both last-placement planes are empty.
  for (int i = 0; i < 225; ++i) {
    ASSERT_EQ(out[kSelfPlacementPlane * 225 + i], 0.0f);
    ASSERT_EQ(out[kOppPlacementPlane * 225 + i], 0.0f);
  }
}

// ===========================================================================
// Binary-log / DataLoader / movegen round-trip tests
// ===========================================================================

// Plays a random highest-scoring PLAY, else passes. Never exchanges.
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

// The two positions a turn can be sampled at.
enum class PositionKind : uint8_t {
  kPreMove = 0,   // the player is about to move
  kPostMove = 1,  // the player has moved but not yet drawn
};

// One position from an independent replay of a GameLogStorage, taking each
// mover's rack from its logged rack_before. The ground truth that
// GameStateEncoder replays are checked against.
struct LiveSnapshot {
  scribblez::Board board;
  scribblez::Rack rack_active;
  scribblez::Move last_opp_move;
  int score_active = 0;
  int score_opp = 0;
  int turn_index = 0;
  int active_player = 0;
  PositionKind kind = PositionKind::kPreMove;
};

std::vector<LiveSnapshot> live_replay_all_snapshots(const scribblez::GameLogStorage& log) {
  using namespace scribblez;

  std::vector<LiveSnapshot> out;
  Board board;
  Move last_by[2] = {Move{}, Move{}};

  for (size_t i = 0; i < log.turns.size(); ++i) {
    const TurnRecord& turn = log.turns[i];
    const int active = turn.player;
    const int opp = 1 - active;
    const int prev_active = turn.cumulative_scores[active] - turn.score_delta;
    const int prev_opp = turn.cumulative_scores[opp];

    LiveSnapshot pre;
    pre.board = board;
    pre.rack_active = turn.rack_before;
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

    if (turn.move.type() == MoveType::PLAY) board.apply(turn.move);
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

// Exits if movegen yields different play sets (by move_key) on the
// reconstructed and live positions.
void check_movegen_equiv(const scribblez::Dictionary& dict, const scribblez::Board& reconstructed,
                         const scribblez::Rack& reconstructed_rack, const scribblez::Board& live,
                         const scribblez::Rack& live_rack, const char* context) {
  scribblez::MoveGenerator gen_r(reconstructed, dict);
  scribblez::MoveGenerator gen_l(live, dict);
  auto m_r = gen_r.generate(reconstructed_rack);
  auto m_l = gen_l.generate(live_rack);
  auto k_r = key_set(m_r);
  auto k_l = key_set(m_l);
  if (k_r != k_l) {
    std::cerr << "movegen mismatch [" << context << "]: reconstructed=" << k_r.size()
              << " live=" << k_l.size() << "\n";
    std::exit(1);
  }
}
}  // anonymous namespace

// With opp_leave_input, the row is the base row plus a trailing block of the
// opponent's known per-tile rack counts.
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
  base_enc.encode_input(base_enc.active_player(), rack, base_row.data());
  open_enc.encode_input(open_enc.active_player(), rack, opp, open_row.data());

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

// Under a hidden-leaves spec the opponent-leave overload ignores the leave, so
// callers can pass it without branching on the spec.
TEST(InputLayout, HiddenLeavesIgnoresOppLeave) {
  Dictionary d = medium_dict();
  const InputEncodingSpec spec{&d};
  GameStateEncoder enc{spec};
  enc.apply_move(make_play_full(7, 7, /*horizontal=*/true, 0b111, 12,
                                {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('T'))}));
  const Rack rack = rack_from("RSE");

  std::vector<float> plain(input_floats(spec), -1.0f);
  std::vector<float> with_leave(input_floats(spec), -1.0f);
  enc.encode_input(enc.active_player(), rack, plain.data());
  enc.encode_input(enc.active_player(), rack, rack_from("QIZAA"), with_leave.data());
  ASSERT_EQ(plain, with_leave);
}

// Replaying a game log through GameStateEncoder reproduces every position of an
// independent replay: board, scores, last opponent move, and the legal-play set.
TEST(Encoder, ExtractPositionsMovegenRoundtrip) {
  Dictionary dict = medium_dict();

  const std::vector<uint64_t> seeds = {42, 1337, 0xDEADBEEFULL};
  long positions_compared = 0;

  for (uint64_t seed : seeds) {
    scribblez::GameLogStorage log = play_test_game(dict, seed);
    ASSERT_FALSE(log.turns.empty());

    auto live_snaps = live_replay_all_snapshots(log);

    scribblez::GameStateEncoder enc{scribblez::InputEncodingSpec{&dict}};
    // The encoder does not track racks, so the test tracks them alongside it.
    std::array<scribblez::Rack, 2> racks = {log.initial_racks[0], log.initial_racks[1]};

    size_t snap_idx = 0;
    for (size_t k = 0; k < log.turns.size(); ++k) {
      const auto& turn = log.turns[k];

      ASSERT_LT(snap_idx, live_snaps.size());
      const LiveSnapshot& pre = live_snaps[snap_idx++];
      ASSERT_EQ(pre.kind, PositionKind::kPreMove);
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

      if (turn.move.type() == scribblez::MoveType::PLAY) {
        ASSERT_LT(snap_idx, live_snaps.size());
        const LiveSnapshot& post = live_snaps[snap_idx++];
        ASSERT_EQ(post.kind, PositionKind::kPostMove);

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

// End to end: write games through BinaryLogWriter, load the .slog with
// DataLoader, and check that (a) every row's labels match some (game, POV) and
// (b) replaying the raw on-disk turns reproduces every live position.
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

  constexpr int kGames = 3;
  std::vector<scribblez::GameLogStorage> logs;
  {
    scribblez::binlog::BinaryLogWriter writer(dir.string(), /*games_per_file=*/kGames);
    for (int i = 0; i < kGames; ++i) {
      scribblez::GameLogStorage log = play_test_game(dict, /*seed=*/100ULL + i);
      writer.append(scribblez::GameLogStorage(log));
      logs.push_back(std::move(log));
    }
  }  // The writer's destructor flushes the file.

  std::vector<fs::path> slogs;
  for (const auto& ent : fs::directory_iterator(dir)) {
    if (ent.path().extension() == ".slog") slogs.push_back(ent.path());
  }
  ASSERT_EQ(slogs.size(), 1);

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
  // Each turn that began with tiles in the bag is one sample position.
  int64_t total_positions = 0;
  for (const auto& log : logs)
    for (const auto& turn : log.turns)
      if (turn.bag_size_before > 0) ++total_positions;
  ASSERT_GT(total_positions, 0);
  ASSERT_EQ(int64_t(hdr->num_sample_positions), total_positions);

  scribblez::binlog::DataLoader::Params dl_params;
  dl_params.spec = {&dict};
  dl_params.num_worker_threads = 2;
  dl_params.num_prefetch_threads = 1;
  scribblez::binlog::DataLoader loader(dl_params);
  loader.add_file(slog.string(), total_positions, fsize);
  ASSERT_EQ(loader.num_positions(), total_positions);

  const int row_size = kRowFloats;

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

  const int n_samples = int(total_positions) * 2;
  std::vector<float> rows;
  rows.insert(rows.end(), pre_rows.begin(), pre_rows.end());
  rows.insert(rows.end(), post_rows.begin(), post_rows.end());

  std::set<std::tuple<int, int, int, int>> valid_labels;  // (W, D, L, score_diff)
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

  const int label_off = kInputFloats;
  for (int i = 0; i < n_samples; ++i) {
    const float* row = rows.data() + int64_t(i) * row_size;
    const int w = row[label_off + 0];
    const int dd = row[label_off + 1];
    const int l = row[label_off + 2];
    const int sd = row[label_off + scribblez::kWldFloats];
    ASSERT_EQ(w + dd + l, 1);
    ASSERT_EQ(valid_labels.count({w, dd, l, sd}), 1);
  }

  // Replay each game from the raw on-disk records, read the same way DataLoader
  // reads them, and compare against the live snapshots.
  const auto* metas = reinterpret_cast<const scribblez::binlog::GameMetadata*>(
    raw.data() + sizeof(scribblez::binlog::FileHeader));
  long compared = 0;
  for (uint32_t gi = 0; gi < hdr->num_games; ++gi) {
    const auto& gm = metas[gi];
    // Games are stored in append order.
    ASSERT_EQ(gm.num_turns, logs[gi].turns.size());

    const auto* ir =
      reinterpret_cast<const scribblez::binlog::InitialRacks*>(raw.data() + gm.start_offset);
    const auto* turns = reinterpret_cast<const scribblez::binlog::TurnBlob*>(
      raw.data() + gm.start_offset + sizeof(scribblez::binlog::InitialRacks));

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
  ASSERT_GE(compared, total_positions);
  std::cout << "  file+DataLoader round-trip OK (" << kGames << " games, " << total_positions
            << " positions, " << n_samples << " loader rows)\n";
}

// play_test_game with a random opening of `plies` moves.
static scribblez::GameLogStorage play_random_opening_test_game(const scribblez::Dictionary& dict,
                                                               uint64_t seed, int plies) {
  TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
  TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
  scribblez::Game g(a0, a1, dict, seed);
  g.set_random_opening(plies);
  g.play();
  return g.extract_log();
}

// Game::set_random_opening records how many random plies were played (fewer
// only if the game ends sooner), and the game is reproducible from its seed.
TEST(Game, RandomOpening) {
  Dictionary dict = medium_dict();

  for (int plies : {0, 1, 3, 6}) {
    scribblez::GameLogStorage log = play_random_opening_test_game(dict, /*seed=*/321, plies);
    ASSERT_FALSE(log.turns.empty());
    ASSERT_EQ(log.num_random_opening_plies, std::min<int>(plies, int(log.turns.size())));

    scribblez::GameLogStorage again = play_random_opening_test_game(dict, /*seed=*/321, plies);
    ASSERT_EQ(again.turns.size(), log.turns.size());
    for (size_t k = 0; k < log.turns.size(); ++k)
      ASSERT_TRUE(moves_equal_for_replay(again.turns[k].move, log.turns[k].move));
  }
  std::cout << "  Game random opening OK\n";
}

// generate_legal_exchanges yields each distinct non-empty sub-multiset of the
// rack once, and nothing when the bag holds fewer than RACK_SIZE tiles.
TEST(Movegen, GenerateLegalExchanges) {
  Dictionary dict = medium_dict();
  Board board;
  Rack rack;  // AAB?: (2+1)(1+1)(1+1) - 1 = 11 distinct exchanges
  rack.add(Tile::from_char('A'));
  rack.add(Tile::from_char('A'));
  rack.add(Tile::from_char('B'));
  rack.add(BLANK);
  Rack opp;

  MoveRequest req{board, dict, rack, opp, 0, 0, /*bag_size=*/50};
  const std::vector<Move> exchanges = generate_legal_exchanges(req);
  ASSERT_EQ(exchanges.size(), 11);

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

  MoveRequest starved{board, dict, rack, opp, 0, 0, /*bag_size=*/RACK_SIZE - 1};
  ASSERT_TRUE(generate_legal_exchanges(starved).empty());
  std::cout << "  generate_legal_exchanges OK (" << exchanges.size() << " exchanges)\n";
}

// BinaryLogWriter records a random-opening game's eligible turn region: it
// begins at the last random ply and ends after the last turn that began with
// tiles in the bag. The header's num_sample_positions sums the region widths.
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

  // A blank played as Q reads as Q but scores zero and comes from a rack blank.
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

  ASSERT_TRUE(Glyph::empty().is_empty());
  ASSERT_FALSE(Glyph::blank().is_empty());
  ASSERT_TRUE(Glyph::blank().is_blank());
  ASSERT_EQ(Glyph::blank().rack_tile(), BLANK);

  // Code 0 is empty, so zero-initialized storage (e.g. a default Board) is
  // all-empty. Much code relies on this.
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
  return true;
}

TEST(Rack, Invariants) {
  // The tile array stays sorted through adds and removes in arbitrary order.
  Rack r;
  ASSERT_TRUE(r.empty());
  ASSERT_EQ(r.size(), 0);
  ASSERT_EQ(r.point_value(), 0);
  ASSERT_FALSE(r.remove(Tile::from_char('A')));

  const char* in = "QAZZB?A";
  for (char c : std::string(in)) {
    r.add(c == '?' ? BLANK : Tile::from_char(c));
  }
  ASSERT_EQ(r.size(), 7);
  ASSERT_TRUE(rack_is_sorted(r));
  ASSERT_EQ(r.to_string(), "AABQZZ?");
  ASSERT_EQ(r.count(Tile::from_char('A')), 2);
  ASSERT_EQ(r.count(Tile::from_char('Z')), 2);
  ASSERT_EQ(r.count(Tile::from_char('B')), 1);
  ASSERT_EQ(r.count(Tile::from_char('X')), 0);
  ASSERT_EQ(r.blanks(), 1);

  int expected = TILE_VALUES[Tile::from_char('A')] * 2 + TILE_VALUES[Tile::from_char('B')] +
                 TILE_VALUES[Tile::from_char('Q')] + TILE_VALUES[Tile::from_char('Z')] * 2;
  ASSERT_EQ(r.point_value(), expected);

  ASSERT_TRUE(r.remove(Tile::from_char('A')));
  ASSERT_EQ(r.count(Tile::from_char('A')), 1);
  ASSERT_EQ(r.size(), 6);
  ASSERT_TRUE(rack_is_sorted(r));
  ASSERT_TRUE(r.remove(BLANK));
  ASSERT_EQ(r.blanks(), 0);
  ASSERT_TRUE(rack_is_sorted(r));
  ASSERT_FALSE(r.remove(BLANK));

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
  // Bag::kTotalTiles must match TILE_COUNTS. The constructor checks this only
  // with a DEBUG_ASSERT, so this test is what pins it in release builds.
  Bag b(/*seed=*/42);
  int total = 0;
  for (int c : TILE_COUNTS) total += c;
  ASSERT_EQ(b.size(), total);
  ASSERT_EQ(b.size(), 100);
  ASSERT_EQ(b.size(), Bag::kTotalTiles);
  for (int i = 0; i < 27; ++i) ASSERT_EQ(b.counts()[i], TILE_COUNTS[i]);

  // The draw sequence is reproducible from the seed.
  Bag b1(/*seed=*/12345);
  Bag b2(/*seed=*/12345);
  for (int i = 0; i < 100; ++i) {
    auto t1 = b1.draw();
    auto t2 = b2.draw();
    ASSERT_TRUE(t1.has_value() && t2.has_value());
    ASSERT_EQ(*t1, *t2);
  }
  ASSERT_EQ(b1.size(), 0);
  ASSERT_FALSE(b1.draw().has_value());

  Bag bA(1), bB(2);
  bool any_diff = false;
  for (int i = 0; i < 100; ++i) {
    auto a = bA.draw();
    auto bb = bB.draw();
    if (a != bb) any_diff = true;
  }
  ASSERT_TRUE(any_diff);

  // Draining the bag yields exactly TILE_COUNTS.
  std::array<int, 27> drawn{};
  Bag b3(/*seed=*/777);
  while (auto t = b3.draw()) ++drawn[*t];
  for (int i = 0; i < 27; ++i) ASSERT_EQ(drawn[i], TILE_COUNTS[i]);
  ASSERT_EQ(b3.size(), 0);

  Bag b4(/*seed=*/9999);
  while (b4.draw().has_value()) {
  }
  ASSERT_EQ(b4.size(), 0);
  b4.put_back(Tile::from_char('Q'));
  ASSERT_EQ(b4.size(), 1);
  auto got = b4.draw();
  ASSERT_TRUE(got.has_value() && *got == Tile::from_char('Q'));
}

// Board::apply places glyphs only on the square mask's cells, skipping tiles
// already on the board, and a PASS changes nothing.
TEST(Board, ApplyInterleavesCrossTiles) {
  Board b;
  b.set(7, 8, Glyph::of(Tile::from_char('A')));

  // CAT through the existing A.
  Move m = make_play_full(7, 7, /*horizontal=*/true, 0b101, 0,
                          {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('T'))});

  b.apply(m);

  ASSERT_EQ(b.at(7, 7).letter(), Tile::from_char('C'));
  ASSERT_EQ(b.at(7, 8).letter(), Tile::from_char('A'));  // unchanged
  ASSERT_EQ(b.at(7, 9).letter(), Tile::from_char('T'));
  Move pass;
  Board snapshot = b;
  b.apply(pass);
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      ASSERT_EQ(b.at(r, c).code(), snapshot.at(r, c).code());
    }
  }
}

TEST(Move, MainWordThroughCross) {
  Board b;
  b.set(7, 7, Glyph::of(Tile::from_char('C')));
  b.set(7, 8, Glyph::of(Tile::from_char('A')));
  b.set(7, 9, Glyph::of(Tile::from_char('T')));

  // A one-tile hook: main_word extends back through the existing tiles.
  Move hook = make_play_full(7, 10, /*horizontal=*/true, 0b1, 0, {Glyph::of(Tile::from_char('S'))});
  ASSERT_EQ(hook.main_word(b), "CATS");

  // Tiles on both ends interleave with the existing ones. main_word does not
  // check legality, so the non-word BCATS is fine.
  Move through = make_play_full(7, 6, /*horizontal=*/true, 0b10001, 0,
                                {Glyph::of(Tile::from_char('B')),    // placed at (7,6)
                                 Glyph::of(Tile::from_char('S'))});  // placed at (7,10)
  ASSERT_EQ(through.main_word(b), "BCATS");

  // A blank renders as its designated letter.
  Move with_blank = make_play_full(7, 10, /*horizontal=*/true, 0b1, 0,
                                   {Glyph::played(Tile::from_char('S'), /*is_blank=*/true)});
  ASSERT_EQ(with_blank.main_word(b), "CATS");

  Move pass;
  ASSERT_TRUE(pass.main_word(b).empty());
  TileCounts xch_tiles;
  xch_tiles.add(Tile::from_char('A'));
  Move xch = Move::exchange(xch_tiles);
  ASSERT_TRUE(xch.main_word(b).empty());
}

// A blank placed as C scores less than a real C in the same play (CAT onto an
// existing AT).
TEST(Movegen, BlankScoresZero) {
  Dictionary d = Dictionary::build_from_words({"CAT", "CATS", "BAT", "BATS"});

  Board b;
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('T'))}));

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
      ASSERT_EQ(m.num_glyphs(), 1);
      ASSERT_TRUE(m.glyph(0).is_blank());
      score_blank = m.score();
      break;
    }
  }
  ASSERT_GE(score_blank, 0);
  ASSERT_LT(score_blank, score_real);
}

// ===========================================================================
// Game end conditions
// ===========================================================================

namespace {

// Always passes, forcing a stalemate end.
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

// The tiles `player` kept at their most recent turn before `turn`: what
// face-up leaves makes public. Empty before they have acted.
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
    // The variant is a header flag, not a format version: a file with no flags
    // reads as standard Scrabble.
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
  // Needs the real lexicon: on a small dictionary greedy agents almost always
  // stalemate before emptying the bag.
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
    const int winner = last.player;  // the player who went out
    const int loser = 1 - winner;
    // Tournament convention: the player going out gains twice the opponent's
    // remaining tile values; the opponent's score is unchanged.
    const int bonus = 2 * log.final_racks[loser].point_value();

    ASSERT_EQ(log.final_scores[winner], last.cumulative_scores[winner] + bonus);
    ASSERT_EQ(log.final_scores[loser], last.cumulative_scores[loser]);
    ASSERT_TRUE(log.final_racks[winner].empty());
  }
  ASSERT_TRUE(found_out);
}

TEST(Game, EndStalematePenalty) {
  // Six consecutive zero-score turns end the game, and each player loses their
  // own remaining rack value from a score of zero.
  Dictionary dict = medium_dict();
  AlwaysPassAgent a0(0, "P0");
  AlwaysPassAgent a1(0, "P1");
  scribblez::Game g(a0, a1, dict, /*seed=*/424242ULL);
  g.play();
  const scribblez::GameLogStorage log = g.extract_log();

  ASSERT_EQ(log.end_reason, "stalemate");
  ASSERT_EQ(log.turns.size(), 6);
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

// Game::play_from, which starts a rollout from a mid-game position: it keeps
// the known leave, fills both racks from the given pool, plays to a natural
// end, and is deterministic per seed. Also checks Bag::remove, which callers
// use to take known tiles out of the pool.
TEST(Game, PlayFrom) {
  {
    Bag bag(123);
    const Tile a = Tile::from_char('A');
    for (int i = bag.counts()[a.index()]; i > 0; --i) bag.remove(a);
    ASSERT_EQ(bag.counts()[a.index()], 0);
    while (auto t = bag.draw()) ASSERT_NE(t->index(), a.index());
  }

  const Dictionary d = medium_dict();
  const Board board;
  const Rack leave = rack_from("ING");  // seat 0 just moved and kept ING
  const std::array<Rack, 2> known = {leave, Rack{}};
  const std::array<int, 2> scores = {120, 95};

  GameLogStorage logs[2];
  for (int run = 0; run < 2; ++run) {  // the same seed twice
    const uint64_t seed = 7;
    Bag pool(seed);
    for (int i = 0; i < leave.size(); ++i) pool.remove(leave.tiles()[i]);
    TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
    TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
    scribblez::Game g(a0, a1, d, seed);
    g.play_from(board, scores, known, pool, /*to_move=*/1);
    logs[run] = g.extract_log();
  }

  ASSERT_FALSE(logs[0].end_reason.empty());
  ASSERT_EQ(logs[0].initial_racks[0].size(), RACK_SIZE);
  ASSERT_EQ(logs[0].initial_racks[1].size(), RACK_SIZE);
  for (int i = 0; i < leave.size(); ++i)
    ASSERT_TRUE(rack_contains(logs[0].initial_racks[0], leave.tiles()[i]));
  ASSERT_EQ(logs[0].final_scores, logs[1].final_scores);
  ASSERT_EQ(logs[0].end_reason, logs[1].end_reason);
}

// Game::set_max_plies sets a truncated rollout's horizon. Play stops after
// exactly the cap with no end-of-game score adjustment, and leave() gives the
// last mover's rack after the move but before drawing.
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
  std::array<int, 2> expected = scores;
  for (int i = 0; i < log.num_records; ++i)
    expected[log.records[i].player] += log.records[i].score_delta;
  ASSERT_EQ(log.final_scores, expected);
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

// The ply cap never truncates in the endgame: once the bag is empty the game
// plays to a natural end. The leaf model that values a truncated rollout is
// trained only on positions with tiles in the bag (see Game::set_max_plies).
TEST(Game, MaxPliesSparesTheEndgame) {
  const Dictionary d = medium_dict();
  const Board board;
  // Both racks known and two tiles in the bag, so the bag empties within the
  // first couple of plies.
  const std::array<Rack, 2> known = {rack_from("CATSEIQ"), rack_from("RATESIN")};
  const uint64_t seed = 7;
  Bag pool(seed);
  {
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
    // Truncation is allowed only at a ply that began with tiles in the bag.
    ASSERT_GT(log.records[log.num_records - 1].bag_size_before, 0);
  } else {
    // The expected path: the cap fell inside the endgame and was ignored.
    ASSERT_GE(log.num_records, 3);
    ASSERT_TRUE(std::string(log.end_reason) == "out" || std::string(log.end_reason) == "stalemate");
  }
}

// ===========================================================================
// Training targets
// ===========================================================================

namespace {

// Label row layout (AllTargets order):
//   [wld(3), score_diff(1),
//    opp_next(1), self_next(1), opp_win(1), self_win(1),   // footprint class index
//    opp_placement_mask(N), self_placement_mask(N)]        // N = kFootprintClasses
// Each side's legality mask serves both its next and win heads. The mask marks
// kExtraClass illegal; the loss makes it legal for the win head.
constexpr int kClassBase = kWldFloats + kScoreDiffFloats;
constexpr int kMaskBase = kClassBase + 4 * kPlacementClassFloats;

// An EncodeContext with the POV, final scores and encoder set. The next moves
// are unset; callers testing the placement class targets set them.
scribblez::EncodeContext scores_view(const GameStateEncoder& enc, const InputEncodingSpec& spec,
                                     int fs_active, int fs_opp, int active_player) {
  scribblez::EncodeContext v{};
  v.enc = &enc;
  v.spec = spec;
  v.active_player = active_player;
  v.final_score_p0 = active_player == 0 ? fs_active : fs_opp;
  v.final_score_p1 = active_player == 0 ? fs_opp : fs_active;
  return v;
}

// An empty-board encoder for the label tests. Only the legality masks read the
// board, and on an empty board they do not depend on the dictionary.
struct LabelFixture {
  Dictionary dict = Dictionary::build_from_words({"CAT", "CATS", "BAT"});
  InputEncodingSpec spec{&dict};
  GameStateEncoder enc{spec, Board{}, std::array<int, 2>{0, 0}, 0};

  scribblez::EncodeContext view(int fs_active, int fs_opp, int active_player) {
    return scores_view(enc, spec, fs_active, fs_opp, active_player);
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

  // Large differentials are stored unclipped.
  encode_labels_flat(fx.view(620, 0, 0), flat.data());
  check_score_diff(620);
  encode_labels_flat(fx.view(0, 620, 0), flat.data());
  check_score_diff(-620);

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

  // With no next move the next heads are kPassClass, which both masks allow.
  // Both masks mark kExtraClass illegal.
  auto v = fx.view(/*fs_active=*/100, /*fs_opp=*/80, /*active_player=*/0);
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(flat[opp_next], float(kPassClass));
  ASSERT_EQ(flat[self_next], float(kPassClass));
  ASSERT_EQ(opp_mask[kPassClass], 1.0f);
  ASSERT_EQ(self_mask[kPassClass], 1.0f);
  ASSERT_EQ(opp_mask[kExtraClass], 0.0f);
  ASSERT_EQ(self_mask[kExtraClass], 0.0f);

  // An opponent PLAY maps to its footprint class, which the opp mask must allow
  // (a masked-out target would give the loss -log(0)). The class decodes back
  // to the play's cells.
  Move next_play = make_play_full(4, 2, /*horizontal=*/true, 0b111, 0,
                                  {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                                   Glyph::of(Tile::from_char('C'))});
  v.opp_next_move = next_play;
  v.has_opp_next_move = true;
  encode_labels_flat(v, flat.data());
  const int cls = int(flat[opp_next]);
  ASSERT_EQ(cls, footprint_class(next_play));
  ASSERT_LT(cls, kAnchoredFootprints);
  ASSERT_EQ(opp_mask[cls], 1.0f);
  std::array<std::pair<int, int>, kFootprintMaxK> cells;
  const int n = footprint_cells(cls, fx.enc.board(), cells);
  ASSERT_EQ(n, 3);
  ASSERT_EQ(cells[0], std::make_pair(4, 2));
  ASSERT_EQ(cells[1], std::make_pair(4, 3));
  ASSERT_EQ(cells[2], std::make_pair(4, 4));

  // In the transposed frame the class transposes with the move: anchor (4,2)
  // becomes (2,4) and the play becomes vertical.
  const GameStateEncoder enc_t = fx.enc.transpose();
  v.enc = &enc_t;
  v.opp_next_move = next_play.transpose();
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_next]), (2 * 15 + 4) * kSlotsPerCell + (kFootprintMaxK + (3 - 2)));
  v.enc = &fx.enc;
  v.opp_next_move = next_play;

  // An EXCHANGE maps to kPassClass.
  TileCounts xch_tiles;
  xch_tiles.add(Tile::from_char('A'));
  v.opp_next_move = Move::exchange(xch_tiles);
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_next]), kPassClass);

  // Win heads: the played footprint if that seat won, else kExtraClass.
  v.opp_next_move = next_play;

  // The active player (0) is winning, so opp_win is kExtraClass.
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_win]), kExtraClass);

  // With the opponent winning, opp_win equals opp_next.
  v.final_score_p0 = 80;
  v.final_score_p1 = 100;
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_win]), int(flat[opp_next]));
  ASSERT_LT(int(flat[opp_win]), kAnchoredFootprints);

  // The same for the mover's own next play.
  Move self_play =
    make_play_full(7, 3, /*horizontal=*/false, 0b11, 0,
                   {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('E'))});
  v.self_next_move = self_play;
  v.has_self_next_move = true;

  encode_labels_flat(v, flat.data());  // mover (0) losing
  ASSERT_EQ(int(flat[self_next]), footprint_class(self_play));
  ASSERT_EQ(int(flat[self_win]), kExtraClass);
  // The self mask must allow the played footprint too.
  ASSERT_EQ(self_mask[footprint_class(self_play)], 1.0f);
  ASSERT_EQ(self_mask[kExtraClass], 0.0f);

  v.final_score_p0 = 100;
  v.final_score_p1 = 80;  // mover wins
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[self_next]), footprint_class(self_play));
  ASSERT_EQ(int(flat[self_win]), int(flat[self_next]));

  // A draw counts as not winning for both conjunctions; the marginals hold.
  v.final_score_p0 = 90;
  v.final_score_p1 = 90;
  encode_labels_flat(v, flat.data());
  ASSERT_EQ(int(flat[opp_win]), kExtraClass);
  ASSERT_EQ(int(flat[self_win]), kExtraClass);
  ASSERT_EQ(int(flat[opp_next]), footprint_class(next_play));
  ASSERT_EQ(int(flat[self_next]), footprint_class(self_play));
}

// ===========================================================================
// DataLoader
// ===========================================================================

// A one-game .slog with a single eligible turn: p0 plays Q at (3,5), then p1
// passes. The loader's one row is turn 0 post-move, from p0's POV with a leave
// of six As. The Q sits off the diagonal, so the position is not
// transpose-symmetric. The other fields let the test build the reference
// encoding.
struct SymFixture {
  std::filesystem::path path;
  int64_t fsize;
  scribblez::Rack active_rack;
  scribblez::Move self_move;
  int final_score_p0;
  int final_score_p1;
  int active_player;
};

static SymFixture write_one_position_slog(const std::filesystem::path& dir) {
  using namespace scribblez::binlog;
  using namespace scribblez;

  // No draws are recorded, so p0's rack after the Q is exactly the six As.
  Rack p0_init;
  p0_init.add(Tile::from_char('Q'));
  for (int i = 0; i < 6; ++i) p0_init.add(Tile::from_char('A'));
  Rack p1_init;

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
  hdr.num_sample_positions = 1;

  GameMetadata gm{};
  gm.start_offset = sizeof(FileHeader) + sizeof(GameMetadata);
  gm.num_turns = 2;
  gm.sampled_turn = 0;  // eval-only; training uses the eligible region
  gm.eligible_begin = 0;
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

  std::vector<float> ref_normal(kInputFloats, 0.0f);
  std::vector<float> ref_flipped(kInputFloats, 0.0f);
  {
    GameStateEncoder ref_enc{InputEncodingSpec{&dict}};
    ref_enc.apply_move(fix.self_move);
    ref_enc.encode_input(fix.active_player, fix.active_rack, ref_normal.data());
    ref_enc.transpose().encode_input(fix.active_player, fix.active_rack, ref_flipped.data());
  }
  ASSERT_NE(std::memcmp(ref_normal.data(), ref_flipped.data(), kInputFloats * sizeof(float)), 0);

  // The labels are not transpose-invariant either: the placement class targets
  // are all pass/not-win here, but the legality masks read the asymmetric board,
  // so a flipped row carries transposed masks.
  GameStateEncoder label_enc{InputEncodingSpec{&dict}};
  label_enc.apply_move(fix.self_move);
  const GameStateEncoder label_enc_t = label_enc.transpose();
  const InputEncodingSpec label_spec{&dict};
  float ref_labels[kLabelFloats];
  float ref_labels_flipped[kLabelFloats];
  encode_labels_flat(
    scores_view(label_enc, label_spec, fix.final_score_p0, fix.final_score_p1, fix.active_player),
    ref_labels);
  encode_labels_flat(
    scores_view(label_enc_t, label_spec, fix.final_score_p0, fix.final_score_p1, fix.active_player),
    ref_labels_flipped);
  ASSERT_NE(std::memcmp(ref_labels, ref_labels_flipped, kLabelFloats * sizeof(float)), 0);

  DataLoader::Params params;
  params.spec = {&dict};
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);
  loader.add_file(fix.path.string(), /*num_positions=*/1, fix.fsize);

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

  // With apply_symmetry, the transpose is a per-row coin flip seeded by the
  // epoch seed, so many seeds must produce both frames.
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
      ASSERT_TRUE(is_normal || is_flipped);
      // The labels must be in the same frame as the input.
      const float* want = is_normal ? ref_labels : ref_labels_flipped;
      if (is_normal)
        ++normal_count;
      else
        ++flipped_count;
      ASSERT_EQ(std::memcmp(row.data() + kInputFloats, want, kLabelFloats * sizeof(float)), 0);
    }
    // A spurious failure needs 200 identical fair flips: probability 2^-199.
    ASSERT_GT(normal_count, 0);
    ASSERT_GT(flipped_count, 0);
    std::cout << "  DataLoader per-row symmetry: " << normal_count << " normal / " << flipped_count
              << " flipped (of " << n << ")\n";
  }
}

// A game's rows start at eligible_begin: with the region [1, 2), the single row
// is the turn-1 post-move position (POV p1), not turn 0.
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

  // p0 plays Q, then p1 plays C; each keeps six As.
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
  gm.eligible_begin = 1;
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

  std::vector<float> ref_row(kInputFloats, 0.0f);
  {
    GameStateEncoder ref_enc{InputEncodingSpec{&dict}};
    ref_enc.apply_move(q_play);
    ref_enc.apply_move(c_play);
    Rack leave;
    for (int i = 0; i < 6; ++i) leave.add(Tile::from_char('A'));
    ref_enc.encode_input(/*active_player=*/1, leave, ref_row.data());
  }
  // The legality masks read the board, so the label encoder must also be in the
  // turn-1 post-move state.
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

// Self-play games written through BinaryLogWriter into `num_files` .slog files.
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

// An epoch's rows are a function of its seed alone, across loaders and across
// repeated epochs on one loader.
TEST(DataLoader, EpochDeterminism) {
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

  DataLoader loader1(params);
  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = fs::file_size(p);
    loader1.add_file(p.string(), hdr.num_games, fsize);
  }
  auto data1 = run_epoch(loader1);

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

  auto data3 = run_epoch(loader1);
  ASSERT_EQ(data3.size(), data1.size());
  ASSERT_EQ(std::memcmp(data1.data(), data3.data(), data1.size() * sizeof(float)), 0);

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

// Each epoch visits every row exactly once: two differently seeded epochs hold
// the same rows in different orders.
TEST(DataLoader, EpochCoverage) {
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
  // Each game has several eligible turns, so rows outnumber games.
  const int64_t total_positions = loader.num_positions();
  ASSERT_GT(total_positions, fix.total_games);

  const int row_sz = kRowFloats;
  auto drain_epoch = [&](uint64_t seed) {
    DataLoader::EpochConfig cfg;
    cfg.batch_size = 3;
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

  ASSERT_EQ(int64_t(epoch1.size()), total_positions * row_sz);
  ASSERT_EQ(int64_t(epoch2.size()), total_positions * row_sz);

  ASSERT_NE(std::memcmp(epoch1.data(), epoch2.data(), epoch1.size() * sizeof(float)), 0);

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

// A memory budget of one file forces the loader to evict and reload files
// throughout a shuffled epoch. Every row must still come out, deterministically.
TEST(DataLoader, EpochMemoryBudgetStress) {
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

  Dictionary dict = medium_dict();
  DataLoader::Params params;
  params.spec = {&dict};
  params.memory_budget = max_fsize + 1;
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);

  for (int i = 0; i < int(fix.slog_paths.size()); ++i) {
    loader.add_file(fix.slog_paths[i].string(), file_info[i].first, file_info[i].second);
  }
  const int64_t total_positions = loader.num_positions();

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
    // Residency may exceed the one-file budget by at most one more file.
    ASSERT_LE(loader.resident_bytes(), 2 * max_fsize + 100);
  }
  ASSERT_EQ(rows_decoded, total_positions);

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
    cfg.apply_symmetry = false;
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
  auto d3 = run_with_seed(100);

  ASSERT_EQ(d1.size(), d2.size());
  ASSERT_EQ(d1.size(), d3.size());
  ASSERT_GT(d1.size(), 0);
  ASSERT_EQ(std::memcmp(d1.data(), d3.data(), d1.size() * sizeof(float)), 0);
  ASSERT_NE(std::memcmp(d1.data(), d2.data(), d1.size() * sizeof(float)), 0);

  std::cout << "  epoch seed-shuffle OK\n";
}

// =========================================================================
// LeaveValues and HastyEquity tests
// =========================================================================

// A minimal .klv2 holding three single-tile leaves: ? = 12.0, A = 1.5 and
// B = -2.5. Macondo's leave KWG numbers the blank as machine letter 0, ahead of
// the letters, so the blank entry exercises that mapping.
// KWG node bits: 0..21 arc_index, 22 is_end, 23 accepts, 24..31 tile.
struct KlvFixture {
  std::filesystem::path path;
};

KlvFixture write_synthetic_klv(const std::filesystem::path& dir) {
  std::filesystem::path p = dir / "synthetic.klv2";
  std::ofstream f(p, std::ios::binary | std::ios::trunc);

  auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto write_f32 = [&](float v) { f.write(reinterpret_cast<const char*>(&v), 4); };

  write_u32(4);                                          // node count
  write_u32((0u << 24) | (1u << 22) | (0u << 23) | 1u);  // root: arcs start at node 1
  write_u32((0u << 24) | (0u << 22) | (1u << 23) | 0u);  // ?
  write_u32((1u << 24) | (0u << 22) | (1u << 23) | 0u);  // A
  write_u32((2u << 24) | (1u << 22) | (1u << 23) | 0u);  // B, last sibling
  write_u32(3);                                          // leave count, then values in word order
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

  Rack a;
  a.add(Tile::from_char('A'));
  ASSERT_LT(std::abs(lv.lookup(a) - 1.5f), 1e-4f);

  Rack b;
  b.add(Tile::from_char('B'));
  ASSERT_LT(std::abs(lv.lookup(b) - (-2.5f)), 1e-4f);

  // A mis-mapped blank would silently look up as 0.
  Rack blank;
  blank.add(BLANK);
  ASSERT_LT(std::abs(lv.lookup(blank) - 12.0f), 1e-4f);

  Rack empty;
  ASSERT_EQ(lv.lookup(empty), 0.0f);

  // A leave missing from the table is worth 0.
  Rack c;
  c.add(Tile::from_char('C'));
  ASSERT_EQ(lv.lookup(c), 0.0f);

  fs::remove_all(tmp);
}

TEST(LeaveValues, RealKwg) {
  // The NWL23 leave values ship with the Macondo checkout.
  const std::string klv_path = HastyEquity::default_leaves_path("NWL23");
  if (!std::filesystem::exists(klv_path)) {
    GTEST_SKIP() << "no leaves.klv2 at " << klv_path;
  }

  LeaveValues lv = LeaveValues::load(klv_path);

  Rack blank_leave;
  blank_leave.add(BLANK);
  float blank_val = lv.lookup(blank_leave);
  ASSERT_GT(blank_val, 20.0f);  // about 25 in Macondo's NWL23 table

  Rack empty;
  ASSERT_EQ(lv.lookup(empty), 0.0f);
}

// A named pre-endgame table that is missing or malformed is a setup error,
// not a silent opt-out of the adjustment.
TEST(HastyEquity, BadPegFileThrows) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_heq_badpeg";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);

  EXPECT_THROW(HastyEquity::init(fix.path.string(), (tmp / "missing.json").string()),
               util::Exception);
  const fs::path malformed = tmp / "malformed.json";
  std::ofstream(malformed) << "{}";
  EXPECT_THROW(HastyEquity::init(fix.path.string(), malformed.string()), util::Exception);
  fs::remove_all(tmp);
}

TEST(HastyEquity, Components) {
  // The equity components one at a time, on the synthetic leaves and an empty
  // pre-endgame table (so the PEG term is always 0).
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_heq_XXXXXX";
  fs::create_directories(tmp);

  KlvFixture fix = write_synthetic_klv(tmp);
  std::filesystem::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }

  HastyEquity::init(fix.path.string(), peg_path.string());

  const HastyEquity& eq = HastyEquity::instance();
  Board board;
  Rack opp;

  // Opening adjustment: a seven-A opening over columns 4..10.
  Move all_out = make_play_full(7, 4, /*horizontal=*/true, 0b1111111, 50,
                                {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('A')),
                                 Glyph::of(Tile::from_char('A'))});

  Rack rack_7a;
  for (int i = 0; i < 7; ++i) rack_7a.add(Tile::from_char('A'));

  // The leave is empty (worth 0). Columns 6 and 8 are penalised opening
  // squares, each holding a vowel: 2 * -0.7 = -1.4.
  double e_mid = eq.equity(all_out, board, 86, opp, rack_7a);
  ASSERT_LT(std::abs(e_mid - (50.0 - 1.4)), 1e-3);

  // An opening A on the center column, which is not penalised.
  Move one_a = make_play_full(7, 7, /*horizontal=*/true, 0b1, 2, {Glyph::of(Tile::from_char('A'))});

  Rack rack_1a;
  rack_1a.add(Tile::from_char('A'));
  double e_one = eq.equity(one_a, board, 86, opp, rack_1a);
  ASSERT_LT(std::abs(e_one - 2.0), 1e-3);

  // Endgame, not going out: the leave's KLV value is ignored, and the mover pays
  // twice the leave's tile points plus 10. Leaving B (3 points): -2 * 3 - 10.
  Move play_a_endgame =
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 2, {Glyph::of(Tile::from_char('A'))});

  Rack rack_ab;
  rack_ab.add(Tile::from_char('A'));
  rack_ab.add(Tile::from_char('B'));
  Board board_with_tiles;  // not an opening, so no opening adjustment
  board_with_tiles.set(0, 0, Glyph::of(Tile::from_char('Q')));
  double e_eg = eq.equity(play_a_endgame, board_with_tiles, 0, opp, rack_ab);
  ASSERT_LT(std::abs(e_eg - (2.0 - 16.0)), 1e-3);

  // Endgame, going out: gain twice the opponent's rack, here a Q (10 points).
  Move out_play =
    make_play_full(7, 7, /*horizontal=*/true, 0b11, 5,
                   {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B'))});

  Rack opp_q;
  opp_q.add(Tile::from_char('Q'));
  double e_out = eq.equity(out_play, board_with_tiles, 0, opp_q, rack_ab);
  ASSERT_LT(std::abs(e_out - (5.0 + 20.0)), 1e-3);

  fs::remove_all(tmp);
}

// A mid-game EXCHANGE's equity is just the value of the kept leave, so a
// mis-keyed blank leave shows up directly as a wrong (typically 0) exchange
// equity. The single-move and batched paths must both get it right.
TEST(HastyEquity, ExchangeBlankLeave) {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_heq_xch_XXXXXX";
  fs::create_directories(tmp);

  KlvFixture fix = write_synthetic_klv(tmp);  // ? = 12.0
  std::filesystem::path peg_path = tmp / "peg.json";
  {
    std::ofstream pf(peg_path);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg_path.string());
  const HastyEquity& eq = HastyEquity::instance();

  Board board;
  Rack opp;

  // Exchanging the A keeps the blank.
  Rack rack_a_blank;
  rack_a_blank.add(Tile::from_char('A'));
  rack_a_blank.add(BLANK);

  TileCounts surrender_a;
  surrender_a.add(Tile::from_char('A'));
  Move exch_a = Move::exchange(surrender_a);

  double single = eq.equity(exch_a, board, 50, opp, rack_a_blank);
  ASSERT_LT(std::abs(single - 12.0), 1e-3);

  std::vector<Move> moves{exch_a};
  std::vector<double> batched = eq.equities(moves, board, 50, opp, rack_a_blank);
  ASSERT_EQ(batched.size(), 1);
  ASSERT_LT(std::abs(batched[0] - 12.0), 1e-3);

  fs::remove_all(tmp);
}

// A row encoded straight from a live game's log (the streaming path) is
// bit-identical to the row decoded after writing the game to a .slog (the disk
// path). Both go through PositionEncoder, so a mismatch means the two log views
// differ.
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

    // The writer picks the sampled turn that both paths encode.
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
      enc.encode_row<PositionEvalTask>(storage.view(), sampled, post_move, /*transpose=*/false,
                                       row_stream.data());

      for (int i = 0; i < row_floats; ++i) ASSERT_EQ(row_disk[i], row_stream[i]);
      ++compared;
    }

    // Empty the directory so the next seed's lookup finds only its own file.
    for (const auto& ent : fs::directory_iterator(dir)) fs::remove(ent.path());
  }
  ASSERT_EQ(compared, 6);
  std::cout << "  streaming/disk encode equivalence OK (" << compared << " rows)\n";
}

// Many producers and tiny slots, so rows often straddle slot boundaries. Every
// row index is written and read exactly once, and the consumed rows are exactly
// [0, total), which a slot overwritten while the consumer held it would break.
TEST(StreamingRowBuffer, Concurrency) {
  using namespace scribblez::binlog;
  const int n_slots = 2, rows_per_slot = 4, row_floats = 1;
  const int slots_to_consume = 64;
  std::vector<std::vector<float>> bufs(n_slots,
                                       std::vector<float>(rows_per_slot * row_floats, -1.0f));
  std::vector<float*> slots;
  for (auto& b : bufs) slots.push_back(b.data());
  StreamingRowBuffer ring(slots.data(), n_slots, rows_per_slot, row_floats);

  // Cap production at exactly the rows the consumer will read. Unbounded
  // producers could fill later slot generations before earlier ones, and the
  // first slots_to_consume slots read would then not be rows [0, total).
  const uint64_t total_rows = uint64_t(slots_to_consume) * rows_per_slot;
  std::atomic<uint64_t> work{0};
  const int K = 8;
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

  ASSERT_FALSE(dup);
  ASSERT_EQ(int(seen.size()), slots_to_consume * rows_per_slot);
  for (uint64_t v = 0; v < total_rows; ++v) ASSERT_EQ(seen.count(v), 1);
  std::cout << "  StreamingRowBuffer concurrency OK (" << seen.size() << " rows, K=" << K << ")\n";
}

// stop() wakes every producer blocked on a full ring, and the consumer's
// wait_full_slot() then returns -1.
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

// pick_sampled_turn chooses only turns in the eligible region (see
// GameMetadata::eligible_begin) and returns -1 when it is empty.
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

  // Two random plies: the region starts at the last one, turn 1.
  s.num_random_opening_plies = 2;
  ASSERT_EQ(eligible_span(s.view()).begin, 1);
  for (int i = 0; i < 20; ++i) ASSERT_EQ(pick_sampled_turn(s.view(), rng), 1);

  // A game that ended during its random opening has an empty eligible region.
  s.num_random_opening_plies = 3;
  ASSERT_EQ(pick_sampled_turn(s.view(), rng), -1);
  s.num_random_opening_plies = 0;

  GameLogStorage z;
  z.turns.resize(2);  // every bag_size_before is 0
  ASSERT_EQ(pick_sampled_turn(z.view(), rng), -1);

  // pick_any_turn, which the max-move-per-lane task samples with, ignores bag
  // size and the random opening.
  std::array<bool, 3> seen{};
  for (int i = 0; i < 200; ++i) {
    const int t = pick_any_turn(s.view(), rng);
    ASSERT_TRUE(t >= 0 && t < 3);
    seen[t] = true;
  }
  ASSERT_TRUE(seen[0] && seen[1] && seen[2]);
  GameLogStorage empty;
  ASSERT_EQ(pick_any_turn(empty.view(), rng), -1);
  std::cout << "  pick_sampled_turn / pick_any_turn eligibility OK\n";
}

// ShadowMoveGen's per-anchor generation, run over every anchor, yields exactly
// MoveGenerator::generate's plays. Each anchor's score bound must be admissible
// (never below a play generated there), which is what makes best-first pruning
// by the bound exact.
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
          ASSERT_LE(int(m.score()), a.score_bound_by_size[m.num_glyphs()]);
        }
        for (Move& m : am) shadow.push_back(std::move(m));
      }

      ASSERT_EQ(key_set(full), key_set(shadow));

      ++positions;
      total_moves += full.size();
      board.apply(t.move);
    }
  }
  ASSERT_GT(positions, 0);
  std::cout << "  ShadowMoveGen matches full generate + bound admissible (" << positions
            << " positions, " << total_moves << " moves)\n";
}

// HastyBot's pruned search picks the same move as full generation plus an
// equity argmax, across NWL23 self-play games. Skipped without the NWL23 lexicon
// and leaves.
namespace {
class ShadowCheckAgent : public scribblez::Agent {
 public:
  ShadowCheckAgent(int tid, const std::string& name)
      : scribblez::Agent(tid, name), bot_({.thread_id = tid, .name = name}) {}
  scribblez::MoveDecision make_move(const scribblez::MoveRequest& req) override {
    const scribblez::Move shadow = bot_.make_move(req).move;
    const scribblez::Move ref = scribblez::hasty_best_move_reference(req);
    EXPECT_EQ(move_key(shadow), move_key(ref));
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

// For blank-free racks, WordMap generation (wmp_generate) yields the same plays
// as the GADDAG generator, as a whole, per anchor, and per extent.
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
        ASSERT_EQ(key_set(full), key_set(wmp));

        // Per anchor, so WordMap generation can drive the shadow best-first loop.
        ShadowMoveGen smg(board, dict);
        WmpSubracks subracks;
        int rack_tiles = 0;
        wmp_rack_subracks(rack, subracks, rack_tiles);
        for (const ShadowAnchor& a : smg.anchors(rack)) {
          std::vector<Move> g, w;
          smg.generate_anchor(a, rack, g);
          wmp_generate_anchor(board, wm, subracks, rack_tiles, a, w);
          ASSERT_EQ(key_set(g), key_set(w));
        }

        // Together the extents yield exactly the full play set, and each extent's
        // score bound is admissible, so the best-first early exit over extents is
        // exact.
        std::vector<Move> extent_union;
        for (const ShadowExtent& e : smg.extents(rack, &wm)) {
          std::vector<Move> em;
          wmp_generate_extent(board, wm, subracks, e, em);
          for (const Move& m : em) {
            ASSERT_LE(int(m.score()), e.score_bound);
            extent_union.push_back(m);
          }
        }
        ASSERT_EQ(key_set(full), key_set(extent_union));
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

// On NWL23 positions from HastyBot self-play, WordMap generation yields the
// GADDAG's play set for blank-free racks, and HastyBot picks the same move as
// hasty_best_move_wmp for every rack. Skipped without the NWL23 lexicon and
// leaves.
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

  // Racks with a blank fall back to the GADDAG path.
  HastyBotAgent blank_bot({.thread_id = 0, .name = "blankcheck"});
  for (const CapturedPos& p : blanked) {
    const MoveRequest req{p.board, dict, p.rack, p.opp_rack, p.my_score, p.opp_score, p.bag_size};
    ASSERT_EQ(move_key(blank_bot.make_move(req).move), move_key(hasty_best_move_wmp(req)));
  }

  HastyBotAgent bot({.thread_id = 0, .name = "wmpcheck"});
  long total_moves = 0;
  for (const CapturedPos& p : positions) {
    MoveGenerator gen(p.board, dict);
    const std::vector<Move> full = gen.generate(p.rack);
    const std::vector<Move> wmp = wmp_generate(p.board, dict, wm, p.rack);
    ASSERT_EQ(key_set(full), key_set(wmp));
    total_moves += full.size();

    const MoveRequest req{p.board, dict, p.rack, p.opp_rack, p.my_score, p.opp_score, p.bag_size};
    ASSERT_EQ(move_key(bot.make_move(req).move), move_key(hasty_best_move_wmp(req)));
  }
  ASSERT_FALSE(positions.empty());
  std::cout << "  WMP/GADDAG equivalence OK (" << positions.size() << " blank-free + "
            << blanked.size() << " blanked positions, " << total_moves << " plays)\n";
}

TEST(Util, Helpers) {
  ASSERT_EQ(util::round_up_pow2(0), 1);
  ASSERT_EQ(util::round_up_pow2(1), 1);
  ASSERT_EQ(util::round_up_pow2(2), 2);
  ASSERT_EQ(util::round_up_pow2(3), 4);
  ASSERT_EQ(util::round_up_pow2(5), 8);
  ASSERT_EQ(util::round_up_pow2(8), 8);
  ASSERT_EQ(util::round_up_pow2(9), 16);
  ASSERT_EQ(util::round_up_pow2(1u << 20), (1u << 20));
  ASSERT_EQ(util::round_up_pow2((1u << 20) + 1), (1u << 21));

  ASSERT_EQ(util::align_up(0, 8), 0);
  ASSERT_EQ(util::align_up(1, 8), 8);
  ASSERT_EQ(util::align_up(7, 8), 8);
  ASSERT_EQ(util::align_up(8, 8), 8);
  ASSERT_EQ(util::align_up(9, 8), 16);
  ASSERT_EQ(util::align_up(7, 1), 7);

  int sum_dr = 0, sum_dc = 0;
  for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
    ASSERT_NE((dr == 0), (dc == 0));
    ASSERT_TRUE(dr >= -1 && dr <= 1 && dc >= -1 && dc <= 1);
    sum_dr += dr;
    sum_dc += dc;
  }
  ASSERT_TRUE(sum_dr == 0 && sum_dc == 0);
}

// HastyEquity::equities() (batched) agrees value for value with equity() (one
// move at a time), so both give the same argmax. A top-k=1 agent that ranks with
// the batched call then plays exactly HastyBot's move; this checks that without
// linking TensorRT.
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
  Board board;
  MoveGenerator gen(board, d);
  Rack my_rack = rack_from("CATSOHE");
  std::vector<Move> plays = gen.generate(my_rack);
  ASSERT_GE(plays.size(), 2);

  Rack opp;
  const int bag_size = 80;

  std::vector<double> batch = eq.equities(plays, board, bag_size, opp, my_rack);
  ASSERT_EQ(batch.size(), plays.size());
  std::vector<double> per_move(plays.size());
  for (size_t i = 0; i < plays.size(); ++i) {
    per_move[i] = eq.equity(plays[i], board, bag_size, opp, my_rack);
    ASSERT_LT(std::abs(batch[i] - per_move[i]), 1e-9);
  }

  int hasty_pick = 0;
  for (size_t i = 1; i < per_move.size(); ++i) {
    if (per_move[i] > per_move[hasty_pick]) hasty_pick = int(i);
  }
  int topk1_pick = 0;
  for (size_t i = 1; i < batch.size(); ++i) {
    if (batch[i] > batch[topk1_pick]) topk1_pick = int(i);
  }
  ASSERT_EQ(hasty_pick, topk1_pick);

  fs::remove_all(tmp);
}

// With no legal PLAY, HastyBot exchanges rather than passes whenever the bag
// allows it: both score 0, but an exchange gives a chance at a better rack.
// DDGPTWZ has no play on an empty board with tiny_dict().
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

  Dictionary dict = tiny_dict();
  Board board;
  Rack rack = rack_from("DDGPTWZ");
  Rack opp;

  MoveRequest req{board, dict, rack, opp, 0, 0, /*bag_size=*/80};
  ASSERT_TRUE(generate_legal_plays(req).empty());

  HastyBotAgent agent(HastyBotAgent::Params{.thread_id = 0, .name = "Hasty"});
  const Move chosen = agent.make_move(req).move;
  ASSERT_EQ(chosen.type(), MoveType::EXCHANGE);

  fs::remove_all(tmp);
}

// The same on a real mid-game board, from HastyBot self-play on NWL23: after
// these 26 turns the mover's AEFIORX has no legal PLAY. Skipped without the
// NWL23 lexicon and leaves.
TEST(HastyBotAgent, ExchangesOnRealMidGamePositionWithNoLegalPlay) {
  const std::string kwg_path = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves_path = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(kwg_path).good() || !std::ifstream(leaves_path).good()) {
    GTEST_SKIP() << "no NWL23 kwg/leaves";
  }
  // init(), not ensure_initialized(): other tests in this binary load synthetic
  // leaves into the process-wide singleton, and ensure_initialized() would keep
  // them.
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
  ASSERT_TRUE(generate_legal_plays(req).empty());

  HastyBotAgent agent(HastyBotAgent::Params{.thread_id = 0, .name = "Hasty"});
  const Move chosen = agent.make_move(req).move;
  ASSERT_EQ(chosen.type(), MoveType::EXCHANGE);
}

// HastyBot weighs exchanges against plays by equity, not only as a fallback
// when no play exists. IIIIIIH's only play, HI, keeps IIIII, and with the real
// NWL23 leaves that is bad enough that exchanging must win.
TEST(HastyBotAgent, ExchangesDuplicateHeavyRackOverItsOnlyPlay) {
  const std::string kwg_path = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves_path = HastyEquity::default_leaves_path("NWL23");
  if (!std::ifstream(kwg_path).good() || !std::ifstream(leaves_path).good()) {
    GTEST_SKIP() << "no NWL23 kwg/leaves";
  }
  // init(), not ensure_initialized(): other tests in this binary load synthetic
  // leaves into the process-wide singleton, and ensure_initialized() would keep
  // them.
  HastyEquity::init(leaves_path, HastyEquity::default_peg_path());
  Dictionary dict = Dictionary::load_kwg(kwg_path);

  Board board;
  Rack rack = rack_from("IIIIIIH");
  Rack opp;

  MoveRequest req{board, dict, rack, opp, 0, 0, /*bag_size=*/80};
  ASSERT_FALSE(generate_legal_plays(req).empty());

  HastyBotAgent agent(HastyBotAgent::Params{.thread_id = 0, .name = "Hasty"});
  const Move chosen = agent.make_move(req).move;
  ASSERT_EQ(chosen.type(), MoveType::EXCHANGE);
}

// ===========================================================================
// SimRunner + sim-observation log
// ===========================================================================

// play_from's returned_to_bag (tiles the mover just exchanged) joins the bag
// only after both racks are filled, and every tile is conserved through the
// game.
TEST(Game, PlayFromReturnedToBag) {
  const Dictionary d = medium_dict();
  const Board board;

  // Seat 0 exchanged Q and Z and kept AB.
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

  // The distribution has one Q and one Z, both out of the pool at refill time.
  const GameLog log = g.log();
  for (int p = 0; p < 2; ++p) {
    ASSERT_FALSE(rack_contains(log.initial_racks[p], Tile::from_char('Q')));
    ASSERT_FALSE(rack_contains(log.initial_racks[p], Tile::from_char('Z')));
  }

  // Every tile handed to play_from is on the board, on a rack or in the bag at
  // the end, so the returned tiles joined exactly once.
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

  // Candidates of all three move types.
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

  // medium_dict() has many 2-letter words, so some opponent replies are real
  // placements (anchored classes), not passes.
  int64_t total_opp = 0;
  for (int i = 0; i < kAnchoredFootprints; ++i) total_opp += obs[0].opp_next_count[i];
  ASSERT_GT(total_opp, 0);

  // Results do not depend on the thread count.
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

// accumulate_rollout buckets each rollout move at footprint_class(move) in the
// untransposed frame: opp_reply into the opp counts (win weight p_loss),
// self_next into the self counts (win weight p_win), a PASS into kPassClass.
// The classes are computed by hand because the invariants SimRunner.Basic
// checks would still hold with opp/self swapped or the frame transposed.
TEST(SimRunner, AccumulateRolloutBucketsFootprints) {
  const Glyph g[3] = {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                      Glyph::of(Tile::from_char('C'))};
  RolloutResult r;
  // Horizontal, 2 tiles at row 3, cols 6-7: anchor (3,6), slot 1.
  r.opp_reply = Move::play(/*horizontal=*/true, /*start=*/3,
                           /*square_mask=*/uint16_t((1 << 6) | (1 << 7)), /*score=*/10, g, 2);
  // Vertical, 3 tiles at col 5, rows 2-4: anchor (2,5), slot
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
  EXPECT_FLOAT_EQ(obs.opp_win_count[opp_cls], 0.5f);  // p_loss
  EXPECT_EQ(obs.self_next_count[self_cls], 1);
  EXPECT_FLOAT_EQ(obs.self_win_count[self_cls], 0.25f);  // p_win
  int64_t opp_total = 0, self_total = 0;
  for (int i = 0; i < SimObservation::kClasses; ++i) {
    opp_total += obs.opp_next_count[i];
    self_total += obs.self_next_count[i];
  }
  EXPECT_EQ(opp_total, 1);
  EXPECT_EQ(self_total, 1);

  // A 1-tile play takes the orientation-free slot 0 whichever axis it declares,
  // and a missing move (a default Move is a PASS) buckets into kPassClass.
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

// Leaf-model stub for truncated rollouts: every row reads WLD (0.7, 0.1, 0.2)
// and a final score delta of +100, from the POV of the player to move at the
// horizon. Constant outputs make the flip to the root player's POV at odd
// horizons exactly checkable.
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

// Leaf stub whose outputs depend on the row's score-diff input. Rollouts then
// contribute distinct fractional values, so the reduction-order determinism and
// common-random-number checks are not vacuous.
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
      sd[0] = s * scribblez::kScoreDiffInputScale;  // predicts the current diff holds
      sd[1] = 5.0f;
    }
    rows_seen += batch.count;
  }
};

// One output element to overwrite (typically with a non-finite value) in an
// otherwise constant leaf readout, to exercise the runner's error guard.
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

// With a constant leaf, every rollout is cut at the horizon (a game from the
// opening cannot end within 4 plies), so the observations are exact multiples
// of the stub's outputs, flipped to the root mover's POV at an odd horizon.
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
    // The leaf reads the position after the horizon ply from that ply's
    // mover's POV. The opponent moves first, so at horizon 4 the last ply is
    // the root mover's own (opp, self, opp, self) and the readout carries over;
    // at horizon 3 it is the opponent's, so win/loss and the delta sign flip.
    const double p_win = horizon % 2 == 0 ? double(0.7f) : double(0.2f);
    const double p_loss = horizon % 2 == 0 ? double(0.2f) : double(0.7f);
    const double delta = horizon % 2 == 0 ? 100.0 : -100.0;
    for (const SimObservation& o : obs) {
      ASSERT_EQ(int(o.n), params.rollouts);
      ASSERT_DOUBLE_EQ(o.wins, o.n * p_win);
      ASSERT_DOUBLE_EQ(o.draws, o.n * double(0.1f));
      ASSERT_DOUBLE_EQ(o.losses, o.n * p_loss);
      ASSERT_DOUBLE_EQ(o.delta_sum, o.n * delta);
      // The stub predicts sigma = 5, so the second moment is mean^2 + sigma^2,
      // unaffected by the POV flip.
      ASSERT_DOUBLE_EQ(o.delta_sq_sum, o.n * (100.0 * 100.0 + 5.0 * 5.0));
      for (int i = 0; i < SimObservation::kClasses; ++i) {
        ASSERT_NEAR(o.opp_win_count[i], p_loss * o.opp_next_count[i], 1e-3);
        ASSERT_NEAR(o.self_win_count[i], p_win * o.self_next_count[i], 1e-3);
      }
    }
  }
  fs::remove_all(tmp);
}

// Truncated observations are exactly identical across thread counts (the
// reduction order is fixed), identical for duplicate candidates (common random
// numbers), and independent of which other candidates were simmed. The worker
// threads share one leaf service.
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
  ASSERT_EQ(std::memcmp(&obs[0], &obs[1], sizeof(SimObservation)), 0);

  {
    SimRunner::Params p1 = params;
    p1.threads = 1;
    const std::vector<SimObservation> obs1 = SimRunner(d, p1).run(pos, candidates, base_seed);
    for (size_t c = 0; c < obs.size(); ++c)
      ASSERT_EQ(std::memcmp(&obs[c], &obs1[c], sizeof(SimObservation)), 0);
  }
  {
    const std::vector<SimObservation> alone =
      SimRunner(d, params).run(pos, {candidates[2]}, base_seed);
    ASSERT_EQ(std::memcmp(&alone[0], &obs[2], sizeof(SimObservation)), 0);
  }
  fs::remove_all(tmp);
}

// A horizon past every game's natural end changes nothing: the leaf service is
// never called and the observations are byte-identical to an untruncated run's.
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
  truncated.horizon_plies = 350;
  truncated.leaf_service = &leaf;
  const std::vector<SimObservation> obs_truncated =
    SimRunner(d, truncated).run(pos, candidates, 400);

  ASSERT_EQ(leaf.rows_seen, 0);
  for (size_t c = 0; c < obs_terminal.size(); ++c)
    ASSERT_EQ(std::memcmp(&obs_terminal[c], &obs_truncated[c], sizeof(SimObservation)), 0);
  fs::remove_all(tmp);
}

// A horizon and a leaf service must be set together, and the horizon must be at
// least kMinHorizonPlies.
TEST(SimRunner, ValidatesTruncationParams) {
  SimRunner::Params p;
  p.horizon_plies = 4;
  ASSERT_THROW(SimRunner::validate(p), std::runtime_error);
  ConstantLeafService leaf;
  p.leaf_service = &leaf;
  p.horizon_plies = 0;
  ASSERT_THROW(SimRunner::validate(p), std::runtime_error);
  p.horizon_plies = SimRunner::kMinHorizonPlies - 1;
  ASSERT_THROW(SimRunner::validate(p), std::runtime_error);
  p.horizon_plies = SimRunner::kMinHorizonPlies;
  SimRunner::validate(p);
}

// A non-finite leaf readout in any field the runner consumes is a hard error,
// whether NaN or the inf that an FP16 overflow produces. Rollouts run on worker
// threads, so the throw must reach the calling thread rather than terminate the
// process.
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
    {1, 0, inf},  // score-diff mean: inf passes an isnan check
    {1, 1, inf},  // score-diff std, which feeds delta_sq
    {1, 1, nan},
    {0, 2, nan},  // loss probability, with the win probability finite
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

// A known 7-tile opponent leave is the opponent's whole rack. The rollout
// policy is deterministic, so the opponent's first reply to a candidate is the
// same in every rollout and each footprint class count is exactly 0 or n.
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
    for (int i = 0; i < SimObservation::kClasses; ++i)
      ASSERT_TRUE(o.opp_next_count[i] == 0 || o.opp_next_count[i] == o.n);
    // For some candidate the reply is a real placement, not a pass.
    for (int i = 0; i < kAnchoredFootprints; ++i)
      if (o.opp_next_count[i] == o.n) any_reply = true;
  }
  ASSERT_TRUE(any_reply);

  SimRunner::Params p1 = params;
  p1.threads = 1;
  const std::vector<SimObservation> obs1 = SimRunner(d, p1).run(pos, candidates, /*base_seed=*/9);
  for (size_t c = 0; c < obs.size(); ++c)
    ASSERT_EQ(std::memcmp(&obs[c], &obs1[c], sizeof(SimObservation)), 0);

  fs::remove_all(tmp);
}

// A partial known leave: rollouts keep the opponent's known tiles and sample
// only the rest of their rack. The usual observation invariants hold.
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
  pos.opp_leave = rack_from("ZI");

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

// opp_leave_from_replay is the opponent's current rack minus what they drew
// after their last move, and empty before they have acted.
TEST(SimRunner, OppLeaveFromReplay) {
  using scribblez::binlog::opp_leave_from_replay;
  TurnRecord records[2] = {};
  records[0].player = 1;  // the opponent's move at turn 0
  records[0].drawn = rack_from("AB");
  GameLog g{};
  g.records = records;
  g.num_records = 2;

  const Rack now = rack_from("CABDEFG");
  const Rack leave = opp_leave_from_replay(g, /*sampled_turn=*/1, now);
  ASSERT_EQ(leave.size(), 5);
  Rack expect = rack_from("CDEFG");
  for (int i = 0; i < expect.size(); ++i) ASSERT_TRUE(rack_contains(leave, expect.tiles()[i]));

  ASSERT_EQ(opp_leave_from_replay(g, /*sampled_turn=*/0, now).size(), 0);
}

// util/metaprogramming.h's consteval reflection helpers. The checks are
// compile-time; the TEST wrapper only makes the coverage visible in the suite.
namespace metaprog_test {
struct Sample {
 public:
  uint32_t plain;

 private:
  std::array<uint16_t, 3> squares_;

 public:
  // Uses squares_ to silence unused-private-field warnings.
  const void* touch() const { return &squares_; }
};

consteval bool helpers_hold() {
  if (scribblez::util::num_members<Sample>() != 2) return false;
  const auto members = scribblez::util::nonstatic_data_members<Sample>();
  if (!scribblez::util::type_is<uint32_t>(std::meta::type_of(members[0]))) return false;
  // Private members reflect too, and member_name drops the trailing underscore.
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

// The format-layout JSON served over the FFI to Python readers. Its sizes and
// constants come from the compiler, so this checks the document structure the
// Python side walks: struct entries, nested-struct references, subarray
// fields and constants.
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
  // histograms are integer counts; the win histograms and outcome sums are
  // fractional because truncated rollouts add leaf probabilities.
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

// The Glyph code table is replicated in Python (sim_evidence/sobs.py
// glyph_char) rather than served over the FFI, because it is effectively
// frozen. This test and its twin in py/tests/test_format_layout.py pin both
// copies; change them together.
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
  constexpr uint32_t kCells = move_set_eval::kPlaneWidth;

  const Move m1 = make_play_full(4, 2, /*horizontal=*/true, 0b111, 24,
                                 {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                                  Glyph::of(Tile::from_char('C'))});
  TileCounts xchg_tiles;
  xchg_tiles.add(Tile::from_char('A'));
  const Move m2 = Move::exchange(xchg_tiles);
  const std::vector<float> targets = {0.7f, 0.1f, 0.2f, 33.5f,  41.0f,
                                      0.2f, 0.0f, 0.8f, -12.0f, 55.5f};
  // Distinct per-plane maxima; candidate 1's last plane is all zero, so its
  // quantization scale is 0.
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
    // A swept position records its legal-move count, so a sweep truncated by
    // the candidate cap shows as a shortfall.
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
  ASSERT_EQ(p0.header->num_legal_moves, 0u);  // not recorded
  ASSERT_EQ(r.position(1).header->num_legal_moves, 9184u);
  ASSERT_EQ(r.move_at(p0, 0), m1);
  ASSERT_EQ(r.move_at(p0, 1), m2);
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < int(kFloats); ++j) {
      ASSERT_EQ(r.targets_at(p0, c)[j], targets[c * kFloats + j]);
    }
  }

  // Planes are 8-bit absmax-quantized: each dequantizes within half a step,
  // and the plane max is exact.
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

  // A version mismatch throws rather than misparsing a stale file.
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(4);  // TargetFileHeader::version
    const uint16_t bad = 0xFFFF;
    f.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
  }
  ASSERT_THROW(move_set_eval::TargetReader r2(path), std::runtime_error);

  fs::remove_all(tmp);
}

// Full-sweep files carry no planes: with record_planes 0 a record is just the
// Move and its value targets.
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

// The full sweep caps candidates by static-equity rank, but always keeps every
// exchange and the played move, and preserves rank order so the stored order is
// still a static-equity ranking.
TEST(MoveSetEvalCandidates, FullSweepCapKeepsExchangesAndRankOrder) {
  std::vector<Move> ranked = ranked_plays(10);
  const Move buried_exchange = exchange_of('Q');
  const Move buried_play = ranked[8];
  ranked.insert(ranked.begin() + 6, exchange_of('A'));  // inside the cap
  ranked.push_back(buried_exchange);                    // beyond it

  const move_set_eval::Selection sel =
    move_set_eval::full_sweep_candidates(ranked, buried_play, /*cap=*/4);
  const std::vector<Move>& swept = sel.candidates;

  // The head under the cap, then the exchanges and the played move from beyond
  // it, all in `ranked`'s order.
  ASSERT_EQ(swept.size(), 7u);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(swept[size_t(i)], ranked[i]);
  EXPECT_EQ(swept[4], ranked[6]);  // the first exchange, ranked past the cap
  EXPECT_EQ(swept[5], buried_play);
  EXPECT_EQ(swept[6], buried_exchange);
  // The shortfall against the recorded legal count is what the cap dropped.
  EXPECT_EQ(sel.num_legal_moves, ranked.size());

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

// Forced candidates (moves already simmed for the trajectory sidecar) come right
// after the played move, and duplicates are dropped throughout, so nothing is
// labeled twice.
TEST(MoveSetEvalCandidates, StratifiedForceIncludesSimmedCandidates) {
  const std::vector<Move> ranked = ranked_plays(40);
  const Move played = ranked[0];
  const std::vector<Move> forced = {ranked[17], played, ranked[35]};
  std::mt19937_64 rng(7);
  const move_set_eval::StratumQuotas quotas;
  const move_set_eval::Selection sel =
    move_set_eval::stratified_candidates(ranked, played, quotas, rng, forced);
  const std::vector<Move>& out = sel.candidates;
  // Forced candidates add to the budget rather than shrinking a stratum:
  // played + 2 distinct forced + 4 top, 4 mid, 4 tail (no exchanges exist).
  ASSERT_EQ(out.size(), 15u);
  EXPECT_EQ(out[0], played);
  EXPECT_EQ(out[1], ranked[17]);
  EXPECT_EQ(out[2], ranked[35]);  // the duplicate of `played` was skipped
  for (int i = 1; i <= quotas.top; ++i) {
    EXPECT_NE(std::find(out.begin(), out.end(), ranked[size_t(i)]), out.end());
  }
  for (size_t i = 0; i < out.size(); ++i) {
    for (size_t j = i + 1; j < out.size(); ++j) EXPECT_NE(out[i], out[j]);
  }
}

// select_sim_candidates reports each candidate's 0-based static-equity rank, so
// a consumer can ask how deep in the ranking the sim's favourite sat.
TEST(SimCandidates, FlatRecipeIsTheRankedPrefix) {
  const std::vector<Move> ranked = ranked_plays(5);
  std::mt19937_64 rng(1);
  SimCandidateRecipe recipe;
  recipe.top_k = 3;
  const SimCandidates sel = select_sim_candidates(ranked, ranked[0], recipe, rng);
  EXPECT_EQ(sel.moves, std::vector<Move>(ranked.begin(), ranked.begin() + 3));
  EXPECT_EQ(sel.equity_ranks, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(sel.num_legal_moves, 5u);

  recipe.top_k = 9;  // more moves than exist
  EXPECT_EQ(select_sim_candidates(ranked, ranked[0], recipe, rng).moves, ranked);
}

TEST(SimCandidates, StratifiedRecipeRanksEveryStratum) {
  std::vector<Move> ranked = ranked_plays(40);
  ranked.push_back(exchange_of('Q'));
  std::mt19937_64 rng(7);
  SimCandidateRecipe recipe;
  recipe.quotas = move_set_eval::StratumQuotas{
    .top = 2, .mid = 3, .tail = 4, .exchange = 1, .mid_rank_limit = 10};
  const SimCandidates sel = select_sim_candidates(ranked, ranked[0], recipe, rng);
  ASSERT_EQ(sel.moves.size(), 11u);  // played + 2 + 3 + 4 + 1
  ASSERT_EQ(sel.equity_ranks.size(), sel.moves.size());
  for (size_t i = 0; i < sel.moves.size(); ++i) {
    ASSERT_GE(sel.equity_ranks[i], 0);
    EXPECT_EQ(ranked[size_t(sel.equity_ranks[i])], sel.moves[i]);
  }
  EXPECT_EQ(sel.equity_ranks[0], 0);  // the played move leads
  const auto in_band = [&](int lo, int hi) {
    return std::count_if(sel.equity_ranks.begin(), sel.equity_ranks.end(),
                         [&](int32_t r) { return r >= lo && r < hi; });
  };
  EXPECT_EQ(in_band(0, 3), 3);    // played + top
  EXPECT_EQ(in_band(3, 10), 3);   // mid, up to mid_rank_limit
  EXPECT_EQ(in_band(10, 41), 5);  // tail, plus the exchange ranked last
}

// A played move the generator never enumerates (a PASS chosen while other moves
// were legal) has no rank.
TEST(SimCandidates, UnrankedPlayedMoveGetsMinusOne) {
  const std::vector<Move> ranked = ranked_plays(6);
  std::mt19937_64 rng(3);
  SimCandidateRecipe recipe;
  recipe.quotas = move_set_eval::StratumQuotas{};
  const SimCandidates sel = select_sim_candidates(ranked, Move{}, recipe, rng);
  EXPECT_EQ(sel.equity_ranks[0], -1);
}

// summarize_rollouts: outcome and margin sums, histogram bin edges, and the
// adjacency count of next moves played next to the candidate's own tiles.
TEST(RolloutSummary, ReducesOutcomesScoresAndAdjacency) {
  const Glyph a = Glyph::of(Tile::from_char('A'));
  const Move candidate = make_play_full(7, 7, /*horizontal=*/true, 0b1, 10, {a});
  const Move hook = make_play_full(7, 8, true, 0b1, 35, {a});  // beside it
  const Move far = make_play_full(0, 0, true, 0b1, 104, {a});
  const Move bingo = make_play_full(14, 0, true, 0b1111111, 72, {a, a, a, a, a, a, a});

  std::vector<RolloutResult> rollouts(3);
  rollouts[0] = {.opp_reply = far, .self_next = hook, .p_win = 1, .delta = 30, .delta_sq = 900};
  rollouts[1] = {
    .opp_reply = bingo, .self_next = Move{}, .p_loss = 1, .delta = -201, .delta_sq = 40401};
  rollouts[2] = {.opp_reply = hook, .self_next = far, .p_draw = 1, .delta = 0, .delta_sq = 0};
  rollouts[0].opp_stranded = 10;   // the mover played out against a stuck Q: +20
  rollouts[1].self_stranded = 30;  // the opponent played out: -60, below the floor
  rollouts[2].self_stranded = 3;   // nobody played out: 5 - 3 = +2
  rollouts[2].opp_stranded = 5;
  const RolloutSummary s = summarize_rollouts(candidate, rollouts);

  EXPECT_EQ(s.n, 3u);
  EXPECT_EQ(s.wins, 1);
  EXPECT_EQ(s.draws, 1);
  EXPECT_EQ(s.losses, 1);
  EXPECT_EQ(s.delta_sum, -171);
  EXPECT_EQ(s.delta_hist[0], 1u);   // below the -200 floor
  EXPECT_EQ(s.delta_hist[9], 1u);   // 0 in [0, 25)
  EXPECT_EQ(s.delta_hist[10], 1u);  // 30 in [25, 50)
  EXPECT_EQ(s.opp_reply.score_sum, 104 + 72 + 35);
  EXPECT_EQ(s.opp_reply.score_hist[kScoreBins - 1], 1u);  // 104: the open-ended top bin
  EXPECT_EQ(s.opp_reply.score_hist[7], 1u);               // 72
  EXPECT_EQ(s.opp_reply.bingos, 1u);
  EXPECT_EQ(s.opp_reply.adjacent, 1u);
  EXPECT_EQ(s.self_next.non_plays, 1u);  // rollout 1's missing move (a default Move)
  EXPECT_EQ(s.self_next.adjacent, 1u);
  EXPECT_EQ(s.opp_stranded_sum, 15);
  EXPECT_EQ(s.self_went_out, 1u);
  EXPECT_EQ(s.opp_went_out, 1u);
  EXPECT_EQ(s.end_swing_sum, 20 - 60 + 2);
  EXPECT_EQ(s.end_swing_hist[0], 1u);  // -60
  EXPECT_EQ(s.end_swing_hist[6], 1u);  // +2 in [0, 10)
  EXPECT_EQ(s.end_swing_hist[8], 1u);  // +20 in [20, 30)
}

// paired_win_diff: win = 1, draw = 1/2, differenced rollout by rollout.
TEST(RolloutSummary, PairedWinDiffMoments) {
  std::vector<RolloutResult> a(3), b(3);
  a[0].p_win = 1;   // vs a loss: +1
  a[1].p_draw = 1;  // vs a win: -1/2
  b[1].p_win = 1;
  a[2].p_win = 1;  // vs a win: 0
  b[2].p_win = 1;
  const PairedWinDiff d = paired_win_diff(a, b);
  EXPECT_DOUBLE_EQ(d.sum, 0.5);
  EXPECT_DOUBLE_EQ(d.sq_sum, 1.25);
}

// is_high_value_setup on the play it was defined from: Sokol's K6 AC.TA
// (positions/NWL23/interesting-positions/ACETA.gcg) keeps the Z and lays its A's
// beside the triple-letter squares J6 and J10, where ZA then hooks. The same
// word one column left puts the A's on those squares and sets up nothing; 9K TIZ
// spends the Z. Skipped without the NWL23 lexicon and leaves.
TEST(SetupPlays, AcetaIsAHighValueSetup) {
  namespace fs = std::filesystem;
  const std::string kwg = SCRIBBLEZ_DEFAULT_KWG;
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  if (!fs::exists(kwg) || !fs::exists(leaves)) GTEST_SKIP() << "no NWL23 kwg/leaves";
  Dictionary dict = Dictionary::load_kwg(kwg);
  HastyEquity::init(leaves, HastyEquity::default_peg_path());

  const std::string gcg =
    "#player1 Will Will\n#player2 Joshua Joshua\n"
    ">Will: EEEFGKR 8H GREEK +30 30\n>Joshua: AACITTZ K6 AC.TA +7 7\n";
  ParsedGcgPosition pos;
  std::string error;
  ASSERT_TRUE(read_gcg_position_at(gcg, 1, /*open_leaves=*/false, &pos, &error)) << error;
  MoveRequest req{pos.board, dict, pos.rack, Rack{}, pos.scores[1], pos.scores[0], pos.bag_size};
  std::map<std::string, bool> setup;
  for (const Move& m : equity_top_k(req, std::numeric_limits<int>::max()))
    setup[move_notation(pos.board, m)] = is_high_value_setup(pos.board, dict, pos.rack, m);

  ASSERT_TRUE(setup.contains("K6 AC.TA"));
  EXPECT_TRUE(setup.at("K6 AC.TA"));
  EXPECT_FALSE(setup.at("J6 AC.TA"));
  EXPECT_FALSE(setup.at("9K TIZ"));
}

// off_policy_draws (the trajectory's off-policy picks, docs/roadmap.md item 4)
// draws `count` distinct indices uniformly from the moves not yet taken, and
// marks them taken. There is no stratification: exchanges and the tail appear
// at their natural frequency in the move list.
TEST(MoveSetEvalCandidates, OffPolicyDrawsUniformlyExcludingTaken) {
  std::vector<Move> ranked = ranked_plays(20);
  TileCounts xchg;
  xchg.add(Tile::from_char('A'));
  ranked.push_back(Move::exchange(xchg));
  const size_t n = ranked.size();
  std::vector<char> taken(n, 0);
  for (size_t i : {size_t{0}, size_t{5}, size_t{12}}) taken[i] = 1;  // anchor and on-policy
  const std::vector<char> pre = taken;
  std::mt19937_64 rng(3);
  const std::vector<size_t> draws =
    move_set_eval::off_policy_draws(ranked, /*count=*/4, rng, &taken);

  ASSERT_EQ(draws.size(), 4u);
  for (size_t a = 0; a < draws.size(); ++a) {
    EXPECT_LT(draws[a], n);
    EXPECT_FALSE(pre[draws[a]]);
    EXPECT_TRUE(taken[draws[a]]);
    for (size_t b = a + 1; b < draws.size(); ++b) EXPECT_NE(draws[a], draws[b]);
  }
}

// A count larger than the untaken pool yields exactly the remaining pool,
// including an exchange.
TEST(MoveSetEvalCandidates, OffPolicyDrawsWholeUntakenPoolIncludingNonPlays) {
  std::vector<Move> ranked = ranked_plays(4);
  TileCounts xchg;
  xchg.add(Tile::from_char('A'));
  ranked.push_back(Move::exchange(xchg));  // index 4
  std::vector<char> taken(ranked.size(), 0);
  taken[0] = taken[1] = 1;
  std::mt19937_64 rng(1);
  const std::vector<size_t> draws =
    move_set_eval::off_policy_draws(ranked, /*count=*/10, rng, &taken);
  ASSERT_EQ(draws.size(), 3u);
  std::vector<size_t> sorted = draws;
  std::ranges::sort(sorted);
  EXPECT_EQ(sorted, (std::vector<size_t>{2u, 3u, 4u}));
}

// count == 0 is a valid config and must draw nothing.
TEST(MoveSetEvalCandidates, OffPolicyDrawsCountZeroDrawsNothing) {
  const std::vector<Move> ranked = ranked_plays(6);
  std::vector<char> taken(ranked.size(), 0);
  taken[0] = 1;
  const std::vector<char> pre = taken;
  std::mt19937_64 rng(1);
  const std::vector<size_t> draws =
    move_set_eval::off_policy_draws(ranked, /*count=*/0, rng, &taken);
  EXPECT_TRUE(draws.empty());
  EXPECT_EQ(taken, pre);
}

// On-policy proposals are a temperature softmax over every unsimmed candidate,
// not over a top-ranked head: the deployed agent argmaxes over the full
// support, so a training proposal must be reachable at any rank. Equal win
// equities make the softmax uniform, so over 500 seeds some proposal lands
// deeper than rank 64. Checked here so it does not depend on the GPU-gated
// end-to-end test.
TEST(EvidenceTrajectory, ProposalsDrawBeyondTheRetiredPoolCap) {
  constexpr int kN = 100;
  const std::vector<Move> ranked = ranked_plays(kN);  // descending score
  const std::vector<float> win_equity(kN, 0.5f);
  evidence::TrajectoryOptions opt;
  opt.on_policy_min = 1;  // exactly one on-policy proposal, at chosen[1]
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
    EXPECT_EQ(roles[1], SimObsRole::kOnPolicy);
    deepest_proposal = std::max(deepest_proposal, chosen[1]);
  }
  EXPECT_GT(deepest_proposal, 64u);
}

// The per-game sampling shuffle depends only on (seed, game), so a smaller
// sample is a prefix of a larger one. The move-set target generator relies on
// this to find every trajectory-sidecar position in its own sample (see
// slog_sampling.h); checked here so it does not depend on the GPU-gated
// end-to-end test.
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

// The stored std stays finite when FP16 teacher inference overflows it to +inf
// (see kSdStdCap); ordinary values pass through.
TEST(MoveSetEvalTargetLog, SdStdClamp) {
  ASSERT_EQ(move_set_eval::clamped_sd_std(std::numeric_limits<float>::infinity()),
            move_set_eval::kSdStdCap);
  ASSERT_EQ(move_set_eval::clamped_sd_std(41.0f), 41.0f);
  ASSERT_EQ(move_set_eval::clamped_sd_std(move_set_eval::kSdStdCap), move_set_eval::kSdStdCap);
}

// A .mset records whether the games it labels were played with face-up leaves,
// taken from its source .slog's header, so a training corpus can be held to one
// variant.
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

// With opp_leave_input, the target generator's candidate rows carry the
// opponent's leave as recovered by replay, so an open-leaves teacher sees the
// input it was trained on rather than a zeroed block.
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
  // A PLAY of A, blank-as-B, C; an exchange of D, A and a blank; a PASS.
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

  // Letters are 1..26 whether or not the tile is a blank; 0 is padding.
  ASSERT_TRUE(tile_mask[0] == 1 && tile_mask[1] == 1 && tile_mask[2] == 1);
  ASSERT_TRUE(tile_mask[3] == 0 && tile_mask[6] == 0);
  ASSERT_EQ(letters[0], Tile::from_char('A').index() + 1);
  ASSERT_EQ(letters[1], Tile::from_char('B').index() + 1);
  ASSERT_EQ(letters[2], Tile::from_char('C').index() + 1);
  ASSERT_TRUE(blanks[0] == 0 && blanks[1] == 1 && blanks[2] == 0);
  ASSERT_EQ(squares[0], 4 * BOARD_SIZE + 2);
  ASSERT_EQ(squares[2], 4 * BOARD_SIZE + 4);
  // Scalars: resulting score differential, tiles / 7, is_play.
  ASSERT_LT(std::abs(scalars[0] - 34.0f / kScoreDiffInputScale), 1e-6f);
  ASSERT_LT(std::abs(scalars[1] - 3.0f / 7.0f), 1e-6f);
  ASSERT_EQ(scalars[2], 1.0f);

  // An exchange encodes its surrendered tiles, sorted, so same-size exchanges
  // are distinguishable; squares stay 0. An exchanged blank has no letter, only
  // the blank flag.
  const int e = mset::kMoveMaxPlaced;
  ASSERT_TRUE(tile_mask[e + 0] == 1 && tile_mask[e + 1] == 1 && tile_mask[e + 2] == 1);
  ASSERT_TRUE(tile_mask[e + 3] == 0 && tile_mask[e + 6] == 0);
  ASSERT_EQ(letters[e + 0], Tile::from_char('A').index() + 1);
  ASSERT_EQ(letters[e + 1], Tile::from_char('D').index() + 1);
  ASSERT_EQ(letters[e + 2], 0);
  ASSERT_TRUE(blanks[e + 0] == 0 && blanks[e + 1] == 0 && blanks[e + 2] == 1);
  for (int j = 0; j < mset::kMoveMaxPlaced; ++j) ASSERT_EQ(squares[e + j], 0);
  ASSERT_LT(std::abs(scalars[mset::kMoveScalars + 0] - (-5.0f) / kScoreDiffInputScale), 1e-6f);
  ASSERT_LT(std::abs(scalars[mset::kMoveScalars + 1] - 3.0f / 7.0f), 1e-6f);
  ASSERT_EQ(scalars[mset::kMoveScalars + 2], 0.0f);

  const int p = 2 * mset::kMoveMaxPlaced;
  for (int j = 0; j < mset::kMoveMaxPlaced; ++j) ASSERT_EQ(tile_mask[p + j], 0);
  ASSERT_LT(std::abs(scalars[2 * mset::kMoveScalars + 0] - (-5.0f) / kScoreDiffInputScale), 1e-6f);
  ASSERT_EQ(scalars[2 * mset::kMoveScalars + 1], 0.0f);
  ASSERT_EQ(scalars[2 * mset::kMoveScalars + 2], 0.0f);
}

// One move's slice of the batch encode_cross_check_deltas arrays.
struct CrossDeltaView {
  const uint8_t* axes;
  const int32_t* squares;
  const uint32_t* old_masks;
  const uint32_t* new_masks;
  const uint8_t* delta_mask;
};

// The pre-move cross-check planes as the delta entries say `m` leaves them:
// zeroed on the squares it fills, and rewritten to the new mask on each entry.
// Checks each entry's old mask against the planes it overwrites on the way.
std::vector<float> patch_cross_check_planes(const float* pre_planes, const Move& m,
                                            const CrossDeltaView& d) {
  std::vector<float> planes(pre_planes, pre_planes + kCrossCheckPlanes * kBoardCells);
  visit_placed_squares(m, [&](int r, int c) {
    for (int l = 0; l < kCrossCheckPlanes; ++l) planes[l * kBoardCells + r * kBoardSide + c] = 0.0f;
  });
  for (int i = 0; i < move_set::kMoveMaxCrossDeltas; ++i) {
    if (!d.delta_mask[i]) continue;
    float* block = planes.data() + d.axes[i] * kHorizontalCrossCheckPlanes * kBoardCells;
    for (int l = 0; l < 26; ++l) {
      float& cell = block[l * kBoardCells + d.squares[i]];
      EXPECT_EQ(cell, float((d.old_masks[i] >> l) & 1u));
      cell = float((d.new_masks[i] >> l) & 1u);
    }
  }
  return planes;
}

// The defining property: a candidate's pre-move planes patched with its delta
// entries are the cross-check planes of the post-move row the teacher is fed.
// Turn 0 covers the empty-board apply, which rebuilds every cache entry.
TEST(CrossCheckDelta, PatchedPreMovePlanesEqualTheTeachersPostMovePlanes) {
  namespace mset = move_set;
  Dictionary dict = medium_dict();
  const GameLogStorage storage = play_test_game(dict, /*seed=*/4242ULL);
  const GameLog g = storage.view();
  const InputEncodingSpec spec{&dict};
  const size_t row_floats = input_floats(spec);
  const size_t cross0 = size_t(spatial_block_plane0(SpatialBlockId::kCrossChecks)) * kBoardCells;
  const size_t cross_bytes = sizeof(float) * kCrossCheckPlanes * kBoardCells;

  int max_entries = 0;
  for (int turn = 0; turn < std::min(g.num_records, 10); ++turn) {
    binlog::PositionEncoder pos(spec);
    const int mover = pos.replay_to_sampled(g, turn, /*post_move=*/false);
    const Board& board = pos.enc().board();
    board.ensure_movegen_caches(dict);

    // A PASS leaves the board alone, so its row carries the pre-move planes.
    std::vector<Move> candidates = {Move::pass()};
    const std::vector<Move> plays = MoveGenerator(board, dict).generate(pos.rack(mover));
    const size_t stride = plays.size() / 100 + 1;
    for (size_t i = 0; i < plays.size(); i += stride) candidates.push_back(plays[i]);

    const size_t n = candidates.size();
    std::vector<float> rows(n * row_floats);
    binlog::encode_candidate_rows(pos, g, turn, mover, candidates, rows.data());

    const size_t width = mset::kMoveMaxCrossDeltas;
    std::vector<uint8_t> axes(n * width), delta_mask(n * width);
    std::vector<int32_t> squares(n * width);
    std::vector<uint32_t> old_masks(n * width), new_masks(n * width);
    mset::encode_cross_check_deltas(board, dict, candidates.data(), int64_t(n), axes.data(),
                                    squares.data(), old_masks.data(), new_masks.data(),
                                    delta_mask.data());

    for (size_t c = 0; c < n; ++c) {
      const size_t at = c * width;
      const CrossDeltaView d{axes.data() + at, squares.data() + at, old_masks.data() + at,
                             new_masks.data() + at, delta_mask.data() + at};
      const std::vector<float> patched =
        patch_cross_check_planes(rows.data() + cross0, candidates[c], d);
      ASSERT_EQ(std::memcmp(patched.data(), rows.data() + c * row_floats + cross0, cross_bytes), 0)
        << "turn " << turn << ", candidate " << c;

      // Real entries lead, strictly ordered by (axis, square); pads are zero.
      const int entries = int(std::count(d.delta_mask, d.delta_mask + width, uint8_t(1)));
      for (int i = 0; i < int(width); ++i) {
        ASSERT_EQ(d.delta_mask[i], i < entries ? 1 : 0);
        if (i >= entries) {
          ASSERT_EQ(d.axes[i] + d.squares[i] + d.old_masks[i] + d.new_masks[i], 0u);
        }
        if (i > 0 && i < entries) {
          ASSERT_LT(std::pair(d.axes[i - 1], d.squares[i - 1]), std::pair(d.axes[i], d.squares[i]));
        }
      }
      max_entries = std::max(max_entries, entries);
    }
    ASSERT_EQ(std::count(delta_mask.begin(), delta_mask.begin() + width, uint8_t(1)), 0);
  }
  ASSERT_GT(max_entries, 4) << "no multi-tile play was exercised";
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
  o1.wins = 9.25;  // fractional, as truncated rollouts produce
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
    w.add_position(4, 0, {m2}, {o2}, 16, 1000);  // no roles
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
  ASSERT_EQ(p0.header->flags, 0u);  // reserved
  SimObsRecord rec;                 // copied out of the packed file view before comparing
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
  ASSERT_EQ(rec.role, SimObsRole::kAnchor);  // the default without roles

  // A version mismatch throws rather than misparsing a stale file.
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(4);  // SimObsFileHeader::version
    const uint16_t bad = 0xFFFF;
    f.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
  }
  ASSERT_THROW(SimObsReader r2(path), std::runtime_error);

  fs::remove_all(tmp);
}

// Decodes turn 0 of a one-game .slog buffer in which p0 starts with
// `initial_score_p0` points and both turns are passes, and returns the score
// differential read back from the score-diff input. Nothing else scores, so the
// result is the handicap.
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
  gm.sampled_turn = 0;
  gm.initial_score_p0 = initial_score_p0;

  InitialRacks ir{};
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

  const float* sd = output.data() + kSpatialFloats + kScoreDiffOffset;
  return std::lround(sd[0] * kScoreDiffInputScale);
}

// A handicap stored in GameMetadata's initial scores reaches the replayed
// position's score-diff input.
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

// The max-move-per-lane task's per-lane targets. Each legal play belongs to the
// row it lies along (horizontal) or the column (vertical); a single-tile play
// belongs to each direction in which it forms a word. The best lane max must
// equal the best play's score.
TEST(Lane, Targets) {
  const Dictionary d = tiny_dict();

  // CAT -> CATS is in CENTER's row only; the S forms no vertical word.
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
    ASSERT_TRUE((t.rows[CENTER].placed[CENTER + 3] >> sk) & 1u);
    ASSERT_EQ(t.rows[CENTER].max_score, best_move_score(moves));
    ASSERT_FALSE(t.cols[CENTER + 3].has_move);
    ASSERT_EQ(lane_global_max(t), best_move_score(moves));
  }

  // A single tile forming words both ways lands in its row and its column, at
  // the same score.
  {
    Board b;
    b.set(CENTER, CENTER - 1, Glyph::of(Tile::from_char('A')));  // A to S's left
    b.set(CENTER - 1, CENTER, Glyph::of(Tile::from_char('A')));  // A above S
    const Rack r = rack_from("S");
    const LaneTargets t = compute_lane_targets(b, r, d);
    const int sk = Tile::from_char('S').index();

    ASSERT_TRUE(t.rows[CENTER].has_move);
    ASSERT_TRUE(t.cols[CENTER].has_move);
    ASSERT_TRUE((t.rows[CENTER].placed[CENTER] >> sk) & 1u);  // lane cell is the column
    ASSERT_TRUE((t.cols[CENTER].placed[CENTER] >> sk) & 1u);  // lane cell is the row
    ASSERT_EQ(t.rows[CENTER].max_score, t.cols[CENTER].max_score);
  }

  // A single tile cannot open the game, so every lane is empty.
  {
    Board b;
    const LaneTargets t = compute_lane_targets(b, rack_from("S"), d);
    for (const auto& lane : t.rows) ASSERT_FALSE(lane.has_move);
    for (const auto& lane : t.cols) ASSERT_FALSE(lane.has_move);
  }

  // The flat label encoding of the CATS position: CENTER's row lane carries the
  // S occupancy, the score bin and the mask bit; other lanes are all zeros.
  {
    Board b;
    b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                      {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                       Glyph::of(Tile::from_char('T'))}));
    const LaneTargets t = compute_lane_targets(b, rack_from("S"), d);
    std::vector<float> row(kLaneLabelFloats, -1.0f);
    encode_lane_targets(t, row.data());

    const float* occ = row.data();
    const float* score = occ + kLaneOccupancyFloats;
    const float* mask = score + kLaneScoreFloats;
    const int row_id = CENTER;  // axis 0, lane CENTER
    const int sk = Tile::from_char('S').index();

    const float* lane_occ = occ + row_id * kLaneLen * kLaneTileKinds;
    ASSERT_EQ(lane_occ[(CENTER + 3) * kLaneTileKinds + sk], 1.0f);
    ASSERT_EQ(lane_occ[(CENTER + 3) * kLaneTileKinds + Tile::from_char('C').index()], 0.0f);
    ASSERT_EQ(lane_occ[CENTER * kLaneTileKinds + sk], 0.0f);  // CAT was already on the board

    ASSERT_EQ(mask[row_id], 1.0f);
    ASSERT_EQ(score[row_id], float(std::min(t.rows[CENTER].max_score, kLaneScoreBins - 1)));

    ASSERT_EQ(mask[0], 0.0f);
    ASSERT_EQ(score[0], 0.0f);
    for (int i = 0; i < kLaneLen * kLaneTileKinds; ++i) ASSERT_EQ(occ[i], 0.0f);

    // On the transposed board the play is vertical, so it moves to the axis-1
    // lane CENTER, same cell.
    std::vector<float> frow(kLaneLabelFloats, -1.0f);
    encode_lane_targets(compute_lane_targets(b.transpose(), rack_from("S"), d), frow.data());
    const float* focc = frow.data();
    const float* v_lane = focc + (kLanesPerAxis + CENTER) * kLaneLen * kLaneTileKinds;
    const float* h_lane = focc + CENTER * kLaneLen * kLaneTileKinds;
    ASSERT_EQ(v_lane[(CENTER + 3) * kLaneTileKinds + sk], 1.0f);
    for (int i = 0; i < kLaneLen * kLaneTileKinds; ++i) ASSERT_EQ(h_lane[i], 0.0f);
    ASSERT_EQ((focc + kLaneOccupancyFloats)[kLanesPerAxis + CENTER], score[row_id]);
  }

  // Along a random game: the best lane max equals the best play's score, and
  // some lane has a move iff any legal play exists.
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
        b = Board();
        continue;
      }
      const Move* best = &moves.front();
      for (const auto& m : moves)
        if (m.score() > best->score()) best = &m;
      b.apply(*best);
    }
  }
}

// compute_lane_best_moves keeps the plays tied for each lane's maximum. It
// agrees with compute_lane_targets on every lane, keeps only maximal plays, and
// a kept play's word and origin are recoverable from the pre-move board.
TEST(Lane, BestMoves) {
  const Dictionary d = tiny_dict();

  Board b;
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                     Glyph::of(Tile::from_char('T'))}));
  const Rack r = rack_from("S");

  const LaneTargets t = compute_lane_targets(b, r, d);
  const LaneBestMovesSet bm = compute_lane_best_moves(b, r, d);

  for (int i = 0; i < kLanesPerAxis; ++i) {
    ASSERT_EQ(bm.rows[i].has_move, t.rows[i].has_move);
    ASSERT_EQ(bm.cols[i].has_move, t.cols[i].has_move);
    ASSERT_EQ(bm.rows[i].max_score, t.rows[i].max_score);
    ASSERT_EQ(bm.cols[i].max_score, t.cols[i].max_score);
  }

  for (const auto& lane : bm.rows)
    for (const Move& m : lane.moves) ASSERT_EQ(int(m.score()), lane.max_score);
  for (const auto& lane : bm.cols)
    for (const Move& m : lane.moves) ASSERT_EQ(int(m.score()), lane.max_score);

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

// parse_gcg_analysis_position takes the board after all recorded moves, with
// the next player to move and that player's rack from the #Rack header. The
// lane-analysis JSON carries the web board plus per-lane ground truth and
// maximal plays.
TEST(Lane, Analysis) {
  // After P1's CAT at 8H (the center square), P2 is on move with #Rack2.
  const std::string gcg =
    "#player1 P1 Player One\n"
    "#player2 P2 Player Two\n"
    "#Rack1 _______\n"
    "#Rack2 EINRSTU\n"
    ">P1: AACATTX 8H CAT +5 5\n";

  GcgAnalysisPosition pos;
  std::string error;
  ASSERT_TRUE(parse_gcg_analysis_position(gcg, &pos, &error));
  ASSERT_EQ(pos.on_move, 1);
  ASSERT_EQ(pos.rack, rack_from("EINRSTU"));
  ASSERT_FALSE(pos.board.at(CENTER, CENTER).is_empty());      // C
  ASSERT_FALSE(pos.board.at(CENTER, CENTER + 2).is_empty());  // T
  ASSERT_TRUE(pos.board.at(CENTER, CENTER + 3).is_empty());

  const Dictionary d = tiny_dict();
  Board b;
  b.apply(make_play(CENTER, CENTER, /*horizontal=*/true,
                    {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                     Glyph::of(Tile::from_char('T'))}));
  const std::string js = lane_analysis_json(b, rack_from("S"), /*on_move=*/0, d);
  const boost::json::value v = boost::json::parse(js);
  const boost::json::object& o = v.as_object();
  ASSERT_TRUE(o.contains("board"));
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

// The premium plane (0-based, within the premium block) for `p`, or -1.
static int prem_plane_offset(Premium p) {
  if (p == Premium::DLS) return 0;
  if (p == Premium::TLS) return 1;
  if (p == Premium::DWS) return 2;
  if (p == Premium::TWS) return 3;
  return -1;
}

// The max-move-per-lane model's input: the 31 board planes (letters, blank
// marker, premiums) and 27 raw rack counts, with no cross-check planes.
TEST(MaxMovePerLane, InputEncoder) {
  Board b;
  b.set(7, 7, Glyph::of(Tile::from_char('C')));
  b.set(7, 8, Glyph::of(Tile::from_char('A')));
  b.set(7, 9, Glyph::of(Tile::from_char('T')));
  b.set(5, 5, Glyph::played(Tile::from_char('S'), /*is_blank=*/true));
  const Rack rack = rack_from("AAB?");

  using Enc = MaxMovePerLaneInputEncoder;
  const int A = Tile::from_char('A').index();
  const int B = Tile::from_char('B').index();
  const int C = Tile::from_char('C').index();
  const int Sx = Tile::from_char('S').index();
  const int cells = Enc::kBoardCells;
  auto cell = [](int r, int c) { return r * BOARD_SIZE + c; };

  std::vector<float> out(Enc::kInputFloats, -1.0f);
  Enc::encode(b, rack, out.data());

  // A blank sets its designated letter's plane too.
  ASSERT_EQ(out[C * cells + cell(7, 7)], 1.0f);
  ASSERT_EQ(out[A * cells + cell(7, 8)], 1.0f);
  ASSERT_EQ(out[Sx * cells + cell(5, 5)], 1.0f);
  ASSERT_EQ(out[A * cells + cell(7, 7)], 0.0f);

  ASSERT_EQ(out[BoardPlanes::kBlankMarkerPlane * cells + cell(5, 5)], 1.0f);
  ASSERT_EQ(out[BoardPlanes::kBlankMarkerPlane * cells + cell(7, 7)], 0.0f);

  // Premium planes match the board's premiums everywhere, including under
  // played tiles.
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const int want = prem_plane_offset(b.premium_at(r, c));
      for (int off = 0; off < BoardPlanes::kPremiumPlanes; ++off) {
        const float v = out[(BoardPlanes::kPremiumPlane0 + off) * cells + cell(r, c)];
        ASSERT_EQ(v, (off == want ? 1.0f : 0.0f));
      }
    }
  }

  const float* counts = out.data() + Enc::kSpatialFloats;
  ASSERT_EQ(counts[A], 2.0f);
  ASSERT_EQ(counts[B], 1.0f);
  ASSERT_EQ(counts[26], 1.0f);  // blank
  ASSERT_EQ(counts[C], 0.0f);

  std::vector<float> flipped(Enc::kInputFloats, -1.0f);
  Enc::encode(b.transpose(), rack, flipped.data());
  ASSERT_EQ(flipped[A * cells + cell(8, 7)], 1.0f);
  ASSERT_EQ(flipped[A * cells + cell(7, 8)], 0.0f);
  ASSERT_EQ(flipped[C * cells + cell(7, 7)], 1.0f);  // on the diagonal, unchanged
  const float* fcounts = flipped.data() + Enc::kSpatialFloats;
  ASSERT_TRUE(fcounts[A] == 2.0f && fcounts[26] == 1.0f);
}

// A max-move-per-lane training row is the task's input encoding followed by
// the per-lane labels, in both frames.
TEST(MaxMovePerLane, TaskRow) {
  const Dictionary d = tiny_dict();

  GameStateEncoder gse{InputEncodingSpec{&d}};
  gse.apply_move(make_play(CENTER, CENTER, /*horizontal=*/true,
                           {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                            Glyph::of(Tile::from_char('T'))}));
  const Rack rack = rack_from("S");

  for (bool transposed : {false, true}) {
    const GameStateEncoder enc = transposed ? gse.transpose() : gse;
    EncodeContext ctx{};
    ctx.enc = &enc;
    ctx.pov_rack = &rack;
    ctx.spec = {&d};

    std::vector<float> row(MaxMovePerLaneTask::kRowFloats, -1.0f);
    MaxMovePerLaneTask::encode_row(ctx, row.data());

    std::vector<float> ref_in(MaxMovePerLaneInputEncoder::kInputFloats);
    MaxMovePerLaneInputEncoder::encode(enc.board(), rack, ref_in.data());
    std::vector<float> ref_lab(kLaneLabelFloats);
    encode_lane_targets(compute_lane_targets(enc.board(), rack, d), ref_lab.data());

    ASSERT_EQ(MaxMovePerLaneTask::kInputFloats, int(ref_in.size()));
    ASSERT_EQ(MaxMovePerLaneTask::kLabelFloats, int(ref_lab.size()));
    for (int i = 0; i < MaxMovePerLaneTask::kInputFloats; ++i) ASSERT_EQ(row[i], ref_in[i]);
    for (int i = 0; i < MaxMovePerLaneTask::kLabelFloats; ++i)
      ASSERT_EQ(row[MaxMovePerLaneTask::kInputFloats + i], ref_lab[i]);
  }
}

// read_trajectory_decision on a position-set .gcg (a copy of
// positions/NWL23/face-up-trajectory-set/egotize-lane.gcg): the seat, rack,
// scores and known leave match that set's README, the row uses the open-leaves
// layout, and the bundle has a notation for every legal move, including
// HastyBot's E11 GAVE through the A of INCASED ("E11 G.VE"). Skipped without
// the NWL23 lexicon and leaves.
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
  // Without opp_leave_input, the row only lacks the opponent-leave block.
  const InputEncodingSpec hidden{&dict, false};
  EXPECT_EQ(input_floats(arm), input_floats(hidden) + kOppLeaveCountFloats);
}

// Evidence staging, the C++ port of evidence.py's build_evidence_inputs,
// checked against hand-computed values for a two-candidate evidence set over
// three scored candidates. This catches drift from the Python normalization (a
// missing /rollouts, an unscaled delta, a wrong softmax or sigmoid, a
// mis-gathered move encoding, a footprint class on the wrong channel).
// test_proposal_inference_parity.cpp checks the whole path end to end.
TEST(EvidenceStaging, MatchesHandComputedNormalization) {
  using namespace evidence;
  constexpr int kChannels = 2;
  constexpr int kScored = 3;
  constexpr int kCells = kEvidencePlaneCells;

  // Cached model predictions for three scored candidates, gathered by each
  // evidence candidate's scored index.
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
  // Sentinel-filled, because real buffers are reused from turn to turn: the
  // padding checks below must see stale data cleared, not a fresh zero buffer.
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

  // Padding rows (2, 3) are zeroed in every buffer.
  for (int row = 2; row < kMaxE; ++row) {
    const float* pad_planes = ev_planes.data() + size_t(row) * kNumEvidencePlanes * kCells;
    const float* pad_scalars = ev_scalars.data() + size_t(row) * kNumEvidenceScalars;
    EXPECT_FLOAT_EQ(pad_planes[(kNumObservedPlanes + 2) * kCells + 7], 0.0f);  // a predicted cell
    EXPECT_FLOAT_EQ(pad_planes[0], 0.0f);
    EXPECT_FLOAT_EQ(pad_scalars[0], 0.0f);
    EXPECT_FLOAT_EQ(pad_scalars[kNumEvidenceScalars - 1], 0.0f);
  }

  // move_enc gathered by scored index: 2, then 0.
  EXPECT_FLOAT_EQ(ev_move_enc[0], 7.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[1], 8.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[2], 1.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[3], 2.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[4], 0.0f);
  EXPECT_FLOAT_EQ(ev_move_enc[7], 0.0f);

  // Candidate 0 planes: observed histogram counts / rollouts on channel
  // (head * kSlotsPerCell + slot) at the class's cell, predicted probabilities
  // copied through, and the candidate's footprint one-hot at (slot, anchor cell).
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

  // Candidate 1 (PASS): the pass class has no cell, so the observed and
  // footprint planes stay empty.
  const float* p1 = ev_planes.data() + size_t(kNumEvidencePlanes) * kCells;
  EXPECT_FLOAT_EQ(p1[(0 * kSlotsPerCell + 2) * kCells + 5], 0.0f);
  EXPECT_FLOAT_EQ(p1[(kFootBase + 0) * kCells + (7 * BOARD_SIZE + 7)], 0.0f);

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

  // One more candidate than the padded width throws.
  const std::vector<Move> too_many(kMaxE + 1);
  const std::vector<SimObservation> obs_many(kMaxE + 1);
  const std::vector<int> idx_many(kMaxE + 1, 0);
  EXPECT_THROW(stage_evidence(too_many, obs_many, idx_many, pred, kMaxE, out), std::runtime_error);
  // So does a length mismatch, in either the observations or scored_indices.
  EXPECT_THROW(stage_evidence(std::vector<Move>(2), std::vector<SimObservation>(1),
                              std::vector<int>(2), pred, kMaxE, out),
               std::runtime_error);
  EXPECT_THROW(stage_evidence(std::vector<Move>(2), std::vector<SimObservation>(2),
                              std::vector<int>(1), pred, kMaxE, out),
               std::runtime_error);

  // Exactly the padded width is accepted.
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

  // delta_sq_sum / n (75) below mean^2 (100) gives a negative variance; the std
  // must clamp to exactly 0, never NaN.
  SimObservation neg_var;
  neg_var.n = 2;
  neg_var.delta_sum = 20.0;      // mean 10
  neg_var.delta_sq_sum = 150.0;  // 150/2 - 100 = -25
  stage_evidence(std::vector<Move>(1), std::vector<SimObservation>{neg_var}, std::vector<int>{0},
                 pred, kMaxE, out);
  EXPECT_FLOAT_EQ(sc[4], 0.0f);  // delta_std, clamped
  EXPECT_TRUE(std::isfinite(sc[4]));

  // The empty set (the deployment loop's first pass) masks and zeroes every
  // buffer. Sentinel-filled first, as in MatchesHandComputedNormalization.
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
