// Minimal hand-rolled tests for the engine. Exits nonzero on failure.

#include "scribblez/agent.h"
#include "scribblez/bag.h"
#include "scribblez/binary_log.h"
#include "scribblez/block_decoder.h"
#include "scribblez/board.h"
#include "scribblez/board_planes.h"
#include "scribblez/contingent_map.h"
#include "scribblez/data_loader.h"
#include "scribblez/dictionary.h"
#include "scribblez/game.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/glyph.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/lane_analysis.h"
#include "scribblez/lane_targets.h"
#include "scribblez/leave_values.h"
#include "scribblez/macondo_bot.h"
#include "scribblez/max_move_per_lane_input_encoder.h"
#include "scribblez/max_move_per_lane_task.h"
#include "scribblez/movegen.h"
#include "scribblez/position_encoder.h"
#include "scribblez/rack.h"
#include "scribblez/streaming_row_buffer.h"
#include "scribblez/tile_counts.h"
#include "scribblez/training_targets.h"
#include "scribblez/training_task.h"
#include "util/grid.h"
#include "util/math.h"
#include "util/string.h"

#include <boost/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::cerr << "CHECK failed: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

using namespace scribblez;

// Layout shorthands for the FULL input layout, which is what these tests
// encode; derived from the block registry (the spec's dict is irrelevant to
// layout math). Blocks the registry does not name individually (the board
// sub-planes, the two cross-check families, the per-move metadata halves) are
// located relative to their block starts.
static const InputEncodingSpec kFullLayout{nullptr, true};
static const int kInputFloats = input_floats(kFullLayout);
static const int kSpatialFloats = spatial_floats(kFullLayout);
static const int kSpatialPlanes = spatial_planes(kFullLayout);
static const int kRowFloats = kInputFloats + kLabelFloats;
static const int kBlankMarkerPlane = BoardPlanes::kBlankMarkerPlane;
static const int kPremiumPlane0 = BoardPlanes::kPremiumPlane0;
static const int kSelfPlacementPlane =
  spatial_block_plane0(kFullLayout, SpatialBlockId::kSelfPlacement);
static const int kOppPlacementPlane =
  spatial_block_plane0(kFullLayout, SpatialBlockId::kOppPlacement);
static const int kHorizontalCrossCheckPlane0 =
  spatial_block_plane0(kFullLayout, SpatialBlockId::kCrossChecks);
static const int kVerticalCrossCheckPlane0 =
  kHorizontalCrossCheckPlane0 + kHorizontalCrossCheckPlanes;
static const int kRackCountOffset = scalar_block_offset(kFullLayout, ScalarBlockId::kRackCounts);
static const int kUnseenPoolOffset = scalar_block_offset(kFullLayout, ScalarBlockId::kUnseenPool);
static const int kScoreDiffOffset = scalar_block_offset(kFullLayout, ScalarBlockId::kScoreDiff);
static const int kMoveMetaOffset = scalar_block_offset(kFullLayout, ScalarBlockId::kMoveMeta);

static Dictionary tiny_dict() {
  return Dictionary::build_from_words({"CAT", "CATS", "AT",     "AS",     "BAT", "BATS", "HE",
                                       "TO",  "ON",   "NO",     "IT",     "IS",  "OAT",  "OATS",
                                       "HAT", "HATS", "RAT",    "RATS",   "DOG", "GOD",  "GO",
                                       "OD",  "DO",   "AERIES", "PARTIED"});
}

static void test_dict_basic() {
  Dictionary d = tiny_dict();
  CHECK(d.contains("CAT"));
  CHECK(d.contains("cat"));  // case-insensitive
  CHECK(!d.contains("CATX"));
  CHECK(!d.contains("Z"));
  CHECK(d.contains("AERIES"));
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
  for (int i = 0; i < n; ++i) mask |= static_cast<uint16_t>(1u << (lane0 + i));
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
  uint16_t mask = static_cast<uint16_t>(rel_mask << lane0);
  return Move::play(horizontal, start, mask, score, played.data(), n);
}

static void test_movegen_opening() {
  Dictionary d = tiny_dict();
  Board b;
  MoveGenerator gen(b, d);
  // Opening rack with letters CATSO -> can play CAT, CATS, etc., must cover center.
  Rack r = rack_from("CATSOHE");
  auto moves = gen.generate(r);
  CHECK(!moves.empty());
  // Every opening move must cover the center square (CENTER, CENTER).
  // The board is empty, so placements are at consecutive squares from start.
  for (const auto& m : moves) {
    CHECK(m.type() == MoveType::PLAY);
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
    CHECK(covers);
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
  CHECK(best_move != nullptr);
  CHECK(d.contains(best_move->main_word(b)));
}

static void test_movegen_cross_word() {
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
  CHECK(!moves.empty());
  bool found_cats = false;
  for (const auto& m : moves) {
    if (m.main_word(b) == "CATS") found_cats = true;
  }
  CHECK(found_cats);
}

static void test_bingo_bonus() {
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
  CHECK(found_partied);
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

static void test_gaddag_vs_dawg_inmemory() {
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

static void test_board_caches_incremental_matches_full() {
  Dictionary d = medium_dict();
  cache_consistency_stress(d, "medium_dict", 99887766u, /*games=*/30, /*steps_per_game=*/10);
}

// If the real lexicon is present locally, cross-validate against it too and
// sanity-check a few known NWL words. The path comes from a compile-time define
// (SCRIBBLEZ_DEFAULT_KWG, set by CMake to data/lexica/NWL23.kwg). Skipped (not
// failed) when the define is absent or the file is missing -- the .kwg binary
// is not committed.
static void test_real_kwg_optional() {
#ifdef SCRIBBLEZ_DEFAULT_KWG
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) {
    std::cout << "  (no lexicon at " << path << "; skipping real-lexicon cross-validation)\n";
    return;
  }
  Dictionary d = Dictionary::load_kwg(path);
  CHECK(d.contains("QI"));
  CHECK(d.contains("MUZJIKS"));
  CHECK(d.contains("PARTIED"));
  CHECK(!d.contains("QXZ"));
  cross_validate(d, "real-kwg", 99u, /*games=*/6, /*steps_per_game=*/8);
#else
  std::cout << "  (SCRIBBLEZ_DEFAULT_KWG undefined; skipping real-lexicon cross-validation)\n";
#endif
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

static void test_encoder_basic_layout() {
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
  GameStateEncoder enc{InputEncodingSpec{&d, true}};
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
  CHECK(out[c_plane * 225 + 7 * 15 + 7] == 1.0f);
  CHECK(out[d_plane * 225 + 3 * 15 + 3] == 1.0f);  // blank-as-D still lights the D plane

  // Blank-marker plane is index 26: 1 at (3,3), 0 at (7,7).
  CHECK(out[26 * 225 + 3 * 15 + 3] == 1.0f);
  CHECK(out[26 * 225 + 7 * 15 + 7] == 0.0f);

  // Premium planes (27..30) are board-static; just verify they are emitted as
  // 0/1 (no garbage left over from the -1.0 sentinel).
  for (int p = 27; p <= 30; ++p) {
    for (int i = 0; i < 225; ++i) {
      float v = out[p * 225 + i];
      CHECK(v == 0.0f || v == 1.0f);
    }
  }

  // Self last-placement plane (31): only (7,7) is lit (p0's own most recent
  // move). Opponent last-placement plane (32): only (3,3) is lit (p1's most
  // recent move).
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      const float self_expected = (r == 7 && c == 7) ? 1.0f : 0.0f;
      const float opp_expected = (r == 3 && c == 3) ? 1.0f : 0.0f;
      CHECK(out[kSelfPlacementPlane * 225 + r * 15 + c] == self_expected);
      CHECK(out[kOppPlacementPlane * 225 + r * 15 + c] == opp_expected);
    }
  }

  const float* scalars = out.data() + kSpatialFloats;

  // Rack: raw per-tile counts at kRackCountOffset.
  CHECK(scalars[kRackCountOffset + Tile::from_char('Q')] == 1.0f);
  CHECK(scalars[kRackCountOffset + Tile::from_char('Z')] == 1.0f);
  CHECK(scalars[kRackCountOffset + 26] == 1.0f);  // blank count in rack
  CHECK(scalars[kRackCountOffset + Tile::from_char('A')] == 0.0f);

  // Unseen pool: per-letter thermometer at kUnseenPoolOffset. The pool is
  // TILE_COUNTS minus board and active_rack only (opp-rack tiles, if any,
  // remain in the pool from the POV).
  const float* pool = scalars + kUnseenPoolOffset;
  float pool_sum = 0.0f;
  for (int i = 0; i < kUnseenPoolThermoFloats; ++i) pool_sum += pool[i];
  CHECK(pool_sum == 95.0f);  // 100 - 2 on board - 3 in rack
  // A: all 9 unseen -> region fully set.
  CHECK(pool[pool_region_start(0) + 0] == 1.0f);
  CHECK(pool[pool_region_start(0) + 8] == 1.0f);
  // C: 1 of 2 unseen (one C on board) -> first slot set, hole at tail.
  CHECK(pool[pool_region_start(Tile::from_char('C')) + 0] == 1.0f);
  CHECK(pool[pool_region_start(Tile::from_char('C')) + 1] == 0.0f);
  // Blank: 1 on board (blank-D) + 1 in rack -> 0 unseen.
  CHECK(pool[pool_region_start(26) + 0] == 0.0f);
  CHECK(pool[pool_region_start(26) + 1] == 0.0f);

  // Score-diff thermometer at kScoreDiffOffset: diff = 50 - 30 = 20.
  const float* sd = scalars + kScoreDiffOffset;
  float sd_sum = 0.0f;
  for (int i = 0; i < kScoreDiffThermoFloats; ++i) sd_sum += sd[i];
  CHECK(sd_sum == static_cast<float>(kScoreDiffClip + 20 + 1));  // bins [0..diff+clip]
  CHECK(sd[0] == 1.0f);                                          // diff >= -clip always
  CHECK(sd[kScoreDiffClip + 20] == 1.0f);
  CHECK(sd[kScoreDiffClip + 20 + 1] == 0.0f);

  // Last-2-move metadata at kMoveMetaOffset: self move (p0's C play) then
  // opponent move (p1's blank-D play); both are 1-glyph PLAYs.
  const float* meta = scalars + kMoveMetaOffset;
  CHECK(meta[static_cast<int>(MoveType::PLAY)] == 1.0f);
  CHECK(meta[static_cast<int>(MoveType::EXCHANGE)] == 0.0f);
  CHECK(meta[static_cast<int>(MoveType::PASS)] == 0.0f);
  CHECK(meta[kMoveMetaTypeFloats] == 1.0f);  // self num_glyphs
  const float* opp_meta = meta + kMoveMetaFloatsPerMove;
  CHECK(opp_meta[static_cast<int>(MoveType::PLAY)] == 1.0f);
  CHECK(opp_meta[kMoveMetaTypeFloats] == 1.0f);  // opp num_glyphs
}

static void test_encoder_last_opp_plane_mask() {
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
  GameStateEncoder enc{InputEncodingSpec{&d, true}};
  enc.apply_move(p0_play);
  enc.apply_move(opp_play);

  Rack active_rack;
  std::vector<float> out(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, out.data());

  const float* plane = out.data() + kOppPlacementPlane * 225;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      const float expected = ((r == 7 && c == 6) || (r == 7 && c == 8)) ? 1.0f : 0.0f;
      CHECK(plane[r * 15 + c] == expected);
    }
  }
  // The opponent's num_glyphs reflects placements (2 = C, T), not cells walked.
  const float* opp_meta = out.data() + kSpatialFloats + kMoveMetaOffset + kMoveMetaFloatsPerMove;
  CHECK(opp_meta[kMoveMetaTypeFloats] == 2.0f);
}

static void test_encoder_flip_symmetry() {
  using namespace scribblez::binlog;

  // p0 single 'B' at (3,5); p1 vertical "AX" at (0,4) (mask=0b11).
  Move p0_play =
    make_play_full(3, 5, /*horizontal=*/true, 0b1, 30, {Glyph::of(Tile::from_char('B'))});

  Move opp_play =
    make_play_full(0, 4, /*horizontal=*/false, 0b11, 12,
                   {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('X'))});

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d, true}};
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
    CHECK(normal[i] == flipped[i]);
  }
  // Every spatial plane (including both placement planes) is transposed under
  // the flip.
  for (int p = 0; p < kSpatialPlanes; ++p) {
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 15; ++c) {
        CHECK(flipped[p * 225 + r * 15 + c] == normal[p * 225 + c * 15 + r]);
      }
    }
  }
}

// Single-tile plays whose only word is perpendicular to the placement axis
// (hooking S onto the I of QI to form vertical IS / SI) come from the
// transposed pass; a single tile that also forms a word along the horizontal
// axis (the S of QIS) is emitted exactly once, from the horizontal pass.
static void test_movegen_single_tile_vertical_hooks() {
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
  CHECK(qis == 1);       // QIS: S at (7,9), horizontal pass
  CHECK(is_below == 1);  // IS: S at (8,8), vertical-only hook
  CHECK(si_above == 1);  // SI: S at (6,8), vertical-only hook
  CHECK(static_cast<int>(plays.size()) == 3);
  const std::vector<Move> dawg = gen.generate(r, GenAlgo::DAWG);
  CHECK(dawg.size() == plays.size());
  std::cout << "test_movegen_single_tile_vertical_hooks passed\n";
}

static void test_encoder_cross_check_planes_qi() {
  using namespace scribblez::binlog;

  Dictionary d = medium_dict();

  // p0 opens with horizontal "QI" at (7,7)..(7,8). We then encode from p1's
  // POV (active after one move) and verify directional per-letter cross-check
  // planes around that word.
  Move qi_play = make_play_full(7, 7, /*horizontal=*/true, 0b11, 22,
                                {Glyph::of(Tile::from_char('Q')), Glyph::of(Tile::from_char('I'))});

  GameStateEncoder enc{InputEncodingSpec{&d, true}};
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
      const char ch = static_cast<char>('A' + l);
      const float expected = has(letters, ch) ? 1.0f : 0.0f;
      CHECK(h_cross_check(Tile::of(l), r, c) == expected);
    }
  };

  const auto assert_vertical_set = [&](int r, int c, const std::initializer_list<char>& letters) {
    for (int l = 0; l < 26; ++l) {
      const char ch = static_cast<char>('A' + l);
      const float expected = has(letters, ch) ? 1.0f : 0.0f;
      CHECK(v_cross_check(Tile::of(l), r, c) == expected);
    }
  };

  // Horizontal hooks after QI:
  //   - right of I: QIS -> only 'S'
  //   - left of Q: none in this fixture dictionary
  assert_horizontal_set(7, 9, {'S'});
  assert_horizontal_set(7, 6, {});

  // Vertical hooks after QI:
  //   - below Q: QI -> only 'I'
  //   - above I: AI BI GI HI KI LI MI OI PI QI SI TI XI
  //   - below I: ID IF IN IS IT
  assert_vertical_set(8, 7, {'I'});
  assert_vertical_set(6, 7, {});
  assert_vertical_set(6, 8, {'A', 'B', 'G', 'H', 'K', 'L', 'M', 'O', 'P', 'Q', 'S', 'T', 'X'});
  assert_vertical_set(8, 8, {'D', 'F', 'N', 'S', 'T'});

  // Occupied squares never carry cross-check planes.
  assert_horizontal_set(7, 7, {});
  assert_horizontal_set(7, 8, {});
  assert_vertical_set(7, 7, {});
  assert_vertical_set(7, 8, {});

  // Spot-check that cells with no cross-check stay zero in both families.
  const Tile a = Tile::from_char('A');
  const Tile z = Tile::from_char('Z');
  CHECK(h_cross_check(a, 0, 0) == 0.0f);
  CHECK(h_cross_check(z, 14, 14) == 0.0f);
  CHECK(v_cross_check(a, 0, 0) == 0.0f);
  CHECK(v_cross_check(z, 14, 14) == 0.0f);
}

// The production replay path (PositionEncoder, used by both the streaming and
// disk pipelines) must emit lexicon-true cross-check planes: it seeds the
// board's move-generation caches from its dictionary. Without that seeding the
// planes degrade silently to all-26-letters adjacency masks, so this checks a
// square whose legal hook set is a strict subset.
static void test_position_encoder_cross_check_planes_lexical() {
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

  PositionEncoder enc(InputEncodingSpec{&d, true});
  std::vector<float> row(kRowFloats, 0.0f);
  enc.encode_row<PostMoveValueTask>(storage.view(), /*sampled_turn=*/0, /*post_move=*/true,
                                    /*flip=*/false, row.data());

  auto h_cross_check = [&row](char ch, int r, int c) {
    return row[(kHorizontalCrossCheckPlane0 + Tile::from_char(ch).index()) * 225 + r * 15 + c];
  };
  // Right of the I, only QIS extends horizontally (per the fixture dictionary):
  // 'S' is set and every other letter is clear.
  for (char ch = 'A'; ch <= 'Z'; ++ch) {
    CHECK(h_cross_check(ch, 7, 9) == (ch == 'S' ? 1.0f : 0.0f));
  }
  std::cout << "test_position_encoder_cross_check_planes_lexical passed\n";
}

static void test_encoder_forced_score_diff_isolation() {
  using namespace scribblez::binlog;

  Move p0_play =
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 17, {Glyph::of(Tile::from_char('A'))});
  Move p1_play =
    make_play_full(7, 8, /*horizontal=*/true, 0b1, 9, {Glyph::of(Tile::from_char('T'))});

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d, true}};
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
  const int score_hi = score_lo + kScoreDiffThermoFloats;

  // Only the score-diff thermometer block should differ.
  for (int i = 0; i < kInputFloats; ++i) {
    if (i >= score_lo && i < score_hi) continue;
    CHECK(normal[i] == forced[i]);
  }

  // Forced block must represent score_diff=123 as a thermometer.
  const float* sd = forced.data() + score_lo;
  CHECK(sd[kScoreDiffClip + 123] == 1.0f);
  CHECK(sd[kScoreDiffClip + 124] == 0.0f);
}

static void test_encoder_nonplay_last_move_metadata() {
  using namespace scribblez::binlog;

  // p0 PASS, p1 EXCHANGE(1 tile). After two plies active is p0 again.
  Move p0_pass = Move::pass();
  TileCounts ex_tiles;
  ex_tiles.add(Tile::from_char('A'));
  Move p1_exchange = Move::exchange(ex_tiles);

  Dictionary d = medium_dict();
  GameStateEncoder enc{InputEncodingSpec{&d, true}};
  enc.apply_move(p0_pass);
  enc.apply_move(p1_exchange);

  Rack active_rack;
  std::vector<float> out(kInputFloats, 0.0f);
  enc.encode_input(enc.active_player(), active_rack, /*apply_flip=*/false, out.data());

  const float* scalars = out.data() + kSpatialFloats;
  const float* self_meta = scalars + kMoveMetaOffset;
  const float* opp_meta = self_meta + kMoveMetaFloatsPerMove;

  CHECK(self_meta[static_cast<int>(MoveType::PLAY)] == 0.0f);
  CHECK(self_meta[static_cast<int>(MoveType::EXCHANGE)] == 0.0f);
  CHECK(self_meta[static_cast<int>(MoveType::PASS)] == 1.0f);
  CHECK(self_meta[kMoveMetaTypeFloats] == 0.0f);

  CHECK(opp_meta[static_cast<int>(MoveType::PLAY)] == 0.0f);
  CHECK(opp_meta[static_cast<int>(MoveType::EXCHANGE)] == 1.0f);
  CHECK(opp_meta[static_cast<int>(MoveType::PASS)] == 0.0f);
  CHECK(opp_meta[kMoveMetaTypeFloats] == 1.0f);

  // Both last-placement planes must be all zero because neither last move is PLAY.
  for (int i = 0; i < 225; ++i) {
    CHECK(out[kSelfPlacementPlane * 225 + i] == 0.0f);
    CHECK(out[kOppPlacementPlane * 225 + i] == 0.0f);
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

  scribblez::Move make_move(const scribblez::MoveRequest& req) override {
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
    pre.turn_index = static_cast<int>(i);
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

// Count of `letter` placed as a natural tile (not a designated blank) by `m`.
static int natural_letter_count(const Move& m, int letter_index) {
  int n = 0;
  for (int i = 0; i < m.num_glyphs(); ++i) {
    const Glyph g = m.glyph(i);
    if (!g.is_blank() && g.letter().index() == letter_index) ++n;
  }
  return n;
}

// The ContingentMap's single phantom-blank generation must reproduce, per
// (drawn letter, lane), the best score a real generation over rack ∪ {letter}
// finds among plays that consume the added copy -- including the rescoring of
// the phantom at the letter's face value -- and, for the rack-alone lanes, the
// plain lane-target reduction.
static void test_contingent_map_matches_per_tile_generation() {
  Dictionary dict = medium_dict();

  int leave_positions = 0, full_positions = 0;
  long columns = 0;
  for (uint64_t seed : std::vector<uint64_t>{11, 77}) {
    GameLogStorage log = play_test_game(dict, seed);
    Board board;
    for (size_t t = 0; t < log.turns.size(); ++t) {
      const TurnRecord& rec = log.turns[t];

      // The full pre-move rack: no replenishment draw is possible, so every
      // contingent column must be empty while the rack-alone lanes still
      // match the plain lane-target reduction.
      if (rec.rack_before.size() == RACK_SIZE && t % 4 == 0 && !board.empty_board()) {
        uint8_t unseen[27];
        compute_unseen_pool(unseen, board, rec.rack_before);
        const ContingentMap cm = ContingentMap::compute(board, rec.rack_before, unseen, dict);
        const LaneTargets lt = compute_lane_targets(board, rec.rack_before, dict);
        for (int lane = 0; lane < kNumLanes; ++lane) {
          const LaneBest& ref =
            lane < kLanesPerAxis ? lt.rows[lane] : lt.cols[lane - kLanesPerAxis];
          CHECK(cm.rack_best(lane).score == (ref.has_move ? ref.max_score : -1));
          for (int kind = 0; kind < kLaneTileKinds; ++kind) {
            CHECK(cm.best(kind, lane).score == -1);
          }
        }
        ++full_positions;
      }

      if (rec.move.type() != MoveType::PLAY) continue;
      board.apply(rec.move);

      // The post-move snapshot -- the training case: rack = the mover's leave,
      // board = after the move.
      Rack leave = rec.rack_before;
      for (int i = 0; i < rec.move.num_glyphs(); ++i) leave.remove(rec.move.glyph(i).rack_tile());
      uint8_t unseen[27];
      compute_unseen_pool(unseen, board, leave);
      const ContingentMap cm = ContingentMap::compute(board, leave, unseen, dict);

      // Rack-alone lanes == the lane-target reduction of a plain generation.
      const LaneTargets lt = compute_lane_targets(board, leave, dict);
      for (int lane = 0; lane < kNumLanes; ++lane) {
        const LaneBest& ref = lane < kLanesPerAxis ? lt.rows[lane] : lt.cols[lane - kLanesPerAxis];
        CHECK(cm.rack_best(lane).score == (ref.has_move ? ref.max_score : -1));
      }

      // Every drawable letter column == a real generation over leave ∪ {L},
      // restricted to plays that consume the added copy.
      for (int L = 0; L < 26; ++L) {
        if (unseen[L] == 0) continue;
        Rack aug = leave;
        aug.add(Tile::of(L));
        MoveGenerator gen(board, dict);
        const std::vector<Move> moves = gen.generate(aug);
        std::array<int, kNumLanes> ref;
        ref.fill(-1);
        PlacedTile placed[RACK_SIZE];
        for (const Move& m : moves) {
          if (natural_letter_count(m, L) <= leave.count(Tile::of(L))) continue;
          const int p = decode_placements(m, placed);
          const LaneAssignments la = compute_lane_assignments(board, m, placed, p);
          for (int i = 0; i < la.count; ++i) {
            const int lane = (la.items[i].horizontal ? 0 : kLanesPerAxis) + la.items[i].lane_index;
            ref[lane] = std::max(ref[lane], static_cast<int>(m.score()));
          }
        }
        for (int lane = 0; lane < kNumLanes; ++lane) {
          CHECK(static_cast<int>(cm.best(L, lane).score) == ref[lane]);
        }
        ++columns;
      }
      ++leave_positions;
    }
  }
  std::cout << "  contingent map parity OK (" << leave_positions << " leaves, " << full_positions
            << " full racks, " << columns << " letter columns)\n";
}

// The base layout is the full layout with the contingent tails spliced out of
// both sections: encoding the same position under both specs must agree
// float-for-float on every shared block. Serving consumers rely on this prefix
// property when running a base-layout model against full-layout rows.
static void test_base_layout_is_full_minus_contingent_tails() {
  Dictionary d = medium_dict();
  const InputEncodingSpec full{&d, true};
  const InputEncodingSpec base{&d, false};

  Move cat = make_play_full(7, 7, /*horizontal=*/true, 0b111, 12,
                            {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                             Glyph::of(Tile::from_char('T'))});
  GameStateEncoder full_enc{full};
  GameStateEncoder base_enc{base};
  full_enc.apply_move(cat);
  base_enc.apply_move(cat);
  Rack rack = rack_from("RSE");

  std::vector<float> full_row(input_floats(full), -1.0f);
  std::vector<float> base_row(input_floats(base), -1.0f);
  full_enc.encode_input(full_enc.active_player(), rack, /*apply_flip=*/false, full_row.data());
  base_enc.encode_input(base_enc.active_player(), rack, /*apply_flip=*/false, base_row.data());

  // Shared spatial prefix, then shared scalar prefix (the full row's scalar
  // block starts after its extra contingent planes).
  CHECK(std::memcmp(base_row.data(), full_row.data(),
                    sizeof(float) * static_cast<size_t>(spatial_floats(base))) == 0);
  CHECK(std::memcmp(base_row.data() + spatial_floats(base), full_row.data() + spatial_floats(full),
                    sizeof(float) * static_cast<size_t>(scalar_floats(base))) == 0);

  // The splice is not vacuous: the full row's contingent tails carry content.
  float tail_sum = 0.0f;
  for (int i = spatial_floats(base); i < spatial_floats(full); ++i)
    tail_sum += std::abs(full_row[i]);
  for (int i = spatial_floats(full) + scalar_floats(base); i < input_floats(full); ++i)
    tail_sum += std::abs(full_row[i]);
  CHECK(tail_sum > 0.0f);
  std::cout << "test_base_layout_is_full_minus_contingent_tails passed\n";
}

// A hand-checked position: horizontal CAT at (7,7..9), rack {R}. Verifies the
// specific contingent entries, the phantom-blank rescoring, and the encoded
// planes' footprint painting and flip symmetry.
static void test_contingent_map_cat_board() {
  Dictionary dict = medium_dict();
  Board board;
  board.apply(make_play_full(7, 7, /*horizontal=*/true, 0b111, 12,
                             {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                              Glyph::of(Tile::from_char('T'))}));
  Rack rack;
  rack.add(Tile::from_char('R'));
  uint8_t unseen[27];
  compute_unseen_pool(unseen, board, rack);
  const ContingentMap cm = ContingentMap::compute(board, rack, unseen, dict);

  const int kS = Tile::from_char('S').index();
  const int kE = Tile::from_char('E').index();
  const int row7 = 7;                  // row lane 7
  const int col8 = kLanesPerAxis + 8;  // column lane 8

  // Drew S -> CATS along row 7: S placed at (7,10), a plain square, so the
  // score is the four face values (C3 A1 T1 S1).
  CHECK(cm.best(kS, row7).score == 6);
  CHECK(cm.best(kS, row7).placed_mask == (1u << 10));
  // Drew a blank -> CATER along row 7 (blank designated E at (7,10), rack R
  // on the (7,11) DLS): 3 + 1 + 1 + 0 + 2.
  CHECK(cm.best(kLaneBlankKind, row7).score == 7);
  // Drew a real E -> the same CATER with the E's face value restored: 8.
  CHECK(cm.best(kE, row7).score == 8);
  // Drew E -> EAR down column 8 (E at (6,8) DLS, board A, rack R at (8,8)
  // DLS): 2 + 1 + 2, with the phantom-E rescoring supplying the doubled E.
  CHECK(cm.best(kE, col8).score == 5);
  CHECK(cm.best(kE, col8).placed_mask == ((1u << 6) | (1u << 8)));
  // Rack alone -> the one-tile vertical hook AR below the A (R at (8,8) DLS):
  // 1 + 2.
  CHECK(cm.rack_best(col8).score == 3);
  CHECK(cm.rack_best(row7).score == -1);

  // Painted planes: the max plane's value at (7,10) is the best entry through
  // that cell -- the drew-E CATER (8); a flip moves it to (10,7). Every value
  // stays in [0,1].
  std::vector<float> planes(kContingentPlanes * kBoardCells, 0.0f);
  cm.encode_planes(/*flip=*/false, planes.data());
  CHECK(planes[7 * BOARD_SIZE + 10] == 8.0f / kContingentScoreClip);
  std::vector<float> flipped(kContingentPlanes * kBoardCells, 0.0f);
  cm.encode_planes(/*flip=*/true, flipped.data());
  CHECK(flipped[10 * BOARD_SIZE + 7] == 8.0f / kContingentScoreClip);
  std::vector<float> scalars(kContingentScalarFloats, -1.0f);
  cm.encode_scalars(scalars.data());
  for (float v : scalars) CHECK(v >= 0.0f && v <= 1.0f);
  CHECK(scalars[kS] == 6.0f / kContingentScoreClip);
  std::cout << "test_contingent_map_cat_board passed\n";
}

// GameStateEncoder, replayed against a live in-memory replay, faithfully
// reproduces every eligible position -- proven by running movegen on both and
// demanding identical legal-play sets at the pre-move snapshot of each turn.
static void test_extract_positions_movegen_roundtrip() {
  Dictionary dict = medium_dict();

  // A handful of games at different seeds; for every eligible position we
  // compare encoder state against the independent live snapshot vector.
  const std::vector<uint64_t> seeds = {42, 1337, 0xDEADBEEFULL};
  long positions_compared = 0;

  for (uint64_t seed : seeds) {
    scribblez::GameLogStorage log = play_test_game(dict, seed);
    CHECK(!log.turns.empty());

    auto live_snaps = live_replay_all_snapshots(log);

    scribblez::GameStateEncoder enc{scribblez::InputEncodingSpec{&dict, true}};
    // The encoder no longer tracks racks (an outside observer cannot see
    // opponent draws). The test, however, has full information, so we
    // maintain a parallel rack pair alongside the encoder.
    std::array<scribblez::Rack, 2> racks = {log.initial_racks[0], log.initial_racks[1]};

    size_t snap_idx = 0;
    for (size_t k = 0; k < log.turns.size(); ++k) {
      const auto& turn = log.turns[k];

      // ---- pre-move snapshot ----
      CHECK(snap_idx < live_snaps.size());
      const LiveSnapshot& pre = live_snaps[snap_idx++];
      CHECK(pre.kind == scribblez::PositionKind::kPreMove);
      const int active = enc.active_player();
      CHECK(active == pre.active_player);
      CHECK(enc.score(active) == pre.score_active);
      CHECK(enc.score(1 - active) == pre.score_opp);
      CHECK(boards_equal(enc.board(), pre.board));
      CHECK(racks_equal(racks[active], pre.rack_active));
      CHECK(moves_equal_for_replay(enc.last_move_by(1 - active), pre.last_opp_move));
      check_movegen_equiv(dict, enc.board(), racks[active], pre.board, pre.rack_active,
                          "GameStateEncoder-pre");
      ++positions_compared;

      // ---- post-move snapshot (PLAY only) ----
      if (turn.move.type() == scribblez::MoveType::PLAY) {
        CHECK(snap_idx < live_snaps.size());
        const LiveSnapshot& post = live_snaps[snap_idx++];
        CHECK(post.kind == scribblez::PositionKind::kPostMove);

        // Materialize post-state from the encoder + parallel racks by hand
        // and compare.
        scribblez::Board post_board = enc.board();
        post_board.apply(turn.move);
        scribblez::Rack post_rack = racks[active];
        const int n = turn.move.num_glyphs();
        for (int g = 0; g < n; ++g) post_rack.remove(turn.move.glyph(g).rack_tile());
        const int post_score = enc.score(active) + turn.move.score();
        CHECK(boards_equal(post_board, post.board));
        CHECK(racks_equal(post_rack, post.rack_active));
        CHECK(post_score == post.score_active);
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
    CHECK(snap_idx == live_snaps.size());
  }
  CHECK(positions_compared > 0);
  std::cout << "  GameStateEncoder replay+movegen round-trip OK (" << positions_compared
            << " positions across " << seeds.size() << " games)\n";
}

// End-to-end: play a game, write it through BinaryLogWriter to disk, register
// the resulting .slog file with DataLoader, decode all rows, and verify that
// (a) every label tail matches a valid (game, active POV) and (b) running
// movegen on a board freshly reconstructed from the .slog file produces the
// same legal-play set as the live game.
static void test_binary_log_file_and_data_loader_roundtrip() {
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
  CHECK(slogs.size() == 1);

  // Verify the on-disk header is self-consistent.
  const fs::path& slog = slogs.front();
  const int64_t fsize = static_cast<int64_t>(fs::file_size(slog));
  std::vector<char> raw(fsize);
  {
    std::ifstream f(slog, std::ios::binary);
    f.read(raw.data(), fsize);
    CHECK(f);
  }
  const auto* hdr = reinterpret_cast<const scribblez::binlog::FileHeader*>(raw.data());
  CHECK(hdr->magic == scribblez::binlog::kMagic);
  CHECK(hdr->version == scribblez::binlog::kVersion);
  CHECK(hdr->num_games == static_cast<uint32_t>(kGames));
  // Training expands each game into one row per eligible (pre-endgame) turn, so
  // the loader's position count is the sum of those across all games.
  int64_t total_positions = 0;
  for (const auto& log : logs)
    for (const auto& turn : log.turns)
      if (turn.bag_size_before > 0) ++total_positions;
  CHECK(total_positions > 0);
  CHECK(static_cast<int64_t>(hdr->num_sample_positions) == total_positions);

  // Register with DataLoader and drain rows via epoch_start/load_batch
  // for both pre-move and post-move phases.
  scribblez::binlog::DataLoader::Params dl_params;
  dl_params.spec = {&dict, true};
  dl_params.num_worker_threads = 2;
  dl_params.num_prefetch_threads = 1;
  scribblez::binlog::DataLoader loader(dl_params);
  loader.add_file(slog.string(), total_positions, fsize);
  CHECK(loader.num_positions() == total_positions);

  const int row_size = kRowFloats;

  // Helper: drain one full epoch into a vector.
  auto drain_epoch = [&](bool post_move) {
    scribblez::binlog::DataLoader::EpochConfig cfg;
    cfg.batch_size = static_cast<int>(total_positions);
    cfg.post_move = post_move;
    cfg.apply_symmetry = false;
    cfg.seed = 1;
    loader.epoch_start(cfg);
    std::vector<float> out(total_positions * row_size);
    int n = loader.load_batch(out.data());
    CHECK(n == static_cast<int>(total_positions));
    CHECK(loader.load_batch(out.data()) == 0);  // epoch exhausted
    return out;
  };

  std::vector<float> pre_rows = drain_epoch(/*post_move=*/false);
  std::vector<float> post_rows = drain_epoch(/*post_move=*/true);

  // Combine for validation.
  const int n_samples = static_cast<int>(total_positions) * 2;
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
    const float* row = rows.data() + static_cast<int64_t>(i) * row_size;
    const int w = static_cast<int>(row[label_off + 0]);
    const int dd = static_cast<int>(row[label_off + 1]);
    const int l = static_cast<int>(row[label_off + 2]);
    // Score-diff target is a single scalar: the clipped final differential.
    const int sd = static_cast<int>(row[label_off + scribblez::kWldFloats]);
    CHECK(w + dd + l == 1);  // exactly one of W/D/L
    CHECK(valid_labels.count({w, dd, l, sd}) == 1);
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
    CHECK(gm.num_turns == logs[gi].turns.size());

    const auto* ir =
      reinterpret_cast<const scribblez::binlog::InitialRacks*>(raw.data() + gm.start_offset);
    const auto* turns = reinterpret_cast<const scribblez::binlog::TurnBlob*>(
      raw.data() + gm.start_offset + sizeof(scribblez::binlog::InitialRacks));

    // Reconstruct the initial racks from the on-disk bytes and replay.
    const Rack& r0_init = ir->p0;
    const Rack& r1_init = ir->p1;
    CHECK(racks_equal(r0_init, logs[gi].initial_racks[0]));
    CHECK(racks_equal(r1_init, logs[gi].initial_racks[1]));

    auto live_snaps = live_replay_all_snapshots(logs[gi]);
    scribblez::GameStateEncoder enc{scribblez::InputEncodingSpec{&dict, true}};
    std::array<scribblez::Rack, 2> racks = {r0_init, r1_init};

    size_t snap_idx = 0;
    for (uint32_t k = 0; k < gm.num_turns; ++k) {
      CHECK(snap_idx < live_snaps.size());
      const LiveSnapshot& pre = live_snaps[snap_idx++];
      const int active = enc.active_player();
      CHECK(boards_equal(enc.board(), pre.board));
      CHECK(racks_equal(racks[active], pre.rack_active));
      check_movegen_equiv(dict, enc.board(), racks[active], pre.board, pre.rack_active,
                          "file-roundtrip");
      ++compared;

      if (turns[k].move.type() == scribblez::MoveType::PLAY) {
        CHECK(snap_idx < live_snaps.size());
        const LiveSnapshot& post = live_snaps[snap_idx++];
        scribblez::Board post_board = enc.board();
        post_board.apply(turns[k].move);
        CHECK(boards_equal(post_board, post.board));
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
    CHECK(snap_idx == live_snaps.size());
  }
  // `compared` includes one pre-snapshot per turn + one post-snapshot per PLAY
  // turn; total_positions counts only one row per turn (shared between pre/post).
  CHECK(compared >= total_positions);
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
static void test_game_random_opening() {
  Dictionary dict = medium_dict();

  for (int plies : {0, 1, 3, 6}) {
    scribblez::GameLogStorage log = play_random_opening_test_game(dict, /*seed=*/321, plies);
    CHECK(!log.turns.empty());
    CHECK(log.num_random_opening_plies == std::min<int>(plies, static_cast<int>(log.turns.size())));

    // Same seed + same opening length -> the identical move sequence.
    scribblez::GameLogStorage again = play_random_opening_test_game(dict, /*seed=*/321, plies);
    CHECK(again.turns.size() == log.turns.size());
    for (size_t k = 0; k < log.turns.size(); ++k)
      CHECK(moves_equal_for_replay(again.turns[k].move, log.turns[k].move));
  }
  std::cout << "  Game random opening OK\n";
}

// generate_legal_exchanges enumerates every distinct non-empty sub-multiset of
// the rack (duplicates don't multiply the list), and is empty when the bag is
// too small to allow exchanging.
static void test_generate_legal_exchanges() {
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
  CHECK(exchanges.size() == 11);

  // Every move is an EXCHANGE of a distinct non-empty sub-multiset of the rack.
  std::set<std::string> seen;
  for (const Move& m : exchanges) {
    CHECK(m.type() == MoveType::EXCHANGE);
    const int n = m.num_glyphs();
    CHECK(n >= 1 && n <= 4);
    std::string tiles;
    for (int i = 0; i < n; ++i) tiles += m.glyph(i).rack_tile().to_char();
    std::sort(tiles.begin(), tiles.end());
    for (char c : tiles) CHECK(c == 'A' || c == 'B' || c == '?');
    seen.insert(tiles);
  }
  CHECK(seen.size() == exchanges.size());

  // Exchanging is illegal when the bag has fewer than RACK_SIZE tiles.
  MoveRequest starved{board, dict, rack, opp, 0, 0, /*bag_size=*/RACK_SIZE - 1};
  CHECK(generate_legal_exchanges(starved).empty());
  std::cout << "  generate_legal_exchanges OK (" << exchanges.size() << " exchanges)\n";
}

// BinaryLogWriter records a random-opening game's eligible region: begin is the
// position after the last random ply, end is the bag-non-empty prefix, and the
// header's num_sample_positions sums the region widths. The DataLoader sizes
// its epoch from the same regions.
static void test_binary_log_random_opening_region() {
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
  CHECK(slogs.size() == 1);
  const int64_t fsize = static_cast<int64_t>(fs::file_size(slogs.front()));
  std::vector<char> raw(fsize);
  {
    std::ifstream f(slogs.front(), std::ios::binary);
    f.read(raw.data(), fsize);
    CHECK(f);
  }
  const auto* hdr = reinterpret_cast<const scribblez::binlog::FileHeader*>(raw.data());
  const auto* metas = reinterpret_cast<const scribblez::binlog::GameMetadata*>(
    raw.data() + sizeof(scribblez::binlog::FileHeader));
  CHECK(hdr->num_games == static_cast<uint32_t>(kGames));

  int64_t expected_rows = 0;
  for (int i = 0; i < kGames; ++i) {
    CHECK(logs[i].num_random_opening_plies == kPlies);
    int prefix = 0;
    for (const auto& turn : logs[i].turns) {
      if (turn.bag_size_before <= 0) break;
      ++prefix;
    }
    CHECK(metas[i].eligible_begin == kPlies - 1);
    CHECK(metas[i].eligible_end == prefix);
    expected_rows += prefix - (kPlies - 1);
  }
  CHECK(static_cast<int64_t>(hdr->num_sample_positions) == expected_rows);

  scribblez::binlog::DataLoader::Params dl_params;
  dl_params.spec = {&dict, true};
  dl_params.num_worker_threads = 1;
  dl_params.num_prefetch_threads = 1;
  scribblez::binlog::DataLoader loader(dl_params);
  loader.add_file(slogs.front().string(), expected_rows, fsize);
  CHECK(loader.num_positions() == expected_rows);
  std::cout << "  BinaryLogWriter random-opening eligible region OK (" << expected_rows
            << " rows across " << kGames << " games)\n";
}

// ===========================================================================
// Foundation types: Tile / Glyph
// ===========================================================================

static void test_tile_glyph_basics() {
  // Tile::from_char round-trips for letters and the blank marker.
  for (char c = 'A'; c <= 'Z'; ++c) {
    Tile t = Tile::from_char(c);
    CHECK(!t.is_blank());
    CHECK(!t.is_empty());
    CHECK(t.to_char() == c);
    // Lowercase is normalized to uppercase.
    CHECK(Tile::from_char(static_cast<char>(c - 'A' + 'a')) == t);
    CHECK(t.value() == TILE_VALUES[t]);
  }
  CHECK(Tile::from_char('?').is_blank());
  CHECK(Tile::from_char('_').is_blank());
  CHECK(BLANK.value() == 0);

  // Glyph: played-as-blank renders the letter but scores zero and consumes
  // a blank from the rack.
  Glyph plain = Glyph::of(Tile::from_char('Q'));
  Glyph blank_q = Glyph::played(Tile::from_char('Q'), /*is_blank=*/true);
  CHECK(plain.letter() == Tile::from_char('Q'));
  CHECK(blank_q.letter() == Tile::from_char('Q'));
  CHECK(!plain.is_blank());
  CHECK(blank_q.is_blank());
  CHECK(plain.value() == TILE_VALUES[Tile::from_char('Q')]);
  CHECK(blank_q.value() == 0);
  CHECK(plain.rack_tile() == Tile::from_char('Q'));
  CHECK(blank_q.rack_tile() == BLANK);
  CHECK(plain != blank_q);
  CHECK(plain == Glyph::of(Tile::from_char('Q')));

  // Empty/unassigned-blank predicates.
  CHECK(Glyph::empty().is_empty());
  CHECK(!Glyph::blank().is_empty());
  CHECK(Glyph::blank().is_blank());
  CHECK(Glyph::blank().rack_tile() == BLANK);

  // Glyph code 0 means empty -- a default-constructed Board is all-empty
  // by virtue of zero-init, which a lot of code relies on.
  Glyph g;
  CHECK(g.is_empty());
  CHECK(g.code() == 0);
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

static void test_rack_invariants() {
  // Sorted-array invariant survives interleaved add/remove in arbitrary order.
  Rack r;
  CHECK(r.empty());
  CHECK(r.size() == 0);
  CHECK(r.point_value() == 0);
  CHECK(!r.remove(Tile::from_char('A')));  // remove-missing returns false

  const char* in = "QAZZB?A";  // 7 tiles incl two A's, two Z's, one blank
  for (char c : std::string(in)) {
    r.add(c == '?' ? BLANK : Tile::from_char(c));
  }
  CHECK(r.size() == 7);
  CHECK(rack_is_sorted(r));
  CHECK(r.to_string() == "AABQZZ?");  // sorted A..Z then '?'
  CHECK(r.count(Tile::from_char('A')) == 2);
  CHECK(r.count(Tile::from_char('Z')) == 2);
  CHECK(r.count(Tile::from_char('B')) == 1);
  CHECK(r.count(Tile::from_char('X')) == 0);
  CHECK(r.blanks() == 1);

  // point_value: blanks contribute 0, everything else its TILE_VALUES entry.
  int expected = TILE_VALUES[Tile::from_char('A')] * 2 + TILE_VALUES[Tile::from_char('B')] +
                 TILE_VALUES[Tile::from_char('Q')] + TILE_VALUES[Tile::from_char('Z')] * 2;
  CHECK(r.point_value() == expected);

  // remove() removes one occurrence and preserves sortedness.
  CHECK(r.remove(Tile::from_char('A')));
  CHECK(r.count(Tile::from_char('A')) == 1);
  CHECK(r.size() == 6);
  CHECK(rack_is_sorted(r));
  CHECK(r.remove(BLANK));
  CHECK(r.blanks() == 0);
  CHECK(rack_is_sorted(r));
  CHECK(!r.remove(BLANK));  // gone now

  // counts() histogram matches per-tile count() probes.
  TileCounts tc = r.counts();
  for (Tile t = Tile::of(0); t < 27; ++t) {
    int via_tc = tc.count(t);
    int via_probe = r.count(t);
    CHECK(via_tc == via_probe);
  }
}

// ===========================================================================
// Bag
// ===========================================================================

static void test_bag_basics() {
  // Initial composition matches TILE_COUNTS and totals to 100 tiles.
  Bag b(/*seed=*/42);
  int total = 0;
  for (int c : TILE_COUNTS) total += c;
  CHECK(b.size() == total);
  CHECK(b.size() == 100);
  for (int i = 0; i < 27; ++i) CHECK(b.counts()[i] == TILE_COUNTS[i]);

  // Same seed -> identical draw sequence (reproducibility).
  Bag b1(/*seed=*/12345);
  Bag b2(/*seed=*/12345);
  for (int i = 0; i < 100; ++i) {
    auto t1 = b1.draw();
    auto t2 = b2.draw();
    CHECK(t1.has_value() && t2.has_value());
    CHECK(*t1 == *t2);
  }
  CHECK(b1.size() == 0);
  CHECK(!b1.draw().has_value());  // empty bag returns nullopt

  // Different seeds -> sequences differ (almost surely; 100 draws is plenty).
  Bag bA(1), bB(2);
  bool any_diff = false;
  for (int i = 0; i < 100; ++i) {
    auto a = bA.draw();
    auto bb = bB.draw();
    if (a != bb) any_diff = true;
  }
  CHECK(any_diff);

  // Tile-count conservation: drain the bag, the per-tile draw counts must
  // equal TILE_COUNTS exactly.
  std::array<int, 27> drawn{};
  Bag b3(/*seed=*/777);
  while (auto t = b3.draw()) ++drawn[*t];
  for (int i = 0; i < 27; ++i) CHECK(drawn[i] == TILE_COUNTS[i]);
  CHECK(b3.size() == 0);

  // put_back round-trip: drain bag, put a tile back, next draw is that tile.
  Bag b4(/*seed=*/9999);
  while (b4.draw().has_value()) {
  }
  CHECK(b4.size() == 0);
  b4.put_back(Tile::from_char('Q'));
  CHECK(b4.size() == 1);
  auto got = b4.draw();
  CHECK(got.has_value() && *got == Tile::from_char('Q'));
}

// ===========================================================================
// Board::apply -- the exact interleave-with-cross-tiles scenario
// ===========================================================================

static void test_board_apply_interleaves_cross_tiles() {
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

  CHECK(b.at(7, 7).letter() == Tile::from_char('C'));
  CHECK(b.at(7, 8).letter() == Tile::from_char('A'));  // unchanged
  CHECK(b.at(7, 9).letter() == Tile::from_char('T'));
  // Apply on a PASS is a no-op.
  Move pass;
  Board snapshot = b;
  b.apply(pass);
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      CHECK(b.at(r, c).code() == snapshot.at(r, c).code());
    }
  }
}

// ===========================================================================
// Move::main_word
// ===========================================================================

static void test_move_main_word_through_cross() {
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
  CHECK(hook.main_word(b) == "CATS");

  // 2. Through-word: place B at (7,6) and S at (7,10) for "BCATS"? Not a
  //    real word -- but main_word doesn't care about legality. We just
  //    verify that interleaving works.
  Move through = make_play_full(7, 6, /*horizontal=*/true, 0b10001, 0,
                                {Glyph::of(Tile::from_char('B')),    // placed at (7,6)
                                 Glyph::of(Tile::from_char('S'))});  // placed at (7,10)
  CHECK(through.main_word(b) == "BCATS");

  // 3. Blank renders as its designated letter (uppercase), like a regular tile.
  Move with_blank = make_play_full(7, 10, /*horizontal=*/true, 0b1, 0,
                                   {Glyph::played(Tile::from_char('S'), /*is_blank=*/true)});
  CHECK(with_blank.main_word(b) == "CATS");

  // 4. PASS / EXCHANGE produce empty strings.
  Move pass;
  CHECK(pass.main_word(b).empty());
  TileCounts xch_tiles;
  xch_tiles.add(Tile::from_char('A'));
  Move xch = Move::exchange(xch_tiles);
  CHECK(xch.main_word(b).empty());
}

// ===========================================================================
// Movegen: blank placement scores zero, even on a letter premium
// ===========================================================================

static void test_movegen_blank_scores_zero() {
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
  CHECK(score_real > 0);

  Rack rack_blank = rack_from("???????");
  auto moves_blank = gen.generate(rack_blank);
  int score_blank = -1;
  for (const auto& m : moves_blank) {
    if (m.main_word(b) == "CAT") {
      // Verify the C placement is a blank.
      CHECK(m.num_glyphs() == 1);
      CHECK(m.glyph(0).is_blank());
      score_blank = m.score();
      break;
    }
  }
  CHECK(score_blank >= 0);
  // The blank-C contributes 0 letter value; the only word score is from the
  // existing A and T (already on the board). With a blank in the placed
  // position the word score equals the unscored A+T sum, possibly multiplied
  // by a word premium under the placed blank. With the real C the score is
  // strictly greater (C alone is worth 3).
  CHECK(score_blank < score_real);
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
  scribblez::Move make_move(const scribblez::MoveRequest&) override {
    return scribblez::Move::pass();
  }
};

}  // namespace

static void test_game_end_rack_out_bonus() {
  // Greedy agents on a small in-memory dict almost always stalemate (they
  // can't form enough words to drain the bag). With the real lexicon, "out"
  // is the normal end condition. Skip gracefully if no real lexicon is
  // available -- the stalemate test below still covers Game::play()'s other
  // end-of-game arithmetic.
#ifdef SCRIBBLEZ_DEFAULT_KWG
  const char* path = SCRIBBLEZ_DEFAULT_KWG;
  if (!std::ifstream(path).good()) {
    std::cout << "  (no lexicon; skipping rack-out bonus test)\n";
    return;
  }
  Dictionary dict = Dictionary::load_kwg(path);
#else
  std::cout << "  (SCRIBBLEZ_DEFAULT_KWG undefined; skipping rack-out bonus test)\n";
  return;
#endif

  bool found_out = false;
  for (uint64_t seed = 0; seed < 20 && !found_out; ++seed) {
    scribblez::GameLogStorage log = play_test_game(dict, seed);
    if (log.end_reason != "out") continue;
    found_out = true;

    const TurnRecord& last = log.turns.back();
    const int winner = last.player;  // out-going player
    const int loser = 1 - winner;
    const int penalty = log.final_racks[loser].point_value();

    CHECK(log.final_scores[winner] == last.cumulative_scores[winner] + penalty);
    CHECK(log.final_scores[loser] == last.cumulative_scores[loser] - penalty);
    CHECK(log.final_racks[winner].empty());
  }
  CHECK(found_out);
}

static void test_game_end_stalemate_penalty() {
  // Two pass-agents: 6 consecutive zero turns trigger "stalemate" and each
  // player's final = cumulative - their own remaining-rack value (cumulative
  // is 0 for both -- nobody scored).
  Dictionary dict = medium_dict();
  AlwaysPassAgent a0(0, "P0");
  AlwaysPassAgent a1(0, "P1");
  scribblez::Game g(a0, a1, dict, /*seed=*/424242ULL);
  g.play();
  const scribblez::GameLogStorage log = g.extract_log();

  CHECK(log.end_reason == "stalemate");
  CHECK(log.turns.size() == 6);  // 6 zero turns (3 per player)
  for (const auto& t : log.turns) CHECK(t.move.type() == MoveType::PASS);
  for (int p = 0; p < 2; ++p) {
    CHECK(log.final_scores[p] == -log.final_racks[p].point_value());
  }
}

static void test_natural_less() {
  using util::natural_less;
  CHECK(natural_less("pos-2", "pos-10"));  // numeric run compares by value, not lexically
  CHECK(!natural_less("pos-10", "pos-2"));
  CHECK(natural_less("pos-2.gcg", "pos-10.gcg"));
  CHECK(natural_less("pos-09", "pos-10"));  // leading zeros ignored
  CHECK(natural_less("a2", "a2b"));         // a prefix sorts before its extension
  CHECK(!natural_less("pos-1", "pos-1"));   // equal -> not less (irreflexive)
  CHECK(natural_less("abc", "abd"));        // non-digit chars compare lexically

  std::vector<std::string> v = {"pos-10", "pos-2", "pos-1", "pos-20", "pos-3"};
  std::sort(v.begin(), v.end(), natural_less);
  CHECK((v == std::vector<std::string>{"pos-1", "pos-2", "pos-3", "pos-10", "pos-20"}));
}

static bool rack_contains(const Rack& r, Tile want) {
  for (int i = 0; i < r.size(); ++i)
    if (r.tiles()[i].index() == want.index()) return true;
  return false;
}

// Bag::remove (build the unseen pool) and Game::play_from (Monte-Carlo rollout from
// a mid-game position): the deal keeps the post-mover's leave, refills both racks
// from the seeded pool, plays to a natural end, and is deterministic per seed.
static void test_game_play_from() {
  // Bag::remove: a fully-removed letter never comes out of the bag.
  {
    Bag bag(123);
    const Tile a = Tile::from_char('A');
    for (int i = bag.counts()[a.index()]; i > 0; --i) bag.remove(a);
    CHECK(bag.counts()[a.index()] == 0);
    while (auto t = bag.draw()) CHECK(t->index() != a.index());
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

  CHECK(!logs[0].end_reason.empty());                   // reached a natural end
  CHECK(logs[0].initial_racks[0].size() == RACK_SIZE);  // post-mover topped up to 7
  CHECK(logs[0].initial_racks[1].size() == RACK_SIZE);  // on-move drew a fresh rack
  for (int i = 0; i < leave.size(); ++i)                // the leave is preserved
    CHECK(rack_contains(logs[0].initial_racks[0], leave.tiles()[i]));
  CHECK(logs[0].final_scores == logs[1].final_scores);  // deterministic per seed
  CHECK(logs[0].end_reason == logs[1].end_reason);
}

// ===========================================================================
// LabelEncoder
// ===========================================================================

namespace {

// Helper: build an EncodeContext with just the fields encode_labels reads for
// heads 0 and 1 (no next move set -> head 2 emits all zeros).
scribblez::EncodeContext make_scores_view(int fs_active, int fs_opp, int active_player,
                                          bool apply_flip = false) {
  using namespace scribblez::binlog;
  EncodeContext v{};
  v.has_next_move = false;
  v.active_player = active_player;
  v.final_score_p0 = active_player == 0 ? fs_active : fs_opp;
  v.final_score_p1 = active_player == 0 ? fs_opp : fs_active;
  v.apply_flip = apply_flip;
  return v;
}

// Convenience: call AllTargets::encode_all into one contiguous buffer of
// size kLabelFloats laid out as [wld(3), score_diff(1), opp_next(225)].
void encode_labels_flat(const scribblez::EncodeContext& view, float* flat) {
  scribblez::AllTargets::encode_all(view, flat);
}

}  // namespace

static void test_encode_labels() {
  using namespace scribblez::binlog;
  float flat[kLabelFloats];

  auto check_score_diff = [&](int diff_signed) {
    // Score-diff target is a single scalar at offset kWldFloats: the final
    // differential clamped to +/- kScoreDiffClip.
    const int expected_clipped = std::clamp(diff_signed, -kScoreDiffClip, kScoreDiffClip);
    CHECK(flat[kWldFloats] == static_cast<float>(expected_clipped));
  };

  // Win.
  auto v_win = make_scores_view(/*fs_active=*/120, /*fs_opp=*/100, /*active_player=*/0);
  encode_labels_flat(v_win, flat);
  CHECK(flat[0] == 1.0f);
  CHECK(flat[1] == 0.0f);
  CHECK(flat[2] == 0.0f);
  check_score_diff(20);

  // Draw.
  auto v_draw = make_scores_view(75, 75, 1);
  encode_labels_flat(v_draw, flat);
  CHECK(flat[0] == 0.0f);
  CHECK(flat[1] == 1.0f);
  CHECK(flat[2] == 0.0f);
  check_score_diff(0);

  // Loss with negative score_diff.
  auto v_loss = make_scores_view(80, 95, 0);
  encode_labels_flat(v_loss, flat);
  CHECK(flat[0] == 0.0f);
  CHECK(flat[1] == 0.0f);
  CHECK(flat[2] == 1.0f);
  check_score_diff(-15);

  // Differentials beyond +/- kScoreDiffClip are clipped (not rejected).
  auto v_huge_win = make_scores_view(/*fs_active=*/kScoreDiffClip + 50, /*fs_opp=*/0, 0);
  encode_labels_flat(v_huge_win, flat);
  check_score_diff(kScoreDiffClip + 50);  // helper clamps; stored value is the clipped diff
  auto v_huge_loss = make_scores_view(0, kScoreDiffClip + 50, 0);
  encode_labels_flat(v_huge_loss, flat);
  check_score_diff(-(kScoreDiffClip + 50));

  // WLD entries are mutually exclusive and sum to 1.0 for every case.
  for (auto [a, b] : std::vector<std::pair<int, int>>{{1, 0}, {0, 0}, {-5, 5}, {200, -200}}) {
    auto v = make_scores_view(a, b, 0);
    encode_labels_flat(v, flat);
    CHECK(flat[0] + flat[1] + flat[2] == 1.0f);
  }

  // With no next move, head 2 (opp_next_placement) is all zeros.
  encode_labels_flat(v_win, flat);
  for (int i = 0; i < kOppNextPlacementFloats; ++i) {
    CHECK(flat[kWldFloats + kScoreDiffFloats + i] == 0.0f);
  }

  // Build a tiny "next move": a horizontal PLAY at (4, 2) covering 3 cells
  // (squares 0, 1, 2 of the move direction; col 2, 3, 4 of row 4). The
  // opp-next-placement head should light up exactly those three cells in
  // canonical orientation, and exactly their transposed cells when
  // apply_flip=true.
  Move next_play = make_play_full(4, 2, /*horizontal=*/true, 0b111, 0,
                                  {Glyph::of(Tile::from_char('A')), Glyph::of(Tile::from_char('B')),
                                   Glyph::of(Tile::from_char('C'))});

  EncodeContext v_with_next{};
  v_with_next.next_move = next_play;
  v_with_next.has_next_move = true;
  v_with_next.active_player = 0;
  v_with_next.final_score_p0 = 100;
  v_with_next.final_score_p1 = 80;
  v_with_next.apply_flip = false;

  encode_labels_flat(v_with_next, flat);
  const float* plane = flat + kWldFloats + kScoreDiffFloats;
  int set_cells = 0;
  for (int i = 0; i < kOppNextPlacementFloats; ++i) {
    if (plane[i] == 1.0f) ++set_cells;
  }
  CHECK(set_cells == 3);
  CHECK(plane[4 * 15 + 2] == 1.0f);
  CHECK(plane[4 * 15 + 3] == 1.0f);
  CHECK(plane[4 * 15 + 4] == 1.0f);

  // apply_flip transposes (r,c) -> (c,r).
  v_with_next.apply_flip = true;
  encode_labels_flat(v_with_next, flat);
  set_cells = 0;
  for (int i = 0; i < kOppNextPlacementFloats; ++i) {
    if (plane[i] == 1.0f) ++set_cells;
  }
  CHECK(set_cells == 3);
  CHECK(plane[2 * 15 + 4] == 1.0f);
  CHECK(plane[3 * 15 + 4] == 1.0f);
  CHECK(plane[4 * 15 + 4] == 1.0f);

  // EXCHANGE / PASS next move -> all zeros.
  TileCounts xch_tiles;
  xch_tiles.add(Tile::from_char('A'));
  v_with_next.next_move = Move::exchange(xch_tiles);
  v_with_next.apply_flip = false;
  encode_labels_flat(v_with_next, flat);
  for (int i = 0; i < kOppNextPlacementFloats; ++i) CHECK(plane[i] == 0.0f);
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
    CHECK(f);
  }
  int64_t fsize = static_cast<int64_t>(std::filesystem::file_size(path));

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

static void test_dataloader_per_row_symmetry() {
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
    GameStateEncoder ref_enc{InputEncodingSpec{&dict, true}};
    ref_enc.apply_move(fix.self_move);
    ref_enc.encode_input(fix.active_player, fix.active_rack, /*apply_flip=*/false,
                         ref_normal.data());
    ref_enc.encode_input(fix.active_player, fix.active_rack, /*apply_flip=*/true,
                         ref_flipped.data());
  }
  // Sanity: the two encodings differ (asymmetric Q placement).
  CHECK(std::memcmp(ref_normal.data(), ref_flipped.data(), kInputFloats * sizeof(float)) != 0);

  // Expected labels for active=p0 (final p0=350 vs p1=200 -> active wins by
  // 150). The move after turn 0 is p1's PASS, so the opp-next-placement head is
  // all-zero, making the whole label tail flip-invariant.
  float ref_labels[kLabelFloats];
  encode_labels_flat(
    make_scores_view(/*fs_active=*/fix.final_score_p0, /*fs_opp=*/fix.final_score_p1,
                     /*active_player=*/fix.active_player),
    ref_labels);

  DataLoader::Params params;
  params.spec = {&dict, true};
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
    CHECK(loader.load_batch(rows.data()) == 1);
    CHECK(std::memcmp(rows.data(), ref_normal.data(), kInputFloats * sizeof(float)) == 0);
    CHECK(std::memcmp(rows.data() + kInputFloats, ref_labels, kLabelFloats * sizeof(float)) == 0);
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
      cfg.seed = static_cast<uint64_t>(i + 100);
      loader.epoch_start(cfg);
      CHECK(loader.load_batch(row.data()) == 1);
      const bool is_normal =
        std::memcmp(row.data(), ref_normal.data(), kInputFloats * sizeof(float)) == 0;
      const bool is_flipped =
        std::memcmp(row.data(), ref_flipped.data(), kInputFloats * sizeof(float)) == 0;
      CHECK(is_normal || is_flipped);  // every row matches one of the two
      if (is_normal)
        ++normal_count;
      else
        ++flipped_count;
      // Labels are flip-invariant.
      CHECK(std::memcmp(row.data() + kInputFloats, ref_labels, kLabelFloats * sizeof(float)) == 0);
    }
    // With n=200 fair coin flips, the probability that one bucket is empty
    // is 2 * 2^-200; the test is effectively deterministic.
    CHECK(normal_count > 0);
    CHECK(flipped_count > 0);
    std::cout << "  DataLoader per-row symmetry: " << normal_count << " normal / " << flipped_count
              << " flipped (of " << n << ")\n";
  }
}

// The DataLoader maps a game's flat rows to turns starting at eligible_begin:
// a two-PLAY game whose metadata declares the region [1, 2) expands to exactly
// one row, and that row is the turn-1 post-move encoding (POV p1), not turn 0.
static void test_dataloader_eligible_begin_offset() {
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
    CHECK(f);
  }
  const int64_t fsize = static_cast<int64_t>(fs::file_size(path));

  Dictionary dict = medium_dict();

  // Reference: replay both moves, then encode turn 1 post-move -- POV p1 with
  // its post-play leave (6 As).
  std::vector<float> ref_row(kInputFloats, 0.0f);
  {
    GameStateEncoder ref_enc{InputEncodingSpec{&dict, true}};
    ref_enc.apply_move(q_play);
    ref_enc.apply_move(c_play);
    Rack leave;
    for (int i = 0; i < 6; ++i) leave.add(Tile::from_char('A'));
    ref_enc.encode_input(/*active_player=*/1, leave, /*apply_flip=*/false, ref_row.data());
  }
  // Labels for POV p1 (final 200 vs 350); turn 1 is the last turn, so the
  // opp-next-placement head is all-zero, matching the scores-only view.
  float ref_labels[kLabelFloats];
  encode_labels_flat(make_scores_view(/*fs_active=*/200, /*fs_opp=*/350, /*active_player=*/1),
                     ref_labels);

  DataLoader::Params params;
  params.spec = {&dict, true};
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);
  loader.add_file(path.string(), /*num_positions=*/1, fsize);
  CHECK(loader.num_positions() == 1);

  DataLoader::EpochConfig cfg;
  cfg.batch_size = 1;
  cfg.post_move = true;
  cfg.apply_symmetry = false;
  cfg.seed = 1;
  loader.epoch_start(cfg);
  std::vector<float> row(kRowFloats, 0.0f);
  CHECK(loader.load_batch(row.data()) == 1);
  CHECK(std::memcmp(row.data(), ref_row.data(), kInputFloats * sizeof(float)) == 0);
  CHECK(std::memcmp(row.data() + kInputFloats, ref_labels, kLabelFloats * sizeof(float)) == 0);
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
  CHECK(static_cast<int>(fix.slog_paths.size()) == num_files);
  return fix;
}

static void test_epoch_determinism() {
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
  params.spec = {&dict, true};
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
      all_data.insert(all_data.end(), batch.begin(),
                      batch.begin() + static_cast<size_t>(n) * kRowFloats);
    }
    return all_data;
  };

  // First run.
  DataLoader loader1(params);
  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = static_cast<int64_t>(fs::file_size(p));
    loader1.add_file(p.string(), hdr.num_games, fsize);
  }
  auto data1 = run_epoch(loader1);

  // Second run: fresh loader, same files, same seed.
  DataLoader loader2(params);
  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = static_cast<int64_t>(fs::file_size(p));
    loader2.add_file(p.string(), hdr.num_games, fsize);
  }
  auto data2 = run_epoch(loader2);

  CHECK(data1.size() == data2.size());
  CHECK(data1.size() > 0);
  CHECK(std::memcmp(data1.data(), data2.data(), data1.size() * sizeof(float)) == 0);

  // Third run: same loader, same seed again -- must also be identical.
  auto data3 = run_epoch(loader1);
  CHECK(data3.size() == data1.size());
  CHECK(std::memcmp(data1.data(), data3.data(), data1.size() * sizeof(float)) == 0);

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
      data4.insert(data4.end(), batch.begin(), batch.begin() + static_cast<size_t>(n) * kRowFloats);
    }
    CHECK(data4.size() == data1.size());
    CHECK(std::memcmp(data1.data(), data4.data(), data1.size() * sizeof(float)) != 0);
  }

  std::cout << "  epoch determinism OK (" << data1.size() / kRowFloats << " rows)\n";
}

static void test_epoch_coverage() {
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
  params.spec = {&dict, true};
  params.num_worker_threads = 2;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);

  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = static_cast<int64_t>(fs::file_size(p));
    loader.add_file(p.string(), hdr.num_games, fsize);
  }
  // Each game expands to one row per eligible turn; the loader knows the total.
  const int64_t total_positions = loader.num_positions();
  CHECK(total_positions > fix.total_games);  // strictly more rows than games

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
      data.insert(data.end(), batch.begin(), batch.begin() + static_cast<size_t>(n) * row_sz);
    }
    return data;
  };

  std::vector<float> epoch1 = drain_epoch(7777);
  std::vector<float> epoch2 = drain_epoch(8888);

  // Both epochs must contain exactly total_positions rows.
  CHECK(static_cast<int64_t>(epoch1.size()) == total_positions * row_sz);
  CHECK(static_cast<int64_t>(epoch2.size()) == total_positions * row_sz);

  // The two epochs have different seeds, so should be in different order.
  CHECK(std::memcmp(epoch1.data(), epoch2.data(), epoch1.size() * sizeof(float)) != 0);

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
    CHECK(matched);
  }
  for (int64_t i = 0; i < total_positions; ++i) CHECK(found[i]);

  std::cout << "  epoch coverage OK (" << total_positions << " positions)\n";
}

static void test_epoch_memory_budget_stress() {
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
    int64_t fsize = static_cast<int64_t>(fs::file_size(p));
    file_info.emplace_back(hdr.num_games, fsize);
    if (fsize > max_fsize) max_fsize = fsize;
  }

  // Budget = just one file (largest). This forces eviction on every file switch.
  Dictionary dict = medium_dict();
  DataLoader::Params params;
  params.spec = {&dict, true};
  params.memory_budget = max_fsize + 1;  // allow exactly one file at a time
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);

  for (int i = 0; i < static_cast<int>(fix.slog_paths.size()); ++i) {
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
    CHECK(loader.resident_bytes() <= 2 * max_fsize + 100);
  }
  CHECK(rows_decoded == total_positions);

  // Verify determinism: same seed produces same data.
  loader.epoch_start(cfg);
  std::vector<float> run1;
  while (true) {
    int n = loader.load_batch(batch.data());
    if (n == 0) break;
    run1.insert(run1.end(), batch.begin(), batch.begin() + static_cast<size_t>(n) * kRowFloats);
  }

  loader.epoch_start(cfg);
  std::vector<float> run2;
  while (true) {
    int n = loader.load_batch(batch.data());
    if (n == 0) break;
    run2.insert(run2.end(), batch.begin(), batch.begin() + static_cast<size_t>(n) * kRowFloats);
  }
  CHECK(run1.size() == run2.size());
  CHECK(std::memcmp(run1.data(), run2.data(), run1.size() * sizeof(float)) == 0);

  std::cout << "  epoch memory-budget stress OK (" << rows_decoded
            << " rows, budget=" << params.memory_budget << " bytes, " << fix.slog_paths.size()
            << " files)\n";
}

static void test_epoch_shuffles_across_seeds() {
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
  params.spec = {&dict, true};
  params.num_worker_threads = 2;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);

  for (auto& p : fix.slog_paths) {
    std::ifstream f(p, std::ios::binary);
    FileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    int64_t fsize = static_cast<int64_t>(fs::file_size(p));
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
      data.insert(data.end(), batch.begin(), batch.begin() + static_cast<size_t>(n) * kRowFloats);
    }
    return data;
  };

  auto d1 = run_with_seed(100);
  auto d2 = run_with_seed(200);
  auto d3 = run_with_seed(100);  // same as d1

  CHECK(d1.size() == d2.size());
  CHECK(d1.size() == d3.size());
  CHECK(d1.size() > 0);
  // Same seed -> identical.
  CHECK(std::memcmp(d1.data(), d3.data(), d1.size() * sizeof(float)) == 0);
  // Different seed -> different ordering.
  CHECK(std::memcmp(d1.data(), d2.data(), d1.size() * sizeof(float)) != 0);

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
  CHECK(f);
  return KlvFixture{p};
}

static void test_leave_values_synthetic() {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "scribblez_test_klv_XXXXXX";
  fs::create_directories(tmp);

  KlvFixture fix = write_synthetic_klv(tmp);
  LeaveValues lv = LeaveValues::load(fix.path.string());

  // Leave "A"
  Rack a;
  a.add(Tile::from_char('A'));
  CHECK(std::abs(lv.lookup(a) - 1.5f) < 1e-4f);

  // Leave "B"
  Rack b;
  b.add(Tile::from_char('B'));
  CHECK(std::abs(lv.lookup(b) - (-2.5f)) < 1e-4f);

  // Leave "?" (blank). Macondo's leave KWG numbers the blank as machine letter
  // 0; a regression here means blank-bearing leaves silently look up as 0.
  Rack blank;
  blank.add(BLANK);
  CHECK(std::abs(lv.lookup(blank) - 12.0f) < 1e-4f);

  // Empty leave → 0
  Rack empty;
  CHECK(lv.lookup(empty) == 0.0f);

  // Unknown leave (C) → 0
  Rack c;
  c.add(Tile::from_char('C'));
  CHECK(lv.lookup(c) == 0.0f);

  fs::remove_all(tmp);
  std::cout << "test_leave_values_synthetic passed\n";
}

static void test_leave_values_real_kwg_optional() {
  // Use the real NWL23 KLV if available; skip otherwise.
  // The default leaves file lives alongside the KWG.
  std::string kwg_path = SCRIBBLEZ_DEFAULT_KWG;
  std::filesystem::path klv_path = std::filesystem::path(kwg_path).parent_path().parent_path() /
                                   "strategy" / "NWL23" / "leaves.klv2";
  if (!std::filesystem::exists(klv_path)) {
    std::cout << "test_leave_values_real_kwg_optional: SKIPPED (no leaves.klv2 "
                 "at "
              << klv_path << ")\n";
    return;
  }

  LeaveValues lv = LeaveValues::load(klv_path.string());

  // Single blank is a well-known leave with strongly positive value.
  Rack blank_leave;
  blank_leave.add(BLANK);
  float blank_val = lv.lookup(blank_leave);
  CHECK(blank_val > 20.0f);  // known to be ~24..26 in Macondo NWL23

  // Empty leave is 0.
  Rack empty;
  CHECK(lv.lookup(empty) == 0.0f);

  std::cout << "test_leave_values_real_kwg_optional passed (blank leave = " << blank_val << ")\n";
}

static void test_hasty_equity_components() {
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
  CHECK(std::abs(e_mid - (50.0 - 1.4)) < 1e-3);

  // --- leave equity with non-empty leave (uses synthetic KLV: A=1.5, B=-2.5)
  // Play a single A, leaving AAAAAA (6 A's). Our synthetic KLV only has
  // single-tile leaves so the 6-tile leave returns 0.
  Move one_a = make_play_full(7, 7, /*horizontal=*/true, 0b1, 2, {Glyph::of(Tile::from_char('A'))});

  // rack = single A; leave = empty after playing it.
  Rack rack_1a;
  rack_1a.add(Tile::from_char('A'));
  double e_one = eq.equity(one_a, board, 86, opp, rack_1a);
  // score=2, leave=empty(0), opening: center col=7 not in {2,6,8,12} → 0
  CHECK(std::abs(e_one - 2.0) < 1e-3);

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
  CHECK(std::abs(e_eg - (2.0 - 16.0)) < 1e-3);

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
  CHECK(std::abs(e_out - (5.0 + 20.0)) < 1e-3);

  fs::remove_all(tmp);
  std::cout << "test_hasty_equity_components passed\n";
}

// Regression test for blank-bearing exchange equity. An EXCHANGE's equity is
// just the leave value of the tiles kept (score 0, no opening/peg/endgame
// adjustments mid-game), so a mis-keyed blank leave surfaces directly as a
// wrong (typically 0) exchange equity in the web move list.
static void test_hasty_equity_exchange_blank_leave() {
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
  CHECK(std::abs(single - 12.0) < 1e-3);

  std::vector<Move> moves{exch_a};
  std::vector<double> batched = eq.equities(moves, board, 50, opp, rack_a_blank);
  CHECK(batched.size() == 1);
  CHECK(std::abs(batched[0] - 12.0) < 1e-3);

  fs::remove_all(tmp);
  std::cout << "test_hasty_equity_exchange_blank_leave passed\n";
}

// The keystone streaming guarantee: a row encoded directly from a live game's
// GameLog view (the streaming path) is BIT-IDENTICAL to the row the disk
// pipeline produces by writing that game to a .slog and decoding it back. Both
// funnel through PositionEncoder, so any divergence would mean the view built
// from the .slog buffer differs from the one built from self-play storage.
static void test_streaming_disk_encode_equivalence() {
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
    CHECK(!slog.empty());

    const int64_t fsize = static_cast<int64_t>(fs::file_size(slog));
    std::vector<char> raw(fsize);
    {
      std::ifstream f(slog, std::ios::binary);
      f.read(raw.data(), fsize);
      CHECK(f);
    }
    const auto* metas = reinterpret_cast<const GameMetadata*>(raw.data() + sizeof(FileHeader));
    const int sampled = static_cast<int>(metas[0].sampled_turn);

    for (bool post_move : {false, true}) {
      const uint8_t flip = 0;
      std::vector<float> row_disk(row_floats, 0.0f);
      BlockDecoder decoder(InputEncodingSpec{&dict, true});
      decoder.decode(raw.data(), "eq", /*local_start=*/0, /*n_rows=*/1, &flip, post_move,
                     /*output_row_start=*/0, row_disk.data());

      std::vector<float> row_stream(row_floats, 0.0f);
      PositionEncoder enc(InputEncodingSpec{&dict, true});
      enc.encode_row<PostMoveValueTask>(storage.view(), sampled, post_move, /*flip=*/false,
                                        row_stream.data());

      for (int i = 0; i < row_floats; ++i) CHECK(row_disk[i] == row_stream[i]);
      ++compared;
    }

    // Fresh dir per seed so directory_iterator finds exactly one file.
    for (const auto& ent : fs::directory_iterator(dir)) fs::remove(ent.path());
  }
  CHECK(compared == 6);
  std::cout << "  streaming/disk encode equivalence OK (" << compared << " rows)\n";
}

// StreamingRowBuffer: many producers, tiny slots (frequent boundary crossings).
// Every global row must be written exactly once and read back in a contiguous
// set, with no slot overwritten while the consumer holds it.
static void test_streaming_row_buffer_concurrency() {
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
  const uint64_t total_rows = static_cast<uint64_t>(slots_to_consume) * rows_per_slot;
  std::atomic<uint64_t> work{0};
  const int K = 8;  // many producers + tiny slots -> frequent slot-boundary crossings
  std::vector<std::thread> producers;
  for (int t = 0; t < K; ++t) {
    producers.emplace_back([&] {
      while (work.fetch_add(1, std::memory_order_relaxed) < total_rows) {
        uint64_t r = ring.claim_row();
        if (r == StreamingRowBuffer::kNoRow) break;
        ring.row_dest(r)[0] = static_cast<float>(r);
        ring.commit_row(r);
      }
    });
  }

  std::set<uint64_t> seen;
  bool dup = false;
  for (int i = 0; i < slots_to_consume; ++i) {
    int slot = ring.wait_full_slot();
    CHECK(slot >= 0);
    for (int k = 0; k < rows_per_slot; ++k) {
      uint64_t v = static_cast<uint64_t>(slots[slot][k]);
      if (!seen.insert(v).second) dup = true;
    }
    ring.release_slot(slot);
  }
  for (auto& p : producers) p.join();

  CHECK(!dup);  // each global row written and read exactly once
  CHECK(static_cast<int>(seen.size()) == slots_to_consume * rows_per_slot);
  for (uint64_t v = 0; v < total_rows; ++v) CHECK(seen.count(v) == 1);  // exactly [0, total)
  std::cout << "  StreamingRowBuffer concurrency OK (" << seen.size() << " rows, K=" << K << ")\n";
}

// StreamingRowBuffer shutdown: stop() must wake every blocked producer (and the
// consumer) so nothing hangs, even with producers parked on backpressure.
static void test_streaming_row_buffer_shutdown() {
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
        ring.row_dest(r)[0] = static_cast<float>(r);
        ring.commit_row(r);
      }
      exited.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // No consumer: producers fill both slots, then park on backpressure. stop()
  // must release them all.
  ring.stop();
  for (auto& p : producers) p.join();
  CHECK(exited.load() == K);
  CHECK(ring.wait_full_slot() == -1);
  std::cout << "  StreamingRowBuffer shutdown OK\n";
}

// pick_sampled_turn only chooses turns in the eligible region -- within the
// bag-non-empty prefix and not before the last random-opening ply -- and
// returns -1 when the region is empty.
static void test_pick_sampled_turn_eligibility() {
  using namespace scribblez;
  using namespace scribblez::binlog;

  GameLogStorage s;
  s.turns.resize(3);  // value-initialized: bag_size_before == 0
  s.turns[0].bag_size_before = 9;
  s.turns[1].bag_size_before = 5;
  std::mt19937_64 rng(123);
  CHECK(eligible_span(s.view()).begin == 0);
  CHECK(eligible_span(s.view()).end == 2);
  for (int i = 0; i < 20; ++i) {
    const int t = pick_sampled_turn(s.view(), rng);
    CHECK(t == 0 || t == 1);
  }

  // A random opening excludes the positions before its last ply: with 2 random
  // plies, only turns >= 1 remain eligible.
  s.num_random_opening_plies = 2;
  CHECK(eligible_span(s.view()).begin == 1);
  for (int i = 0; i < 20; ++i) CHECK(pick_sampled_turn(s.view(), rng) == 1);

  // A game that ended during its random opening has an empty eligible region.
  s.num_random_opening_plies = 3;
  CHECK(pick_sampled_turn(s.view(), rng) == -1);
  s.num_random_opening_plies = 0;

  GameLogStorage z;
  z.turns.resize(2);  // all ineligible
  CHECK(pick_sampled_turn(z.view(), rng) == -1);

  // pick_any_turn (max-move-per-lane sampling) ignores bag size: every turn is eligible,
  // and the choice covers the whole range. An empty game yields -1.
  std::array<bool, 3> seen{};
  for (int i = 0; i < 200; ++i) {
    const int t = pick_any_turn(s.view(), rng);
    CHECK(t >= 0 && t < 3);
    seen[t] = true;
  }
  CHECK(seen[0] && seen[1] && seen[2]);  // bag_size_before == 0 turns included
  GameLogStorage empty;
  CHECK(pick_any_turn(empty.view(), rng) == -1);
  std::cout << "  pick_sampled_turn / pick_any_turn eligibility OK\n";
}

// ShadowMoveGen, summed over every anchor (no pruning), reproduces exactly the
// move set of MoveGenerator::generate, and every anchor's score bound is
// admissible (>= the score of each play canonically anchored there). The latter
// is the invariant that makes best-first equity pruning exact.
static void test_shadow_movegen_matches_full() {
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
          CHECK(static_cast<int>(m.score()) <= a.score_bound_by_size[m.num_glyphs()]);
        }
        for (Move& m : am) shadow.push_back(std::move(m));
      }

      CHECK(key_set(board, full) == key_set(board, shadow));

      ++positions;
      total_moves += static_cast<long>(full.size());
      board.apply(t.move);
    }
  }
  CHECK(positions > 0);
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
  scribblez::Move make_move(const scribblez::MoveRequest& req) override {
    const scribblez::Move shadow = bot_.make_move(req);
    const scribblez::Move ref = scribblez::hasty_best_move_reference(req);
    CHECK(move_key(req.board, shadow) == move_key(req.board, ref));
    ++comparisons;
    return shadow;
  }
  long comparisons = 0;

 private:
  scribblez::HastyBotAgent bot_;
};
}  // namespace

static void test_hasty_shadow_matches_reference() {
  namespace fs = std::filesystem;
  std::string kwg;
  for (const char* cand : {
#ifdef SCRIBBLEZ_DEFAULT_KWG
         SCRIBBLEZ_DEFAULT_KWG,
#endif
         "/workspace/mount/lexica/NWL23.kwg",
         "/workspace/mount/macondo/data/lexica/gaddag/NWL23.kwg"}) {
    std::error_code ec;
    if (fs::exists(cand, ec)) {
      kwg = cand;
      break;
    }
  }
  const std::string leaves = scribblez::HastyEquity::default_leaves_path("NWL23");
  const std::string peg = scribblez::HastyEquity::default_peg_path();
  if (kwg.empty() || !fs::exists(leaves)) {
    std::cout << "  (no NWL23 kwg/leaves; skipping HastyBot shadow-equivalence)\n";
    return;
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
  CHECK(comparisons > 0);
  std::cout << "  HastyBot shadow search matches reference (" << comparisons << " positions)\n";
}

// wmp_generate (WordMap anagram lookup) produces exactly the same set of legal
// plays as the GADDAG move generator, for blank-free racks. Always runs against
// the in-memory medium dictionary.
static void test_wmp_generate_matches_full() {
  using namespace scribblez;
  Dictionary dict = medium_dict();
  WordMap wm = WordMap::build(dict);
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
        CHECK(key_set(board, full) == key_set(board, wmp));

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
          CHECK(key_set(board, g) == key_set(board, w));
        }

        // The per-extent partition covers every legal play exactly once, and each
        // extent's score bound never underestimates its plays (so the best-first
        // early-exit over extents is exact).
        std::vector<Move> extent_union;
        for (const ShadowExtent& e : smg.extents(rack, &wm)) {
          std::vector<Move> em;
          wmp_generate_extent(board, wm, subracks, e, em);
          for (const Move& m : em) {
            CHECK(static_cast<int>(m.score()) <= e.score_bound);
            extent_union.push_back(m);
          }
        }
        CHECK(key_set(board, full) == key_set(board, extent_union));
        ++positions;
      }
      board.apply(t.move);
    }
  }
  CHECK(positions > 0);
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
  scribblez::Move make_move(const scribblez::MoveRequest& req) override {
    auto& dst = req.my_rack.counts().blanks() == 0 ? sink_ : blanked_sink_;
    dst.push_back(
      {req.board, req.my_rack, req.opp_rack, req.my_score, req.opp_score, req.bag_size});
    return bot_.make_move(req);
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
static void test_wmp_matches_gaddag_real_lexicon() {
  namespace fs = std::filesystem;
  using namespace scribblez;
  std::string kwg;
  for (const char* cand : {
#ifdef SCRIBBLEZ_DEFAULT_KWG
         SCRIBBLEZ_DEFAULT_KWG,
#endif
         "/workspace/mount/lexica/NWL23.kwg",
         "/workspace/mount/macondo/data/lexica/gaddag/NWL23.kwg"}) {
    std::error_code ec;
    if (fs::exists(cand, ec)) {
      kwg = cand;
      break;
    }
  }
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  const std::string peg = HastyEquity::default_peg_path();
  if (kwg.empty() || !fs::exists(leaves)) {
    std::cout << "  (no NWL23 kwg/leaves; skipping WMP/GADDAG equivalence)\n";
    return;
  }
  Dictionary dict = Dictionary::load_kwg(kwg);
  HastyEquity::init(leaves, peg);
  WordMap wm = WordMap::build(dict);

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
    CHECK(move_key(p.board, blank_bot.make_move(req)) ==
          move_key(p.board, hasty_best_move_wmp(req, wm)));
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
    CHECK(key_set(p.board, full) == key_set(p.board, wmp));
    total_moves += static_cast<long>(full.size());

    const MoveRequest req{p.board, dict, p.rack, p.opp_rack, p.my_score, p.opp_score, p.bag_size};
    CHECK(move_key(p.board, bot.make_move(req)) == move_key(p.board, hasty_best_move_wmp(req, wm)));
  }
  CHECK(!positions.empty());
  std::cout << "  WMP/GADDAG equivalence OK (" << positions.size() << " blank-free + "
            << blanked.size() << " blanked positions, " << total_moves << " plays)\n";
}

static void test_util_helpers() {
  // round_up_pow2: exact powers map to themselves; everything else rounds up.
  CHECK(util::round_up_pow2(0) == 1);
  CHECK(util::round_up_pow2(1) == 1);
  CHECK(util::round_up_pow2(2) == 2);
  CHECK(util::round_up_pow2(3) == 4);
  CHECK(util::round_up_pow2(5) == 8);
  CHECK(util::round_up_pow2(8) == 8);
  CHECK(util::round_up_pow2(9) == 16);
  CHECK(util::round_up_pow2(1u << 20) == (1u << 20));
  CHECK(util::round_up_pow2((1u << 20) + 1) == (1u << 21));

  // align_up to a power-of-two boundary.
  CHECK(util::align_up(0, 8) == 0);
  CHECK(util::align_up(1, 8) == 8);
  CHECK(util::align_up(7, 8) == 8);
  CHECK(util::align_up(8, 8) == 8);
  CHECK(util::align_up(9, 8) == 16);
  CHECK(util::align_up(7, 1) == 7);

  // plane_index: row-major vs transpose across the diagonal.
  CHECK(util::plane_index(2, 3, 15, false) == 2 * 15 + 3);
  CHECK(util::plane_index(2, 3, 15, true) == 3 * 15 + 2);
  CHECK(util::plane_index(4, 4, 15, false) == util::plane_index(4, 4, 15, true));

  // The four orthogonal neighbor deltas are unit steps with zero net sum.
  int sum_dr = 0, sum_dc = 0;
  for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
    CHECK((dr == 0) != (dc == 0));  // exactly one axis moves
    CHECK(dr >= -1 && dr <= 1 && dc >= -1 && dc <= 1);
    sum_dr += dr;
    sum_dc += dc;
  }
  CHECK(sum_dr == 0 && sum_dc == 0);

  std::cout << "test_util_helpers passed\n";
}

// Backs the invariant "NeuralAgent with --top-k=1 plays exactly HastyBot's
// move" without instantiating the (TensorRT-linked) agent. At k=1 both agents
// reduce to the same equity argmax over the legal plays, via two different
// HastyEquity entry points: HastyBot scores moves one at a time with
// HastyEquity::equity(), while NeuralAgent ranks the batch from
// HastyEquity::equities(). This checks (a) the batch and per-move APIs
// agree value-for-value and (b) their argmax -- the move each agent returns --
// is identical.
static void test_topk1_selection_matches_hastybot() {
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
  CHECK(plays.size() >= 2);  // a meaningful argmax needs >1 candidate

  Rack opp;  // empty
  const int bag_size = 80;

  // Batch path (what NeuralAgent uses) must match the per-move path (what
  // HastyBot uses) value-for-value.
  std::vector<double> batch = eq.equities(plays, board, bag_size, opp, my_rack);
  CHECK(batch.size() == plays.size());
  std::vector<double> per_move(plays.size());
  for (size_t i = 0; i < plays.size(); ++i) {
    per_move[i] = eq.equity(plays[i], board, bag_size, opp, my_rack);
    CHECK(std::abs(batch[i] - per_move[i]) < 1e-9);
  }

  // HastyBot's selection: first move with strictly-greatest per-move equity.
  int hasty_pick = 0;
  for (size_t i = 1; i < per_move.size(); ++i) {
    if (per_move[i] > per_move[hasty_pick]) hasty_pick = static_cast<int>(i);
  }
  // NeuralAgent k=1 selection: top-1 of the batch ranking (same rule).
  int topk1_pick = 0;
  for (size_t i = 1; i < batch.size(); ++i) {
    if (batch[i] > batch[topk1_pick]) topk1_pick = static_cast<int>(i);
  }
  CHECK(hasty_pick == topk1_pick);

  fs::remove_all(tmp);
  std::cout << "test_topk1_selection_matches_hastybot passed (" << plays.size()
            << " candidates, pick=" << hasty_pick << ")\n";
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
  gm.initial_score_p0 = static_cast<int16_t>(initial_score_p0);

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
  BlockDecoder dec(InputEncodingSpec{&dict, true});
  dec.decode(buf.data(), "handicap-test", /*local_start=*/0, /*n_rows=*/1, &flip,
             /*post_move=*/false, /*output_row_start=*/0, output.data());

  // Thermometer invariant: the number of set slots equals
  // (clipped_diff + kScoreDiffClip + 1).
  const float* sd = output.data() + kSpatialFloats + kScoreDiffOffset;
  int ones = 0;
  for (int i = 0; i < kScoreDiffThermoBins; ++i) ones += sd[i] > 0.5f ? 1 : 0;
  return ones - kScoreDiffClip - 1;
}

// A head-start handicap stored in GameMetadata must reach the replayed
// position's score-differential input (the decoder seeds its score
// accumulator from the metadata's initial scores).
static void test_handicap_shifts_score_diff_input() {
  CHECK(decode_handicap_score_diff(0) == 0);
  CHECK(decode_handicap_score_diff(80) == 80);
  std::cout << "test_handicap_shifts_score_diff_input passed\n";
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
static void test_lane_targets() {
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

    CHECK(t.rows[CENTER].has_move);
    CHECK((t.rows[CENTER].placed[CENTER + 3] >> sk) & 1u);  // S newly placed after CAT
    CHECK(t.rows[CENTER].max_score == best_move_score(moves));
    CHECK(!t.cols[CENTER + 3].has_move);
    CHECK(lane_global_max(t) == best_move_score(moves));
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

    CHECK(t.rows[CENTER].has_move);
    CHECK(t.cols[CENTER].has_move);
    CHECK((t.rows[CENTER].placed[CENTER] >> sk) & 1u);  // lane cell == column
    CHECK((t.cols[CENTER].placed[CENTER] >> sk) & 1u);  // lane cell == row
    CHECK(t.rows[CENTER].max_score == t.cols[CENTER].max_score);
  }

  // 3. A single tile cannot open the game (no word formed), so every lane is
  //    empty on an empty board.
  {
    Board b;
    const LaneTargets t = compute_lane_targets(b, rack_from("S"), d);
    for (const auto& lane : t.rows) CHECK(!lane.has_move);
    for (const auto& lane : t.cols) CHECK(!lane.has_move);
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
    CHECK(lane_occ[(CENTER + 3) * kLaneTileKinds + sk] == 1.0f);
    CHECK(lane_occ[(CENTER + 3) * kLaneTileKinds + Tile::from_char('C').index()] == 0.0f);
    CHECK(lane_occ[CENTER * kLaneTileKinds + sk] == 0.0f);  // CAT tiles are not "placed"

    CHECK(mask[row_id] == 1.0f);
    CHECK(score[row_id] ==
          static_cast<float>(std::min(t.rows[CENTER].max_score, kLaneScoreBins - 1)));

    // An empty lane (row 0) is all zeros: mask off, score 0, no occupancy.
    CHECK(mask[0] == 0.0f);
    CHECK(score[0] == 0.0f);
    for (int i = 0; i < kLaneLen * kLaneTileKinds; ++i) CHECK(occ[i] == 0.0f);

    // Flip is a rows<->cols swap: the horizontal CATS play that lived in axis-0
    // lane CENTER now lives in axis-1 (vertical) lane CENTER, same cell.
    std::vector<float> frow(kLaneLabelFloats, -1.0f);
    encode_lane_targets(t, /*flip=*/true, frow.data());
    const float* focc = frow.data();
    const float* v_lane = focc + (kLanesPerAxis + CENTER) * kLaneLen * kLaneTileKinds;
    const float* h_lane = focc + CENTER * kLaneLen * kLaneTileKinds;
    CHECK(v_lane[(CENTER + 3) * kLaneTileKinds + sk] == 1.0f);                     // now vertical
    for (int i = 0; i < kLaneLen * kLaneTileKinds; ++i) CHECK(h_lane[i] == 0.0f);  // cols empty
    CHECK((focc + kLaneOccupancyFloats)[kLanesPerAxis + CENTER] == score[row_id]);
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

      CHECK(lane_global_max(t) == best_move_score(moves));
      bool any_lane = false;
      for (const auto& lane : t.rows) any_lane = any_lane || lane.has_move;
      for (const auto& lane : t.cols) any_lane = any_lane || lane.has_move;
      CHECK(any_lane == !moves.empty());

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
static void test_lane_best_moves() {
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
    CHECK(bm.rows[i].has_move == t.rows[i].has_move);
    CHECK(bm.cols[i].has_move == t.cols[i].has_move);
    CHECK(bm.rows[i].max_score == t.rows[i].max_score);
    CHECK(bm.cols[i].max_score == t.cols[i].max_score);
  }

  // Every kept move scores exactly its lane's max (no sub-maximal plays retained).
  for (const auto& lane : bm.rows)
    for (const Move& m : lane.moves) CHECK(static_cast<int>(m.score()) == lane.max_score);
  for (const auto& lane : bm.cols)
    for (const Move& m : lane.moves) CHECK(static_cast<int>(m.score()) == lane.max_score);

  // CENTER's row holds the maximal play CATS; its word and origin recover from the
  // pre-move board.
  const LaneBestMoves& row = bm.rows[CENTER];
  CHECK(row.has_move);
  CHECK(!row.moves.empty());
  bool found_cats = false;
  for (const Move& m : row.moves)
    if (m.main_word(b) == "CATS") {
      found_cats = true;
      CHECK(m.horizontal());
      CHECK(m.word_origin(b) == std::make_pair(CENTER, CENTER));
    }
  CHECK(found_cats);
}

// The GCG -> analysis-position bridge and the lane-analysis JSON. parse_* takes the
// board after all recorded moves with the next player to move and reads that
// player's rack from the #Rack header (the move log clears it during replay); the
// JSON carries the web board plus per-lane ground truth and maximal plays.
static void test_lane_analysis() {
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
  CHECK(parse_gcg_analysis_position(gcg, &pos, &error));
  CHECK(pos.on_move == 1);                              // P2 to move after P1's single play
  CHECK(pos.rack == rack_from("EINRSTU"));              // from #Rack2
  CHECK(!pos.board.at(CENTER, CENTER).is_empty());      // C
  CHECK(!pos.board.at(CENTER, CENTER + 2).is_empty());  // T
  CHECK(pos.board.at(CENTER, CENTER + 3).is_empty());   // nothing past CAT

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
  CHECK(o.contains("board"));  // web GameState for rendering
  CHECK(o.at("on_move").as_int64() == 0);
  const boost::json::object& la = o.at("lane_analysis").as_object();
  const boost::json::array& rows = la.at("rows").as_array();
  CHECK(rows.size() == static_cast<size_t>(kLanesPerAxis));
  const boost::json::object& center_row = rows[CENTER].as_object();
  CHECK(center_row.at("has_move").as_bool());
  bool json_has_cats = false;
  for (const auto& mv : center_row.at("best_moves").as_array())
    if (mv.as_object().at("word").as_string() == "CATS") json_has_cats = true;
  CHECK(json_has_cats);
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

static void test_max_move_per_lane_input_encoder() {
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
  CHECK(out[C * cells + cell(7, 7)] == 1.0f);
  CHECK(out[A * cells + cell(7, 8)] == 1.0f);
  CHECK(out[Sx * cells + cell(5, 5)] == 1.0f);
  CHECK(out[A * cells + cell(7, 7)] == 0.0f);

  // Blank-marker plane: set under the blank only.
  CHECK(out[BoardPlanes::kBlankMarkerPlane * cells + cell(5, 5)] == 1.0f);
  CHECK(out[BoardPlanes::kBlankMarkerPlane * cells + cell(7, 7)] == 0.0f);

  // Premium planes agree with Board::PREMIUM at every cell (reported even under
  // a played tile), and exactly one premium plane is set per premium square.
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const int want = prem_plane_offset(b.premium_at(r, c));
      for (int off = 0; off < BoardPlanes::kPremiumPlanes; ++off) {
        const float v = out[(BoardPlanes::kPremiumPlane0 + off) * cells + cell(r, c)];
        CHECK(v == (off == want ? 1.0f : 0.0f));
      }
    }
  }

  // Rack scalars: raw counts, blank in slot 26.
  const float* counts = out.data() + Enc::kSpatialFloats;
  CHECK(counts[A] == 2.0f);
  CHECK(counts[B] == 1.0f);
  CHECK(counts[26] == 1.0f);  // blank
  CHECK(counts[C] == 0.0f);

  // Flip transposes the spatial planes but leaves rack scalars untouched.
  std::vector<float> flipped(Enc::kInputFloats, -1.0f);
  Enc::encode(b, rack, /*flip=*/true, flipped.data());
  CHECK(flipped[A * cells + cell(8, 7)] == 1.0f);  // (7,8) -> (8,7)
  CHECK(flipped[A * cells + cell(7, 8)] == 0.0f);
  CHECK(flipped[C * cells + cell(7, 7)] == 1.0f);  // on the diagonal, unchanged
  const float* fcounts = flipped.data() + Enc::kSpatialFloats;
  CHECK(fcounts[A] == 2.0f && fcounts[26] == 1.0f);
}

// The max-move-per-lane training task: one full row is exactly the max-move-per-lane input encoding
// followed by the per-lane labels for the board/rack at the sampled position.
// Checked for both symmetry orientations.
static void test_max_move_per_lane_task_row() {
  const Dictionary d = tiny_dict();

  // CAT on the board (the context exposes the board via its GameStateEncoder).
  GameStateEncoder gse{InputEncodingSpec{&d, true}};
  gse.apply_move(make_play(CENTER, CENTER, /*horizontal=*/true,
                           {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                            Glyph::of(Tile::from_char('T'))}));
  const Rack rack = rack_from("S");

  for (bool flip : {false, true}) {
    EncodeContext ctx{};
    ctx.enc = &gse;
    ctx.pov_rack = &rack;
    ctx.apply_flip = flip;
    ctx.spec = {&d, true};

    std::vector<float> row(MaxMovePerLaneTask::kRowFloats, -1.0f);
    MaxMovePerLaneTask::encode_row(ctx, row.data());

    std::vector<float> ref_in(MaxMovePerLaneInputEncoder::kInputFloats);
    MaxMovePerLaneInputEncoder::encode(gse.board(), rack, flip, ref_in.data());
    std::vector<float> ref_lab(kLaneLabelFloats);
    encode_lane_targets(compute_lane_targets(gse.board(), rack, d), flip, ref_lab.data());

    CHECK(MaxMovePerLaneTask::kInputFloats == static_cast<int>(ref_in.size()));
    CHECK(MaxMovePerLaneTask::kLabelFloats == static_cast<int>(ref_lab.size()));
    for (int i = 0; i < MaxMovePerLaneTask::kInputFloats; ++i) CHECK(row[i] == ref_in[i]);
    for (int i = 0; i < MaxMovePerLaneTask::kLabelFloats; ++i)
      CHECK(row[MaxMovePerLaneTask::kInputFloats + i] == ref_lab[i]);
  }
}

int main() {
  test_util_helpers();
  test_dict_basic();
  test_lane_targets();
  test_lane_best_moves();
  test_lane_analysis();
  test_max_move_per_lane_input_encoder();
  test_max_move_per_lane_task_row();
  test_shadow_movegen_matches_full();
  test_wmp_generate_matches_full();
  test_wmp_matches_gaddag_real_lexicon();
  test_hasty_shadow_matches_reference();
  test_movegen_opening();
  test_movegen_cross_word();
  test_bingo_bonus();
  test_gaddag_vs_dawg_inmemory();
  test_board_caches_incremental_matches_full();
  test_real_kwg_optional();
  test_encoder_basic_layout();
  test_encoder_last_opp_plane_mask();
  test_encoder_flip_symmetry();
  test_movegen_single_tile_vertical_hooks();
  test_encoder_cross_check_planes_qi();
  test_position_encoder_cross_check_planes_lexical();
  test_contingent_map_matches_per_tile_generation();
  test_contingent_map_cat_board();
  test_base_layout_is_full_minus_contingent_tails();
  test_encoder_forced_score_diff_isolation();
  test_encoder_nonplay_last_move_metadata();
  test_extract_positions_movegen_roundtrip();
  test_binary_log_file_and_data_loader_roundtrip();
  test_game_random_opening();
  test_generate_legal_exchanges();
  test_binary_log_random_opening_region();
  test_handicap_shifts_score_diff_input();
  test_tile_glyph_basics();
  test_rack_invariants();
  test_bag_basics();
  test_board_apply_interleaves_cross_tiles();
  test_move_main_word_through_cross();
  test_movegen_blank_scores_zero();
  test_game_end_rack_out_bonus();
  test_game_end_stalemate_penalty();
  test_game_play_from();
  test_natural_less();
  test_encode_labels();
  test_dataloader_per_row_symmetry();
  test_dataloader_eligible_begin_offset();
  test_epoch_determinism();
  test_epoch_coverage();
  test_epoch_memory_budget_stress();
  test_epoch_shuffles_across_seeds();
  test_leave_values_synthetic();
  test_leave_values_real_kwg_optional();
  test_hasty_equity_components();
  test_hasty_equity_exchange_blank_leave();
  test_streaming_disk_encode_equivalence();
  test_streaming_row_buffer_concurrency();
  test_streaming_row_buffer_shutdown();
  test_pick_sampled_turn_eligibility();
  test_topk1_selection_matches_hastybot();
  std::cout << "All tests passed.\n";
  return 0;
}
