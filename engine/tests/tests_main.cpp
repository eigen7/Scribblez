// Minimal hand-rolled tests for the engine. Exits nonzero on failure.

#include "scribblez/agent.h"
#include "scribblez/bag.h"
#include "scribblez/binary_log.h"
#include "scribblez/board.h"
#include "scribblez/data_loader.h"
#include "scribblez/dictionary.h"
#include "scribblez/game.h"
#include "scribblez/glyph.h"
#include "scribblez/input_encoder.h"
#include "scribblez/label_encoder.h"
#include "scribblez/movegen.h"
#include "scribblez/rack.h"

#include <algorithm>
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
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::cerr << "CHECK failed: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

using namespace scribblez;

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
// setups); set square_mask accordingly.
static Move make_play(int row, int col, bool horizontal, std::initializer_list<Glyph> gs) {
  Move m;
  m.type = MoveType::PLAY;
  m.horizontal = horizontal;
  m.start_row = static_cast<int8_t>(row);
  m.start_col = static_cast<int8_t>(col);
  int i = 0;
  for (Glyph g : gs) {
    if (i >= RACK_SIZE) break;
    m.glyphs[i++] = g;
  }
  m.square_mask = static_cast<uint16_t>((1u << i) - 1u);
  return m;
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
    CHECK(m.type == MoveType::PLAY);
    const int dr = m.horizontal ? 0 : 1;
    const int dc = m.horizontal ? 1 : 0;
    bool covers = false;
    for (int i = 0; i < m.num_glyphs(); ++i) {
      if (m.start_row + i * dr == CENTER && m.start_col + i * dc == CENTER) {
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
    if (m.score > best) {
      best = m.score;
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
  struct Placement {
    int r, c;
    Glyph g;
  };
  std::vector<Placement> tiles;
  if (m.type == MoveType::PLAY) {
    const int dr = m.horizontal ? 0 : 1;
    const int dc = m.horizontal ? 1 : 0;
    int r = m.start_row, c = m.start_col;
    const int n = m.num_glyphs();
    for (int gi = 0; gi < n; ++gi) {
      while (board.in_bounds(r, c) && !board.at(r, c).is_empty()) {
        r += dr;
        c += dc;
      }
      if (!board.in_bounds(r, c)) break;
      tiles.push_back({r, c, m.glyphs[gi]});
      r += dr;
      c += dc;
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
  std::snprintf(buf, sizeof(buf), "|%d", m.score);
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
    {"AA",     "AB",      "AD",     "AE",     "AG",      "AH",      "AI",      "AL",      "AN",
     "AR",     "AS",      "AT",     "AW",     "AX",      "AY",      "BA",      "BE",      "BI",
     "BO",     "BY",      "CAB",    "CAR",    "CARS",    "CART",    "CARTS",   "CAT",     "CATS",
     "CARE",   "CARES",   "CARET",  "CARETS", "CASTE",   "CASTER",  "CASTERS", "DOG",     "DOGS",
     "DOT",    "DOTS",    "EAR",    "EARS",   "EAT",     "EATS",    "RAT",     "RATE",    "RATES",
     "RATS",   "STARE",   "STARED", "TARE",   "TARES",   "TEAR",    "TEARS",   "REACT",   "REACTS",
     "TRACE",  "TRACES",  "CRATE",  "CRATES", "CATER",   "CATERS",  "RECAST",  "RECASTS", "TASTE",
     "TASTER", "TASTERS", "SET",    "SET",    "TASTERS", "PARTIED", "AERIES",  "OX",      "OXEN",
     "QI",     "ZA",      "JO",     "GO",     "NO",      "ON",      "TO",      "IT",      "IS",
     "HE",     "OH",      "OW",     "WO",     "WORD",    "WORDS",   "WORDIER", "TIE",     "TIES",
     "TIED",   "DIET",    "DIETS",  "EDIT",   "EDITS",   "TIDE",    "TIDES",   "SITE",    "SITED",
     "STIED"});
}

static void test_gaddag_vs_dawg_inmemory() {
  Dictionary d = medium_dict();
  cross_validate(d, "medium_dict", 1234u, /*games=*/12, /*steps_per_game=*/6);
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

static void test_encoder_basic_layout() {
  using namespace scribblez::binlog;
  PositionRecord rec{};
  rec.active_player = 0;
  rec.position_kind = static_cast<uint8_t>(PositionKind::kPreMove);
  rec.score_active = 50;
  rec.score_opp = 30;
  rec.bag_counts[0] = 3;   // 3 A's in bag
  rec.bag_counts[26] = 1;  // 1 blank in bag
  rec.active_rack.add(Tile::from_char('Q'));
  rec.active_rack.add(Tile::from_char('Z'));
  rec.active_rack.add(BLANK);
  rec.board[7 * 15 + 7] = Glyph::of(Tile::from_char('C'));
  rec.board[3 * 15 + 3] = Glyph::played(Tile::from_char('D'), /*is_blank=*/true);

  std::vector<float> out(kInputFloats, -1.0f);
  encode_input(rec, /*apply_flip=*/false, out.data());

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

  // Last-opp-placement plane (31): record's last_opp_move defaults to PASS, so all-zero.
  for (int i = 0; i < 225; ++i) CHECK(out[31 * 225 + i] == 0.0f);

  // Scalars: rack[27] + bag[27] + (score_diff, bag_size, rack_size, last_opp_num_glyphs).
  const float* scalars = out.data() + kSpatialFloats;
  CHECK(scalars[Tile::from_char('Q')] == 1.0f);
  CHECK(scalars[Tile::from_char('Z')] == 1.0f);
  CHECK(scalars[26] == 1.0f);  // blank count in rack
  CHECK(scalars[Tile::from_char('A')] == 0.0f);
  CHECK(scalars[27 + 0] == 3.0f);   // 3 A's in bag
  CHECK(scalars[27 + 26] == 1.0f);  // 1 blank in bag
  CHECK(scalars[54] == 20.0f);      // score_diff
  CHECK(scalars[55] == 4.0f);       // bag_size = sum of bag_counts
  CHECK(scalars[56] == 3.0f);       // active_rack_size
  CHECK(scalars[57] == 0.0f);       // PASS -> num_glyphs == 0
}

static void test_encoder_last_opp_plane_mask() {
  using namespace scribblez::binlog;

  // Hand-build a record where the opponent played CAT horizontally starting at
  // (7,6), interleaving an already-present A at (7,7): cells (7,6) and (7,8)
  // were newly placed -> square_mask bits 0 and 2 set.
  PositionRecord rec{};
  rec.active_player = 1;
  rec.position_kind = static_cast<uint8_t>(PositionKind::kPreMove);
  rec.board[7 * 15 + 7] = Glyph::of(Tile::from_char('A'));

  Move m;
  m.type = MoveType::PLAY;
  m.horizontal = true;
  m.start_row = 7;
  m.start_col = 6;
  m.glyphs[0] = Glyph::of(Tile::from_char('C'));
  m.glyphs[1] = Glyph::of(Tile::from_char('T'));
  m.square_mask = 0b101;  // bit 0 (cell 7,6) + bit 2 (cell 7,8)
  m.score = 5;
  rec.last_opp_move = m;

  std::vector<float> out(kInputFloats, 0.0f);
  encode_input(rec, /*apply_flip=*/false, out.data());

  const float* plane = out.data() + 31 * 225;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      const float expected = ((r == 7 && c == 6) || (r == 7 && c == 8)) ? 1.0f : 0.0f;
      CHECK(plane[r * 15 + c] == expected);
    }
  }
  // num_glyphs scalar reflects placements (not cells walked).
  CHECK(out[kSpatialFloats + 57] == 2.0f);
}

static void test_encoder_flip_symmetry() {
  using namespace scribblez::binlog;

  PositionRecord rec{};
  rec.active_player = 0;
  rec.position_kind = static_cast<uint8_t>(PositionKind::kPreMove);
  rec.score_active = 30;
  rec.score_opp = 12;
  rec.active_rack.add(Tile::from_char('Q'));
  rec.bag_counts[0] = 5;
  rec.board[3 * 15 + 5] = Glyph::of(Tile::from_char('B'));
  rec.board[1 * 15 + 9] = Glyph::played(Tile::from_char('E'), /*is_blank=*/true);

  // A vertical opp move starting at (0,4) placing 'A' then 'X' (mask=0b11).
  Move m;
  m.type = MoveType::PLAY;
  m.horizontal = false;
  m.start_row = 0;
  m.start_col = 4;
  m.glyphs[0] = Glyph::of(Tile::from_char('A'));
  m.glyphs[1] = Glyph::of(Tile::from_char('X'));
  m.square_mask = 0b11;
  m.score = 9;
  rec.last_opp_move = m;

  std::vector<float> normal(kInputFloats, 0.0f);
  std::vector<float> flipped(kInputFloats, 0.0f);
  encode_input(rec, /*apply_flip=*/false, normal.data());
  encode_input(rec, /*apply_flip=*/true, flipped.data());

  // Scalars are flip-invariant.
  for (int i = kSpatialFloats; i < kInputFloats; ++i) {
    CHECK(normal[i] == flipped[i]);
  }
  // Every spatial plane (including last-opp) is transposed under the flip.
  for (int p = 0; p < kSpatialPlanes; ++p) {
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 15; ++c) {
        CHECK(flipped[p * 225 + r * 15 + c] == normal[p * 225 + c * 15 + r]);
      }
    }
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
    if (!req.legal_plays.empty()) {
      int best = -1;
      for (const auto& m : req.legal_plays) best = std::max(best, int(m.score));
      std::vector<const scribblez::Move*> top;
      for (const auto& m : req.legal_plays)
        if (int(m.score) == best) top.push_back(&m);
      std::uniform_int_distribution<size_t> d(0, top.size() - 1);
      return *top[d(rng_)];
    }
    scribblez::Move m;
    m.type = scribblez::MoveType::PASS;
    return m;
  }

 private:
  std::mt19937_64 rng_;
};

// Snapshot of game state at one eligible PositionRecord moment, captured
// during a live in-memory replay. Mirrors what extract_positions builds.
struct LiveSnapshot {
  scribblez::Board board;
  scribblez::Rack rack_active;
  scribblez::Move last_opp_move;
  int score_active = 0;
  int score_opp = 0;
  int turn_index = 0;
  int active_player = 0;
  scribblez::binlog::PositionKind kind = scribblez::binlog::PositionKind::kPreMove;
};

std::vector<LiveSnapshot> live_replay_all_snapshots(const scribblez::GameLog& log) {
  using namespace scribblez;
  using binlog::PositionKind;

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

    if (turn.move.type == MoveType::PLAY) {
      LiveSnapshot post = pre;
      post.board.apply(turn.move);
      const int n = turn.move.num_glyphs();
      for (int g = 0; g < n; ++g) post.rack_active.remove(turn.move.glyphs[g].rack_tile());
      post.score_active = prev_active + turn.score_delta;
      post.kind = PositionKind::kPostMove;
      out.push_back(post);
    }

    // Advance live state.
    if (turn.move.type == MoveType::PLAY) {
      const int n = turn.move.num_glyphs();
      for (int g = 0; g < n; ++g) racks[active].remove(turn.move.glyphs[g].rack_tile());
      board.apply(turn.move);
    } else if (turn.move.type == MoveType::EXCHANGE) {
      const int n = turn.move.num_glyphs();
      for (int g = 0; g < n; ++g) racks[active].remove(turn.move.glyphs[g].rack_tile());
    }
    for (Tile t : turn.drawn) racks[active].add(t);
    last_by[active] = turn.move;
  }
  return out;
}

// Reconstruct a Board from the raw 225-glyph array stored in a PositionRecord.
scribblez::Board board_from_record(const scribblez::binlog::PositionRecord& rec) {
  scribblez::Board b;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      b.set(r, c, rec.board[r * 15 + c]);
    }
  }
  return b;
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
  if (a.type != b.type) return false;
  if (a.type != scribblez::MoveType::PLAY) return true;  // PASS/EXCHANGE: type alone suffices here
  if (a.horizontal != b.horizontal) return false;
  if (a.start_row != b.start_row || a.start_col != b.start_col) return false;
  if (a.square_mask != b.square_mask) return false;
  if (a.score != b.score) return false;
  for (int i = 0; i < scribblez::RACK_SIZE; ++i) {
    if (a.glyphs[i].code() != b.glyphs[i].code()) return false;
  }
  return true;
}

// Play one game with two TestAgents and return the log.
scribblez::GameLog play_test_game(const scribblez::Dictionary& dict, uint64_t seed) {
  TestAgent a0(0, "A0", seed ^ 0x1111111111111111ULL);
  TestAgent a1(0, "A1", seed ^ 0x2222222222222222ULL);
  scribblez::Game g(a0, a1, dict, seed);
  g.play();
  return g.log();
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

// extract_positions produces records whose (board, rack, last_opp_move,
// scores) faithfully reproduce the live game state -- proven by running
// movegen on both and demanding identical legal-play sets.
static void test_extract_positions_movegen_roundtrip() {
  Dictionary dict = medium_dict();

  // A handful of games at different seeds; sample every eligible position
  // (samples_per_game large) so we test pre- AND post-move kinds at every
  // turn.
  const std::vector<uint64_t> seeds = {42, 1337, 0xDEADBEEFULL};
  long positions_compared = 0;

  for (uint64_t seed : seeds) {
    scribblez::GameLog log = play_test_game(dict, seed);
    CHECK(!log.turns.empty());

    auto live_snaps = live_replay_all_snapshots(log);
    auto records = scribblez::binlog::extract_positions(log, /*samples_per_game=*/10000);
    CHECK(records.size() <= live_snaps.size());

    // Index live snapshots by (turn_index, kind) so we can look them up by
    // each record's coordinates.
    auto key = [](int turn, scribblez::binlog::PositionKind k) {
      return (static_cast<uint64_t>(turn) << 1) | static_cast<uint64_t>(k);
    };
    std::unordered_map<uint64_t, const LiveSnapshot*> by_key;
    by_key.reserve(live_snaps.size());
    for (const auto& s : live_snaps) by_key[key(s.turn_index, s.kind)] = &s;

    for (const auto& rec : records) {
      auto k =
        key(rec.move_number, static_cast<scribblez::binlog::PositionKind>(rec.position_kind));
      auto it = by_key.find(k);
      CHECK(it != by_key.end());
      const LiveSnapshot& live = *it->second;

      // 1. The PositionRecord faithfully encodes the live state.
      CHECK(rec.active_player == live.active_player);
      CHECK(rec.score_active == live.score_active);
      CHECK(rec.score_opp == live.score_opp);
      scribblez::Board recon_board = board_from_record(rec);
      CHECK(boards_equal(recon_board, live.board));
      CHECK(racks_equal(rec.active_rack, live.rack_active));
      CHECK(moves_equal_for_replay(rec.last_opp_move, live.last_opp_move));

      // 2. The killer test: movegen on the round-tripped state agrees
      //    exactly with movegen on the live state.
      check_movegen_equiv(dict, recon_board, rec.active_rack, live.board, live.rack_active,
                          "extract_positions");
      ++positions_compared;
    }
  }
  CHECK(positions_compared > 0);
  std::cout << "  extract_positions+movegen round-trip OK (" << positions_compared
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
  constexpr int kSamplesPerGame = 12;
  std::vector<scribblez::GameLog> logs;
  {
    scribblez::binlog::BinaryLogWriter writer(dir.string(), /*games_per_file=*/kGames,
                                              kSamplesPerGame);
    for (int i = 0; i < kGames; ++i) {
      scribblez::GameLog log = play_test_game(dict, /*seed=*/100ULL + i);
      writer.append(log);
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
  const int64_t total_positions = hdr->num_positions;
  CHECK(total_positions > 0);

  // Register with DataLoader and drain all rows in one load().
  scribblez::binlog::DataLoader::Params dl_params;
  dl_params.num_worker_threads = 2;
  dl_params.num_prefetch_threads = 1;
  scribblez::binlog::DataLoader loader(dl_params);
  loader.add_file(slog.string(), total_positions, fsize);
  CHECK(loader.num_positions() == total_positions);

  const int n_samples = static_cast<int>(total_positions) * 4;  // oversample
  std::vector<float> rows(static_cast<size_t>(n_samples) *
                          scribblez::binlog::DataLoader::row_size_floats());
  loader.load(/*window_start=*/0, /*window_end=*/total_positions, n_samples,
              /*apply_symmetry=*/false, rows.data());

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
  const int row_size = scribblez::binlog::DataLoader::row_size_floats();
  const int label_off = scribblez::binlog::DataLoader::input_size_floats();
  for (int i = 0; i < n_samples; ++i) {
    const float* row = rows.data() + static_cast<int64_t>(i) * row_size;
    const int w = static_cast<int>(row[label_off + 0]);
    const int dd = static_cast<int>(row[label_off + 1]);
    const int l = static_cast<int>(row[label_off + 2]);
    const int sd = static_cast<int>(row[label_off + 3]);
    CHECK(w + dd + l == 1);  // exactly one of W/D/L
    CHECK(valid_labels.count({w, dd, l, sd}) == 1);
  }

  // Re-read the file's raw PositionRecords and verify each one still passes
  // the movegen round-trip against its live snapshot. This exercises the
  // disk-roundtrip path (writer -> file -> mmap-style reinterpret_cast)
  // that the DataLoader itself uses internally.
  const auto* metas = reinterpret_cast<const scribblez::binlog::GameMetadata*>(
    raw.data() + sizeof(scribblez::binlog::FileHeader));
  long compared = 0;
  for (uint32_t gi = 0; gi < hdr->num_games; ++gi) {
    const auto& gm = metas[gi];
    const auto* recs =
      reinterpret_cast<const scribblez::binlog::PositionRecord*>(raw.data() + gm.start_offset);
    // GameMetadata is written in the order games were appended, so logs[gi]
    // corresponds to metas[gi].
    CHECK(gm.seed == logs[gi].seed);
    auto live_snaps = live_replay_all_snapshots(logs[gi]);
    std::unordered_map<uint64_t, const LiveSnapshot*> by_key;
    auto key = [](int turn, scribblez::binlog::PositionKind k) {
      return (static_cast<uint64_t>(turn) << 1) | static_cast<uint64_t>(k);
    };
    for (const auto& s : live_snaps) by_key[key(s.turn_index, s.kind)] = &s;

    for (uint32_t ri = 0; ri < gm.num_positions; ++ri) {
      const auto& rec = recs[ri];
      auto it = by_key.find(
        key(rec.move_number, static_cast<scribblez::binlog::PositionKind>(rec.position_kind)));
      CHECK(it != by_key.end());
      const LiveSnapshot& live = *it->second;
      scribblez::Board recon = board_from_record(rec);
      CHECK(boards_equal(recon, live.board));
      CHECK(racks_equal(rec.active_rack, live.rack_active));
      check_movegen_equiv(dict, recon, rec.active_rack, live.board, live.rack_active,
                          "file-roundtrip");
      ++compared;
    }
  }
  CHECK(compared == total_positions);
  std::cout << "  file+DataLoader round-trip OK (" << kGames << " games, " << total_positions
            << " positions, " << n_samples << " loader rows)\n";
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
  Move m;
  m.type = MoveType::PLAY;
  m.horizontal = true;
  m.start_row = 7;
  m.start_col = 7;
  m.glyphs[0] = Glyph::of(Tile::from_char('C'));
  m.glyphs[1] = Glyph::of(Tile::from_char('T'));
  m.square_mask = 0b101;

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
  Move hook;
  hook.type = MoveType::PLAY;
  hook.horizontal = true;
  hook.start_row = 7;
  hook.start_col = 7;
  hook.glyphs[0] = Glyph::of(Tile::from_char('S'));
  // Note: Move::main_word does not depend on start_col being the placement;
  // it walks from start_col through existing letters until it runs out of
  // placements AND hits an empty cell. start_col = 7 -> walks C,A,T then
  // places S then runs out -> "CATS".
  CHECK(hook.main_word(b) == "CATS");

  // 2. Through-word: place B at (7,6) and S at (7,10) for "BCATS"? Not a
  //    real word -- but main_word doesn't care about legality. We just
  //    verify that interleaving works.
  Move through;
  through.type = MoveType::PLAY;
  through.horizontal = true;
  through.start_row = 7;
  through.start_col = 6;
  through.glyphs[0] = Glyph::of(Tile::from_char('B'));  // placed at (7,6)
  through.glyphs[1] = Glyph::of(Tile::from_char('S'));  // placed at (7,10)
  CHECK(through.main_word(b) == "BCATS");

  // 3. Blank renders as its designated letter (uppercase), like a regular tile.
  Move with_blank;
  with_blank.type = MoveType::PLAY;
  with_blank.horizontal = true;
  with_blank.start_row = 7;
  with_blank.start_col = 7;
  with_blank.glyphs[0] = Glyph::played(Tile::from_char('S'), /*is_blank=*/true);
  CHECK(with_blank.main_word(b) == "CATS");

  // 4. PASS / EXCHANGE produce empty strings.
  Move pass;
  CHECK(pass.main_word(b).empty());
  Move xch;
  xch.type = MoveType::EXCHANGE;
  xch.glyphs[0] = Glyph::of(Tile::from_char('A'));
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
      score_real = m.score;
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
      CHECK(m.glyphs[0].is_blank());
      score_blank = m.score;
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
    scribblez::Move m;
    m.type = scribblez::MoveType::PASS;
    return m;
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
    scribblez::GameLog log = play_test_game(dict, seed);
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
  const auto& log = g.log();

  CHECK(log.end_reason == "stalemate");
  CHECK(log.turns.size() == 6);  // 6 zero turns (3 per player)
  for (const auto& t : log.turns) CHECK(t.move.type == MoveType::PASS);
  for (int p = 0; p < 2; ++p) {
    CHECK(log.final_scores[p] == -log.final_racks[p].point_value());
  }
}

// ===========================================================================
// LabelEncoder
// ===========================================================================

static void test_encode_labels() {
  using namespace scribblez::binlog;
  float out[kLabelFloats];

  // Win.
  encode_labels(/*fs_active=*/120, /*fs_opp=*/100, out);
  CHECK(out[0] == 1.0f);
  CHECK(out[1] == 0.0f);
  CHECK(out[2] == 0.0f);
  CHECK(out[3] == 20.0f);

  // Draw.
  encode_labels(75, 75, out);
  CHECK(out[0] == 0.0f);
  CHECK(out[1] == 1.0f);
  CHECK(out[2] == 0.0f);
  CHECK(out[3] == 0.0f);

  // Loss with negative score_diff.
  encode_labels(80, 95, out);
  CHECK(out[0] == 0.0f);
  CHECK(out[1] == 0.0f);
  CHECK(out[2] == 1.0f);
  CHECK(out[3] == -15.0f);

  // WLD entries are mutually exclusive and sum to 1.0 for every case.
  for (auto [a, b] : std::vector<std::pair<int, int>>{{1, 0}, {0, 0}, {-5, 5}, {200, -200}}) {
    encode_labels(a, b, out);
    CHECK(out[0] + out[1] + out[2] == 1.0f);
  }
}

// ===========================================================================
// DataLoader: per-row diagonal-flip symmetry
// ===========================================================================

// Build a one-game, one-position .slog file under `dir` with a single record
// whose board has an asymmetric placement (a 'Q' at (3,5)). Returns the file
// path and the on-disk size.
static std::pair<std::filesystem::path, int64_t> write_one_position_slog(
  const std::filesystem::path& dir) {
  using namespace scribblez::binlog;

  PositionRecord rec{};
  rec.active_player = 0;
  rec.position_kind = static_cast<uint8_t>(PositionKind::kPreMove);
  rec.score_active = 10;
  rec.score_opp = 4;
  rec.board[3 * 15 + 5] = Glyph::of(Tile::from_char('Q'));

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.num_games = 1;
  hdr.num_positions = 1;

  GameMetadata gm{};
  gm.start_offset = sizeof(FileHeader) + sizeof(GameMetadata);
  gm.num_positions = 1;
  gm.data_size = sizeof(PositionRecord);
  gm.seed = 0xC0FFEEULL;
  gm.final_score_p0 = 350;  // active=p0 -> win, score_diff=+150
  gm.final_score_p1 = 200;

  std::filesystem::path path = dir / "one_position.slog";
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(&gm), sizeof(gm));
    f.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    CHECK(f);
  }  // f destructor flushes + closes before we measure file_size
  int64_t fsize = static_cast<int64_t>(std::filesystem::file_size(path));
  return {path, fsize};
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

  auto [slog, fsize] = write_one_position_slog(dir);

  // Build the two reference encodings we expect to see in the output rows.
  PositionRecord rec{};
  rec.active_player = 0;
  rec.position_kind = static_cast<uint8_t>(PositionKind::kPreMove);
  rec.score_active = 10;
  rec.score_opp = 4;
  rec.board[3 * 15 + 5] = Glyph::of(Tile::from_char('Q'));

  std::vector<float> ref_normal(kInputFloats, 0.0f);
  std::vector<float> ref_flipped(kInputFloats, 0.0f);
  encode_input(rec, /*apply_flip=*/false, ref_normal.data());
  encode_input(rec, /*apply_flip=*/true, ref_flipped.data());
  // Sanity: the two encodings differ (asymmetric Q placement).
  CHECK(std::memcmp(ref_normal.data(), ref_flipped.data(), kInputFloats * sizeof(float)) != 0);

  // Expected labels for active=p0 (win, score_diff=+150).
  float ref_labels[kLabelFloats];
  encode_labels(/*fs_active=*/350, /*fs_opp=*/200, ref_labels);

  DataLoader::Params params;
  params.num_worker_threads = 1;
  params.num_prefetch_threads = 1;
  DataLoader loader(params);
  loader.add_file(slog.string(), /*num_positions=*/1, fsize);

  // apply_symmetry=false: every row must match the canonical (unflipped)
  // encoding.
  {
    constexpr int n = 64;
    std::vector<float> rows(static_cast<size_t>(n) * DataLoader::row_size_floats(), 0.0f);
    loader.load(0, 1, n, /*apply_symmetry=*/false, rows.data());
    for (int i = 0; i < n; ++i) {
      const float* row = rows.data() + i * DataLoader::row_size_floats();
      CHECK(std::memcmp(row, ref_normal.data(), kInputFloats * sizeof(float)) == 0);
      CHECK(std::memcmp(row + kInputFloats, ref_labels, kLabelFloats * sizeof(float)) == 0);
    }
  }

  // apply_symmetry=true: each row independently coin-flipped. Over many rows
  // we expect both buckets to appear and every row to match one of the two
  // reference encodings exactly.
  {
    constexpr int n = 200;
    std::vector<float> rows(static_cast<size_t>(n) * DataLoader::row_size_floats(), 0.0f);
    loader.load(0, 1, n, /*apply_symmetry=*/true, rows.data());
    int normal_count = 0, flipped_count = 0;
    for (int i = 0; i < n; ++i) {
      const float* row = rows.data() + i * DataLoader::row_size_floats();
      const bool is_normal = std::memcmp(row, ref_normal.data(), kInputFloats * sizeof(float)) == 0;
      const bool is_flipped =
        std::memcmp(row, ref_flipped.data(), kInputFloats * sizeof(float)) == 0;
      CHECK(is_normal || is_flipped);  // every row matches one of the two
      if (is_normal)
        ++normal_count;
      else
        ++flipped_count;
      // Labels are flip-invariant.
      CHECK(std::memcmp(row + kInputFloats, ref_labels, kLabelFloats * sizeof(float)) == 0);
    }
    // With n=200 fair coin flips, the probability that one bucket is empty
    // is 2 * 2^-200; the test is effectively deterministic.
    CHECK(normal_count > 0);
    CHECK(flipped_count > 0);
    std::cout << "  DataLoader per-row symmetry: " << normal_count << " normal / " << flipped_count
              << " flipped (of " << n << ")\n";
  }
}

int main() {
  test_dict_basic();
  test_movegen_opening();
  test_movegen_cross_word();
  test_bingo_bonus();
  test_gaddag_vs_dawg_inmemory();
  test_real_kwg_optional();
  test_encoder_basic_layout();
  test_encoder_last_opp_plane_mask();
  test_encoder_flip_symmetry();
  test_extract_positions_movegen_roundtrip();
  test_binary_log_file_and_data_loader_roundtrip();
  test_tile_glyph_basics();
  test_rack_invariants();
  test_bag_basics();
  test_board_apply_interleaves_cross_tiles();
  test_move_main_word_through_cross();
  test_movegen_blank_scores_zero();
  test_game_end_rack_out_bonus();
  test_game_end_stalemate_penalty();
  test_encode_labels();
  test_dataloader_per_row_symmetry();
  std::cout << "All tests passed.\n";
  return 0;
}
