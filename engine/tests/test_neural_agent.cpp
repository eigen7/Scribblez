// Unit tests for NeuralAgent, with the model replaced by a scripted EvalService
// stub -- no ONNX, no TensorRT, no GPU.
//
//  * top-K mode (top_k > 0): among the top-K equity candidates the agent plays
//    the one the stub rates highest under the configured objective, and a play
//    outside the top-K is never evaluated.
//  * all-moves mode (top_k == 0): every legal play is evaluated with no equity
//    pre-filter, so the agent can play a move HastyBot ranks last.
//  * chunked evaluation: with a small batch limit the agent still scores every
//    candidate (across multiple evaluate() calls) and picks the global best.
//  * encode_candidate() parity: it produces exactly the row an independent
//    GameStateEncoder replay -- and the training BlockDecoder -- produce, so the
//    agent feeds the model the position (POV, leave, post-move state) it was
//    trained on.
//  * temperature controls selection: temperature 0 is the greedy argmax, while a
//    high temperature samples across the candidates.

#include "scribblez/agent.h"
#include "scribblez/binary_log.h"
#include "scribblez/block_decoder.h"
#include "scribblez/board.h"
#include "scribblez/data_loader.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/glyph.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/move.h"
#include "scribblez/neural_agent.h"
#include "scribblez/nn/eval_service.h"
#include "scribblez/rack.h"
#include "scribblez/tile.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::cerr << "CHECK failed: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

using namespace scribblez;

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

// Build a PLAY Move with an explicit per-tile layout. `rel_mask` is relative to
// the first lane cell (bit 0 == the start cell); `gs` are the placed glyphs in
// word order (count == popcount(rel_mask)).
static Move make_play_full(int row, int col, bool horizontal, uint16_t rel_mask, uint16_t score,
                           std::initializer_list<Glyph> gs) {
  std::array<Glyph, RACK_SIZE> played{};
  int n = 0;
  for (Glyph g : gs) {
    if (n >= RACK_SIZE) break;
    played[n++] = g;
  }
  const int lane0 = horizontal ? col : row;
  const int start = horizontal ? row : col;
  uint16_t mask = static_cast<uint16_t>(rel_mask << lane0);
  return Move::play(horizontal, start, mask, score, played.data(), n);
}

// A single-tile horizontal play at (row, col): start() == row, square at col.
static Move single_tile_play(int row, int col, uint16_t score, char letter) {
  return make_play_full(row, col, /*horizontal=*/true, 0b1, score,
                        {Glyph::of(Tile::from_char(letter))});
}

// Write a synthetic Macondo .klv2 leave file (blank=12.0, A=1.5, B=-2.5; every
// other leave looks up as 0) and initialize the HastyEquity singleton from it,
// so the agent's candidate ranking works. Returns the temp dir to remove.
static std::filesystem::path init_equity() {
  namespace fs = std::filesystem;
  fs::path tmp = fs::temp_directory_path() / "scribblez_test_neural_agent_XXXXXX";
  fs::create_directories(tmp);

  fs::path klv = tmp / "synthetic.klv2";
  std::ofstream f(klv, std::ios::binary | std::ios::trunc);
  auto write_u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto write_f32 = [&](float v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  write_u32(4);                                           // kwg_node_count
  write_u32((0u << 24) | (1u << 22) | (0u << 23) | 1u);  // root
  write_u32((0u << 24) | (0u << 22) | (1u << 23) | 0u);  // ? (blank)
  write_u32((1u << 24) | (0u << 22) | (1u << 23) | 0u);  // A
  write_u32((2u << 24) | (1u << 22) | (1u << 23) | 0u);  // B
  write_u32(3);                                          // num_leaves
  write_f32(12.0f);
  write_f32(1.5f);
  write_f32(-2.5f);
  CHECK(f);
  f.close();  // flush to disk before HastyEquity::init reopens the file to read

  fs::path peg = tmp / "peg.json";
  std::ofstream(peg) << "[]";
  HastyEquity::init(klv.string(), peg.string());
  return tmp;
}

// An EvalService that returns pre-set evals, one per candidate in *per-call*
// order, ignoring the input rows. Suits tests that re-script and call make_move
// repeatedly on the same agent (the scripted index resets every call).
class StubEvalService : public nn::EvalService {
 public:
  std::vector<nn::Eval> scripted;
  void evaluate(const float* /*inputs*/, int count, nn::Eval* out) override {
    for (int i = 0; i < count; ++i) {
      out[i] = (i < static_cast<int>(scripted.size())) ? scripted[i] : nn::Eval{};
    }
  }
};

// An EvalService that returns pre-set evals indexed in *global* candidate order
// across however many chunked evaluate() calls a single make_move() makes, and
// records the total rows seen, the largest chunk, and the call count. Suits the
// all-moves and chunking tests (one make_move per agent).
class CountingStubEvalService : public nn::EvalService {
 public:
  std::vector<nn::Eval> scripted;
  int total_rows = 0;
  int max_chunk = 0;
  int calls = 0;

  void evaluate(const float* /*inputs*/, int count, nn::Eval* out) override {
    ++calls;
    max_chunk = std::max(max_chunk, count);
    for (int i = 0; i < count; ++i) {
      const int g = total_rows + i;
      out[i] = (g < static_cast<int>(scripted.size())) ? scripted[g] : nn::Eval{};
    }
    total_rows += count;
  }
};

static nn::Eval eval_with(float score_diff_mean, float win_prob) {
  nn::Eval e;
  e.score_diff_mean = score_diff_mean;
  e.win_prob = win_prob;
  return e;
}

static nn::Eval sd(float score_diff_mean) { return eval_with(score_diff_mean, 0.0f); }

static void test_topk_selection_uses_objective() {
  std::filesystem::path tmp = init_equity();

  // Two single-tile consonant plays with very different scores, so their static
  // equities are just their scores: a high-equity play at row 7, a low-equity
  // play at row 5. Slot 0 is the higher.
  Board board;  // empty
  Rack my_rack = rack_from("TR");
  Rack opp;
  std::vector<Move> plays = {single_tile_play(7, 7, 30, 'T'), single_tile_play(5, 5, 5, 'R')};
  MoveRequest req{board, my_rack, opp, /*my_score=*/0, /*opp_score=*/0, /*bag_size=*/50, plays};

  const uint16_t high_start = 7, low_start = 5;

  auto stub = std::make_unique<StubEvalService>();
  StubEvalService* sp = stub.get();
  NeuralAgent agent(/*thread_id=*/0, "stub", std::move(stub), /*top_k=*/2,
                    NeuralAgent::Objective::kScoreDiff);
  agent.begin_game();

  // Value head prefers the LOW-equity play -> the agent overrides HastyBot.
  sp->scripted = {eval_with(/*sd=*/1.0f, /*wp=*/0.0f), eval_with(/*sd=*/9.0f, /*wp=*/0.0f)};
  Move got = agent.make_move(req);
  CHECK(got.start() == low_start);

  // Value head prefers the HIGH-equity play -> the agent agrees with HastyBot.
  sp->scripted = {eval_with(9.0f, 0.0f), eval_with(1.0f, 0.0f)};
  got = agent.make_move(req);
  CHECK(got.start() == high_start);

  // The win-prob objective reads win_prob, not score_diff_mean.
  auto stub2 = std::make_unique<StubEvalService>();
  StubEvalService* sp2 = stub2.get();
  NeuralAgent agent_wp(/*thread_id=*/0, "stub-wp", std::move(stub2), /*top_k=*/2,
                       NeuralAgent::Objective::kWinProb);
  agent_wp.begin_game();
  sp2->scripted = {eval_with(/*sd=*/9.0f, /*wp=*/0.1f), eval_with(/*sd=*/1.0f, /*wp=*/0.9f)};
  got = agent_wp.make_move(req);
  CHECK(got.start() == low_start);

  std::filesystem::remove_all(tmp);
  std::cout << "test_topk_selection_uses_objective passed\n";
}

static void test_topk_excludes_low_equity_play() {
  std::filesystem::path tmp = init_equity();

  // Three plays; top_k=2 keeps the top-2 by equity (play0, play1) and drops the
  // lowest (play2) before evaluation, even though the stub would rate it best.
  Board board;
  Rack my_rack = rack_from("TRN");
  Rack opp;
  std::vector<Move> plays = {single_tile_play(7, 7, 40, 'T'), single_tile_play(6, 6, 22, 'R'),
                             single_tile_play(5, 5, 5, 'N')};
  MoveRequest req{board, my_rack, opp, 0, 0, /*bag_size=*/50, plays};

  auto stub = std::make_unique<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  // Candidate order (top-2 by equity): play0, play1. play2 is never scored.
  sp->scripted = {sd(1.0f), sd(5.0f)};
  NeuralAgent agent(/*thread_id=*/0, "topk", std::move(stub), /*top_k=*/2,
                    NeuralAgent::Objective::kScoreDiff);
  agent.begin_game();

  Move got = agent.make_move(req);
  CHECK(got.start() == 6);     // play1 (a survivor), not the excluded play2
  CHECK(sp->total_rows == 2);  // only the top-2 were evaluated

  std::filesystem::remove_all(tmp);
  std::cout << "test_topk_excludes_low_equity_play passed\n";
}

static void test_all_moves_evaluated() {
  std::filesystem::path tmp = init_equity();

  // top_k = 0: every legal play is evaluated. The stub prefers the LOWEST-equity
  // play (play2) -- a move a top-K<3 agent would never even see.
  Board board;
  Rack my_rack = rack_from("TRN");
  Rack opp;
  std::vector<Move> plays = {single_tile_play(7, 7, 40, 'T'), single_tile_play(6, 6, 22, 'R'),
                             single_tile_play(5, 5, 5, 'N')};
  MoveRequest req{board, my_rack, opp, 0, 0, /*bag_size=*/50, plays};

  auto stub = std::make_unique<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  sp->scripted = {sd(1.0f), sd(2.0f), sd(9.0f)};  // play2 best
  NeuralAgent agent(/*thread_id=*/0, "full", std::move(stub), /*top_k=*/0,
                    NeuralAgent::Objective::kScoreDiff);
  agent.begin_game();

  Move got = agent.make_move(req);
  CHECK(got.start() == 5);     // played the model's pick, not HastyBot's
  CHECK(sp->total_rows == 3);  // every legal play was evaluated

  std::filesystem::remove_all(tmp);
  std::cout << "test_all_moves_evaluated passed\n";
}

static void test_chunked_evaluation() {
  std::filesystem::path tmp = init_equity();

  // Five plays scored with a batch limit of 2 -> the agent scores them across
  // multiple evaluate() calls, yet still picks the global best (play3).
  Board board;
  Rack my_rack = rack_from("TRNSL");
  Rack opp;
  std::vector<Move> plays = {single_tile_play(3, 3, 10, 'T'), single_tile_play(4, 4, 11, 'R'),
                             single_tile_play(5, 5, 12, 'N'), single_tile_play(6, 6, 30, 'S'),
                             single_tile_play(7, 7, 9, 'L')};
  MoveRequest req{board, my_rack, opp, 0, 0, /*bag_size=*/50, plays};

  auto stub = std::make_unique<CountingStubEvalService>();
  CountingStubEvalService* sp = stub.get();
  sp->scripted = {sd(1.0f), sd(2.0f), sd(3.0f), sd(9.0f), sd(4.0f)};  // index 3 best
  NeuralAgent agent(/*thread_id=*/0, "chunk", std::move(stub), /*top_k=*/0,
                    NeuralAgent::Objective::kScoreDiff, /*temperature=*/0.0, /*seed=*/0,
                    /*max_batch=*/2);
  agent.begin_game();

  Move got = agent.make_move(req);
  CHECK(got.start() == 6);     // play3, the globally best-rated candidate
  CHECK(sp->total_rows == 5);  // all five scored
  CHECK(sp->max_chunk <= 2);   // never exceeded the batch limit
  CHECK(sp->calls >= 3);       // ceil(5 / 2) == 3 chunks

  std::filesystem::remove_all(tmp);
  std::cout << "test_chunked_evaluation passed (calls=" << sp->calls << ")\n";
}

// Remove a play's tiles from a rack copy (the leave the encoder sees).
static Rack leave_after(const Rack& rack, const Move& mv) {
  Rack leave = rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());
  return leave;
}

static void test_encode_candidate_matches_replay() {
  using binlog::GameStateEncoder;
  using binlog::kInputFloats;

  // A short move history applied to both the agent (via observe_move) and an
  // independent reference encoder.
  Move move_a = make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                               {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                                Glyph::of(Tile::from_char('T'))});
  Move move_b =
    make_play_full(9, 7, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('S'))});

  // encode_candidate uses only the tracked encoder, so no model/service is run.
  NeuralAgent agent(/*thread_id=*/0, "stub", std::make_unique<StubEvalService>(), /*top_k=*/4,
                    NeuralAgent::Objective::kScoreDiff);
  agent.begin_game();
  agent.observe_move(move_a);
  agent.observe_move(move_b);

  GameStateEncoder ref;
  ref.apply_move(move_a);
  ref.apply_move(move_b);
  const int my_seat = ref.active_player();

  // The candidate the agent is about to score, with the rack it holds.
  Move candidate =
    make_play_full(0, 0, /*horizontal=*/true, 0b11, 8,
                   {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('O'))});
  Rack my_rack = rack_from("DONERST");

  std::vector<float> agent_row(kInputFloats);
  agent.encode_candidate(candidate, my_rack, my_seat, agent_row.data());

  std::vector<float> ref_row(kInputFloats);
  GameStateEncoder post = ref;
  post.apply_move(candidate);
  post.encode_input(my_seat, leave_after(my_rack, candidate), /*apply_flip=*/false, ref_row.data());

  for (size_t i = 0; i < agent_row.size(); ++i) CHECK(agent_row[i] == ref_row[i]);
  std::cout << "test_encode_candidate_matches_replay passed (" << kInputFloats << " floats)\n";
}

// Append the raw bytes of a trivially-copyable value to a byte buffer.
template <class T>
static void append_pod(std::vector<char>& buf, const T& v) {
  const char* p = reinterpret_cast<const char*>(&v);
  buf.insert(buf.end(), p, p + sizeof(T));
}

// Serialize a single game into an in-memory .slog buffer the real BlockDecoder
// can read: FileHeader, one GameMetadata (with a caller-chosen sampled_turn),
// then the game's InitialRacks and TurnBlob[]. Scores are left at zero because
// this fixture only compares the input row, not the score-derived targets.
static std::vector<char> build_slog(const binlog::InitialRacks& ir,
                                    const std::vector<binlog::TurnBlob>& turns,
                                    uint32_t sampled_turn) {
  binlog::FileHeader hdr{};
  hdr.magic = binlog::kMagic;
  hdr.version = binlog::kVersion;
  hdr.num_games = 1;

  binlog::GameMetadata gm{};
  gm.start_offset = sizeof(binlog::FileHeader) + sizeof(binlog::GameMetadata);
  gm.num_turns = static_cast<uint32_t>(turns.size());
  gm.sampled_turn = sampled_turn;

  std::vector<char> buf;
  append_pod(buf, hdr);
  append_pod(buf, gm);
  append_pod(buf, ir);
  for (const binlog::TurnBlob& t : turns) append_pod(buf, t);
  return buf;
}

static void test_encode_candidate_matches_training_decoder() {
  using binlog::kInputFloats;
  using binlog::kRowFloats;

  // A three-turn game. The sampled turn (2, post-move) is player 0's, so the
  // decoded POV is player 0 and both players already have a prior move, which
  // exercises the last-self / last-opp placement-plane features.
  Move move0 = make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                              {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                               Glyph::of(Tile::from_char('T'))});
  Move move1 =
    make_play_full(0, 0, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('S'))});
  Move move2 = make_play_full(2, 2, /*horizontal=*/true, 0b11, 8,
                              {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('O'))});
  const uint32_t sampled_turn = 2;
  const int mover = static_cast<int>(sampled_turn % 2);  // turn k is played by k % 2

  // Initial racks and post-turn draws chosen so the decoder reconstructs
  // player 0's pre-move rack at turn 2 as DONERST: starting CATERST, play CAT
  // (leave ERST), draw DON. Player 1 holds an S to play on turn 1.
  binlog::InitialRacks ir{};
  ir.p0 = rack_from("CATERST");
  ir.p1 = rack_from("SAINTED");

  binlog::TurnBlob t0{};
  t0.move = move0;
  t0.drawn = rack_from("DON");
  binlog::TurnBlob t1{};
  t1.move = move1;
  binlog::TurnBlob t2{};
  t2.move = move2;

  std::vector<char> buf = build_slog(ir, {t0, t1, t2}, sampled_turn);

  // Training path: decode the post-move sampled row (no symmetry flip).
  binlog::BlockDecoder dec;
  const uint8_t flips[1] = {0};
  std::vector<float> dec_row(kRowFloats, 0.0f);
  dec.decode(buf.data(), "test.slog", /*local_start=*/0, /*n_rows=*/1, flips, /*post_move=*/true,
             /*output_row_start=*/0, dec_row.data());

  // Inference path: the agent observes turns 0..1, then encodes the move played
  // at the sampled turn from the rack it holds there (CATERST -> ... -> DONERST).
  NeuralAgent agent(/*thread_id=*/0, "stub", std::make_unique<StubEvalService>(), /*top_k=*/4,
                    NeuralAgent::Objective::kScoreDiff);
  agent.begin_game();
  agent.observe_move(move0);
  agent.observe_move(move1);

  std::vector<float> agent_row(kInputFloats, 0.0f);
  agent.encode_candidate(move2, rack_from("DONERST"), mover, agent_row.data());

  bool any_nonzero = false;
  for (int i = 0; i < kInputFloats; ++i) {
    CHECK(agent_row[i] == dec_row[i]);
    any_nonzero = any_nonzero || agent_row[i] != 0.0f;
  }
  CHECK(any_nonzero);  // guard against a vacuous all-zero match
  std::cout << "test_encode_candidate_matches_training_decoder passed (" << kInputFloats
            << " floats)\n";
}

static void test_temperature_sampling_spreads() {
  std::filesystem::path tmp = init_equity();

  // Two single-tile plays; equity = score, so the top-2 are slot 0 (row 7,
  // score 30) and slot 1 (row 5, score 5). The stub then rates slot 0 higher.
  Board board;
  Rack my_rack = rack_from("TR");
  Rack opp;
  std::vector<Move> plays = {single_tile_play(7, 7, 30, 'T'), single_tile_play(5, 5, 5, 'R')};
  MoveRequest req{board, my_rack, opp, /*my_score=*/0, /*opp_score=*/0, /*bag_size=*/50, plays};
  const uint16_t high_start = 7, low_start = 5;

  // Temperature 0 -> always the model-preferred candidate (slot 0).
  auto greedy_stub = std::make_unique<StubEvalService>();
  StubEvalService* gp = greedy_stub.get();
  NeuralAgent greedy(/*thread_id=*/0, "greedy", std::move(greedy_stub), /*top_k=*/2,
                     NeuralAgent::Objective::kScoreDiff, /*temperature=*/0.0);
  greedy.begin_game();
  gp->scripted = {eval_with(/*sd=*/2.0f, 0.0f), eval_with(/*sd=*/0.0f, 0.0f)};
  for (int i = 0; i < 50; ++i) CHECK(greedy.make_move(req).start() == high_start);

  // High temperature -> both candidates are sampled, but the higher-rated one
  // (slot 0) still dominates. Seeded for reproducibility.
  auto sampler_stub = std::make_unique<StubEvalService>();
  StubEvalService* sp = sampler_stub.get();
  NeuralAgent sampler(/*thread_id=*/0, "sampler", std::move(sampler_stub), /*top_k=*/2,
                      NeuralAgent::Objective::kScoreDiff, /*temperature=*/5.0, /*seed=*/12345);
  sampler.begin_game();
  sp->scripted = {eval_with(/*sd=*/2.0f, 0.0f), eval_with(/*sd=*/0.0f, 0.0f)};
  int high = 0, low = 0;
  for (int i = 0; i < 400; ++i) {
    const uint16_t s = sampler.make_move(req).start();
    if (s == high_start)
      ++high;
    else if (s == low_start)
      ++low;
  }
  CHECK(high > 0 && low > 0);  // both arms explored
  CHECK(high > low);           // higher-valued arm dominates

  std::filesystem::remove_all(tmp);
  std::cout << "test_temperature_sampling_spreads passed (high=" << high << " low=" << low << ")\n";
}

int main() {
  test_topk_selection_uses_objective();
  test_topk_excludes_low_equity_play();
  test_all_moves_evaluated();
  test_chunked_evaluation();
  test_encode_candidate_matches_replay();
  test_encode_candidate_matches_training_decoder();
  test_temperature_sampling_spreads();
  std::cout << "All neural-agent tests passed.\n";
  return 0;
}
