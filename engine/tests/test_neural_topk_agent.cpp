// Unit tests for NeuralTopKAgent's selection and encoding, with the model
// replaced by a scripted EvalService stub -- no ONNX, no TensorRT, no GPU.
//
//  * Suite 2.2: among the top-K equity candidates the agent plays the one the
//    stub rates highest under the configured objective (score-diff vs win-prob),
//    even when that overrides HastyBot's equity ranking.
//  * Suite 2.3: encode_candidate() produces exactly the row an independent
//    GameStateEncoder replay of the same moves produces -- so the agent feeds
//    the model the position (POV, leave, post-move state) it was trained on.

#include "scribblez/agent.h"
#include "scribblez/board.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/glyph.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/move.h"
#include "scribblez/neural_top_k_agent.h"
#include "scribblez/nn/eval_service.h"
#include "scribblez/rack.h"
#include "scribblez/tile.h"

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
  const int start = horizontal ? row : col;
  const int lane0 = horizontal ? col : row;
  uint16_t mask = static_cast<uint16_t>(rel_mask << lane0);
  return Move::play(horizontal, start, mask, score, played.data(), n);
}

// Write a synthetic Macondo .klv2 leave file (blank=12.0, A=1.5, B=-2.5; every
// other leave looks up as 0) and initialize the HastyEquity singleton from it,
// so the agent's candidate ranking works. Returns the temp dir to remove.
static std::filesystem::path init_equity() {
  namespace fs = std::filesystem;
  fs::path tmp = fs::temp_directory_path() / "scribblez_test_topk_agent_XXXXXX";
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

  fs::path peg = tmp / "peg.json";
  std::ofstream(peg) << "[]";
  HastyEquity::init(klv.string(), peg.string());
  return tmp;
}

// An EvalService that returns pre-set evals, one per candidate in call order,
// ignoring the input rows entirely. Lets a test dictate the value head's verdict
// and assert on which move the agent then plays.
class StubEvalService : public nn::EvalService {
 public:
  std::vector<nn::Eval> scripted;
  void evaluate(const float* /*inputs*/, int count, nn::Eval* out) override {
    for (int i = 0; i < count; ++i) {
      out[i] = (i < static_cast<int>(scripted.size())) ? scripted[i] : nn::Eval{};
    }
  }
};

static nn::Eval eval_with(float score_diff_mean, float win_prob) {
  nn::Eval e;
  e.score_diff_mean = score_diff_mean;
  e.win_prob = win_prob;
  return e;
}

static void test_topk_selection_uses_objective() {
  std::filesystem::path tmp = init_equity();

  // Two single-tile consonant plays with very different scores, so their static
  // equities are just their scores (no leave/opening/endgame terms apply): a
  // high-equity play at row 7, a low-equity play at row 5. Slot 0 is the higher.
  Board board;  // empty
  Rack my_rack = rack_from("TR");
  Rack opp;
  std::vector<Move> plays = {
    make_play_full(7, 7, /*horizontal=*/true, 0b1, 30, {Glyph::of(Tile::from_char('T'))}),
    make_play_full(5, 5, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('R'))})};
  MoveRequest req{board, my_rack, opp, /*my_score=*/0, /*opp_score=*/0, /*bag_size=*/50, plays};

  const uint16_t high_start = 7, high_mask = 1u << 7;
  const uint16_t low_start = 5, low_mask = 1u << 5;

  auto stub = std::make_unique<StubEvalService>();
  StubEvalService* sp = stub.get();
  NeuralTopKAgent agent(/*thread_id=*/0, "stub", std::move(stub), /*top_k=*/2,
                        NeuralTopKAgent::Objective::kScoreDiff);
  agent.begin_game({0, 0});

  // Value head prefers the LOW-equity play -> the agent overrides HastyBot.
  sp->scripted = {eval_with(/*sd=*/1.0f, /*wp=*/0.0f), eval_with(/*sd=*/9.0f, /*wp=*/0.0f)};
  Move got = agent.make_move(req);
  CHECK(got.start() == low_start);
  CHECK(got.square_mask() == low_mask);

  // Value head prefers the HIGH-equity play -> the agent agrees with HastyBot.
  sp->scripted = {eval_with(9.0f, 0.0f), eval_with(1.0f, 0.0f)};
  got = agent.make_move(req);
  CHECK(got.start() == high_start);
  CHECK(got.square_mask() == high_mask);

  // The win-prob objective reads win_prob, not score_diff_mean: score-diff favors
  // slot 0 but win-prob favors slot 1, so the win-prob agent plays slot 1.
  auto stub2 = std::make_unique<StubEvalService>();
  StubEvalService* sp2 = stub2.get();
  NeuralTopKAgent agent_wp(/*thread_id=*/0, "stub-wp", std::move(stub2), /*top_k=*/2,
                           NeuralTopKAgent::Objective::kWinProb);
  agent_wp.begin_game({0, 0});
  sp2->scripted = {eval_with(/*sd=*/9.0f, /*wp=*/0.1f), eval_with(/*sd=*/1.0f, /*wp=*/0.9f)};
  got = agent_wp.make_move(req);
  CHECK(got.start() == low_start);
  CHECK(got.square_mask() == low_mask);

  std::filesystem::remove_all(tmp);
  std::cout << "test_topk_selection_uses_objective passed\n";
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
  NeuralTopKAgent agent(/*thread_id=*/0, "stub", std::make_unique<StubEvalService>(), /*top_k=*/4,
                        NeuralTopKAgent::Objective::kScoreDiff);
  agent.begin_game({0, 0});
  agent.observe_move(move_a);
  agent.observe_move(move_b);

  GameStateEncoder ref({0, 0});
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

int main() {
  test_topk_selection_uses_objective();
  test_encode_candidate_matches_replay();
  std::cout << "All neural-topk-agent tests passed.\n";
  return 0;
}
