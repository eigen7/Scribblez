// Unit tests for NeuralTopKAgent's selection and encoding, with the model
// replaced by a scripted EvalService stub -- no ONNX, no TensorRT, no GPU.
//
//  * Suite 2.2: among the top-K equity candidates the agent plays the one the
//    stub rates highest under the configured objective (score-diff vs win-prob),
//    even when that overrides HastyBot's equity ranking.
//  * Suite 2.3: encode_candidate() produces exactly the row an independent
//    GameStateEncoder replay of the same moves produces -- so the agent feeds
//    the model the position (POV, leave, post-move state) it was trained on.

#include "test_support.h"

#include "scribblez/agent.h"
#include "scribblez/board.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/neural_top_k_agent.h"
#include "scribblez/nn/eval_service.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace scribblez;
using namespace scribblez::test_support;

namespace {

// An EvalService that returns pre-set evals, one per candidate in call order,
// ignoring the input rows entirely. Lets a test dictate the value head's
// verdict and assert on which move the agent then plays.
class StubEvalService : public nn::EvalService {
 public:
  std::vector<nn::Eval> scripted;
  void evaluate(const float* /*inputs*/, int count, nn::Eval* out) override {
    for (int i = 0; i < count; ++i) {
      out[i] = (i < static_cast<int>(scripted.size())) ? scripted[i] : nn::Eval{};
    }
  }
};

nn::Eval eval_with(float score_diff_mean, float win_prob) {
  nn::Eval e;
  e.score_diff_mean = score_diff_mean;
  e.win_prob = win_prob;
  return e;
}

// Initialize the HastyEquity singleton from the synthetic KLV + an empty PEG so
// the agent's candidate ranking works. Returns the temp dir (caller removes it).
std::filesystem::path init_equity() {
  namespace fs = std::filesystem;
  fs::path tmp = fs::temp_directory_path() / "scribblez_test_topk_agent_XXXXXX";
  fs::create_directories(tmp);
  KlvFixture fix = write_synthetic_klv(tmp);
  fs::path peg = tmp / "peg.json";
  {
    std::ofstream pf(peg);
    pf << "[]";
  }
  HastyEquity::init(fix.path.string(), peg.string());
  return tmp;
}

// Two single-tile consonant plays with very different scores (so their static
// equities are just their scores -- no leave/opening/endgame terms apply): a
// high-equity play at row 7 and a low-equity play at row 5.
const uint16_t kHighStart = 7;
const uint16_t kHighMask = static_cast<uint16_t>(1u << 7);
const uint16_t kLowStart = 5;
const uint16_t kLowMask = static_cast<uint16_t>(1u << 5);

std::vector<Move> two_candidate_plays() {
  return {make_play_full(7, 7, /*horizontal=*/true, 0b1, 30, {Glyph::of(Tile::from_char('T'))}),
          make_play_full(5, 5, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('R'))})};
}

void test_topk_selection_uses_objective() {
  std::filesystem::path tmp = init_equity();

  Board board;  // empty; consonant plays incur no opening adjustment
  Rack my_rack = rack_from("TR");
  Rack opp;
  std::vector<Move> plays = two_candidate_plays();
  MoveRequest req{board, my_rack, opp, /*my_score=*/0, /*opp_score=*/0, /*bag_size=*/50, plays};

  // Candidate slot 0 is the higher-equity play (row 7), slot 1 the lower (row 5).
  auto stub = std::make_unique<StubEvalService>();
  StubEvalService* sp = stub.get();
  NeuralTopKAgent agent(/*thread_id=*/0, "stub", std::move(stub), /*top_k=*/2,
                        NeuralTopKAgent::Objective::kScoreDiff);
  agent.begin_game({0, 0});

  // Value head prefers the LOW-equity play -> the agent overrides HastyBot.
  sp->scripted = {eval_with(/*sd=*/1.0f, /*wp=*/0.0f), eval_with(/*sd=*/9.0f, /*wp=*/0.0f)};
  Move got = agent.make_move(req);
  CHECK(got.start() == kLowStart);
  CHECK(got.square_mask() == kLowMask);

  // Value head prefers the HIGH-equity play -> the agent agrees with HastyBot.
  sp->scripted = {eval_with(9.0f, 0.0f), eval_with(1.0f, 0.0f)};
  got = agent.make_move(req);
  CHECK(got.start() == kHighStart);
  CHECK(got.square_mask() == kHighMask);

  // Win-prob objective reads win_prob, not score_diff_mean: here score-diff
  // favors slot 0 but win-prob favors slot 1, so the win-prob agent plays slot 1.
  auto stub2 = std::make_unique<StubEvalService>();
  StubEvalService* sp2 = stub2.get();
  NeuralTopKAgent agent_wp(/*thread_id=*/0, "stub-wp", std::move(stub2), /*top_k=*/2,
                           NeuralTopKAgent::Objective::kWinProb);
  agent_wp.begin_game({0, 0});
  sp2->scripted = {eval_with(/*sd=*/9.0f, /*wp=*/0.1f), eval_with(/*sd=*/1.0f, /*wp=*/0.9f)};
  got = agent_wp.make_move(req);
  CHECK(got.start() == kLowStart);
  CHECK(got.square_mask() == kLowMask);

  std::filesystem::remove_all(tmp);
  std::cout << "test_topk_selection_uses_objective passed\n";
}

// Remove a play's tiles from a rack copy (the leave the encoder sees).
Rack leave_after(const Rack& rack, const Move& mv) {
  Rack leave = rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());
  return leave;
}

void test_encode_candidate_matches_replay() {
  using binlog::GameStateEncoder;
  using binlog::kInputFloats;

  // A short move history applied to both the agent (via observe_move) and an
  // independent reference encoder.
  Move move_a = make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                               {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                                Glyph::of(Tile::from_char('T'))});
  Move move_b =
    make_play_full(9, 7, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('S'))});

  auto stub = std::make_unique<StubEvalService>();
  NeuralTopKAgent agent(/*thread_id=*/0, "stub", std::move(stub), /*top_k=*/4,
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

}  // namespace

int main() {
  test_topk_selection_uses_objective();
  test_encode_candidate_matches_replay();
  std::cout << "All neural-topk-agent tests passed.\n";
  return 0;
}
