#pragma once

// Shared checks that a model-driven agent feeds its model exactly what the
// training pipeline encodes for the same position: its board row against the
// row the BlockDecoder reconstructs by replay (NeuralAgent, MsetSimAgent,
// UltimateBotAgent), and its per-candidate move features against
// move_set::encode_move (MsetSimAgent, UltimateBotAgent).

#include "agent/agent.h"
#include "data/binary_log.h"
#include "data/block_decoder.h"
#include "data/data_loader.h"  // kLabelFloats
#include "encoding/input_encoder.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game_fixture.h"
#include "lexicon/dictionary.h"
#include "sim_agent_fixture.h"
#include "training/move_set_encoder.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace scribblez::testing {

// A three-turn game sampled at turn 2 (player 0's), so both players have a
// prior move and the last-move placement planes are exercised. Player 0's rack
// at turn 2 replays to DONERST: CATERST, plays CAT (moves[0]), draws DON.
// Player 1 holds the S it plays on turn 1.
inline constexpr uint32_t kThreeTurnSampledTurn = 2;

inline std::array<Move, 3> three_turn_moves() {
  return {make_play_full(7, 7, /*horizontal=*/true, 0b111, 10,
                         {Glyph::of(Tile::from_char('C')), Glyph::of(Tile::from_char('A')),
                          Glyph::of(Tile::from_char('T'))}),
          make_play_full(0, 0, /*horizontal=*/true, 0b1, 5, {Glyph::of(Tile::from_char('S'))}),
          make_play_full(2, 2, /*horizontal=*/true, 0b11, 8,
                         {Glyph::of(Tile::from_char('D')), Glyph::of(Tile::from_char('O'))})};
}

// The sampled mover's rack at turn 2.
inline Rack three_turn_mover_rack() { return rack_from("DONERST"); }

// The training row the BlockDecoder reconstructs for the three-turn game's
// sampled turn, untransposed: the pre-move row, or with `post_move` the row
// after the turn-2 move. That move is moves[2] unless `move2` overrides it
// (e.g. with an exchange from the DONERST rack). `dict` must be the agent's,
// for the cross-check planes.
inline std::vector<float> decode_three_turn_row(const Dictionary& dict,
                                                std::array<int, 2> initial_scores, bool post_move,
                                                const Move& move2 = three_turn_moves()[2]) {
  const std::array<Move, 3> moves = three_turn_moves();
  binlog::InitialRacks ir{};
  ir.p0 = rack_from("CATERST");
  ir.p1 = rack_from("SAINTED");
  binlog::TurnBlob t0{};
  t0.move = moves[0];
  t0.drawn = rack_from("DON");
  binlog::TurnBlob t1{};
  t1.move = moves[1];
  binlog::TurnBlob t2{};
  t2.move = move2;
  const std::vector<char> buf = build_slog(ir, {t0, t1, t2}, kThreeTurnSampledTurn, initial_scores);

  binlog::BlockDecoder dec(InputEncodingSpec{&dict});
  const uint8_t transposes[1] = {0};
  std::vector<float> row(size_t(input_floats(InputEncodingSpec{&dict}) + kLabelFloats), 0.0f);
  dec.decode(buf.data(), "test.slog", /*local_start=*/0, /*n_rows=*/1, transposes, post_move,
             /*output_row_start=*/0, row.data());
  return row;
}

// The agent's input row equals the decoder row's leading input floats exactly,
// and is not all zero (an all-zero match would prove nothing).
inline void expect_input_rows_match(const std::vector<float>& agent_row,
                                    const std::vector<float>& dec_row) {
  bool any_nonzero = false;
  for (size_t i = 0; i < agent_row.size(); ++i) {
    ASSERT_EQ(agent_row[i], dec_row[i]) << "input float " << i;
    any_nonzero = any_nonzero || agent_row[i] != 0.0f;
  }
  ASSERT_TRUE(any_nonzero);
}

// The pre-move row a move-set agent (MsetSimAgent, UltimateBotAgent) encodes
// with encode_board_row must equal the decoder's pre-move row for the
// three-turn game. `Service` is the stub model the agent is built with.
template <typename AgentT, typename Service>
void check_pre_move_row_matches_decoder(std::array<int, 2> initial_scores) {
  Dictionary dict = opening_dict();
  const std::vector<float> dec_row =
    decode_three_turn_row(dict, initial_scores, /*post_move=*/false);

  typename AgentT::Params p;
  p.name = "Agent";
  p.dict = &dict;
  AgentT agent(p, std::make_unique<Service>());
  const std::array<Move, 3> moves = three_turn_moves();
  agent.begin_game({initial_scores});
  agent.observe_move(moves[0]);
  agent.observe_move(moves[1]);

  // The agent takes board and scores from its own replay of observed moves;
  // only the rack comes from the request, so the rest of it is arbitrary.
  const Rack no_leave;
  const Board board;
  const MoveRequest req{board,           dict,           three_turn_mover_rack(), no_leave,
                        /*my_score=*/10,
                        /*opp_score=*/5, /*bag_size=*/50};
  std::vector<float> agent_row(size_t(input_floats(InputEncodingSpec{&dict})), 0.0f);
  agent.encode_board_row(req, agent_row.data());
  expect_input_rows_match(agent_row, dec_row);
}

// Candidate `i`'s features in `got` are what encode_move, the encoder the
// training rows go through, makes of `move` at `pre_diff`.
inline void expect_candidate_features_match(const move_set::MoveFeatureArrays& got, size_t i,
                                            const Move& move, int pre_diff) {
  int32_t letters[move_set::kMoveMaxPlaced];
  uint8_t blanks[move_set::kMoveMaxPlaced];
  int32_t squares[move_set::kMoveMaxPlaced];
  uint8_t tile_mask[move_set::kMoveMaxPlaced];
  float scalars[move_set::kMoveScalars];
  move_set::encode_move(move, pre_diff, letters, blanks, squares, tile_mask, scalars);

  const size_t tile_base = i * move_set::kMoveMaxPlaced;
  for (int t = 0; t < move_set::kMoveMaxPlaced; ++t) {
    EXPECT_EQ(got.letters[tile_base + t], letters[t]) << "move " << i << " tile " << t;
    EXPECT_EQ(got.blanks[tile_base + t], blanks[t]) << "move " << i << " tile " << t;
    EXPECT_EQ(got.squares[tile_base + t], squares[t]) << "move " << i << " tile " << t;
    EXPECT_EQ(got.tile_mask[tile_base + t], tile_mask[t]) << "move " << i << " tile " << t;
  }
  for (int s = 0; s < move_set::kMoveScalars; ++s) {
    EXPECT_EQ(got.scalars[i * move_set::kMoveScalars + s], scalars[s])
      << "move " << i << " scalar " << s;
  }
}

// expect_candidate_features_match over the whole candidate set.
inline void expect_move_features_match(const move_set::MoveFeatureArrays& got,
                                       const std::vector<Move>& candidates, int pre_diff) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    expect_candidate_features_match(got, i, candidates[i], pre_diff);
  }
}

}  // namespace scribblez::testing
