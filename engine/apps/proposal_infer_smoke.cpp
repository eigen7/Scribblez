// Standalone sanity-check for the move proposal model's two-graph evidence-path
// runtime (roadmap item 3) -- mset_infer_smoke's counterpart for the split
// cache/step graphs. Loads a cache/step ONNX pair, runs the cache over a
// synthetic candidate set, conditions on one synthetic evidence set, and prints
// the plain and conditioned predictions.
//
// Usage:
//   proposal_infer_smoke <cache.onnx> <step.onnx> [num_moves] [num_evidence] [FP32|FP16]
//
// A successful run confirms the whole path end to end on a real checkpoint:
// both ONNX parses and the cache/step compatibility check, two engine builds +
// plan caches, the board/g/move_enc host handoff, the evidence staging
// (agent/evidence_staging.h), and the decode. FP32 is the item-3 serving
// precision; FP16 is available as a spot check, with the fusion-graph caveat in
// docs/fp16_safe_serving.md. The moves and observations are synthetic -- what a
// Move/SimObservation encodes to has its own tests; this tool is about the
// engine path.

#include "agent/move_proposal_runtime.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "nn/trt_util.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"
#include "util/misc.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using scribblez::Move;
using scribblez::SimObservation;

// A candidate set spanning the shapes the model sees: plays of 1..7 tiles, and
// every fifth candidate an exchange (tiles, no squares). Mirrors
// mset_infer_smoke's synthetic set so the two smoke tools stay comparable.
scribblez::move_set::MoveFeatureArrays synthetic_candidates(int num_moves) {
  using namespace scribblez::move_set;
  MoveFeatureArrays moves;
  moves.count = num_moves;
  moves.letters.assign(size_t(num_moves) * kMoveMaxPlaced, 0);
  moves.blanks.assign(size_t(num_moves) * kMoveMaxPlaced, 0);
  moves.squares.assign(size_t(num_moves) * kMoveMaxPlaced, 0);
  moves.tile_mask.assign(size_t(num_moves) * kMoveMaxPlaced, 0);
  moves.scalars.assign(size_t(num_moves) * kMoveScalars, 0.0f);

  for (int m = 0; m < num_moves; ++m) {
    const bool is_play = m % 5 != 0;
    const int tiles = m % kMoveMaxPlaced + 1;
    for (int t = 0; t < tiles; ++t) {
      const size_t slot = size_t(m) * kMoveMaxPlaced + t;
      moves.letters[slot] = (m + t) % 26 + 1;
      moves.tile_mask[slot] = 1;
      if (is_play) moves.squares[slot] = (m * kMoveMaxPlaced + t) % kMoveCells;
    }
    float* scalars = moves.scalars.data() + size_t(m) * kMoveScalars;
    scalars[0] = float(m - num_moves / 2) / 100.0f;
    scalars[1] = float(tiles) / kMoveMaxPlaced;
    scalars[2] = is_play ? 1.0f : 0.0f;
  }
  return moves;
}

// One synthetic evidence candidate: a horizontal 2-tile play with a plausible
// rollout observation, tied to scored candidate `scored_index`.
SimObservation synthetic_observation(int j) {
  SimObservation obs;
  obs.n = 40 + j;
  obs.wins = 20 + j;
  obs.draws = 5;
  obs.losses = obs.n - obs.wins - obs.draws;
  obs.delta_sum = 12.0 * obs.n;
  obs.delta_sq_sum = (12.0 * 12.0 + 25.0) * obs.n;
  for (int cell = 0; cell < SimObservation::kCells; ++cell) {
    obs.opp_next_count[cell] = uint16_t(cell % 7);
    obs.self_next_count[cell] = uint16_t(cell % 3);
  }
  return obs;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <cache.onnx> <step.onnx> [num_moves] [num_evidence] [FP32|FP16]\n";
    return 1;
  }

  const int num_moves = std::max(argc > 3 ? std::atoi(argv[3]) : 12, 1);
  const int num_evidence = std::clamp(argc > 4 ? std::atoi(argv[4]) : 3, 0, num_moves);
  const std::string precision = argc > 5 ? argv[5] : "FP32";

  try {
    scribblez::agent::MoveProposalRuntime::Params params;
    params.cache_onnx_path = argv[1];
    params.step_onnx_path = argv[2];
    params.precision = scribblez::nn::parse_precision(precision);
    scribblez::agent::MoveProposalRuntime runtime(params);
    runtime.load();

    // An all-zero board row at the model's own width; the candidates are what
    // this tool varies.
    const size_t row_floats =
      size_t(runtime.spatial_planes()) * scribblez::kBoardCells + runtime.scalar_floats();
    const std::vector<float> board(row_floats, 0.0f);
    const scribblez::move_set::MoveFeatureArrays moves = synthetic_candidates(num_moves);

    const auto& plain = runtime.encode(board.data(), moves);
    std::cout << "encoded " << plain.num_moves << " candidates (C=" << runtime.channels()
              << ", E=" << runtime.max_evidence() << ")\n";

    // Evidence over the first `num_evidence` candidates (scattered indices would
    // do too; the parity test covers those).
    std::vector<Move> ev_moves;
    std::vector<SimObservation> ev_obs;
    std::vector<int> scored_indices;
    for (int j = 0; j < num_evidence; ++j) {
      const scribblez::Glyph g0 = scribblez::Glyph::of(scribblez::Tile::from_char('A'));
      const scribblez::Glyph g1 = scribblez::Glyph::of(scribblez::Tile::from_char('B'));
      const scribblez::Glyph played[] = {g0, g1};
      ev_moves.push_back(Move::play(/*horizontal=*/true, /*start=*/7,
                                    /*square_mask=*/uint16_t((1 << 7) | (1 << 8)),
                                    /*score=*/uint16_t(20 + j), played, /*num_played=*/2));
      ev_obs.push_back(synthetic_observation(j));
      scored_indices.push_back(j);
    }
    const auto& conditioned = runtime.condition(ev_moves, ev_obs, scored_indices);

    for (int m = 0; m < std::min(num_moves, 8); ++m) {
      const float* pw = plain.wld.data() + size_t(m) * 3;
      const float* cw = conditioned.wld.data() + size_t(m) * 3;
      std::cout << "move " << m << ": plain win_prob=" << pw[0] + 0.5f * pw[1]
                << "  conditioned win_prob=" << cw[0] + 0.5f * cw[1]
                << "  gain=" << conditioned.gain[m] << "\n";
    }
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
