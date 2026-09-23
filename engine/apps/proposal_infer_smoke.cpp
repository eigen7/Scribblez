// proposal_infer_smoke: exercises the move proposal model's two-graph runtime
// (cache graph, then step graph; docs/roadmap.md item 3) on a real checkpoint,
// with no game logic. It loads a cache/step ONNX pair as one shared
// MoveProposalNets, opens `sessions` sessions over it, and runs each through
// encode (a synthetic candidate set) and condition (a synthetic evidence set).
// It prints the first few candidates' plain and conditioned predictions, plus
// what the setup costs:
//   - device memory taken by the loaded pair;
//   - host memory held by the sessions' retained caches, both as the sum of
//     their vectors and as the resident-set growth.
// Run it with sessions=12 and a deployment-sized num_moves to see what a
// 12-thread match costs in memory.
//
//   proposal_infer_smoke cache.onnx step.onnx [num_moves=12] [num_evidence=3]
//                        [FP32|BF16|FP16, default FP32] [sessions=1] [max_rows]
//
// max_rows bounds the cache graph's batch only. The moves and observations are
// synthetic; their encoders have their own tests.

#include "agent/move_proposal_nets.h"
#include "agent/move_proposal_session.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "nn/cuda_util.h"
#include "nn/trt_util.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"
#include "util/misc.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using scribblez::Move;
using scribblez::SimObservation;
using scribblez::agent::EvidenceSet;
using scribblez::agent::MoveProposalNets;
using scribblez::agent::MoveProposalPredictions;
using scribblez::agent::MoveProposalSession;

// The same synthetic candidate set as mset_infer_smoke's, so the two tools'
// outputs stay comparable.
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

// A plausible rollout observation, varied slightly by `j`.
SimObservation synthetic_observation(int j) {
  SimObservation obs;
  obs.n = 40 + j;
  obs.wins = 20 + j;
  obs.draws = 5;
  obs.losses = obs.n - obs.wins - obs.draws;
  obs.delta_sum = 12.0 * obs.n;
  obs.delta_sq_sum = (12.0 * 12.0 + 25.0) * obs.n;
  for (int cls = 0; cls < SimObservation::kClasses; ++cls) {
    obs.opp_next_count[cls] = uint16_t(cls % 7);
    obs.self_next_count[cls] = uint16_t(cls % 3);
  }
  return obs;
}

// Evidence on the first `num_evidence` candidates, each a horizontal 2-tile
// play. The parity test covers scattered candidate indices.
EvidenceSet synthetic_evidence(int num_evidence) {
  EvidenceSet evidence;
  for (int j = 0; j < num_evidence; ++j) {
    const scribblez::Glyph g0 = scribblez::Glyph::of(scribblez::Tile::from_char('A'));
    const scribblez::Glyph g1 = scribblez::Glyph::of(scribblez::Tile::from_char('B'));
    const scribblez::Glyph played[] = {g0, g1};
    evidence.add(Move::play(/*horizontal=*/true, /*start=*/7,
                            /*square_mask=*/uint16_t((1 << 7) | (1 << 8)),
                            /*score=*/uint16_t(20 + j), played, /*num_played=*/2),
                 synthetic_observation(j), j);
  }
  return evidence;
}

// This process's resident set, in bytes, off /proc/self/statm.
size_t resident_bytes() {
  std::ifstream statm("/proc/self/statm");
  size_t pages = 0, resident = 0;
  statm >> pages >> resident;
  return resident * size_t(sysconf(_SC_PAGESIZE));
}

double mib(size_t bytes) { return double(bytes) / (1024.0 * 1024.0); }

// The bytes one session's retained cache holds.
size_t cache_bytes(const scribblez::agent::MoveProposalCache& c) {
  return sizeof(float) * (c.move_enc.size() + c.wld.size() + c.score_diff.size() + c.planes.size() +
                          c.board.size() + c.g.size());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <cache.onnx> <step.onnx> [num_moves] [num_evidence] [FP32|BF16|FP16] [sessions] "
                 "[max_rows]\n";
    return 1;
  }

  const int num_moves = std::max(argc > 3 ? std::atoi(argv[3]) : 12, 1);
  const int num_evidence = std::clamp(argc > 4 ? std::atoi(argv[4]) : 3, 0, num_moves);
  const std::string precision = argc > 5 ? argv[5] : "FP32";
  const int num_sessions = std::max(argc > 6 ? std::atoi(argv[6]) : 1, 1);

  try {
    MoveProposalNets::Params params;
    params.cache_onnx_path = argv[1];
    params.step_onnx_path = argv[2];
    params.precision = scribblez::nn::parse_precision(precision);
    // The step graph keeps its own row bound (step_max_rows).
    if (argc > 7) params.max_rows = std::max(std::atoi(argv[7]), 1);

    const size_t device_before = scribblez::nn::device_memory_used();
    std::shared_ptr<MoveProposalNets> nets = MoveProposalNets::create(params);
    std::cout << "loaded pair (C=" << nets->channels() << ", E=" << nets->max_evidence()
              << ", trained width " << nets->trained_max_evidence()
              << ", max_rows=" << nets->max_rows() << "/" << nets->step_max_rows()
              << "): device memory +" << mib(scribblez::nn::device_memory_used() - device_before)
              << " MiB\n";

    // An all-zero board row at the model's own width; only the candidates vary.
    const size_t row_floats =
      size_t(nets->spatial_planes()) * scribblez::kBoardCells + nets->scalar_floats();
    const std::vector<float> board(row_floats, 0.0f);
    const scribblez::move_set::MoveFeatureArrays moves = synthetic_candidates(num_moves);
    const EvidenceSet evidence = synthetic_evidence(num_evidence);

    // Warm the pair on a throwaway session first, so its first-use costs
    // (first-touched pinned buffers, lazily loaded CUDA modules) are not
    // charged to the measured sessions.
    {
      MoveProposalSession warmup(nets);
      warmup.encode(board.data(), moves);
      warmup.condition(evidence);
    }
    const size_t host_before = resident_bytes();
    std::vector<std::unique_ptr<MoveProposalSession>> sessions;
    const MoveProposalPredictions* plain = nullptr;
    const MoveProposalPredictions* conditioned = nullptr;
    size_t sessions_bytes = 0;
    for (int s = 0; s < num_sessions; ++s) {
      sessions.push_back(std::make_unique<MoveProposalSession>(nets));
      const MoveProposalPredictions& p = sessions.back()->encode(board.data(), moves);
      const MoveProposalPredictions& c = sessions.back()->condition(evidence);
      if (s == 0) {
        plain = &p;
        conditioned = &c;
      }
      sessions_bytes += cache_bytes(sessions.back()->cache());
    }
    std::cout << num_sessions << " session(s) x " << num_moves << " candidates: retained caches "
              << mib(sessions_bytes) << " MiB (resident set +"
              << mib(resident_bytes() - host_before) << " MiB)\n";

    for (int m = 0; m < std::min(num_moves, 8); ++m) {
      const float* pw = plain->wld.data() + size_t(m) * 3;
      const float* cw = conditioned->wld.data() + size_t(m) * 3;
      std::cout << "move " << m << ": plain win_prob=" << pw[0] + 0.5f * pw[1]
                << "  conditioned win_prob=" << cw[0] + 0.5f * cw[1]
                << "  gain=" << conditioned->gain[m] << "\n";
    }
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
