// Standalone sanity-check for the move proposal model's two-graph evidence-path
// runtime (roadmap item 3) -- mset_infer_smoke's counterpart for the split
// cache/step graphs. Loads a cache/step ONNX pair as the shared
// MoveProposalNets, opens `sessions` sessions over it, runs each over a
// synthetic candidate set, conditions each on one synthetic evidence set, and
// prints the plain and conditioned predictions plus what the whole thing cost:
// the device memory the loaded pair took, and the host memory the sessions'
// retained caches took -- both as the exact sum of their vectors and as the
// resident-set delta, measured after the pair has been warmed on a throwaway
// session so its first-use footprint (first-touched pinned buffers, lazily
// loaded CUDA modules) is not charged to the sessions.
//
// Usage:
//   proposal_infer_smoke <cache.onnx> <step.onnx> [num_moves] [num_evidence]
//                        [FP32|FP16] [sessions] [max_rows]
//
// A successful run confirms the whole path end to end on a real checkpoint:
// both ONNX parses and the cache/step compatibility check, two engine builds +
// plan caches, the board/g/move_enc host handoff, the evidence staging
// (agent/evidence_staging.h), and the decode. FP32 is the item-3 serving
// precision; FP16 is available as a spot check, with the fusion-graph caveat in
// docs/fp16_safe_serving.md. The moves and observations are synthetic -- what a
// Move/SimObservation encodes to has its own tests; this tool is about the
// engine path. The memory readouts are the numbers roadmap item 6's runtime
// restructure was sized by: run at sessions=12 and the deployment num_moves to
// see what a 12-thread match costs.

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
  for (int cls = 0; cls < SimObservation::kClasses; ++cls) {
    obs.opp_next_count[cls] = uint16_t(cls % 7);
    obs.self_next_count[cls] = uint16_t(cls % 3);
  }
  return obs;
}

// Evidence over the first `num_evidence` candidates (scattered indices would
// do too; the parity test covers those).
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
              << " <cache.onnx> <step.onnx> [num_moves] [num_evidence] [FP32|FP16] [sessions] "
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
    // The cache graph's row bound only; the step graph keeps its own default
    // (step_max_rows), which the memory question is not about.
    if (argc > 7) params.max_rows = std::max(std::atoi(argv[7]), 1);

    const size_t device_before = scribblez::nn::device_memory_used();
    std::shared_ptr<MoveProposalNets> nets = MoveProposalNets::create(params);
    std::cout << "loaded pair (C=" << nets->channels() << ", E=" << nets->max_evidence()
              << ", max_rows=" << nets->max_rows() << "/" << nets->step_max_rows()
              << "): device memory +" << mib(scribblez::nn::device_memory_used() - device_before)
              << " MiB\n";

    // An all-zero board row at the model's own width; the candidates are what
    // this tool varies.
    const size_t row_floats =
      size_t(nets->spatial_planes()) * scribblez::kBoardCells + nets->scalar_floats();
    const std::vector<float> board(row_floats, 0.0f);
    const scribblez::move_set::MoveFeatureArrays moves = synthetic_candidates(num_moves);
    const EvidenceSet evidence = synthetic_evidence(num_evidence);

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
