#pragma once

#include "game/board.h"
#include "lexicon/dictionary.h"
#include "training/footprint.h"

namespace scribblez {

// The four placement heads, in PositionEvaluationSpec::AuxOutputs / .mset plane
// order: opp_next, self_next, opp_win, self_win.
inline constexpr int kPlacementHeads = 4;

// Collapse the four placement heads' raw footprint logits at one post-move state
// into the four per-cell occupancy marginals the .mset teacher target stores --
// the PR1 bridge that lets the categorical footprint heads keep feeding the
// per-cell (15,15) student/distillation stack unchanged.
//
// `raw` is kPlacementHeads x kFootprintClasses (each head's raw logits, the
// service's undecoded aux output); `out` is kPlacementHeads x
// (kFootprintSide*kFootprintSide). Per head: drive illegal footprints to zero
// with the legality mask (opp heads exact-ish from `board`, self heads
// opp-move-invariant; see training/footprint_mask.h), softmax over the legal
// classes, then scatter each footprint's probability onto the board cells it
// covers (footprint_cells). So out[h][cell] is Pr[the next move covers cell] for
// a plays head and Pr[covers cell AND that seat wins] for a win head -- the same
// per-cell marginal the old Bernoulli heads emitted, now derived from the
// footprint distribution.
//
// `board` gets its movegen caches bound from `dict` on demand. `flip` is the
// frame the logits were produced in (the generator encodes unflipped); the
// output planes are in that same frame.
void collapse_footprint_planes(const Board& board, const Dictionary& dict, bool flip,
                               const float* raw, float* out);

}  // namespace scribblez
