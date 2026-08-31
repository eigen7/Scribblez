#pragma once

#include "game/board.h"
#include "lexicon/dictionary.h"
#include "training/footprint.h"

#include <cstdint>

namespace scribblez {

// The four placement heads, in PositionEvaluationSpec::AuxOutputs / .mset plane
// order: opp_next, self_next, opp_win, self_win.
inline constexpr int kPlacementHeads = 4;

// Collapse the four placement heads' raw footprint logits at one post-move state
// into the four per-cell occupancy marginals the .mset teacher target stores --
// the bridge that lets the categorical footprint heads keep feeding the
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
// `available_counts` is the opponent's 27-count tile availability (the unseen
// pool; see opp_footprint_mask), applied to the two OPP heads so a footprint
// whose hooks no available tile can satisfy is masked out and its mass
// renormalizes onto viable footprints. `available_counts == nullptr` disables
// availability (board legality only) -- what the .mset teacher path passes, since
// that collapse consumer is being retired and does not fund the extra plumbing.
// The self heads never take availability (two plies out, post-redraw).
//
// `board` gets its movegen caches bound from `dict` on demand. `flip` is the
// frame the logits were produced in (the generator encodes unflipped); the
// output planes are in that same frame.
void collapse_footprint_planes(const Board& board, const Dictionary& dict,
                               const uint8_t* available_counts, bool flip, const float* raw,
                               float* out);

// The four heads' MASKED footprint distributions -- the same mask + masked-softmax
// collapse_footprint_planes applies (identical `available_counts`/`flip`
// semantics), but written per class instead of scattered onto cells. `raw` is
// kPlacementHeads x kFootprintClasses; `out` is kPlacementHeads x
// kFootprintClasses, each head a distribution over the 2927 classes with illegal
// footprints at zero. This is the exact target the student distills against,
// exposed so its per-footprint sparsity/fidelity can be measured (the per-cell
// collapse hides the footprint distribution).
void masked_placement_distributions(const Board& board, const Dictionary& dict,
                                    const uint8_t* available_counts, bool flip, const float* raw,
                                    float* out);

}  // namespace scribblez
