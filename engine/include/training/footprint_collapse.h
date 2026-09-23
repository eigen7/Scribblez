#pragma once

#include "game/board.h"
#include "lexicon/dictionary.h"
#include "training/footprint.h"

#include <cstdint>

namespace scribblez {

// Turns the placement heads' raw footprint logits into distributions a consumer
// can read: masked per class, or collapsed onto board cells. The model emits raw
// logits, so every consumer masks and softmaxes itself; these are the engine's
// shared implementations.
//
// All three functions share one masking regime. Each head gets its side's
// legality mask (training/footprint_mask.h): the opp heads from the ply on
// `board`, the self heads from the ply after it. `available_counts` is the
// opponent's 27-count tile pool (the unseen tiles). It gates the opp heads
// directly, so a footprint whose hooks no available tile can fill is masked and
// its mass renormalizes onto the rest. It gates the self heads only through the
// opponent's ply, whose reach seeds theirs. nullptr disables availability,
// leaving board legality only. `raw`, where taken, is kPlacementHeads x
// kFootprintClasses undecoded logits from an encode of this same board, so the
// outputs share its frame. `board` gets its move-generation caches built from
// `dict` on demand.

// Collapses each head onto the board: out[h][cell] is the probability that the
// next move covers `cell` (plays heads), or covers it AND that seat goes on to
// win (win heads). This drives the dashboard's occupancy overlay. Self
// footprints are decoded on `board` too, i.e. as if the opponent passed.
// `out` is kPlacementHeads x kFootprintCells.
void collapse_footprint_planes(const Board& board, const Dictionary& dict,
                               const uint8_t* available_counts, const float* raw, float* out);

// Each head's masked footprint distribution, per class rather than collapsed
// onto cells: illegal footprints at zero. With null `available_counts` this is
// the .mset distillation target the student trains against. `out` is
// kPlacementHeads x kFootprintClasses.
void masked_placement_distributions(const Board& board, const Dictionary& dict,
                                    const uint8_t* available_counts, const float* raw, float* out);

// Each head's legality per cell: out[h][cell] is 1.0 iff some footprint the
// head's mask keeps covers `cell`, else 0.0. `out` is kPlacementHeads x
// kFootprintCells.
void collapse_footprint_legal_cells(const Board& board, const Dictionary& dict,
                                    const uint8_t* available_counts, float* out);

}  // namespace scribblez
