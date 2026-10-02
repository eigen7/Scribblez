#pragma once

// Candidate selection for SupremeBot M1a's held-out transfer test
// (docs/plans/supreme_bot_m1a.md): HastyBot-equity strata, plus coupled pairs,
// moves that share one factor and differ in another, so that what should
// transfer between them is known in advance.

#include "game/move.h"
#include "sim/slog_position_simmer.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

// How many candidates each stratum gets. Ranks index the full static-equity
// ranking, best first.
struct TransferRecipe {
  int top = 6;        // plays from ranks [0, top_ranks)
  int middle = 5;     // plays from ranks [top_ranks, middle_ranks)
  int exchanges = 3;  // distinct keep-sets, whatever their rank
  int low = 2;        // plays from ranks [middle_ranks, end)
  int top_ranks = 10;
  int middle_ranks = 100;

  int size() const { return top + middle + exchanges + low; }
};

enum class Stratum : uint8_t { kTop, kMiddle, kExchange, kLow };

// The factor a coupled pair shares.
enum class Coupling : uint8_t {
  kNone,
  kPlayExchange,     // a play and the exchange of exactly its tiles: the same leave
  kSameTiles,        // the same tiles at two placements: the same leave, another region
  kSameLaneOneTile,  // one lane, overlapping, tiles differing by one: a shared region
};
inline constexpr int kCouplingKinds = 3;  // the kinds after kNone

struct CoupledPair {
  int a = 0;  // indices into the moves the pair was found among
  int b = 0;
  Coupling kind = Coupling::kNone;
};

Stratum stratum_of(const Move& m, int equity_rank, const TransferRecipe& recipe);

// The factor `a` and `b` share, or kNone.
Coupling coupling_of(const Move& a, const Move& b);

// Every coupled pair among `moves`.
std::vector<CoupledPair> find_couplings(const std::vector<Move>& moves);

// The coupled pairs a position offers, by kind (index = kind - 1): each
// anchored on a play in ranks [0, recipe.middle_ranks) of `ranked`, its
// partner of any rank, as rank indices.
using AnchoredCouplings = std::array<std::vector<std::array<int, 2>>, kCouplingKinds>;
AnchoredCouplings anchored_couplings(const std::vector<Move>& ranked, const TransferRecipe& recipe);

// Selects up to recipe.size() candidates from `ranked`: first one coupled
// pair of each kind the position offers, anchored in the top and middle ranks,
// then each stratum filled to its quota, a short stratum's slots going to
// random remaining moves. Candidates come back in rank order.
SimCandidateSelector transfer_selector(const TransferRecipe& recipe);

}  // namespace scribblez
