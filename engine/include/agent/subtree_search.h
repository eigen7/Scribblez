#pragma once

// What DeepBot's subtrees share, whichever player searches in them
// (agent/bob_subtree.h, agent/alice_subtree.h): a player to move after the
// other's move chooses among its static-equity top moves by sims on samples
// of the other's leave, and the move it chose is then simmed from the
// position as the outer search dealt it.
//
// A subtree's jobs are built a chunk at a time (one of the root's candidates),
// as its labeler asks for them, so a turn never holds them all.

#include "belief/leave_params.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/leave_value_arrays.h"
#include "sim/hasty_rollout_task.h"
#include "sim/pool_labeler.h"
#include "sim/rollout_values.h"
#include "sim/sim_runner.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace scribblez {

class Dictionary;

// The searching player's side of a subtree.
struct SubtreeParams {
  int opp_racks = 20;        // the other player's leaves it samples, one rollout each
  int candidates = 8;        // its own, by static equity
  int late_candidates = 12;  // ... with late_bag or fewer tiles in the bag
  int late_bag = 40;
  bool race = true;  // stop its clearly beaten candidates between rounds
};

// The move a subtree chose, with what the diagnostics read.
struct SubtreeMove {
  Move move;
  int equity_rank = 0;  // among the candidates; 0 is HastyBot's move
  int candidates = 1;   // how many were simmed (1: no sims)
  double q = 1;         // the probability the sims gave that it is the best
};

// The leave inference model a subtree's beliefs come from; an empty `predict`
// draws the other player's whole rack uniformly.
struct BeliefModel {
  belief::LeaveParamsFn predict;
  const LeaveValueArrays* table = nullptr;
};

// The value of a candidate Alice did not search: it has none.
inline constexpr double kPruned = std::numeric_limits<double>::quiet_NaN();

// The index of the largest of `values`, skipping kPruned.
size_t best_candidate(const std::vector<double>& values);

// Seeds by rack index, one domain per use, so every candidate of Alice's meets
// rack i's samples and draws: her candidates are compared on paired samples.
uint64_t rack_seed(uint64_t seed, uint64_t domain, size_t rack);

// The other player to move after `applied` (a move applied from its mover's
// side), holding `rack`; the new mover is player 0.
SimPosition mover_position(const RolloutStart& applied, const Rack& rack);

// The static-equity top moves of `pos`'s mover, N by the bag it sees.
std::vector<Move> mover_candidates(const SimPosition& pos, const SubtreeParams& params,
                                   const Dictionary& dict);

// The candidate the labeler's sims of `moves` found most likely best.
SubtreeMove chosen_move(const std::vector<Move>& moves, const PoolPosition& labeled);

// The sims of the other player's `reply` after `applied`, which it answers
// holding `rack`: the mover of `applied` has its leave seated, and each sim
// draws its replenishment, returns the tiles of its exchange, and draws the
// replier's, all from the bag as the mover found it. That bag is the same
// whichever move it applied, so under one `seed` its moves' sims deal each
// player the same tiles, as far as the counts they draw allow.
PoolJob after_reply_job(const RolloutStart& applied, const Rack& rack, const Move& reply, int sims,
                        const Dictionary& dict, uint64_t seed);

// Hands a labeler jobs built a chunk at a time, remembering the order they
// went out in.
class ChunkedJobs {
 public:
  using Maker = std::function<std::vector<std::pair<size_t, PoolJob>>(size_t chunk)>;

  ChunkedJobs(size_t chunks, Maker make) : chunks_(chunks), make_(std::move(make)) {}

  std::optional<PoolJob> next();
  // The keys of the jobs handed out, in order.
  const std::vector<size_t>& keys() const { return keys_; }

 private:
  size_t chunks_;
  size_t chunk_ = 0;
  Maker make_;
  std::deque<std::pair<size_t, PoolJob>> buffered_;
  std::vector<size_t> keys_;
};

// Labels every job `jobs` hands out, calling done(key, record) in order.
void label_all(RolloutValues& values, const PoolLabeler::Params& params, ChunkedJobs& jobs,
               const std::function<void(size_t, const PoolPosition&)>& done);

}  // namespace scribblez
