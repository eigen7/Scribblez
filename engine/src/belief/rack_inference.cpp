#include "belief/rack_inference.h"

#include "belief/leave_prior.h"
#include "util/assert.h"
#include "util/math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>

namespace scribblez::belief {

namespace {

Rack combine(const Rack& revealed, const Rack& hidden) {
  Rack rack = revealed;
  for (int i = 0; i < hidden.size(); ++i) rack.add(hidden.tiles()[i]);
  return rack;
}

// Folds scored explanations into a normalized posterior. Duplicates are summed,
// since several racks can keep the same tiles, and entries are ordered by leave
// so a repeated inference reproduces exactly.
RackPosterior build_posterior(const std::vector<ScoredLeave>& scored, bool exhaustive) {
  double max_log = -std::numeric_limits<double>::infinity();
  for (const ScoredLeave& s : scored) max_log = std::max(max_log, s.log_weight);
  if (!std::isfinite(max_log)) return {};

  std::unordered_map<Rack, double> merged;
  for (const ScoredLeave& s : scored) merged[s.leave] += std::exp(s.log_weight - max_log);
  double total = 0.0;
  for (const auto& [leave, weight] : merged) total += weight;

  std::vector<RackPosterior::Entry> entries;
  entries.reserve(merged.size());
  for (const auto& [leave, weight] : merged) entries.push_back({leave, weight / total});
  std::sort(
    entries.begin(), entries.end(),
    [](const RackPosterior::Entry& a, const RackPosterior::Entry& b) { return a.leave < b.leave; });
  return RackPosterior(std::move(entries), exhaustive);
}

// Appends the readings of one hypothesis, adding `log_prior` to each reading's
// log likelihood.
void score_hypothesis(const EquityLikelihood& likelihood, const Rack& hidden, double log_prior,
                      std::vector<ScoredLeave>* out) {
  const size_t begin = out->size();
  likelihood.explain(combine(likelihood.revealed(), hidden), out);
  for (size_t i = begin; i < out->size(); ++i) (*out)[i].log_weight += log_prior;
}

std::vector<ScoredLeave> score_enumerated(const EquityLikelihood& likelihood,
                                          const TileCounts& pool, int hidden) {
  std::vector<ScoredLeave> scored;
  for (const ScoredLeave& h : enumerate_leaves(pool, hidden))
    score_hypothesis(likelihood, h.leave, h.log_weight, &scored);
  return scored;
}

std::vector<ScoredLeave> score_sampled(const EquityLikelihood& likelihood, const TileCounts& pool,
                                       int hidden, int samples, uint64_t seed) {
  std::vector<ScoredLeave> scored;
  std::mt19937_64 rng(util::splitmix64(seed));
  // Draws come from the prior, so each is weighted by likelihood alone and the
  // prior enters through multiplicity. That is why repeat draws are scored
  // again rather than deduplicated.
  for (int i = 0; i < samples; ++i)
    score_hypothesis(likelihood, draw_leave(pool, hidden, rng), 0.0, &scored);
  return scored;
}

}  // namespace

RackPosterior::RackPosterior(std::vector<Entry> entries, bool exhaustive)
    : entries_(std::move(entries)), exhaustive_(exhaustive) {}

const Rack& RackPosterior::sample(double u) const {
  RELEASE_ASSERT(!entries_.empty());
  double cumulative = 0.0;
  for (const Entry& e : entries_) {
    cumulative += e.weight;
    if (u < cumulative) return e.leave;
  }
  return entries_.back().leave;  // only reachable when u rounds past the total
}

RackInferrer::RackInferrer(const Dictionary& dict, const Params& params)
    : dict_(dict), params_(params) {}

RackPosterior RackInferrer::infer(const OppMoveObservation& obs, uint64_t seed) const {
  if (obs.move.type() == MoveType::PASS) return {};
  // With a non-empty bag their rack was full. With an empty bag the whole
  // unseen pool is their rack, and there is nothing to infer.
  const int bag_size = obs.pool.size() - RACK_SIZE;
  if (bag_size <= 0) return {};

  const EquityLikelihood likelihood(obs.board_before, dict_, bag_size, obs.move,
                                    params_.temperature);
  const int hidden = likelihood.hidden_tiles();
  if (hidden == 0) return {};  // a bingo empties the rack

  TileCounts pool = obs.pool;
  for (int i = 0; i < likelihood.revealed().size(); ++i) {
    const bool drawable = pool.remove(likelihood.revealed().tiles()[i]);
    RELEASE_ASSERT(drawable, "the observation contradicts the pool");
  }
  if (pool.size() < hidden) return {};

  if (count_multisets(pool, hidden, params_.max_enumerated) <= params_.max_enumerated)
    return build_posterior(score_enumerated(likelihood, pool, hidden), /*exhaustive=*/true);
  return build_posterior(score_sampled(likelihood, pool, hidden, params_.samples, seed),
                         /*exhaustive=*/false);
}

}  // namespace scribblez::belief
