#pragma once

// Shared helpers for the model-driven simulating agents' tests (NeuralSimAgent,
// MsetSimAgent, UltimateBotAgent). They script a model over the same opening
// position and check that the sim set follows the model's ranking rather than
// static equity's, so they share the candidate space, the model ranking, and a
// way to script a model that favours chosen candidates.

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "nn/eval_service.h"
#include "sim/sim_runner.h"
#include "stub_eval_service.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace scribblez::testing {

// Dense enough in one rack's letters (CARTES, in these suites) to give many
// opening plays of differing scores.
inline Dictionary opening_dict() {
  return Dictionary::build_from_words(
    {"AE",    "AR",     "AT",    "ARC",    "ARCS", "ARE",   "ART",   "ARTS", "ATE",   "CAR",
     "CARE",  "CARES",  "CARET", "CARETS", "CARS", "CART",  "CARTS", "CAT",  "CATS",  "CATER",
     "CRATE", "CRATES", "EAR",   "EARS",   "EAT",  "EATS",  "ERA",   "ETA",  "RACE",  "RACES",
     "RAT",   "RATE",   "RATES", "RATS",   "SCAR", "SCARE", "SET",   "TARE", "TEA",   "TEAR",
     "TEARS", "TRACE",  "DON",   "DOT",    "DOTS", "NOD",   "SNORT", "TONE", "TONES", "STONED"});
}

// The candidates the agents rank: legal plays and exchanges in descending
// static-equity order, capped at `shortlist` (0 = uncapped).
inline std::vector<Move> shortlist_candidates(const MoveRequest& req, int shortlist) {
  return equity_top_k(req, shortlist == 0 ? std::numeric_limits<int>::max() : shortlist);
}

// Mirrors the agents' model ranking: candidate indices by descending scripted
// value, with ties kept in equity order as their stable_sort does.
inline std::vector<int> model_rank(const std::vector<ScriptedEval>& scripted,
                                   EvalObjective objective) {
  std::vector<int> idx(scripted.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    const ScriptedEval& ea = scripted[size_t(a)];
    const ScriptedEval& eb = scripted[size_t(b)];
    return objective_value(ea.wld.data(), ea.score_diff.data(), objective) >
           objective_value(eb.wld.data(), eb.score_diff.data(), objective);
  });
  return idx;
}

inline ScriptedEval wp(float win_prob) { return {{win_prob, 0.0f, 0.0f}, {}}; }

// Scripted rows for `n` candidates: the `favoured` equity-rank indices get
// descending high win probabilities, the rest a low one.
inline std::vector<ScriptedEval> script_favouring(size_t n, const std::vector<int>& favoured) {
  std::vector<ScriptedEval> scripted(n, wp(0.1f));
  float v = 0.9f;
  for (int idx : favoured) {
    scripted[size_t(idx)] = wp(v);
    v -= 0.05f;
  }
  return scripted;
}

}  // namespace scribblez::testing
