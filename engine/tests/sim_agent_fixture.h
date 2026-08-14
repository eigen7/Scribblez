#pragma once

// Shared fixtures for the model-driven simulating agents' unit tests
// (NeuralSimAgent, MsetSimAgent), which script a model over the same opening
// position and check the same thing of it: that the sim set is the MODEL's top
// K rather than static equity's. Both suites need the candidate space the agent
// ranks, the ranking the agent's stable_sort produces, and a way to script a
// model that favours chosen candidates -- identically, since the two agents are
// meant to make the same decision from the same information.

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

inline Rack rack_from(const std::string& s) {
  Rack r;
  for (char c : s) r.add(c == '?' ? BLANK : Tile::from_char(c));
  return r;
}

// A word list dense enough in one rack's letters to yield many opening plays of
// differing scores -- the candidate sets these suites rank and prune.
inline Dictionary opening_dict() {
  return Dictionary::build_from_words(
    {"AE",    "AR",     "AT",    "ARC",    "ARCS", "ARE",   "ART",   "ARTS", "ATE",   "CAR",
     "CARE",  "CARES",  "CARET", "CARETS", "CARS", "CART",  "CARTS", "CAT",  "CATS",  "CATER",
     "CRATE", "CRATES", "EAR",   "EARS",   "EAT",  "EATS",  "ERA",   "ETA",  "RACE",  "RACES",
     "RAT",   "RATE",   "RATES", "RATS",   "SCAR", "SCARE", "SET",   "TARE", "TEA",   "TEAR",
     "TEARS", "TRACE",  "DON",   "DOT",    "DOTS", "NOD",   "SNORT", "TONE", "TONES", "STONED"});
}

// The candidate space these agents rank: every legal play and exchange in
// descending static-equity order, capped the way they cap it (0 = uncapped).
inline std::vector<Move> shortlist_candidates(const MoveRequest& req, int shortlist) {
  return equity_top_k(req, shortlist == 0 ? std::numeric_limits<int>::max() : shortlist);
}

// The agents' own model ranking: candidate indices in descending scripted-value
// order, ties keeping equity order (their stable_sort).
inline std::vector<int> model_rank(const std::vector<ScriptedEval>& scripted,
                                   EvalObjective objective) {
  std::vector<int> idx(scripted.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    const ScriptedEval& ea = scripted[static_cast<size_t>(a)];
    const ScriptedEval& eb = scripted[static_cast<size_t>(b)];
    return objective_value(ea.wld.data(), ea.score_diff.data(), objective) >
           objective_value(eb.wld.data(), eb.score_diff.data(), objective);
  });
  return idx;
}

inline ScriptedEval wp(float win_prob) { return {{win_prob, 0.0f, 0.0f}, {}}; }

// Scripted rows for a candidate set of `n`: `favoured` (indices into the
// equity ranking) get descending high win probabilities, the rest a low one --
// so the model's preference is separated from static equity's by construction.
inline std::vector<ScriptedEval> script_favouring(size_t n, const std::vector<int>& favoured) {
  std::vector<ScriptedEval> scripted(n, wp(0.1f));
  float v = 0.9f;
  for (int idx : favoured) {
    scripted[static_cast<size_t>(idx)] = wp(v);
    v -= 0.05f;
  }
  return scripted;
}

}  // namespace scribblez::testing
