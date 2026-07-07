# Scribblez documentation map

## Start here

- **[design.md](design.md)** — the north-star design: why existing engines are
  beatable (context-blind leave evaluation, naive rack inference), and the
  target architecture — public belief system, unified Q/V network, GADDAG +
  Monte Carlo search.
- **[roadmap.md](roadmap.md)** — the phased plan from the post-move value model
  (M_post, built) through self-play diversification and evaluation machinery to
  the pre-move model (M_pre), with per-phase status.
- **[roadmap2.md](roadmap2.md)** — the forward plan from here: M_pre, rack
  inference, covariance-guided sim scheduling, the rollout-policy ladder
  (value truncation, self-model plies, endgame solver), and the experiments
  that gate each track.

## The system as built

- **[architecture.md](architecture.md)** — how a self-play game becomes a
  training row: the component chain, the `.slog` format, the
  replay-reconstruction invariant, random openings.
- **[generational_training.md](generational_training.md)** — the generate→train
  lifecycle (rows-clock, sliding window, reuse-driven epochs, live controls),
  plus the forward-looking game-pool producer, resource-contention manager, and
  distributed-worker design it grows into.
- **[react_dashboard.md](react_dashboard.md)** — the training dashboard: React
  shell + Python data API, embedded Bokeh metric figures, and the interactive
  lane-analysis and Positions tabs.

## Design proposals

- **[sim_residual_feedback.md](sim_residual_feedback.md)** — feeding Monte-Carlo
  rollout evidence back into the value models for evidence-conditioned
  re-evaluation, and covariance-guided candidate selection. Steps 1–2 of its
  implementation roadmap are built.
- **[lexical_features_for_value.md](lexical_features_for_value.md)** — giving
  the value models lexical foresight through engineered GADDAG-computed input
  features (the contingent-draw potential map, the cross-check delta) instead
  of network-internal lexical knowledge.

## The lexical-NN experiment track

Probe experiments asking whether a network can internalize or query the
lexicon. Their findings — a compiled-lexicon tool helps only when its structure
matches the task's shape, and a plain network cannot spell a word's interior —
are what justify the engineered-feature approach above.

- **[lexical_tools.md](lexical_tools.md)** — the catalog of compiled-lexicon
  modules (DAWG walks, anagram search, KV memory) and the registry interface.
- **[word_validity_experiments.md](word_validity_experiments.md)** — experiment
  1: classifying real words vs. statistically-matched phonies; the ordered DAWG
  walk wins.
- **[rack_best_experiments.md](rack_best_experiments.md)** — experiment 2: the
  best word in an unordered rack; the anagram tool wins where the ordered walk
  fails, and generation is genuinely harder than discrimination.
- **[lexical_nn.md](lexical_nn.md)** — experiment 3: the max-move-per-lane
  board task (per-lane best-move union + score heads on a CNN + lane
  transformer).
