# Scribblez documentation map

## Start here

- **[design.md](design.md)** — the north-star design: why existing engines are
  beatable (context-blind leave evaluation, naive rack inference), and the
  target architecture — public belief system, unified Q/V network, GADDAG +
  Monte Carlo search.
- **[roadmap.md](roadmap.md)** — the variant we develop in (face-up leaves),
  what is built (the position evaluation model, self-play diversification, the
  evaluation machinery), and the plan from here:
  the move set evaluation model, rack inference, sim scheduling, the
  rollout-policy ladder (value truncation, self-model plies, endgame solver),
  and the experiments that gate each track.

## The system as built

- **[architecture.md](architecture.md)** — how a self-play game becomes a
  training row: the component chain, the `.slog` format, the
  replay-reconstruction invariant, random openings.
- **[generational_training.md](generational_training.md)** — the generate→train
  lifecycle (rows-clock, sliding window, reuse-driven epochs, live controls),
  plus the forward-looking game-pool producer, resource-contention manager, and
  distributed-worker design it grows into.
- **[endgame_bench_results.md](endgame_bench_results.md)** — the endgame
  solver's measured cost/strength curve vs its node budget (methodology, the
  seat-mirrored head-to-head protocol, the shipped default of 400).
- **[react_dashboard.md](react_dashboard.md)** — the training dashboard: React
  shell + Python data API, embedded Bokeh metric figures, and the interactive
  lane-analysis and Positions tabs.
- **[cloud_compute.md](cloud_compute.md)** — distributed data generation on
  rented cloud CPUs (Runpod): the stable dependency-only worker image, per-arch
  code bundles through R2, the fleet CLI, and results syncing back to the
  local mount for unchanged analysis.
- **[master_dashboard.md](master_dashboard.md)** — the React dashboard as the
  single entrypoint for all work: the workload registry, the job-control
  plane (launch/stop local and cloud workers from the browser), and
  workload-specific analysis tabs.
- **[position_eval_workload.md](position_eval_workload.md)** — the training
  workloads on the master dashboard: the workload-spec contract (roles, params,
  stats, tabs), distributed self-play generation via staging + controller-side
  ingest, and the trainer as a singleton consumer worker.

## Design proposals
- **[sim_residual_feedback.md](sim_residual_feedback.md)** — feeding Monte-Carlo
  rollout evidence back into the value models for evidence-conditioned
  re-evaluation, and picking the next candidate to sim via a learned
  proves-best probability. Steps 1–3 of its implementation roadmap are done
  (the kill-test passed).
- **[sim_obs_experiment_results.md](sim_obs_experiment_results.md)** — the
  kill-test's numbers, controls, and conclusions.
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
