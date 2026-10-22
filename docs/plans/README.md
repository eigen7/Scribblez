# Plans and design proposals

Each document here records a decision: the problem, the design chosen, the
alternatives it beat, and how it lands. Each opens with its status. The
system-as-built documents indexed in [../README.md](../README.md) describe
what exists; a plan says why it was built that way. A plan whose work has
landed keeps its rationale and review record even where the code has since
moved on. New plans go here, not in `docs/` proper.

## Modeling and training

- **[sim_residual_feedback.md](sim_residual_feedback.md)**: *built; on
  hiatus before training, superseded by supreme_bot.md.* Feeding Monte Carlo rollout evidence back into the value
  models, re-evaluating the move set conditioned on it, and choosing the next
  candidate to sim with a learned expected-gain ("proves-best") head. The
  kill-test passed; the fusion stage, the move proposal model and the
  sequential agent (UltimateBot) exist. The gen-1 frozen-backbone trial is
  the recorded floor.
- **[rack_conditional_evidence.md](rack_conditional_evidence.md)**:
  *proposed, plan-reviewed, not built; on hiatus, superseded by
  supreme_bot.md.* Extends the sim-residual loop so that
  what the sims of one candidate find transfers to the rest of the turn:
  evidence kept per rack index, an evidence-conditioned reply policy inside
  rollouts, outdated rollouts re-priced or re-run rather than discarded, and
  one acquisition rule over (candidate, rack indices). Built one layer at a
  time against an expert-labeled evaluation set.
- **[supreme_bot.md](supreme_bot.md)**: *the project's direction;
  plan-reviewed, not built.* The unconstrained form of the same goal: every step of every probe
  becomes a token in one context, and a causal transformer reads it to choose
  the moves and draws inside probes and the final pick. No per-node statistics
  or backup rule. Tokens describe content, not tree position, so any probe can
  inform any decision. Built as a learned reader over fixed probes first
  (gated against shrinkage on face-up leaves, then repeated in standard
  Scrabble), then learned move choices, then learned draws.
- **[deep_bot.md](deep_bot.md)**: *proposed, not built.* A two-level
  simulation agent for standard Scrabble. For each candidate and each sampled
  opponent rack, the opponent's reply is chosen in a subtree run on the
  opponent's own beliefs, over a diverse candidate set from a policy-set model
  that picks N moves to maximize the expected best value. Mechanical
  continuations, solved endgames, and a generational rack-inference network.
  Starts with policy-set labels and policy-set recall.
- **[deep_bot_v0.md](deep_bot_v0.md)**: *proposed, not built.* The
  implementation plan for DeepBot's first two models, trained from
  HastyBot-vs-HastyBot games in standard Scrabble. First, a standalone leave
  inference model, with free labels from replay, a 5-move history with
  pre-move cross-checks, and a 29-float output (a weight per letter plus a
  tilt on absolute leave value) over the exact pool prior, trained by a
  streaming pass over a leave table the engine builds in memory. Then the
  policy-set model in both forms, on exhaustively simmed pools whose rollouts stop at a
  win table, with opponent leaves drawn from the leave inference model. Each
  is one dashboard workload that generates, trains, slides its window and
  publishes eval metrics. The win table is a standalone fit that the leave
  inference workload also refits each generation, and every artifact carries
  a format version distinct from its generation.
- **[gain_target_experiment.md](gain_target_experiment.md)**: *proposed,
  not built.* Which gain target the greedy form trains on: the point margin
  today, expected improvement over the means' uncertainty, or the optimistic
  per-rollout expectation. Per-rollout values in the pool record, then the
  rules compared as selection rules on split rollouts.
- **[supreme_bot_m1a.md](supreme_bot_m1a.md)**: *proposed, not built.*
  The build plan for M1a, SupremeBot's held-out transfer test: probe traces
  in the sim runner, a generator workload writing labels and a new `.sprobe`
  probe sidecar, the reader with the face-up teacher as its prior, and the
  evaluation harness. A noise and saturation measurement sets the corpus size
  first.
- **[sim_labeled_candidates.md](sim_labeled_candidates.md)**: *proposed;
  the measurement (PR 0) landed.* A second target stream for the teacher: sim
  outcomes over every simmed candidate at sampled self-play positions, stored
  in the `.slog`, with soft targets and sibling subsampling. Motivated by the
  ACETA setup-play blind spot in `positions/NWL23/interesting-positions/`;
  records the survey of how often plays outside HastyBot's top 10 win.
- **[footprint_native_placement.md](footprint_native_placement.md)**:
  *landed.* Placement made footprint-categorical end to end, with the
  per-cell collapse kept only for visualization: the (15,15,13) spatial
  reshape, per-format storage decisions, the PR slicing and its deviations,
  and the plan-review dissent it resolved.
- **[generational_teacher.md](generational_teacher.md)**: *proposed,
  deferred.* The move-set student tracking an improving teacher: the teacher
  as versioned per-tag state advanced by promotion, teacher-bound corpus
  generations on a pair-aware ingest protocol, and the student training over
  a sliding window.
- **[pov_calibration_bias.md](pov_calibration_bias.md)**: *diagnosis
  settled, fix not built.* The teacher's POV calibration bias (about +0.8%
  win probability / +2.5 points toward the POV player): the evidence chain,
  its decomposition into unpinned drift over a stable, slightly negative
  structure, the candidate fixes and gates, and reproduction recipes.
- **[fp16_safe_serving.md](fp16_safe_serving.md)**: *landed.* The FP16
  activation-overflow incident, resolved by serving BF16: the measured
  monotone activation growth, the BF16/FP16/FP32 comparison, and why the
  model-side containment program was removed.
- **[lexical_features_for_value.md](lexical_features_for_value.md)**:
  *partly built, not adopted.* Lexical foresight for the value models through
  engineered GADDAG-computed input features instead of lexicon knowledge
  inside the network. The contingent-draw potential map was built and
  deleted; the post-move cross-check delta has an encoder and a diagnostic
  but is not yet a model input.

## Cloud compute

- **[cloud_training.md](cloud_training.md)**: *landed, partly superseded.*
  Running trainers on rented GPUs, several tags in parallel: the trainer's
  record/controls contract, the torch runtime and the bucket legs (all still
  in use), and a Runpod train slot, since replaced by cloud_machines.md.
- **[tag_queue.md](tag_queue.md)**: *landed (#276-#280, live-run fixes
  #281-#291).* A pool that owns its
  machines (laptops, and instances it rents up to a cap) and a global FIFO
  tag queue. Tags lease machines, slots come from the workload's layout,
  eligibility is checked against kinds and summed GPU memory, and release
  drains before it removes. End conditions become uniform (-1 = run forever)
  and optional.
- **[dashboard_state_model.md](dashboard_state_model.md)**: *plan-reviewed,
  revised, not built.* What the queue's first live run taught: normalized
  control records in one SQLite database, with constraints, and the tag state
  the operator sees as a projection of them; one writer with snapshot reads;
  an outbox of operations recorded before every action on the world; the pool
  owning every machine; a recorded home for each tag's training state, moved
  automatically by placement; and a simulation over a fake world that checks
  the invariants through random events and restarts.
- **[cloud_machines.md](cloud_machines.md)**: *landed.* AWS instances as
  task-scoped ssh machines that the dashboard launches, idles and terminates;
  the trainer on the ssh kind; spot; Runpod's removal. The provider lives in
  `py/cloud/providers/`.
