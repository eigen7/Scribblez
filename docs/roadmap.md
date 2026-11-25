# Scribblez roadmap

Scribblez aims to beat existing Scrabble engines by replacing their
context-blind static evaluation and naive rack inference with learned,
belief-aware evaluation ([design.md](design.md)). This document is the
**implementation plan**: the agent being built, what exists, what is left, and
the models that have to be trained to feed it. Read it to know where a piece
of work fits and why the pieces are ordered as they are.

**Direction, as of 2026-11-24: DeepBot.** DeepBot plays standard Scrabble with
a learned leaf network and a learned belief about the opponent's rack, and
improves itself by continuous training in the style of AlphaZero. The design
and its build order are [plans/deep_bot.md](plans/deep_bot.md); this document
summarizes them and places them among what exists. Earlier directions are in
[History](#history).

## DeepBot

DeepBot values a move by the leaf network's estimate of the position right
after it. The network is a convolutional or transformer trunk over the board
and both players' tiles (the [leaf network](#the-leaf-network)). The opponent's
rack is unknown, so DeepBot reasons over the leave inference model's belief
about it (the [leave inference model](#the-leave-inference-model)).

The plan takes it in four steps:

- **The belief-input leaf.** The leaf network takes the opponent's leave either
  as a concrete leave or as the leave inference model's belief, and half its
  training rows hide the concrete leave, so one network serves both uses. A
  turn then costs one leaf evaluation per candidate instead of one per
  candidate and drawn leave.
- **Cheap self-play with searched labels.** At most turns self-play plays the
  leaf's best move outright. A random fraction of turns is searched instead:
  each candidate is played a few plies on, over opponent racks drawn from the
  belief, and valued by the leaf where the search stops. The search's best is
  played, and every candidate it valued becomes a training row.
- **Continuous training.** Generators play the newest release while the
  trainer's window spans many releases.
- **The prefilter as research.** Which moves a turn values is chosen by cheap,
  non-neural decision stumps whose union should almost always hold the leaf's
  best. Static equity's top moves are the first; the self-play records the
  data that measures them.

Today's DeepBot (`--type=deepbot`) is the first form: it values static
equity's top candidates, each over leaves drawn from the belief, with no
search.

## Status at a glance

The phases are [plans/deep_bot.md](plans/deep_bot.md#phases)'s.

| Phase | Status |
|------|--------|
| 0. Measurements: the cutoff value's spread, legal-move counts, leaf throughput at small batches | Not started; the spread waits for phase 4's search |
| 1. Small changes: THAN out of the known positions; the opponent's rack panel; the retired workloads removed | Done (2026-11-25) |
| 2. The belief-input leaf: input format version 3, the belief sidecar, the trainer's two modes, the join-plane ablation | Next |
| 3. The greedy player: DeepBot in belief mode, one evaluation per candidate | Not started |
| 4. The labeled-turn search in the engine | Not started |
| 5. The continuous workload, `deep_bot` | Not started |
| 6. The Positions tab: the opponent's rack per release, the search's columns | Not started |

## The variant: standard Scrabble

DeepBot plays standard Scrabble, with hidden leaves. Rack inference is what the
design is about, and a variant that removed it would hide the problem.
Face-up-leaves Scrabble (each player's leave revealed after every turn) stays
a parameter of the position evaluation workload and the game loop, and its
tags and results remain valid for that variant.

## What is already built

- **DeepBot**, the agent ([agent/deep_bot.h](../engine/include/agent/deep_bot.h)),
  with `deepbot-endgame` solving the turns with the bag empty.
- **The leaf network and the leave inference model**, each trained by its own
  workload on HastyBot self-play (`deep_bot_leaf_eval`, `leave_inference`).
  Together they are DeepBot's generation 0.
- **`deep_bot_generation`**, one generation step of DeepBot taken by hand: it
  plays DeepBot on its seeds against itself and trains both models on those
  games. Its policy is fixed for the tag's life; the continuous workload of
  phase 5 replaces it.
- **The Positions tab** of both leaf workloads: the known positions
  ([positions/NWL23/known-positions](../positions/NWL23/known-positions/README.md))
  under each analyzed generation, the leaf's value of every move, and the
  mover's belief about the opponent's rack.
- **The position evaluation model** and its workload (`position_eval`), with
  its placement heads and Monte Carlo ground truth. DeepBot does not use it;
  its architecture findings carry over to the leaf network.
- **Search and baselines**: the sim runner
  ([sim/sim_runner.h](../engine/include/sim/sim_runner.h)), the endgame and
  pre-endgame solvers, HastyBot, and the port of Macondo's BestBot as the
  match baseline.
- **Infrastructure**: the master dashboard and workload registry, the match
  harness, and the cloud fleet ([cloud_compute.md](cloud_compute.md)).

## Models and how they are trained

### The leaf network

- **Is**: [deep_bot_leaf_eval/model.py](../py/scribblez/deep_bot_leaf_eval/model.py),
  a value model of the position after a move, from the mover's side.
- **Predicts**: win/draw/loss, the score difference, and continuation heads
  (each square's fate, going out, end-of-game points).
- **Trained on**: the outcomes of the games it is read in, first HastyBot
  self-play (`deep_bot_leaf_eval`), then DeepBot's own
  (`deep_bot_generation`). Phase 2 adds the belief as an input; phase 5 trains
  it on searched values instead of game results alone.
- **Served**: through TensorRT on the GPU, by the hasty-rollout backend, which
  encodes its rows on the device. Phase 2 either ports the version 3 input
  into that kernel or replaces the path with one that encodes on the CPU.

### The leave inference model

- **Is**: [leave_inference/model.py](../py/scribblez/leave_inference/model.py),
  a tilt of the hypergeometric prior over the leave the opponent kept at their
  last move, as 29 parameters (a weight per tile and a tilt on the leave's
  value).
- **Trained on**: the true leaves replay recovers from self-play, with the
  forgone plays (what the opponent could have played and did not) as input.
- **Role**: DeepBot draws the opponent's leaves from it; from phase 2 the leaf
  network also reads it directly.

## Rack inference

DeepBot's rack inference is the leave inference model. A port of the
algorithm behind Macondo's `SIMMING_INFER_BOT`
([belief/rack_inference.h](../engine/include/belief/rack_inference.h)) also
exists: a hypergeometric prior over draws from the unseen pool with a
temperature-softened static-equity likelihood. Its one consumer is offline,
the hidden-leaves Monte Carlo ground truth of the position evaluation test
sets ([sim/monte_carlo_sim.h](../engine/include/sim/monte_carlo_sim.h)).

## History

Earlier directions, newest first. Their plans keep their rationale; their code
is gone except what [What is already built](#what-is-already-built) lists. The
roadmap as it stood before this rewrite is in `git log -p -- docs/roadmap.md`.

- **SupremeBot** (from 2026-09-29): a learned search in which one
  causal transformer reads every probe of the turn as tokens and makes every
  decision ([plans/supreme_bot.md](plans/supreme_bot.md),
  [plans/supreme_bot_m1a.md](plans/supreme_bot_m1a.md)). Its first milestone
  was a test of whether the reader learns transfer between probes.
- **SubtreeBot** (2026-10 and 2026-11): two-level simulation in which the opponent's reply to each
  candidate is searched as a subtree on the opponent's own beliefs, with sims
  ending at the leaf network ([plans/subtree_bot.md](plans/subtree_bot.md)).
  Its built form used HastyBot's move as the reply and played about level with
  BestBot. Its leaf network and leave inference model are DeepBot's.
- **The evidence-loop agent (UltimateBot)** (to 2026-09-29): candidates simmed
  one at a time, each chosen by a learned expected-gain head reading the
  evidence of the sims so far, over a move-set student distilled from the
  position evaluation model
  ([plans/sim_residual_feedback.md](plans/sim_residual_feedback.md),
  [plans/rack_conditional_evidence.md](plans/rack_conditional_evidence.md)).
  It ran in face-up-leaves Scrabble. The position evaluation model and the
  value-truncated rollouts remain from it.
