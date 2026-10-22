# The greedy form's gain target: an experiment

**Status: proposed 2026-10-22; not built.** An experiment to choose the
training target of the policy-set model's greedy form
([deep_bot.md, "Form 1"](deep_bot.md#form-1-greedy-selection-with-cached-steps);
[deep_bot_v0.md](deep_bot_v0.md), step 8b). It needs per-rollout values in the
`.pool` record, which the pools do not keep today, so it lands in two PRs:
the record change, then the experiment.

Measurements inform the next step; no thresholds are fixed in advance.

## The question

The greedy form builds a set from the seed (the static-equity top move),
adding at each step the move with the largest predicted gain over the best
move chosen so far, `s*`. Two definitions of that gain, for a move `a`:

- **point**, the target today: `max(0, v̂(a) − v̂(s*))`, with `v̂` the pool's
  mean win probability.
- **ei**, expected improvement over the uncertainty in the means:
  `E[max(0, μ_a − μ_s*)]` with `μ_a − μ_s* ~ N(Δ̂, se²)`, where `Δ̂` is the
  mean paired difference against `s*` and `se` its standard error. In closed
  form, `Δ̂·Φ(Δ̂/se) + se·φ(Δ̂/se)`.

**ei** rewards a move that might overturn `s*`, not only one whose estimate
beats it: an unresolved near-tie gets a target in proportion to how
unresolved it is, and a clearly worse move gets about 0. As the sim resolves
the means, `se` goes to 0 and **ei** becomes **point**.

## Why now

Measured on the 8b check: the greedy form trained 8 epochs on 800 pools, the
statistics below from its 100 held-out pools.

- With the seed in every set, 1.6% of candidates beat it on `v̂`, about 10 per
  pool, by a median of 1 point. So about 98% of **point** targets are exactly
  0. The held-out gain loss goes flat after epoch 1 while recall keeps
  improving: the loss mostly measures predicting 0.
- The pools' paired standard errors (median 3.5 points) are larger than the
  margins they label. A near-tie's **point** target is mostly noise.
- The greedy form ends level with the static-equity baseline: recall@8 0.80
  against 0.80, regret@8 0.0023 against 0.0021.

## Considered and dropped: the expectation over rollouts

`E_r[max(0, v_r(a) − v_r(s*))]`, with `v_r` a rollout's value and the rollouts
paired, also puts the max inside an expectation, but over how the two moves
play out deal by deal: the game's own luck, not what is unresolved about the
means. For any random variable, `E[max(0, D)] = (E[D] + E|D|) / 2`, so it is
half the mean margin plus half the mean absolute per-rollout difference. From
the pools' paired standard errors that spread is about 11 to 16 points per
rollout, against margins of about 1 point, so it ranks moves mostly by
volatility. A normal-approximation example: a move 5 points worse than `s*`
with a per-rollout spread of 20 scores 5.7, and a move 2 points better with a
spread of 3 scores 2.5. The final sim picks by the mean, so only the second
could change the decision.

## PR 1: per-rollout values in the pool record

**ei** needs the paired standard error of each move against `s*`, and `s*` is
whichever member of a sampled set is best, so it changes with every set. The
pools keep the paired standard error against the leader only. Pairing matters:
unpaired, the error would include the luck two moves share and be far larger.
So the record keeps every rollout's value, from which any pair's error follows.

- **The record.** `kPoolFormatVersion` 4. A header flag,
  `kPoolFlagRolloutValues`, marks a file that keeps them. Such a file follows
  each position's candidates with every candidate's rollout values, as
  float32 in candidate order, `rollouts` of them per candidate: each
  rollout's `p_win + p_draw/2`. Rollout `i` of every candidate is the same
  deal, as `pool_labels` already assumes.
- **The survey.** `sim_candidate_survey_tool --pool-rollout-values` writes
  them. Off by default; the `policy_set` workload does not set it until a
  trainer reads them.
- **Python.** `pools.Pool` gains `rollout_values`, flattened in candidate
  order, with each candidate's slice given by the cumulative sum of
  `rollouts`. It is `None` for a file without the flag. The trainer ignores
  the field.
- **Tests.** A C++ round trip with and without the flag. A Python read of a
  surveyed chunk checks that the stored values reproduce `win_prob`.
- **Cost.** No policy_set tag exists, so the version bump strands only
  scratch pools.

## PR 2: the experiment

**Pools.** Fresh positions from HastyBot games, surveyed as the 8b check's
were and as production surveys them: opponent leaves inferred by `tilted2`'s
latest generation, rollouts stopped at its win table after 3 plies, 100
rollouts per candidate, raced. On the 8b pools that came to about 1.1
positions/s on 28 threads, so 300 positions take about 5 minutes, and their
rollout values about 26 MB at the 34 rollouts per candidate racing averaged
there. Both figures are estimates until the first chunk is measured. The pools go under
`/workspace/mount`, outside any worktree.

**A split, so the scoring is not the rules' own data.** Each rule sees only
rollouts `[0, k)`, the A part. The truth is the means over `[k, 100)`, the B
part. Scored on the same rollouts it chose by, any rule that tracks the
sample mean would look best.

**Moves racing stopped.** Racing stops a move once it is three paired
standard errors behind the leader at a checkpoint (10, 20, 40 or 70
rollouts). The survey's design note puts the chance of stopping a move truly
equal to the leader well under once per hundred positions. So a stopped move
counts as never best: the B leader is the best B mean among the moves that
ran all 100 rollouts, and a stopped member of a set never counts as the set's
best. A move stopped before k rollouts gives the rules the rollouts it has,
with a correspondingly wider `se`, as production's **ei** would compute it.
On the 8b pools racing stopped the seed in 6.6% of positions, each a position
where the static-equity top move was itself clearly beaten.

**The rules.** Each builds sets of up to 16, starting from the seed, using only
the A part. After each step, `s*` is the best member of the set by its A mean.

| Rule | Next move |
|---|---|
| equity | the next in the static-equity ranking (no sim information; the reference) |
| point | the largest **point** gain, ties at 0 broken by A mean. This is the seed, then the top 15 by A mean |
| ei | the largest **ei** gain, with `Δ̂` and `se` from the paired A differences against `s*` |

**What is reported**, for each rule at N = 2, 4, 8, 16 and k = 10, 20, 40:
- Recall of the B leader.
- Regret: the B leader's mean minus the set's best B mean. It is less
  sensitive than recall to near-ties flipping between the parts.
- **ei**'s difference from **point**, and each from **equity**, paired by
  position, with 95% bootstrap intervals over positions. A pairwise
  comparison on the same positions is tighter than the ±0.02 to 0.03
  standard error of one recall.
- For **ei**: how often a pick is below `s*` on its A mean, and whether such
  picks are the B leader more often than **point**'s picks at the same step.

Small k matters most. A model reads features, not sims, so it knows less about
a pool than even a short sim, and that low-information regime is where
crediting unresolved near-ties should matter most. As k grows, **ei**
converges on **point**.

The tool is `py/tools/gain_target_experiment.py`, which reads pools through
`policy_set.pools`. Its unit tests check each rule's gain against hand-worked
cases, including the closed form for **ei**.

## What it does and does not settle

It compares the two targets as selection rules on sim data. That is what a
greedy model trained on each target would try to reproduce from the board
alone. It does not train the models: what a model can learn from each target
is the next step if **ei** separates from **point**. The greedy trainer would
then take it, so `policy_set` pools would keep rollout values in
production. Racing keeps that to about 34 values per candidate, about 86 KB
per position against today's 23 KB. The comparison would then run on a real
tag's data.
