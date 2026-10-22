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
move chosen so far, `s*`. Three candidate definitions of that gain, for a
move `a`:

- **point**, the target today: `max(0, v̂(a) − v̂(s*))`, with `v̂` the pool's
  mean win probability.
- **ei**, expected improvement over the uncertainty in the means:
  `E[max(0, μ_a − μ_s*)]` with `μ_a − μ_s* ~ N(Δ̂, se²)`, where `Δ̂` is the
  mean paired difference against `s*` and `se` its standard error. In closed
  form, `Δ̂·Φ(Δ̂/se) + se·φ(Δ̂/se)`.
- **optimistic**, the expectation over rollouts: `E_r[max(0, v_r(a) −
  v_r(s*))]`, with `v_r` a rollout's value (`p_win + p_draw/2`) and the
  rollouts paired: rollout `r` deals every candidate the same draw.

Each one puts the max inside an expectation, so each rewards a move that
might overturn `s*`, not only one whose point estimate beats it. They differ
in what the expectation is over. **ei** is over what the sim has not yet
resolved about the two means. **optimistic** is over how the two moves play
out deal by deal, which includes the game's own luck.

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

## What we expect, and why it needs measuring

For any random variable, `E[max(0, D)] = (E[D] + E|D|) / 2`. So
**optimistic** is half the mean margin plus half the mean absolute per-rollout
difference. From the pools' paired standard errors, that spread is about 11 to
16 points per rollout, against margins of about 1 point. A normal-approximation
example: a move 5 points worse than `s*` with a per-rollout spread of 20 scores
5.7, and a move 2 points better with a spread of 3 scores 2.5. The final sim
picks by the mean, so only the second could change the decision. The concern
is that **optimistic** ranks moves by volatility.

The case for it: moves that play out differently from `s*` are where the
sim's verdict is least predictable from `s*` alone, and a set that covers them
may hold the sim-best move more often than a set of near-copies. Which
effect dominates is the experiment's question. **ei** is the middle option.
It credits a move's spread only while the spread leaves the mean unresolved,
and that credit shrinks as more rollouts come in.

## PR 1: per-rollout values in the pool record

- **The record.** `kPoolFormatVersion` 4. A header flag,
  `kPoolFlagRolloutValues`, marks a file that keeps them. Such a file follows
  each position's candidates with every candidate's rollout values, as
  float32 in candidate order, `rollouts` of them per candidate. Rollout `i` of
  every candidate is the same deal, as `pool_labels` already assumes.
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
were (opponent leaves inferred by `tilted2`'s latest generation, rollouts
stopped at its win table after 3 plies), but with **racing off** and 100
rollouts per candidate. Racing would leave the clearly beaten candidates
without the later rollouts the scoring uses. Racing averaged 34 rollouts per
candidate on the 8b pools. Without it, a position costs about 3 times as much,
roughly 0.37 positions/s on 28 threads. That makes 300 positions about 13
minutes and about 76 MB, both estimates to be measured on the first chunk. The
pools go under `/workspace/mount`, outside any worktree.

**A split, so the scoring is not the rules' own data.** Each rule sees only
rollouts `[0, k)`, the A part. The truth is the means over `[k, 100)`, the B
part. Scored on the same rollouts it chose by, any rule that tracks the
sample mean would look best.

**The rules.** Each builds sets of up to 16, starting from the seed, using only
the A part. After each step, `s*` is the best member of the set by its A mean.

| Rule | Next move |
|---|---|
| equity | the next in the static-equity ranking (no sim information; the reference) |
| point | the largest **point** gain, ties at 0 broken by A mean. This is the seed, then the top 15 by A mean |
| ei | the largest **ei** gain, with `Δ̂` and `se` from the paired A differences against `s*` |
| optimistic | the largest **optimistic** gain over the A rollouts |

**What is reported**, for each rule at N = 2, 4, 8, 16 and k = 10, 20, 40:
- Recall of the B leader.
- Regret: the B leader's mean minus the set's best B mean. It is less
  sensitive than recall to near-ties flipping between halves.
- Each rule's difference from **point**, paired by position, with 95%
  bootstrap intervals over positions. A pairwise comparison on the same
  positions is tighter than the ±0.02 to 0.03 standard error of one recall.
- For **optimistic** and **ei**: how often a pick is below `s*` on its A mean,
  and the picks' mean per-rollout spread against `s*`. These test the
  volatility concern directly.

Small k matters most. A model reads features, not sims, so it knows less about
a pool than even a short sim, and that low-information regime is where
optimism should matter most. As k grows, **ei** converges on **point**
(`se` goes to 0); **optimistic** does not, since its limit still weighs each
move's per-rollout spread.

The tool is `py/tools/gain_target_experiment.py`, which reads pools through
`policy_set.pools`. Its unit tests check each rule's gain against hand-worked
cases, including the closed form for **ei** and the identity above for
**optimistic**.

## What it does and does not settle

It compares the targets as selection rules on sim data. That is what a
greedy model trained on each target would try to reproduce from the board
alone. It does not train the models: what a model can learn from each target
is the next step if the rules separate. The greedy trainer would then take
the chosen target, so `policy_set` pools would keep rollout values in
production. Racing keeps that to about 34 values per candidate, about 86 KB
per position against today's 23 KB. The comparison would then run on a real
tag's data.
