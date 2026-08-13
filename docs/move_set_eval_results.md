# Move set evaluation v1: the A3 curves

What the first in-variant distillation run measured. These are curves, not a
verdict. Read them as "how well does the learned filter reproduce the teacher's
ranking, and by how much does it beat the incumbent" — not as "the filter is
ready to replace exact evaluation". The sensitivity sweep that was meant to
convert them into a win-rate bar returned a null
([evaluation_plan.md](evaluation_plan.md)), so the direct comparison against
exact evaluation is what settles that question.

## What the filter is for

Evaluating every legal move with the position evaluation model costs a full
board re-encode per candidate. The move set evaluation model scores the whole
candidate set in one pass and hands the top few to the expensive stage, so its
job is not to be right about values — it is to *not lose the move that
mattered*. That is why the metrics are recall and regret rather than a loss.

The incumbent it replaces is static equity (score + leave value), the ranking
every existing engine shortlists by.

## The run

A `move_set_eval` dashboard tag with a generate worker and a train worker
started together, left to complete unattended.

| | |
|---|---|
| teacher | `position_eval/face-up-leaves` `model_epoch_3016.onnx`, content hash `923dbb811e64` |
| variant | face-up leaves, open-leaves input arm on both teacher and student |
| corpus | 600 `.slog`/`.mset` pairs, 1.5 GB, generated in ~70 min |
| training set | 574 stratified pairs — 2,305,319 positions / 34,225,060 candidates |
| held-out set | 26 full-sweep pairs — 10,396 positions / 6,851,783 candidates |
| student | 6,874,085 params (trunk 192ch × 10 blocks, 4 attention heads) |
| training | 25 passes (5 while the corpus grew, then 20 budgeted epochs), 772,161,340 rows, ~10.5 h |
| optimizer | AdamW, lr 1e-3 flat, weight decay 1e-4, 64 positions/batch, `lambda_sd` 0.004 |

Generation parameters were the workload defaults: 200 games per cycle, the
4/4/4/2 stratified quotas with `mid_rank_limit` 32, `sweep_every` 20,
`sweep_positions_per_game` 2, `sweep_candidate_cap` 1500, greedy HastyBot
self-play with `random_opening_mean` 2.0.

### Why the held-out slice is the full-sweep one

The stratified training sample carries ~15 candidates per position, so scoring
the model on it asks only whether it can rank moves it was trained on — it is
structurally blind to the tail moves a filter exists to catch. The held-out
pairs are instead labeled in the generator's full-sweep mode: every legal
candidate of a position, capped at 1500 by static-equity rank, with the true
legal count stored so truncation stays visible.

Two consequences matter for reading the numbers:

- Because a sweep is stored in equity-rank order, the incumbent baseline here
  is the **exact** static-equity ranking rather than a floor. Every
  student-vs-incumbent comparison below is apples-to-apples.
- Coverage was 0.930 of legal moves, and 1,386 of 10,396 positions (13.3%) hit
  the 1500 cap. The truncated positions are the widest ones — two-blank racks,
  where a filter is most likely to miss — so these numbers are mildly
  optimistic there.

## Results

After the final budgeted epoch, on the held-out sweep:

| metric | learned filter | incumbent (static equity) | |
|---|---|---|---|
| recall@1 | **0.687** | 0.563 | +12.5 pts |
| recall@3 | **0.726** | 0.605 | +12.1 pts |
| recall@5 | **0.744** | 0.624 | +12.1 pts |
| regret@1 | **0.00324** | 0.00896 | 2.8× lower |
| regret@3 | **0.00100** | 0.00353 | 3.5× lower |
| regret@5 | **0.00057** | 0.00227 | 4.0× lower |
| Spearman | **0.875** | 0.802 | |

Recall@K is the fraction of the teacher's top-K that the filter's top-K
retains. Regret@K is the teacher win-equity forfeited by keeping only the top-K
— it prices a miss, where recall counts dropping a near-tie the same as
dropping a uniquely winning move. Both are averaged over positions.

The regret column is the one to weigh: keeping the filter's top 5 costs
0.00057 win-equity against the teacher's own best, a quarter of what the
incumbent's top 5 costs.

### The curve

Recall@1 against training progress, with the incumbent flat at 0.563
throughout. Passes 0–4 ran while the corpus was still growing; the holdout was
frozen from pass 4 onward, so every budgeted epoch is scored against the same
slice.

| pass | corpus (positions) | recall@1 | regret@1 | Spearman | `loss_wld` |
|---|---|---|---|---|---|
| 0 | 389,882 | 0.518 | 0.0121 | 0.708 | 0.5070 |
| 2 | 987,969 | 0.593 | 0.0066 | 0.802 | 0.4975 |
| 4 | 2,305,319 | 0.644 | 0.0046 | 0.851 | 0.4958 |
| 5 (budget 1) | full | 0.665 | 0.0041 | 0.858 | 0.4954 |
| 10 (budget 6) | full | 0.675 | 0.0036 | 0.867 | 0.4948 |
| 15 (budget 11) | full | 0.678 | 0.0036 | 0.871 | 0.4945 |
| 20 (budget 16) | full | 0.684 | 0.0033 | 0.875 | 0.4944 |
| 24 (budget 20) | full | 0.687 | 0.0032 | 0.875 | 0.4943 |

The filter passed the incumbent on both recall@1 and regret@1 at pass 1, while
the corpus was still a quarter of its final size and well before the budgeted
epochs began.

## What this does and does not establish

**Established.** Distillation works in-variant: a 6.9M-parameter student
reproduces the teacher's ranking well enough to beat the incumbent shortlist
decisively on the slice where a filter's failures actually show up. The
previous shakeout's numbers were provisional — generated out-of-variant against
a hidden-leave teacher — and this run is their re-derivation.

**Not established.** Whether that margin is *enough*. The sensitivity sweep that
was to price a recall miss in win-rate terms found no measurable cost at 400
games per arm ([evaluation_plan.md](evaluation_plan.md)), so 0.687 still has no
target attached. What settles it is the direct comparison against per-candidate
exact evaluation at equal rollout budget — a non-inferiority test, since the
filter is ~13× cheaper per turn.

**Not comparable.** The earlier shakeout reported recall@1 0.761 against equity
0.591. That was measured on a *stratified* holdout — ~15 candidates per
position — and the full-sweep task here is far harder. The two numbers should
never be put side by side.

## Convergence: the plateau is the optimizer's, not the corpus's

Training loss and held-out metrics flattened together over the budgeted epochs:

- `loss_wld` — the head that drives the ranking — moved 0.4954 → 0.4943 across
  20 epochs.
- recall@1 over the last eight epochs oscillated in 0.684–0.690 with no trend.
- regret@1 and Spearman were flat from roughly epoch 16.

Only `loss_score_diff` kept falling (13.9 → 9.7); at `lambda_sd` 0.004 it does
not affect the ranking.

Training loss flattening *alongside* the held-out metrics is the signal worth
keeping: data starvation looks like training loss continuing to fall while the
holdout stalls, and that is not what happened. The learning rate was pinned at
1e-3 for all 25 passes and never decayed, so the plateau is evidence about the
schedule rather than about the corpus being exhausted. Deciding between the two
needs a decay from this checkpoint, which the headless trainer cannot currently
do — it builds a fresh model per invocation, so a lower-rate rerun measures
"train slower from scratch", a different question.

More epochs on this corpus buy nothing. Whether more data would is untested.

## Reproducing

Create a `move_set_eval` tag pointing `teacher_model` at an open-leaves
position-eval export, set `face_up_leaves`, add a worker of each role, and
start them. The trainer waits for `warmup_pairs`, keeps pace with the generator
while it runs, and spends its `train_epochs` budget on passes over the finished
corpus, so the tag completes unattended
([workloads/move_set_eval.py](../py/scribblez/workloads/move_set_eval.py),
[move_set_eval/trainer.py](../py/scribblez/move_set_eval/trainer.py)).

The metrics above are the trainer's own per-pass readouts, recorded to the
tag's dashboard DB; `move_set_eval/eval.py` defines them and
`_baseline_ranking` documents exactly what the incumbent column is on each
kind of slice.

### An incidental measurement

The two roles share one GPU. With the generator running, training moved
13.8–14.4k rows/s; once it stopped, 21.2k. Concurrency costs each side roughly
half again its solo throughput, and generation is a small fraction of a run's
wall clock, so running them together buys unattendedness rather than speed.
