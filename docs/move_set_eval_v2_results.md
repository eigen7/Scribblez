# Move set evaluation v2: the planar corpus run

The run that closed [roadmap item 1](roadmap.md#1-per-move-placement-planes):
the first corpus in the `.mset` v2 format (per-candidate quantized placement
planes) and the first student trained with the four per-move plane readouts.
Read alongside [move_set_eval_results.md](move_set_eval_results.md), the v1
curves this run supersedes; the metric definitions and their caveats are the
same.

Conducted by user dshin on his own machine (local dashboard workers, one
overnight session), 2026-08-14.

## The run

A `move_set_eval` dashboard tag with a generate worker and a train worker
started together, left to complete unattended.

| | |
|---|---|
| tag | `move_set_eval/face-up-leaves-planar` |
| teacher | `position_eval/face-up-leaves` `model_epoch_3016.onnx`, content hash `923dbb811e649679` — the same blessed file that labeled both v1 corpora |
| variant | face-up leaves, open-leaves input arm on both teacher and student |
| corpus | 600 `.slog`/`.mset` pairs, 33 GB (32.7 GB `.mset` — the planes — plus 0.1 GB `.slog`), generated in ~1.3 h |
| training set | 570 stratified pairs — 2,289,381 positions / 33,989,532 candidates, planes on every record |
| held-out set | 30 full-sweep pairs (plane-less by design) — 12,000 positions / 8,045,080 candidates, 0.923 mean coverage of legal moves, 1,716 positions cap-truncated |
| student | 7,464,677 params (trunk 192ch × 10 blocks, 4 attention heads, plus the plane readout) |
| training | 25 passes (5 while the corpus grew, then 20 budgeted epochs), 765,889,404 rows, ~13 h |
| optimizer | AdamW, lr 1e-3 flat, weight decay 1e-4, 64 positions/batch, `lambda_sd` 0.004, `lambda_planes` 1.0 |

Generation parameters were the workload defaults, identical to the v1 runs:
200 games per cycle, 4/4/4/2 stratified quotas with `mid_rank_limit` 32,
`sweep_every` 20, `sweep_positions_per_game` 2, `sweep_candidate_cap` 1500,
greedy HastyBot self-play. The only training-side novelty against v1 is the
plane head and its loss term.

## Gate metrics: v2 against v1 and the incumbent

Final budgeted pass, on each run's own full-sweep holdout. The incumbent
baselines agree across the two runs (recall@1 baseline 0.561 vs 0.563), so
the corpora are comparable draws.

| metric | v2 (this run) | v1 | incumbent baseline |
|---|---|---|---|
| recall@1 | **0.713** | 0.689 | 0.561 |
| recall@3 | **0.757** | 0.733 | — |
| recall@5 | **0.771** | 0.753 | — |
| regret@1 | **0.0026** | 0.0031 | 0.0096 |
| Spearman | **0.944** | 0.933 | 0.802 |
| exch_rank_regret | **0.0020** | — | 0.0100 |

The plane head did not tax the value heads: `loss_wld` finished at 0.4937
against v1's 0.4939, and every ranking metric moved the right way. Whether
the improvement is multi-task regularization from the plane targets or partly
a fresh-corpus draw cannot be separated from one run; the direction is what
matters, and it is the safe one — the readouts item 2 depends on came at no
value cost.

## The plane head

`loss_planes` (per-cell BCE against the teacher's quantized planes) fell from
0.0753 on the first pass to 0.0558 on the last — **and was still declining
when the epoch budget ran out**, unlike the value heads, which plateaued the
way v1 did. The plane readout is therefore not saturated at this budget. That
is no blocker — the evidence loop consumes whatever prior the model has — but
if item 2's ablation shows the predicted planes are load-bearing, a longer
budget or a `lambda_planes` sweep on the next corpus generation is cheap
upside.

`plane_bce` does not appear in the holdout metric series, by design: the
full-sweep slice carries no plane targets (`record_planes` 0), so plane
quality for this run is read off the training-side `loss_planes` curve. A
held-out plane read would come from the stratified fallback holdout
(`holdout_every`) on a run configured without sweeps.

## Where this leaves the roadmap

Item 1 is complete: format, generator, readouts, corpus, and trained student
v2 (`model_epoch_0024.onnx`) all exist. Item 2 (evidence conditioning) is
unblocked; its first dependency is `.sobs` sim observations at scale, of
which none exist yet.
