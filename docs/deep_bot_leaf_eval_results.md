# DeepBot leaf evaluation: trunk and join-plane comparison

The first runs of the `deep_bot_leaf_eval` workload: a small network that
values the leaf of a DeepBot rollout ([plans/deep_bot.md](plans/deep_bot.md)),
which the [win table](plans/deep_bot.md#sims) values today. Four arms crossed
two choices: a transformer or a conv trunk, with or without the join planes.

**Result.** Every arm beat the win table on held-out games, by about 0.010
nats of log loss overall and 0.022 nats when 20 or fewer tiles remain unseen.
The four arms finished within 0.0006 nats of each other. That is no larger than
the spread that comes from training on different games, so on this metric
neither the trunk nor the join planes made a measurable difference. The conv
trunk runs about 3x faster under TensorRT, so it is the trunk to serve. The
join planes are undecided: the metric averages over every leaf, while DeepBot
compares sibling leaves, where hook errors would matter most (see
[What the metric does not settle](#what-the-metric-does-not-settle)).

Run by David Shin and Claude on 2026-10-23: localhost, `asus-laptop`, and one
AWS rental, through the tag queue.

## The model

### What it values

A DeepBot rollout plays one move from each player's full rack, Alice and then
Bob, and does not refill either rack. The leaf is the position after Bob's
reply. Both leaves are known, since the sim holds both racks. The **pool** is
every tile on neither the board nor a leave; both refills will come from it.
The model gives the win/draw/loss probabilities from Bob's side, the player
who just moved.

This is the same information as position_eval's open-leaves post-move row, so
training rows come free from any `.slog` by replay. At turn *t*:

- Bob's leave is his rack after his move, before his draw.
- Alice's leave is what she kept at turn *t* − 1.
- Alice's real draw after that move goes back into the pool. It was a uniform
  draw from that pool, so the game's outcome is an unbiased target for the leaf.

Before Alice's first move her leave is empty and her whole rack is that draw,
which is the same form as the leaf after a bingo.

The labels describe the policy that played the games, as the win table's do.
The training games are HastyBot self-play, and HastyBot is the policy DeepBot's
rollouts continue with. Seeing both leaves does not bias the value: HastyBot
never reads its opponent's rack. Averaging the model over DeepBot's sampled
opponent leaves gives the value of the leaf from what the player can see.

### Inputs

The row is written by `LeafEvalInputEncoder`
([leaf_eval_input_encoder.h](../engine/include/encoding/leaf_eval_input_encoder.h)).

| Block | Size | Contents |
|---|---|---|
| board | 31 planes | letters A–Z, a blank marker, premium squares |
| cross-checks | 52 planes | 26 horizontal, 26 vertical |
| join planes | 12 planes | below |
| seat counts | 81 scalars | per-tile counts of my leave, the opponent's leave, and the pool |
| score | 1 scalar | score difference |
| leave values | 2 scalars | HastyBot's static value of each leave |
| pool size | 1 scalar | tiles in the pool |

**Join planes.** An earlier position_eval conv model fixed a hook prior where
it needed to read the rack: it gave a Y hook about a 0.17 chance with no Y
left. To read the rack, a conv trunk has to join a square's cross-check
letters with which seat holds those letters. That takes one gate per letter
and seat, and the model learned the gates only for common tiles. The join
planes do the join in the encoder:

- **Layout:** for each seat (my leave, the opponent's leave, the pool) and each
  play axis, two planes. The first is how well the seat's tiles fit the
  square's cross-check. The second is the highest face value among the fitting
  tiles, divided by 10.
- **Fit:** a leave's fit is 0 or 1. The pool's fit is the fraction of its tiles
  that fit, which is the chance a draw fits.
- **Blanks:** a blank fits any square that allows some letter, at face value 0.
- **Where they are set:** only on hook squares, the empty squares with a tile
  next to them across the play axis.

### Trunks

[model.py](../py/scribblez/deep_bot_leaf_eval/model.py). Both trunks feed the
same heads: win/draw/loss and score difference, from position_eval.

- **transformer:** width 64, 4 layers, 2 heads; 0.30M parameters.
  - Tokens: one per cell, one register per tile kind (its three seat counts,
    normalized, plus a zero flag per seat), and one global token (score, leave
    values, pool size).
  - Layers: pre-norm attention with no position encoding (a learned per-cell
    embedding carries position and the premium squares), then a GELU MLP.
- **conv:** position_eval's residual conv tower, 64 channels × 4 blocks, with
  the scalars injected at the stem; 0.46M parameters.

## Throughput: why the model is this small

DeepBot evaluates one leaf per rollout, so the model's speed is the leaf
budget. The GPU rollout kernel from PR #404 runs 150–600K rollouts/s on this
machine, depending on how many racks hold a blank.

Each size was timed under TensorRT FP16 on the RTX 5000 Ada laptop GPU, with
random weights, measuring GPU compute only, at batch 1,024. Batch 4,096 was
the same or slower.

| Model | Params | MACs | Evals/s |
|---|---|---|---|
| transformer, position_eval's tower (rotary position encoding, RMSNorm, SwiGLU), width 64 × 4 | 0.30M | 80M | 62K |
| **transformer, plain blocks (no position encoding, LayerNorm, GELU), width 64 × 4** | 0.30M | 80M | **78K** |
| transformer, plain blocks, width 96 × 4 | 0.57M | 153M | 50K |
| transformer, plain blocks, width 128 × 6 | 1.26M | 374M | 27K |
| conv 32 × 4 | 0.15M | 22M | 654K |
| conv 48 × 4 | 0.28M | 45M | 360K |
| **conv 64 × 4** | 0.45M | 77M | **256K** |
| conv 64 × 6 | 0.58M | 110M | 187K |
| conv 96 × 6 | 1.18M | 240M | 73K |
| conv 192 × 10 (position_eval's default) | 6.73M | 1.5G | 20K |

**The small transformer is limited by memory traffic, not FLOPs.** In the
profile of position_eval's tower at width 64, the GPU time split as follows:

| Rotary position encoding | RMSNorm | SwiGLU's elementwise part | Matmuls | Fused attention |
|---|---|---|---|---|
| 25% | 20% | 9% | 26% | 15% |

The plain blocks remove the first and shrink the next two. In TensorRT, a
3 × 3 convolution with batch norm and ReLU becomes one tensor-core kernel per
layer, so at about the same MACs the conv trunk runs 3.3x faster. Even so,
transformer XS is 2–8x slower than the rollout kernel, and conv 64 × 4 is
within its range.

## The runs

Four `deep_bot_leaf_eval` tags, identical except for the two knobs. Each tag
generated its own games. Generation is cheap: HastyBot self-play ran at 760–1,000
games/s on localhost.

| | |
|---|---|
| tags | `deep_bot_leaf_eval/leaf-{tf,conv}-{join,nojoin}` |
| games | hidden-leaves HastyBot self-play (greedy, endgame solver on), no random openings; 20,000 games per generation |
| training | one epoch per generation over a 4-generation window, 5 turns per game per generation, so each turn is trained once; batch 256; schedule-free optimizer, default learning rate; 77 generations, 30.2M rows, about 1.5M games per tag |
| held-out set | 2,000 HastyBot games at a fixed seed, every eligible turn; identical across tags |
| baselines | each generation, a win table fitted from the tag's own window, and Macondo's bundled table, scored on the same held-out rows |
| metric | log loss of the mover's expected score (win 1, draw ½) against the outcome, overall and by pool size |

**Where each tag ran.** The per-generation training times differ (31–57 s) by
machine, not by arm.

| Tag | Machine | Time |
|---|---|---|
| leaf-tf-join | localhost | 51 min |
| leaf-tf-nojoin | asus-laptop | 86 min |
| leaf-conv-join | AWS rental | 70 min |
| leaf-conv-nojoin | localhost | 45 min |

**Win-table cell.** The table reads the leaf from the side to move once both
refills are drawn:

- the spread plus the static value of that side's leave;
- the tiles that side cannot see: the bag plus the mover's rack.

A test checks these cells against the engine's own fitting samples for the
same positions, and they agree exactly.

## Results

### Overall

Held-out log loss, mean of the last 5 generations; lower is better.

| Arm | Model | Fitted win table | Macondo table | Model − fitted table |
|---|---|---|---|---|
| transformer + join | **0.4676** | 0.4783 | 0.4796 | −0.0107 |
| transformer | 0.4682 | 0.4782 | 0.4796 | −0.0100 |
| conv + join | 0.4678 | 0.4784 | 0.4796 | −0.0106 |
| conv | 0.4680 | 0.4786 | 0.4796 | −0.0106 |

Other held-out readings, which also barely differ between arms:

- three-way win/draw/loss cross-entropy: 0.4898–0.4902;
- score-difference mean absolute error: 52.3 points.

### By pool size

Each arm's log loss minus its fitted table's, mean of the last 5 generations:

| Tiles in the pool | transformer + join | transformer | conv + join | conv |
|---|---|---|---|---|
| 0–20 | −0.0230 | −0.0210 | −0.0217 | −0.0217 |
| 21–40 | −0.0094 | −0.0093 | −0.0096 | −0.0095 |
| 41–60 | −0.0066 | −0.0067 | −0.0075 | −0.0072 |
| 61–100 | −0.0072 | −0.0066 | −0.0069 | −0.0074 |

In absolute terms, the 0–20 band falls from about 0.294 for the table to
0.271 for the models, about 7% relative. The other bands improve by 1–2%.
Near the end of the game, the board and the exact tiles left matter more than
spread plus leave value, and those are what the table cannot see.

### Learning curves

Held-out model log loss by generation (one generation ≈ 400K rows once the
window is full):

| Generation | 0 | 2 | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 76 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer + join | 0.4738 | 0.4718 | 0.4700 | 0.4700 | 0.4692 | 0.4689 | 0.4687 | 0.4684 | 0.4683 | 0.4678 | 0.4676 |
| transformer | 0.4730 | 0.4709 | 0.4699 | 0.4694 | 0.4690 | 0.4688 | 0.4685 | 0.4684 | 0.4684 | 0.4683 | 0.4681 |
| conv + join | 0.4758 | 0.4710 | 0.4699 | 0.4699 | 0.4696 | 0.4696 | 0.4687 | 0.4684 | 0.4680 | 0.4680 | 0.4679 |
| conv | 0.4744 | 0.4703 | 0.4705 | 0.4700 | 0.4698 | 0.4693 | 0.4687 | 0.4685 | 0.4682 | 0.4681 | 0.4679 |

- Every arm passed the table within the first generation (about 100K rows).
- Every arm reached about 0.4700 by generation 5 (1.8M rows).
- The curves were still falling at the end, by about 0.0002 per 10 generations
  over the last 20.

### How much the arms differ

The spread across arms is 0.0006 nats. Two references for its size:

- **Within a run,** the last 10 generations' model log loss varies by about
  0.0001 (standard deviation).
- **Between runs,** each tag trained on its own games. The fitted table, the
  same procedure in every tag, lands 0.0004 apart across tags (0.4782 to
  0.4786) from that alone. The models' spread is about the same size, so one
  run per arm cannot attribute it.

## Conclusions

- **The model beats the win table everywhere, most of all late in the game.**
  It is a real improvement on DeepBot's current leaf: 0.010 nats overall,
  0.022 nats with 20 or fewer tiles left.
- **The trunk does not matter for value at this size, so serve the conv
  trunk.** It matched the transformer and runs about 3x faster under TensorRT
  (256K against 78K evals/s), within the rollout kernel's range.
- **The join planes are unresolved.** They neither helped nor hurt measurably
  on either trunk.
- **Most of a leaf's outcome is luck.** The score-difference error is 52
  points. A model built on spread plus leave quality starts out about as good
  as these, so absolute log loss can only move a little.

## What the metric does not settle

Average log loss over leaves is not what DeepBot asks of the model. DeepBot
compares the leaves of sibling candidates: different moves from the same
position, under the same rollout draws. What matters is how often the model
orders them the way their outcomes do. The join planes target hook-dependent
errors, which matter most there and are averaged away here.

Two measurements would settle the join planes:

1. **Sibling-leaf ranking:** the arms' agreement with outcome order across
   candidates from the same position, on paired rollouts.
2. **Replicates:** a second seed per arm, to measure the run-to-run spread the
   averages above cannot.

## Found along the way

**`amax` under `torch.compile` with bf16.** The conv arm first trained to NaN.
`amax`'s backward made every conv-trunk gradient non-finite from the first
step. It happened only with the Inductor backend under bf16 autocast:
`aot_eager`, eager, and pooling in FP32 were all clean, across seeds.
`mean_max_pool` ([spatial_trunk.py](../py/scribblez/spatial_trunk.py)) now uses
`max(dim).values`, which sends the gradient to one recorded index instead of
splitting it among recomputed ties. The forward is unchanged. leave_inference's
FP32-pool workaround ([leave_inference/model.py](../py/scribblez/leave_inference/model.py))
is probably the same bug and is still in place.

## Reproducing

The workload's defaults are these runs' settings, except for
`max_rows=30000000` and the two knobs (`trunk`, `join_planes`). The tags'
records, checkpoints and fitted win tables are under
`/workspace/mount/tags/deep_bot_leaf_eval/`. The throughput sweep's scripts
were scratch work and are not in the repo. The table above lists every
configuration timed, and the TensorRT settings were FP16, CUDA graphs, and no
data transfers.
