# Model architectures

Wiring diagrams for the two trained networks and the trunk they share.
Authoritative code:
[spatial_trunk.py](../py/scribblez/spatial_trunk.py),
[position_eval/model.py](../py/scribblez/position_eval/model.py),
[move_set_eval/model.py](../py/scribblez/move_set_eval/model.py).

> **Keep in sync.** An architecture change in either `model.py` belongs in the
> same commit as the corresponding change here. The figures are generated —
> edit [py/tools/plot_model_architectures.py](../py/tools/plot_model_architectures.py)
> and re-run it, never the SVGs.

Symbols used throughout:

| Symbol | Meaning | Default |
|--------|---------|---------|
| `C` | `trunk_channels` | 192 |
| `N` | `num_blocks` in the residual tower | 10 |
| `P_in` / `S_in` | spatial planes / scalar width of the board input | 88 / 992 (full layout), 85 / 936 (base) |
| `B` | batch of board positions (`P` in move-set code) | — |
| `M` | candidate moves in a batch, flattened across positions | — |
| `T` | placed-tile slots per move (`kMoveMaxPlaced` = `RACK_SIZE`) | 7 |
| `‖` | concatenate along the channel/feature dim | — |

---

## 1. Shared spatial trunk

![The shared spatial trunk: conv stem and scalar projection, broadcast-added, then a residual tower](images/arch_spatial_trunk.svg)

`SpatialTrunk` also accepts an optional compiled-lexicon module, which injects a
per-cell residual after the stem. That experiment is deprecated and is left out
of the diagram.

### Tower blocks

![ResBlock and GlobalPoolingResBlock internals](images/arch_tower_blocks.svg)

`mean_max_pool`, used by both the pooling block and the value heads,
concatenates the channel-wise mean and max over the board: `(B, C, H, W)` →
`(B, 2C)`.

---

## 2. `PositionEvalModel`

One board in, six heads out. `wld` is the inference head; the rest are auxiliary
training signal.

![PositionEvalModel: the value summary feeding three FC stacks, plus the shared mask convolution](images/arch_position_eval.svg)

`sd_std_fc` reads a **detached** `v`, so the std loss trains that stack alone —
never the trunk, never the mean.

### Losses

| Head | Target | Loss | Weight |
|------|--------|------|--------|
| `wld` | one-hot win/draw/loss | cross-entropy | 1 |
| `score_diff[:,0]` | observed final differential | Huber (δ=10) | `lambda_sd` = 1 |
| `score_diff[:,1]` | `MAD_TO_STD · \|mean − target\|`, detached | Huber (δ=10) | `lambda_sd` = 1 |
| `*_next_placement` | binary per-cell mask | BCE-with-logits | `lambda_next_placement` = 0.5 each |
| `*_win_placement` | binary per-cell mask | BCE-with-logits | `lambda_win_placement` = 0.5 each |

`MAD_TO_STD = sqrt(π/2)` rescales the absolute-residual target so its optimum is
a Gaussian σ.

---

## 3. `MoveSetEvalModel`

Encode `P` boards once, score `M` candidate moves against them in the same pass.
Moves are flattened with no padding; each carries `move_pos_id ∈ [0, P)`.

![MoveSetEvalModel: board tokens and position summary from the trunk, move queries cross-attending into their own position's board](images/arch_move_set_eval.svg)

Grouping the queries by position keeps one K/V copy per board, so attention's
`W_k`/`W_v` projections are amortized across candidates the same way the trunk
is. The padded `(P, maxK, C)` query grid is the only place padding appears.

### The placement-plane readout (roadmap item 1)

The fused per-move vector (attended embedding + position summary, `4C`) is
projected to one `C`-wide query per plane head and dotted against the 225
board tokens: cell `(h, n)` of the `(M, 4, 225)` logit output is
`query_h · board_token_n`. The contraction runs over the same padded
`(P, maxK)` grid as the cross-attention, so the board tokens are read once
per position. Head order is `training_targets.h`'s placement targets (the
FFI-served `PLANE_NAMES`), matching the teacher masks quantized into the
`.mset` records.

### The move encoder

![MoveEncoder: tile embeddings fused with the move's scalars into one query vector](images/arch_move_encoder.svg)

`move_scalars = [resultant_score_diff, tiles/7, is_play]`; letters are A..Z with
a separate blank flag, so a natural tile and its blank twin share letter
semantics. Layout owned by
[move_set_encoder.h](../engine/include/training/move_set_encoder.h).

### Losses (distillation from the position-eval teacher)

| Head | Target | Loss | Weight |
|------|--------|------|--------|
| `wld` | teacher probabilities (M, 3) | soft cross-entropy | 1 |
| `score_diff[:,0]` | teacher mean | Huber (δ=10) | `lambda_sd` = 0.004 |
| `score_diff[:,1]` | teacher std | Huber (δ=10) | `lambda_sd` = 0.004 |
| `planes` | teacher masks, dequantized (M, 4, 225) | BCE-with-logits | `lambda_planes` = 1 |

Plane targets exist only in stratified (training) records; the full-sweep
evaluation slice is plane-less, so its metrics stay value-based and the
plane-readout quality (`plane_bce`) is read on the stratified fallback
holdout.

Ranking metric: `win_equity = P(win) + 0.5·P(draw)`, applied identically to
student softmax and teacher probabilities.

---

## 4. Side by side

|  | `PositionEvalModel` | `MoveSetEvalModel` |
|--|---------------------|--------------------|
| Trunk | `SpatialTrunk`, shared implementation | same |
| Unit of output | one board | one candidate move |
| Board encodes per output | 1 | 1 / candidate-set |
| Move-conditioning | none (board is post-move) | tile embeddings + cross-attention |
| Heads | wld, score_diff, 4 placement masks | wld, score_diff, 4 placement planes |
| Supervision | game outcomes / observed spread | teacher readouts (`.mset` sidecar) |
| ONNX outputs | `wld`, `score_diff`, 4 mask names | `wld`, `score_diff` (planes deferred to the evidence path) |
