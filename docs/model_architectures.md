# Model architectures

Wiring diagrams for the two trained networks and the trunk they share.
Authoritative code:
[spatial_trunk.py](../py/scribblez/spatial_trunk.py),
[position_eval/model.py](../py/scribblez/position_eval/model.py),
[move_set_eval/model.py](../py/scribblez/move_set_eval/model.py).

> **Keep in sync.** An architecture change in either `model.py` belongs in the
> same commit as the corresponding change here.

Symbols used throughout:

| Symbol | Meaning | Default |
|--------|---------|---------|
| `C` | `trunk_channels` | 192 |
| `N` | `num_blocks` in the residual tower | 10 |
| `P_in` / `S_in` | spatial planes / scalar width of the board input | 88 / 992 (full layout), 85 / 936 (base) |
| `B` | batch of board positions (`P` in move-set code) | — |
| `M` | candidate moves in a batch, flattened across positions | — |
| `T` | placed-tile slots per move (`kMoveMaxPlaced` = `RACK_SIZE`) | 7 |
| `⊕` | concatenate along the channel/feature dim | — |

---

## 1. Shared spatial trunk

```
   input_spatial (B, P_in, 15, 15)          input_scalar (B, S_in)
              │                                      │
      ┌───────▼────────┐                    ┌────────▼────────┐
      │ stem           │                    │ scalar_proj     │
      │ conv3x3 P_in→C │                    │ Linear S_in→C   │
      │ BatchNorm      │                    │ ReLU            │
      │ ReLU           │                    │ Linear C→C      │
      └───────┬────────┘                    └────────┬────────┘
              │ (B, C, 15, 15)                       │ s: (B, C)
              │                                      │
             (+)◀─────── broadcast-add over 15x15 ───┤
              │                                      │
      ┌───────▼──────────────────────────┐           │
      │ lexicon_module      (optional)   │           │
      │ per-lane DAWG walk over rows,    │           │
      │ then columns; x += row + col     │           │
      │ per-cell residual                │           │
      └───────┬──────────────────────────┘           │
              │                                      │
      ┌───────▼──────────────────────────┐           │
      │ residual tower, N blocks         │           │
      │   i % 3 == 2 → GlobalPoolingRes  │           │
      │   otherwise  → ResBlock          │           │
      └───────┬──────────────────────────┘           │
              │                                      │
      BatchNorm → ReLU                               │
              │                                      │
              ▼                                      ▼
        x: (B, C, 15, 15)                       s: (B, C)
```

### Tower blocks

```
ResBlock (pre-activation)

  x ─┬─▶ BN → ReLU → conv3x3 → BN → ReLU → conv3x3 ─▶(+)─▶ out
     └──────────────────── skip ───────────────────────┘
```

```
GlobalPoolingResBlock  (KataGo-style; Cp = C/2, Cs = C − Cp)

  x ──┬──▶ BN → ReLU → conv3x3 (C→C)
      │                    │
      │        ┌───────────┴─────────────┐
      │   channels [0:Cs]         channels [Cs:C]
      │      "spatial"                "pool"
      │         │                        │
      │         │                mean_max_pool (2·Cp)
      │         │                        │
      │         │                Linear → bias (Cs)
      │         │                        │
      │        (+)◀─── broadcast-add ────┘
      │         │
      │   BN → ReLU → conv3x3 (Cs→C)
      │         │
      └────────(+)◀── skip
                │
                ▼ out
```

```
mean_max_pool:  (B, C, H, W) ──▶ [ mean over HxW ⊕ max over HxW ] ──▶ (B, 2C)
```

---

## 2. `PositionEvalModel`

One board in, six heads out. Consumed at inference for `wld`; the rest are
auxiliary training signal.

```
 input_spatial ─┐
                ├─▶ SpatialTrunk ─▶ x (B, C, 15, 15)   s (B, C)
 input_scalar ──┘                     │                  │
                                      │                  │
        ┌─────────────────────────────┤                  │
        │                             │                  │
        │                      mean_max_pool             │
        │                          (B, 2C) ──────⊕───────┘
        │                                        │
        │                                   v (B, 3C)
        │                                        │
        │           ┌────────────────────────────┼──────────────────────┐
        │           │                            │                      │
        │   ┌───────▼────────┐          ┌────────▼───────┐      ┌───────▼────────┐
        │   │ wld_fc         │          │ sd_mean_fc     │      │ sd_std_fc      │
        │   │ Linear 3C→64   │          │ Linear 3C→256  │      │ input: v.detach│
        │   │ ReLU           │          │ ReLU           │      │ Linear 3C→256  │
        │   │ Linear 64→3    │          │ Linear 256→1   │      │ ReLU           │
        │   └───────┬────────┘          └────────┬───────┘      │ Linear 256→1   │
        │           │                            │              │ softplus+1e-3  │
        │           ▼                            │              └───────┬────────┘
        │      wld (B, 3)                        └────────⊕─────────────┘
        │      logits                                     │
        │                                        score_diff (B, 2)
        │                                          [mean, std] in points
        │
 ┌──────▼──────────────┐
 │ mask_conv           │
 │ conv1x1 C→32 (no b) │
 │ BatchNorm → ReLU    │
 │ conv1x1 32→4        │
 └──────┬──────────────┘
        │  (B, 4, 15, 15) logits, split along channels
        ├──▶ opp_next_placement   (B, 15, 15)
        ├──▶ self_next_placement  (B, 15, 15)
        ├──▶ opp_win_placement    (B, 15, 15)
        └──▶ self_win_placement   (B, 15, 15)
```

Gradient note: `sd_std_fc` reads a **detached** `v`, so the std loss trains that
stack alone — never the trunk, never the mean.

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

```
 input_spatial (P, P_in, 15, 15) ─┐
                                  ├─▶ SpatialTrunk ─▶ x (P, C, 15, 15)   s (P, C)
 input_scalar  (P, S_in) ─────────┘                      │                  │
                                                         │                  │
              ┌──────────────────────────────────────────┤                  │
              │                                          │                  │
   flatten(2).transpose + board_pos_emb           mean_max_pool             │
              │                                       (P, 2C) ──────⊕───────┘
              ▼                                                     │
     board (P, 225, C)                                          g (P, 3C)
```

### 3a. Move encoding

```
 move_squares (M, T) ─┐
 move_pos_id  (M,) ───┴─▶ gather board[pos_id, square] ─▶ tile_board (M, T, C)
                                                              │
                          × is_play (move_scalars[:, 2]) ─────┤  exchange / pass
                                                              │  ⇒ zeroed
 move_letters (M, T) ─▶ letter_emb (27→C, padding_idx=0) ─────┤
 move_blanks  (M, T) ─▶ blank_emb  (2→C) ─────────────────────┤
                                                              │
                                              sum ─▶ tile_tok (M, T, C)
                                                              │
                            × move_tile_mask, mean over real tiles
                                                              │
                                                     tile_pool (M, C)
                                                              │
 move_scalars (M, 3) ─▶ Linear 3→C → ReLU → Linear C→C        │
                                     │                        │
                             scalar_feat (M, C) ──────⊕───────┘
                                                      │
                                          fuse: Linear 2C→C
                                                      │
                                                   e (M, C)
```

`move_scalars = [resultant_score_diff, tiles/7, is_play]`; letters are A..Z with
a separate blank flag, so a natural tile and its blank twin share letter
semantics. Layout owned by
[move_set_encoder.h](../engine/include/training/move_set_encoder.h).

### 3b. Cross-attention and head

```
   e (M, C)                                 board (P, 225, C)
      │                                            │
   scatter by (pos_id, rank-within-position)       │
      │                                            │
      ▼                                            │
 queries (P, maxK, C)                              │
      │                                            │
      │        ┌───────────────────────────────────┴──────────┐
      └───────▶│ MultiheadAttention (num_heads = 4)           │
       Q       │   K = V = board          need_weights=False  │
               └───────────────────┬──────────────────────────┘
                                   │ (P, maxK, C)
                    gather by (pos_id, rank)
                                   │
                          attended (M, C) ──────⊕────── g[move_pos_id] (M, 3C)
                                                │
                                        head_in (M, 4C)
                                                │
                                   Linear 4C→C → ReLU → Linear C→5
                                                │
                     ┌──────────────────────────┼────────────────┐
                     ▼                          ▼                ▼
                 out[:, 0:3]                out[:, 3]        out[:, 4]
                 wld (M, 3) logits          sd_mean       softplus + 1e-3
                                                 └───────⊕────────┘
                                                         │
                                             score_diff (M, 2)
```

Grouping the queries by position keeps one K/V copy per board, so attention's
`W_k`/`W_v` projections are amortized across candidates the same way the trunk
is.

### Losses (distillation from the position-eval teacher)

| Head | Target | Loss | Weight |
|------|--------|------|--------|
| `wld` | teacher probabilities (M, 3) | soft cross-entropy | 1 |
| `score_diff[:,0]` | teacher mean | Huber (δ=10) | `lambda_sd` = 0.004 |
| `score_diff[:,1]` | teacher std | Huber (δ=10) | `lambda_sd` = 0.004 |

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
| Heads | wld, score_diff, 4 placement masks | wld, score_diff |
| Supervision | game outcomes / observed spread | teacher readouts (`.mset` sidecar) |
| ONNX outputs | `wld`, `score_diff`, 4 mask names | `wld`, `score_diff` |
