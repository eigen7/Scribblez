# DeepBot phase 2: the belief-input leaf

**Status: proposed 2026-11-25, plan-reviewed; PR 1 built.** This is phase 2 of
[deep_bot.md](deep_bot.md#phases) sliced into pull requests. What each
measures shapes the next; no pass or kill thresholds are fixed in advance.

**Decision.** Four pull requests, in order:

1. **The `.belief` sidecar.** At every turn a leaf row is made from, it holds
   the mover's belief about the opponent's leave and the summaries version 3
   reads. deep_bot_leaf_eval's trainer writes it when a generation arrives,
   under the tag's pinned leave inference model.
2. **Input format version 3, known mode only.** The new row layout, in both of
   its encoders: the CPU encoder and the copy the GPU kernel uses to serve
   DeepBot. The model uses the new per-tile inputs, and DeepBot plays on
   version 3 as it plays now.
3. **Belief mode.** The decoder reads the sidecar and makes a fraction of the
   rows in belief mode. A held-out read compares belief mode against
   DeepBot's average over 16 drawn leaves.
4. **The run**: a version 3 leaf network, DeepBot's generation 0.

**Not in this phase:**
- **Deleting the GPU hasty-rollout path.** It is what keeps a fast leaf fed
  (see [Serving](#serving-why-the-device-encoder-stays)), so this phase keeps
  it.
- **Beliefs for deep_bot_generation's games.** There DeepBot plays, and it
  records the belief it acted on (phase 3).

## Serving: why the device encoder stays

DeepBot's leaf runs through the GPU hasty-rollout backend
(`engine/include/gpu/rollout_backend.h`). It sends each leaf to the GPU as a
compact state (`rollout_abi::LeafState`), and a kernel expands it into the
model's row (`engine/src/gpu/rollout_kernel.cu`, checked against the CPU
encoder by `gpu_kernel_tests`).

A row encoded on the CPU is (95 × 225 + 94) float32s, about 86 KB:
- at the width-64 transformer leaf's roughly 62K evaluations a second, that
  is about 5.3 GB/s to the GPU;
- at the 32×4 conv leaf's roughly 540K, it is about 46 GB/s, beyond a PCIe 4.0
  x16 link.

A CPU-encoded path would also make the game thread that leads a batch encode
every game's leaves itself, in series with the copy and the network call.
So version 3 lands in both encoders, and the kernel test pins them together.

## 1. The `.belief` sidecar

**What it holds**, per game and per turn of the leave inference rows'
region (the turns that begin with tiles in the bag):
- the leave inference model's 29 parameters (`belief::LeaveParams`) for the
  mover's belief about the opponent's kept leave;
- the leave's length, and its `LeaveRowCause`:
  - `valid`: the model's belief;
  - `no_opponent_move`, `empty_leave`: the leave is known: nothing yet, or
    empty;
  - `pass`: the opponent kept all seven tiles, and the leave table has no
    seven-tile leaves. Expected counts are seven times each tile's share of
    the pool, and the value fields are zero with the cause saying why.

  The leaf's rows (`BlockDecoder::row_turns` for kLeafEval) go on past the
  region, but there the bag is empty and the opponent holds every unseen tile.
  The belief is certain, so a belief-mode row there reads that rack as known
  mode does, and the sidecar holds no entry for it. That is why the plan's
  separate cause with a `determined` value was not needed.
- the summaries version 3 reads: each tile's expected count, and the leave
  value's mean and standard deviation, from one `LeaveSampler` pass.
  `LeaveMoments` gains the expected counts.
- **for held-out games only**, 16 leaves drawn as DeepBot draws them
  (stratified quantiles with the turn's jitter, `opp_draws`). They are drawn
  from the leave inference row's own leave pool, which can differ from the
  unseen tiles by the observer's last exchange. The held-out read in PR 3 then
  needs no sampler pass.

**Why the summaries are stored**: one pass takes about 21 ms at length 6,
about 1 ms at length 5, and under 0.2 ms below that (measured through the
FFI). Once per turn when a generation arrives is affordable. Once per row per
epoch in the data loader is not.

**Format**: float16 parameters and value moments, and expected counts as
8-bit fractions of the leave's length: 91 bytes a turn
(`engine/include/data/belief_log.h`). On v8's 1000 held-out games it is
1.8 MB, so about 37 MB for a generation of 20,000 games. That sits beside
about 12 MB of `.slog` and about as much `.forgone`, so a four-generation
window is about 250 MB. That is within `SlogDataset`'s 512 MiB budget, though
the budget is measured, not assumed. The 16 stored draws of the held-out set
add 112 bytes a turn.

**Precision**, measured on those games under leave_inference v1's epoch 292
against float32 parameters and double moments, over 15,492 modelled turns:
- value mean: at most 0.015 points off, 0.0009 on average, against an average
  magnitude of 5.1;
- value SD: at most 0.008 points off, against an average of 8.6;
- expected counts: at most 0.012 tiles off, the 8-bit bound of length / 510.

The stored parameters are float16. A sampler rebuilt from them has the same
moments to within 0.03 points, but it moves 6% of the quantile draws to a
neighbouring leave (up to 94% at a long leave, whose thousands of leaves each
hold little mass). That is why the held-out draws are stored rather than
redrawn from the parameters.

**Header**: a magic, a format version, the leave model's identity (hash, tag,
generation, format version), the leave table's content hash, and the `.slog`
binding `.forgone` uses (`forgone_slog_binding`).

**Writer**: `belief_sidecar_tool`. It reads a `.slog` and its `.forgone`,
builds each turn's leave inference row as training does
(`belief::SlogLeaveBeliefs`), runs the model through TensorRT, and writes the
sidecar. It takes about 3.5 s per 1000 games on 16 threads, so about 70 s
per generation, and about 0.45 GiB of GPU memory with its plan cached.

**Where it runs**: in deep_bot_leaf_eval's trainer, which has a GPU and owns
the model's identity. When a generation is ready, before training on it, it
runs `forgone_play_tool` and then `belief_sidecar_tool` over the generation's
`.slog` files. It does the same for the held-out games it plays, after
`leave_inference/holdout.ensure` (`deep_bot_leaf_eval/beliefs.py`). A
generation whose `.slog` files all have their `.belief` is skipped, so a
restart or a data home seeded from the controller's copy writes only what is
missing. The generators stay CPU-only and need no leave model.

**Changes to the workload:**
- a leave model parameter independent of the matches, pinned at task creation
  (`match_leave_inference_tag` was pinned only while matches were on; it is
  now `leave_inference_tag`, required);
- an `inputs` hook on the train role for that export;
- `.belief` left out of the held-out set's content hash, since TensorRT's
  bytes vary across GPUs.

**This departs from [deep_bot.md](deep_bot.md#the-belief-input-leaf) as
written,** which has the trainer recompute beliefs under its current leave
model; that PR updates it. For deep_bot_leaf_eval the leave model is pinned,
so the belief at training and at play is the same model's. For DeepBot's own
games, the decision recorded on 2026-11-25 is that DeepBot records the belief
it acted on (phase 3). Rows then carry their own release's belief, not the
newest. Within a training window spanning a few releases that gap is small,
and phase 5 measures it.

## 2. Input format version 3, known mode only

**The row** keeps each per-tile block contiguous, so the model's per-tile
registers read them together:
- **Per tile (27 each)**:
  - my leave's counts;
  - the opponent's counts: concrete in known mode, expected in belief mode;
  - the tiles unseen from my side, replacing the pool's counts;
  - the belief's per-tile tilt (`theta`), zero in known mode.
- **Global**:
  - the score difference and my leave's static value;
  - the opponent's leave value mean and standard deviation;
  - the belief's `a` and `b`;
  - the opponent's leave length, the mode flag, the pool size, and the
    bag-after-refills buckets (9).
- **Spatial**: the board and the cross-checks. The join planes are removed
  (decided 2026-11-25, without an ablation). They made up for a conv trunk's
  weakness at joining a square's cross-check letters with who holds them, and
  the transformer can make that join from the cross-checks and the counts.
  Removing them takes their code out of both encoders, and
  `DeepBotLeafEvalParams.join_planes` goes with them.

`kLeafEvalFormatVersion` becomes 3, and exports, checkpoints and the engine's
gate follow it.

**The encoders**: `LeafEvalInput` takes the opponent's side either as a
concrete leave or as a belief record. In this PR only concrete leaves reach
it. The device encoder takes the same change through `LeafState`, and
`gpu_kernel_tests` gains a case for the version 3 rows.

**The model**: the transformer's tile registers read, per tile, my count, the
opponent's count, the unseen count and the tilt, each with its zero flag.
That widens the register projection from `2 × NUM_SEATS`. An expected count
is zero only where the pool has none of the tile, so its zero flag means
"absent from the pool" in belief mode. The conv trunk reads the same blocks as
features.

**Also changed:**
- **`evaluation.py`**: it reads the pool from its slot, which version 3 gives
  to the unseen tiles. It computes the pool as the unseen tiles less the
  opponent's leave length instead, and a test pins `refills_empty_bag` on a
  known version 3 row.
- **`gpu_parity`**: its leaf fixture moves to version 3.

**DeepBot at play**: unchanged in behavior. It plays known mode with drawn
leaves, now on version 3 rows.

**What stops working**: every version 2 export. deep_bot_leaf_eval's tag `v8`
and its exports are deleted. No deep_bot_generation tag remains (`release1`
was deleted on 2026-11-25). Its seeds come from a deep_bot_leaf_eval tag, so
the next one seeds from PR 4's run. `leave_inference` `v1` and the
leaf_eval_hasty_rollouts tags read no leaf export and are unaffected.

## 3. Belief mode

**The loader**: `DataFile` holds a set of sidecars, not `.forgone` alone. The
decoder's per-task question becomes which sidecars a task reads (kLeafEval:
`.belief`). `belief_fraction` (a trainer parameter, default 0.5, half and
half as decided) and the epoch
reach `decode_one` through `EpochConfig`. Each row is in belief mode with that
probability, drawn from a seed fixed by the game, the turn and the epoch.

**The held-out read**: through `decode_rows`, which addresses positions
directly, with a mode argument: belief, known, or drawn *i* (the sidecar's
*i*-th stored leave). On a fixed subsample of held-out positions, three log
losses of the mover's expected score:
- belief mode;
- known mode with the true leave, the floor;
- the mean over the 16 drawn rows, DeepBot's average today.

Belief mode against that average is phase 2's main measurement.

## 4. The run

One deep_bot_leaf_eval tag on version 3, under the pinned leave inference
model. It reports PR 3's held-out read and the existing held-out metrics, and
becomes DeepBot's generation 0 leaf. Phase 0's throughput measurement runs on
it. David launches it.
