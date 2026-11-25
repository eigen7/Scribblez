# DeepBot phase 2: the belief-input leaf

**Status: proposed 2026-11-25, plan-reviewed, not built.** This is phase 2 of
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
4. **The runs**: version 3 leaf networks with and without join planes, which
   are DeepBot's generation 0.

**Not in this phase:**
- **Deleting the GPU hasty-rollout path.** It is what keeps a fast leaf fed
  (see [Serving](#serving-why-the-device-encoder-stays)). Whether it goes
  waits for phase 3's measurement of the greedy player's throughput.
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

**What it holds**, per game and per leaf-row turn (`BlockDecoder::row_turns`
for kLeafEval):
- the leave inference model's 29 parameters (`belief::LeaveParams`) for the
  mover's belief about the opponent's kept leave;
- the leave's length, and a cause of its own:
  - `valid`: the model's belief;
  - `no_opponent_move`, `empty_leave`: the leave is known: nothing yet, or
    empty;
  - `pass`: the opponent kept all seven tiles, and the leave table has no
    seven-tile leaves. Expected counts are seven times each tile's share of
    the pool, and the value fields are zero with the cause saying why;
  - `determined`: past the leave inference rows' region, where the bag is
    empty and the unseen tiles are the opponent's rack.

  This is not `LeaveRowCause`, which lacks the last two.
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
8-bit fractions of the leave's length. That is about 95 bytes a turn, or about
42 MB for a generation of 20,000 games. That sits beside about 12 MB of
`.slog` and about as much `.forgone`, so a four-generation window is about
260 MB. That is within `SlogDataset`'s 512 MiB budget, though the budget is
measured, not assumed. The precision is checked against float32 on a
generation before the format is fixed.

**Header**: a magic, a format version, the leave model's identity (hash, tag,
generation, format version), the leave table's content hash, and the `.slog`
binding `.forgone` uses (`forgone_slog_binding`).

**Writer**: `belief_sidecar_tool`. It reads a `.slog` and its `.forgone`,
builds each turn's leave inference row as training does
(`belief::SlogLeaveBeliefs`), runs the model through TensorRT, and writes the
sidecar. For the turns outside that row's region it has its own path, since
`SlogLeaveBeliefs::query` throws there.

**Where it runs**: in deep_bot_leaf_eval's trainer, which has a GPU and owns
the model's identity. When a generation is ready, before training on it, it
runs `forgone_play_tool` and then `belief_sidecar_tool` over the generation's
`.slog` files. It does the same for the held-out games it plays
(`leave_inference/holdout.ensure`, with a belief-model argument). The
generators stay CPU-only and need no leave model.

**Changes to the workload:**
- a leave model parameter independent of the matches, pinned at task creation
  (`match_leave_inference_tag` is pinned only while matches are on);
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
- **Spatial**: the board, the cross-checks, and join planes for my leave and
  the unseen tiles. The opponent's join planes are removed.
  `DeepBotLeafEvalParams.join_planes` already drops the join planes from what
  the model reads.

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
`.belief`). `belief_fraction` (a trainer parameter, default 0.5) and the epoch
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

## 4. The runs

Two deep_bot_leaf_eval tags on version 3, one with join planes and one
without, under the same pinned leave inference model. Each reports PR 3's
held-out read and the existing held-out metrics. The arm kept becomes
DeepBot's generation 0 leaf. Phase 0's throughput measurement runs on it.
David launches them.

## Open questions

- **`belief_fraction`**: whether half and half is right, or whether belief
  mode, which play will use, deserves more.
- **The sidecar's precision**: whether float16 and 8-bit fractions lose
  anything against float32 (checked in PR 1).
