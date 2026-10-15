# DeepBot v0: building the leave inference model

**Status: proposed 2026-10-15; not built.** The build plan for the first
half of [deep_bot_v0.md](deep_bot_v0.md): the leave inference model `I_0`
and its `lim` workload, plus the v0 win table. The model is designed in
[deep_bot.md](deep_bot.md#the-leave-inference-model). The v0 choices (data
source, workload shape, eval metrics) are in
[deep_bot_v0.md](deep_bot_v0.md#workload-1-lim). This document turns them
into code: what each piece is, where it lives, how its artifacts are
versioned, and how the work splits into PRs. Measurements inform the next
step; no thresholds are fixed in advance.

Two decisions here go beyond deep_bot_v0.md:

- **The win table is a side product of the `lim` workload.** The trainer
  refits it from its window every generation and delivers it beside the
  LIM's ONNX ([The win table](#the-win-table-a-side-product)).
- **Every artifact carries a format version,** distinct from the generation
  that produced it ([Versions](#versions)).

## Rows

A row is one turn, seen by the player to move (the **observer**) before they
move. The target is the opponent's **leave**, the tiles they kept at their
last move.

- **Target.** `binlog::opp_leave_from_replay(g, t, opp_rack_now)` already
  computes the leave: the opponent's rack minus the tiles they drew after
  their last move. `PositionEncoder::make_context` already stores it as
  `ctx.opp_known_leave`.
- **Excluded rows.** A row is dropped when the leave's distribution does not
  depend on the model. A leave of length 0 (a bingo) is certain. A leave of
  length 7 (turn 0, or a pass) is the prior alone, and there is no 7-tile
  table. The trainer reports the share of dropped rows. Every other row is in
  the eligible region, which ends where the bag empties.
- **No random openings.** `random_opening_mean` defaults to 0 for `lim`.
  A random ply in the history, or as the move that set the target leave, is
  not the policy the LIM models. The eligible region then starts at turn 0,
  and every move a row reads was played by HastyBot.
- **Hidden leaves only.** The LIM's target is visible in face-up games, so
  the trainer refuses any `.slog` whose header has `kFlagFaceUpLeaves` set.
  Today, nothing checks that flag on the DataLoader path.

## Engine work

### `LimInputEncoder`

The encoder lives in `engine/include/encoding/lim_input_encoder.h`. It is a
standalone encoder, like `MaxMovePerLaneInputEncoder`, so the position
encoder's `InputEncodingSpec` and its version are untouched.

**Spatial planes (15 × 15):**

| Block | Planes |
|---|---|
| Current board (`BoardPlanes`) | 31 |
| Current cross-checks, horizontal and vertical | 52 |
| Per history move (×5): squares it placed tiles on | 1 |
| Per history move (×5): cross-checks before it was played | 52 |

That is 348 planes. The board and cross-check helpers are
`BoardPlanes::encode`, and `encode_cross_check_planes` and
`encode_placement_plane`, which sit in an anonymous namespace in
`game_state_encoder.cpp` today. They move to `encoding/board_planes.h`, so
the LIM and the position encoder share one convention.

**Scalars:**

- **The current position:** the observer's rack counts (27), the unseen pool
  (the position encoder's thermometer helper), the score differential, the
  bag count, and the length of the leave being predicted (a one-hot over
  1–6).
- **Per history move:**
  - a presence flag (missing early in the game, and then the move's other
    floats are zero);
  - its type (placement, exchange or pass);
  - the tiles it placed (27 counts);
  - the number of tiles it played or exchanged;
  - its score;
  - the bag count before it.

**History slots** are in fixed order back from the current turn: the
opponent's last move, then the observer's, then alternating, for 3 opponent
moves and 2 of the observer's. Players alternate, so a slot's index fixes
whose move it is.

**An opponent's exchange shows its count only.** The tiles they returned are
hidden from the observer, so for an opponent exchange the 27 tile counts stay
zero. The observer's own exchanges show their tiles. The opponent's drawn
tiles never reach the input. One encoder test checks this by encoding the
same public history with different hidden tiles and requiring identical
rows.

**Pre-move cross-checks come from the replay.** `GameStateEncoder` keeps only
each player's last move, so the LIM task snapshots during the replay loop
(`PositionEncoder::replay_to_sampled`). Before applying each of the last 5
moves, it records the board's cross-check planes (after
`ensure_movegen_caches`) and the move. That costs 5 cross-check computations
per row.

**Symmetry.** The DataLoader transposes a random half of the replayed
states. The history snapshots are boards too, so the task transposes them
with the same `Board::transpose()` before encoding. As elsewhere, no encoder
knows about the augmentation.

### The target, the task and the DataLoader mode

- **`OppLeaveTarget`:** 7 sorted tile indices (27 letters with the blank at
  26, padded with −1), plus the length. It is the LIM's only target.
- **`LimTask`:** `encode_row` writes the LIM inputs and the target, and marks
  excluded rows so the loader skips them.
- **`DecodeTask::kLim`:** a third DataLoader task code, wired as
  `kMaxMovePerLane` was: the `BlockDecoder` branch, the row-size getters in
  `data_loader.h`, the `dl_new` mapping, shape queries in the FFI, and
  `"lim": 2` in `NativeDataLoader._TASK_CODES`.

### The leave table

`belief::LeaveTable` holds, for each length from 1 to 6, every leave drawable
from a full bag, in `walk_multisets` order. Each row has 27 counts, a 27-bit
presence mask, and `v(ℓ)` from `HastyEquity::leave_value`. The rows come
from `belief::enumerate_leaves(TILE_COUNTS, k)`.

**It is built in memory, not stored as a file.** Building it takes one
enumeration and one KLV lookup per row, about a second for all 6 lengths.
The trainer gets it as numpy arrays through one FFI call, and the engine
sampler builds the same table at startup. Both come from one C++ function,
so the row order that sampling's pairing and the parity tests depend on
cannot drift between them. The table's content is fixed by the leave values
it reads. The LIM records which leave values those were
([Versions](#versions)), so no table file needs a format of its own.

### Sampling: Algorithm 1, on the CPU

`belief::LimSampler` takes the 29 floats and the observer's unseen pool.

- **One pass over the table** for the leave's length gives each row's log
  weight, the CDF in table order, and `log Z`. Each row costs about 28
  additions, as in the design.
- **`sample(u)`** binary-searches the CDF, then adds a uniform refill from
  the rest of the pool, drawn from the caller's seeded variates.
- **`log_prob(leave)`** returns the leave's log probability, which the parity
  tests use.

v0's only engine consumer is the survey tool: one query per position, about
300 samples. The GPU-batched sampler and Algorithm 2 serve the agent's
~2,000 queries per turn, so they are built when Bob's subtree is
([deep_bot.md](deep_bot.md#build-order), step 4). The CPU sampler is their
parity reference.

### Serving

- **`LimSpec`** in `nn/model_specs.h`:
  - `kGraph = "lim"`, and `kAcceptUntaggedGraph = false`;
  - one input row per position (`input_spatial`, `input_scalar`), and one
    output `lim_params` of 29 floats;
  - `kVersions = {{"lim_format_version", kLimFormatVersion, -1}}`.
- **`LimNet`** drives `NeuralNet<LimSpec>` directly, as
  `move_proposal_nets` does. TrtEvalService's batching earns nothing at one
  call per position.
- **At load,** `LimNet` also compares the model's `leave_values` entry with
  the hash of the KLV that `HastyEquity` loaded. A mismatch throws: a model
  trained against one set of leave values reads its tilt `a` and `b` on
  another.

## Python work: `py/scribblez/lim/`

| File | Contents |
|---|---|
| `model.py` | The encoder (a conv tower over the 348 planes, an MLP over the scalars, fused into a context vector) and the heads |
| `leave_table.py` | The FFI-built table as device tensors, one per length |
| `loss.py` | The tilted-prior NLL ([below](#the-loss)) |
| `trainer.py` | The generational loop, the held-out eval, and win-table fitting |
| `win_table.py` | The fit and the file writer |
| `onnx_export.py` | Export, stamped through `onnx_export_util` |

The conv tower reuses `spatial_trunk.py` if its constructor takes the plane
count. Otherwise it is a small local tower. The input channel count is the
only difference.

### The loss

The tilted prior is an exponential family over the table's rows:

`log p(ℓ) = base(ℓ; pool) + η · T(ℓ) − log Z(η)`

where `base` is the hypergeometric log prior, `η = (θ, a, b)` holds the 29
floats, and `T(ℓ) = (mask, v, −v²)`. The NLL's gradient in `η` is
`E_p[T] − T(ℓ_true)`, so:

- **One streaming pass** over a length's table, in chunks of rows, computes
  `log Z` (a running log-sum-exp) and `E_p[T]` together.
- **A custom `autograd.Function`** returns `log Z − log p̃(ℓ_true)` forward,
  and `(E_p[T] − T(ℓ_true)) · grad` backward. Memory stays at batch × chunk,
  never batch × 737,311.
- **`base` is a matmul.** Each row's counts are one-hot over (letter, count)
  pairs: 216 columns. The batch's `log C(n_l, c)` lookups form a `[B, 216]`
  matrix, and `base` is their product with the rows' one-hot matrix. Counts
  the pool cannot supply take a large negative constant. The pass runs in
  FP32.
- **Rows are grouped by leave length,** with one pass per length present in
  the batch.

The same pass yields the readouts the eval needs. `E_p[mask]` gives the
per-letter presence probabilities, and `E_p[v]` and `E_p[v²]` give the
implied mean and spread of the leave's value. A gradcheck against plain
autograd on a truncated table pins the custom backward.

**Units.** The head's raw outputs are scaled so that `v` enters in tens of
points while training. The export folds the scale back in, so the ONNX emits
`a` and `b` in the design's units (equity points), and `b ≥ 0` by a
softplus.

### The comparison head

`head = autoregressive` replaces the 29 floats with a decoder over the sorted
leave: a 27-way softmax per step, masked by the pool's remaining counts and by
sort order, on the same encoder. It is trained and evaluated in Python only,
and never exported. Its job is to measure what the compact head gives up.

### The workload

The `lim` workload is generational, on the `max_move_per_lane` and
`position_eval` pattern:

- **The spec:** `scheduler=TICK_FOR_TASK`, and a `train` role with
  `ingest=train_ingest:tick`, which is what makes the controller treat the
  tag as generational (`workers._home_trainer`).
- **Roles:**
  - `generate`: `selfplay_gen.generate` with `hasty_spec`, local or ssh, any
    number of workers.
  - `train`: a singleton on a GPU, hosting the data home.
- **Params:**
  - `games_per_generation`, `open_ahead` and `max_rows`, which the scheduler
    reads by name;
  - `window` and `turns_per_game`;
  - `random_opening_mean`, default 0;
  - `holdout_every` and `holdout_size`;
  - `head`;
  - the network and optimizer params, through profiles.
- **`face_up_leaves` is not a param.** `selfplay_gen.generate` takes it as a
  keyword argument instead of reading it off the params, and `lim` passes
  `False`.
- **Registration:** `WORKLOADS` in `workloads/__init__.py`, and
  `END_PARAMS["lim"] = {"max_rows"}` in `test_end_params.py`, which asserts
  that every workload declares its end params.
- **Dashboard:**
  - a `WORKLOAD_TABS.lim` entry (`LossTab`, a training tab on `lim_metrics`,
    and `InfoTab`);
  - a `FIGURES["lim_metrics"]` builder in `dashboard/api.py`;
  - `LIM_QUALITY` metric groups in `dashboard/plots.py`.

**The trainer's loop** is position_eval's:

1. Wait for the generation.
2. Train one epoch over the window.
3. Evaluate on the held-out set.
4. Refit the win table.
5. Export the ONNX.
6. Checkpoint.
7. Stage the record. The delivered outputs are the ONNX and the win table,
   with the record last.
8. Evict generations beyond the window.

### Held-out data

The generational path has no held-out mechanism today; `position_eval`
evaluates on a fixed external set.

- **Splitting.** The scheduler's `_fill` gains an optional per-workload
  predicate. `lim` routes a staged chunk to `holdout/` instead of the open
  generation when `crc32(stem) % holdout_every == 0`, the convention
  `pair_store.split_pair_stems` already uses.
- **The held-out set is frozen** once it holds `holdout_size` positions.
  Later held-out chunks are deleted. deep_bot_v0.md had the trainer read the
  most recent positions instead. v0's policy is fixed, so a frozen set loses
  nothing, keeps the curve comparable across generations, and lets the
  baselines be computed once. The set rolls when a later LIM trains on a
  changing policy.
- **Remote homes.** The held-out directory lives in the data home with the
  generations, so an ssh trainer sees it as it sees them.

### Eval

The metrics are deep_bot_v0.md's, plus a breakdown by leave length and the
win table's:

| Metric | Meaning |
|---|---|
| `eval_nll` | mean NLL of the true leave, in nats |
| `eval_nll_vs_uniform` | minus the hypergeometric prior's NLL |
| `eval_nll_vs_inferrer` | minus `RackInferrer`'s NLL ([below](#the-rackinferrer-baseline)) |
| `eval_nll_bag_*`, `eval_nll_len_*` | by bag band, and by leave length |
| `eval_letter_calibration` | mean absolute gap between `E_p[mask]` and actual presence, per letter |
| `eval_value_calibration` | `E_p[v]` against the true leave's value, and the coverage of its implied spread |
| `eval_wintable_logloss`, `…_macondo` | the refitted table's log loss on the held-out outcomes, and Macondo's bundled table's on the same features |

With `head = autoregressive`, the NLL metrics come from its decoder, and the
calibration metrics are left out.

### The `RackInferrer` baseline

`RackInferrer` reads only the opponent's last move. Its posterior is exact
when the leave space has at most 4,000 leaves. Above that, it scores 2,000
leaves drawn from the prior and can miss the true leave entirely, which would
make the NLL infinite. So the baseline uses a new
`RackInferrer::log_posterior(obs, leave, seed)`:

- **Enumerated:** the exact log posterior of the true leave.
- **Sampled:** `log prior(ℓ) + log L(ℓ) − log Ẑ`, with
  `Ẑ = mean L` over the prior draws. The draws come from the prior, so this
  estimates the normalizer directly. The baseline reports the enumerated
  subset separately, where it is exact.
- **Empty posterior:** for a pass, an empty bag or a bingo, the prior's NLL.
  Rows like these are mostly excluded already.

**An FFI entry point** computes, per held-out row, the uniform NLL, the
inferrer NLL and the enumerated flag. It needs a replay-to-
`OppMoveObservation` helper for `.slog`; only a GCG one exists. The cost is
about 0.1 ms per scored leave, up to 0.4 s per row: a few minutes over a
held-out set on a 28-thread machine. Results are cached beside the held-out
set ([Versions](#versions)), and the set is frozen, so this runs once per
tag.

## The win table: a side product

**Yes, the win table should be a side product.** The `lim` workload already
plays the games the table is fitted from, under the policy the LIM models.
deep_bot.md's generations pair them for that reason ("its games train the
next LIM and fit the next win table"). A separate fitting job would replay
the same corpus, with its own provenance to keep in step. The fit is a scan
over the window's outcomes: seconds of CPU against an epoch of training.
Each generation's table is fitted to the games the matching LIM generation
saw, so a consumer that pins one generation gets a matched pair.

**One feature definition.** The leaf is
`table(spread + v(own leave), tiles unseen)`, from the side to move.

- **One engine function, `win_table_feature(...)`,** computes that pair for
  the sim leaf. An FFI entry applies it to every turn of every game in a set
  of `.slog` files, so the fit and the lookup cannot disagree.
- **"Own leave"** is the tiles the side to move kept at their last move. At
  a sim leaf, the side to move has always moved within the sim. In the fit,
  a player's first turn has no previous leave and is skipped.
- **Every turn, not just the eligible region.** Sims stop past the point
  where the bag empties, so the fit covers turns down to a few tiles unseen.
  With no random openings, every turn's outcome comes from agent play.
  Positions where the game is over are not fitted, since the lookup handles
  them exactly.

**The fit** (`lim/win_table.py`) reads the window's turns. It bins them by
spread (1 point, clamped to ±300) and by tiles unseen (0–93), and pools
neighboring unseen counts while a cell is thin. It makes each unseen column
monotone in spread by isotonic regression, and then smooths across unseen
counts. The fit's details are tuned against the held-out log loss, beside
Macondo's table as the reference.

**Delivery.** The trainer writes `win_table.csv` each generation and
delivers it with that generation's ONNX. A consumer (the `psm` workload)
pins a `(lim tag, generation)` pair through `finalize`, as `move_set_eval`
pins its teacher. The table and the LIM it draws leaves from then come from
the same games. The `win_tables/vN.csv` files under the mount in
deep_bot_v0.md are replaced by these pinned outputs.

## Versions

Each artifact has a **format version**: an integer constant in the code that
writes and reads it. It is bumped whenever the artifact's layout or meaning
changes, for example when a feature is added to the LIM's input. A reader
accepts only its own version. The project keeps no backward compatibility,
so a bump makes older artifacts unloadable by design, and the error names
both versions.

A format version is distinct from the **generation**: which trainer step,
over which policy's games, produced the artifact. Generations are provenance,
recorded beside the format version but never checked as a format.

| Artifact | Format version | Provenance |
|---|---|---|
| LIM ONNX | metadata `lim_format_version` = `kLimFormatVersion`. It covers the encoder's inputs and the 29-float output layout. It is gated by `LimSpec::kVersions`, and a missing entry is rejected. Also stamped: `leave_values` (the KLV's hash), checked by `LimNet` | metadata `tag`, `generation`, `policy` (the generator's player spec), `head` |
| LIM checkpoint | `lim_format_version` and `leave_values` in the checkpoint config. Resuming with a different value is refused, as the evidence trainer refuses a mismatched `move_encoding_version` | `GenerationalState` |
| Win table | a `#`-prefixed header ahead of the CSV grid: `format_version=`, then the provenance lines. The format version covers the grid's shape and the feature definition, so a change to `win_table_feature` bumps it. A new `WinPctTable::load_versioned` checks the version and reuses the grid parser. `macondo_default()` keeps reading Macondo's unversioned file | header lines `tag`, `generation`, `policy`, `games`, `positions` |
| Held-out baseline cache | a `format_version` field, bumped when the baseline's computation changes | the held-out set's chunk list |
| Leave table | none: built in memory by one function and never stored. Its content is fixed by the `leave_values` the LIM records | — |

`kLimFormatVersion` and `kWinTableFormatVersion` are C++ constants, exported
to Python through `format_layout()["constants"]` as the `.slog` and `.mset`
versions are. Python stamps the engine's value and never keeps its own copy.
`.slog` is unchanged.

## PRs

Each PR has its own tests.

1. **Engine rows.**
   - A mechanical commit moves `encode_cross_check_planes` and
     `encode_placement_plane` into `board_planes.h`.
   - Then: `LimInputEncoder`, `OppLeaveTarget`, `LimTask`, `DecodeTask::kLim`
     with its FFI shapes, `LeaveTable` with its FFI export, and
     `kLimFormatVersion` in `format_layout`.
   - Tests:
     - the history slots line up with the replay;
     - pre-move cross-checks equal the board's cross-checks before the move;
     - hidden tiles cannot change a row;
     - transposed rows match transposed boards;
     - missing history encodes as zeros;
     - the target is the true leave;
     - the leave table has the stated row counts (25,254 at length 4,
       737,311 at length 6), a fixed order, and `HastyEquity`'s values.
2. **The LIM trainer and workload.**
   - The `lim` package with the tilted head, the streaming loss and its
     gradcheck.
   - The generational trainer, the held-out split in the scheduler, the eval
     with the uniform baseline and calibration, and the checkpoint version
     gate.
   - The face-up refusal, the workload spec and registration, and the
     dashboard tab and figure.
   - *Reports:* the loss pass's throughput by leave length (this sets the
     batch size and the chunking), and the first NLL-versus-uniform curve.
3. **The win table.**
   - `win_table_feature` and its FFI scan.
   - `lim/win_table.py`.
   - The versioned header and `WinPctTable::load_versioned`.
   - Per-generation refit, delivery and metrics.
   - *Reports:* held-out log loss against Macondo's table, and how much the
     refit moves between generations, which sets how many generations a
     consumer can reuse one table for.
4. **The `RackInferrer` baseline.**
   - `RackPosterior` lookup and `RackInferrer::log_posterior`.
   - The `.slog` observation helper, the FFI entry, the cache, and
     `eval_nll_vs_inferrer`.
5. **Serving.**
   - ONNX export with its metadata, `LimSpec`, `LimNet` with the
     `leave_values` check, and `LimSampler`.
   - Parity fixtures through `add_parity_fixture` (registered in
     `parity_fixtures_cleanup`): ONNX output against PyTorch, and
     `LimSampler`'s `log Z` and `log_prob` against the PyTorch table pass.
   - *Reports:* the CPU time per query for 6-tile leaves, which feeds the
     survey tool's cost estimate.
6. **The comparison head:** `head = autoregressive`.

PRs 3 and 4 depend only on PR 2's workload and can land in either order.
PR 3 unblocks the PSM's survey-tool work (deep_bot_v0.md, step 7), whose
uniform-belief mode needs only the win table. `--lim` needs PR 5.

*Measure, after PR 6:* NLL against the uniform prior and against
`RackInferrer`, by bag band and by leave length, for both heads. The gap
between the heads, concentrated where it is largest (6-tile leaves in
particular), says whether the 29-float output is enough for the agent's
beliefs or whether the tilt needs more terms. A new term is a
`lim_format_version` bump.

## Open questions

- **Leave tables as files.** The plan builds them in memory, so they need no
  format version. If they are wanted on disk (for inspection, or for a
  consumer with no FFI), they get a header with a format version and the
  `leave_values` hash, as the win table does.
- **The input size.** 348 planes is about 310 KB a row in FP32. If loader
  bandwidth binds in PR 2's throughput report, the history's cross-checks
  can shrink to their delta against the current cross-checks, which is zero
  away from the moves played since. That is a `lim_format_version` bump.
