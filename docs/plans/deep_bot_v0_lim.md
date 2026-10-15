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

- **The win table is a side product of the `lim` workload.** The fit is a
  standalone tool, and the trainer also runs it on its window every
  generation, delivering the table beside the LIM's ONNX ([The win
  table](#the-win-table-a-side-product)).
- **Every artifact carries a format version,** distinct from the generation
  that produced it ([Versions](#versions)).

## Rows

A row is one turn, seen by the player to move (the **observer**) before they
move. The target is the opponent's **leave**, the tiles they kept at their
last move.

- **The target.** `binlog::opp_leave_from_replay(g, t, opp_rack_now)`
  computes it: the opponent's rack minus the tiles they drew after their last
  move. It generalizes to `leave_from_replay(g, t, player, rack_now)`, which
  the win table's "own leave" also uses ([below](#the-win-table-a-side-product)),
  so the leave has one definition.
- **Rows without information.** Some rows' leave distributions do not depend
  on the model, and they carry no loss:
  - turn 0, where the opponent has not moved;
  - after a bingo, where the leave is empty and so certain;
  - after a pass, where the leave is the whole rack, drawn from the prior
    alone (there is no 7-tile table).

  The loader cannot drop them, because it plans each batch's rows from the
  `.slog` metadata before decoding them. So `OppLeaveTarget` carries a
  validity flag and the cause, and the loss masks invalid rows. Batch shapes
  stay fixed, which the compiled trainer needs. `max_rows` counts planned
  rows, as in position_eval. The eval reports the share of each cause.
- **No random openings.** `random_opening_mean` defaults to 0 for `lim`.
  A random ply in the history, or as the move that set the target leave, is
  not the policy the LIM models. The eligible region then starts at turn 0,
  and every move a row reads was played by HastyBot.
- **Hidden leaves only.** The LIM's target is visible in face-up games, so
  the trainer refuses any `.slog` whose header has `kFlagFaceUpLeaves` set.
  The header test is factored out of `binlog::load_pending_slogs`, which
  already refuses face-up files for five tools, and not written a third time.

## Engine work

### `LimInputEncoder`

The encoder lives in `engine/include/encoding/lim_input_encoder.h`. It is a
standalone encoder, like `MaxMovePerLaneInputEncoder`, so the position
encoder's `InputEncodingSpec` and its version are untouched.

**Its input is a value, not a replay.** `LimInput` holds the current position
(the board, the observer's rack, the scores and the bag count) and a
`LimHistory`: 5 slots, each with a presence flag, the move, its score, the
bag count before it, and the cross-check planes as they stood before it was
played. The encoder reads only this value. The agent will build it for a
query whose most recent move is Alice's candidate ([deep_bot.md](deep_bot.md#conditioning-on-alices-candidate)),
without a `.slog`, so serving does not need a second encoder.

**Spatial planes (15 × 15):**

| Block | Planes |
|---|---|
| Current board (`BoardPlanes`) | 31 |
| Current cross-checks, horizontal and vertical | 52 |
| Per history move (×5): squares it placed tiles on | 1 |
| Per history move (×5): cross-checks before it was played | 52 |

That is 348 planes.

**Scalars:**

- **The current position:** the observer's rack counts (27), the unseen pool
  (the thermometer), the score differential, the bag count, and the length
  of the leave being predicted (a one-hot over 1–6).
- **Per history move:**
  - a presence flag (missing early in the game, and then the move's other
    floats are zero);
  - its type (placement, exchange or pass);
  - the tiles it placed (27 counts);
  - the number of tiles it played or exchanged;
  - its score;
  - the bag count before it.

**Shared helpers.** The position encoder's helpers sit in an anonymous
namespace in `game_state_encoder.cpp`. A mechanical commit moves every one
the LIM calls into `encoding/board_planes.h`: the cross-check planes, the
placement plane, the rack counts, the unseen-pool thermometer and the
score-differential scalar. The two encoders then share one convention.

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

### Producing `LimHistory` from a `.slog`

`GameStateEncoder` keeps only each player's last move, and a task's
`encode_row` sees only an `EncodeContext`, so the history has to be recorded
during the replay.

- **The recorder.** `PositionEncoder::replay_to_sampled` gains an optional
  recorder argument that only the LIM task passes; the other tasks' replay is
  unchanged. Before applying each of the last 5 moves, the recorder encodes
  the board's cross-check planes (after `ensure_movegen_caches`) straight
  into the slot, and stores the move, its score and the bag count. That is 5
  cross-check computations per row, and no Board copies.
- **Symmetry.** When the DataLoader transposes the replayed state, the
  history is transposed with it: each slot's move by `Move::transpose()`, and
  its cross-check planes by a spatial transpose that swaps the horizontal and
  vertical halves. As elsewhere, no encoder knows about the augmentation.
- **`EncodeContext`** gains a pointer to the history, which only the LIM
  task reads.

### The target, the task and the DataLoader mode

- **`OppLeaveTarget`:** 7 sorted tile indices (27 letters with the blank at
  26, padded with −1), the length, the validity flag and the cause.
- **`LimTask`:** `encode_row` writes the `LimInput` encoding and the target.
- **`DecodeTask::kLim`:** a third DataLoader task code, wired as
  `kMaxMovePerLane` was: the `BlockDecoder` branch, the row-size getters in
  `data_loader.h`, the `dl_new` mapping, shape queries in the FFI, and
  `"lim": 2` in `NativeDataLoader._TASK_CODES`.
- **Addressable decode.** `scribblez_decode_rows` today hard-codes the
  position task's row width. It takes the task too, so the held-out eval can
  decode a fixed list of `(game, turn)` rows and join per-row baselines on
  them.

### The leave table

`belief::LeaveTable` holds, for each length from 1 to 6, every leave drawable
from a full bag, in `walk_multisets` order. The rows come from
`belief::enumerate_leaves(TILE_COUNTS, k)`, unchanged. Each row stores 27
`uint8` counts, a 27-bit presence mask, and `v(ℓ)` from
`HastyEquity::leave_value`.

**It is built in memory, not stored as a file.** Building it takes one
enumeration and one KLV lookup per row, about a second for all 6 lengths.
The trainer gets it as numpy arrays through one FFI call, and the engine
sampler builds the same table at startup. Both come from one C++ function,
so the row order that sampling's pairing and the parity tests depend on
cannot drift between them. The table's content is fixed by the leave values
it reads, and the LIM records which leave values those were
([Versions](#versions)). So no table file needs a format of its own.

**The leave values' identity.** `HastyEquity` keeps no record of which KLV
it loaded. It gains a content hash of the KLV, set at load and exposed
through the FFI. Python stamps that value; it never hashes the file itself.

### Sampling: Algorithm 1, on the CPU

`belief::LimSampler` takes the 29 floats, the leave's length and the
observer's unseen pool.

- **One pass over the table** for that length gives each row's log weight,
  the CDF in table order, and `log Z`. Each row costs about 28 additions, as
  in the design.
- **`sample(u)`** binary-searches the CDF, then adds a uniform refill from
  the rest of the pool, drawn from the caller's seeded variates.
- **`log_prob(leave)`** returns the leave's log probability, which the parity
  tests use.

v0's only engine consumer is the survey tool: one query per position, about
300 samples. The GPU-batched sampler and Algorithm 2 serve the agent's
~2,000 queries per turn, so they are built with Bob's subtree
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
  `HastyEquity`'s hash. A mismatch throws: a model trained against one set of
  leave values reads its tilt `a` and `b` on another.

## Python work: `py/scribblez/lim/`

| File | Contents |
|---|---|
| `model.py` | The encoder (a conv tower over the 348 planes, an MLP over the scalars, fused into a context vector) and the heads |
| `leave_table.py` | The FFI-built table as device tensors, one per length |
| `loss.py` | The tilted-prior NLL ([below](#the-loss)) |
| `trainer.py` | The generational loop, the held-out eval, and the win-table refit |
| `holdout.py` | The held-out games and their baseline cache |
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
- **`base` per chunk.** The table stays on the device as `uint8` counts
  (about 20 MB at length 6). Each chunk's counts are expanded to a one-hot
  over (letter, count) pairs, 216 columns. `base` is then the product of
  that one-hot with the batch's `[B, 216]` matrix of `log C(n_l, c)` values.
  Counts the pool cannot supply take a large negative constant.
- **Full FP32.** The trainers set `float32_matmul_precision("high")`
  process-wide, which would run this matmul in TF32 and leave noise of
  roughly 0.01 nats in each row's NLL. The function runs its pass at highest
  precision and casts its inputs to FP32 (`custom_fwd(cast_inputs=float32)`),
  so neither TF32 nor autocast reaches it.
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
and never exported. The `head` param arrives with it.

It lands before serving. The gap between the two heads says whether the 29
floats are enough, or whether the tilt needs more terms. That decides the
output layout that serving bakes into `LimSpec`, `LimSampler` and the parity
fixtures. The 29-float head stays the design's choice. The comparison
measures what it gives up; it does not compete to replace it.

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
  - `hasty_temperature` (default 0, greedy) and `hasty_top_k`, which
    `hasty_spec` reads;
  - `window` and `turns_per_game`;
  - `random_opening_mean`, default 0;
  - `holdout_games` and `holdout_seed`;
  - the network and optimizer params, through profiles.
- **`face_up_leaves` is not a param.** `selfplay_gen.generate` takes it as a
  keyword argument instead of reading it off the params. `run_generate` (for
  `max_move_per_lane`) and `position_eval.run_generate` pass their own
  param, and `lim` passes `False`.
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

### Held-out data: regenerated, not stored

The generational path has no held-out mechanism today; `position_eval`
evaluates on a fixed external set.

**The held-out set is `holdout_games` games that the trainer plays itself at
startup,** with `run_games` at the fixed `holdout_seed`, HastyBot against
HastyBot, written to a scratch directory on its machine.

- **Reproducible.** `play_game` plays game `g` at seed `base + g`, and the
  bag is seeded from it. With the greedy policy, a fixed seed gives the same
  games on any home.
- **Disjoint from training.** Generators run with seed 0, which draws a fresh
  random base per chunk.
- **Nothing durable to lose.** A held-out directory in the data home would
  need adding to the controller's pull, seed and sweep sets. Without that, it
  is lost when an ssh trainer moves or a spot instance dies. Regeneration
  avoids it, and avoids any change to the scheduler.
- **Fixed for the run.** The same games every generation keep the curve
  comparable and let the baselines be computed once. v0's policy is fixed, so
  a fixed set loses nothing. A later LIM on a changing policy regenerates its
  set from its own policy.
- **Identified by a content hash** of the `.slog` files. The baseline cache
  is keyed by that hash. If a home's regeneration ever differs (say, across
  CPU architectures), the cache misses and is recomputed, and the set's new
  hash is recorded in the run's control events.

The eval decodes the held-out rows through the addressable `decode_rows`,
in a fixed order, so per-row baselines join on `(game, turn)`.

### Eval

The metrics are deep_bot_v0.md's, plus a breakdown by leave length and the
win table's:

| Metric | Meaning |
|---|---|
| `eval_nll` | mean NLL of the true leave over valid rows, in nats |
| `eval_nll_vs_uniform` | minus the hypergeometric prior's NLL |
| `eval_nll_vs_inferrer` | minus `RackInferrer`'s NLL, on rows where the opponent placed tiles ([below](#the-rackinferrer-baseline)) |
| `eval_nll_bag_*`, `eval_nll_len_*` | by bag band, and by leave length |
| `eval_nll_exchange` | on rows where the opponent exchanged; compared with the uniform prior only |
| `eval_invalid_*` | the share of rows without information, by cause |
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
`RackInferrer::log_posterior(obs, leave, seed)`, defined for placements:

- **Enumerated:** the exact log posterior of the true leave.
- **Sampled:** `log prior(ℓ) + log L(ℓ) − log Ẑ`, with
  `Ẑ = mean L` over the prior draws. The draws come from the prior, so this
  estimates the normalizer directly. The enumerated subset is also reported
  on its own, where the baseline is exact. Dropping the sampled path would
  leave mostly short leaves, and long leaves are where the LIM's evidence
  matters most.
- **Exchanges are excluded.** For an exchange, the inferrer's hypotheses are
  whole racks, so the true leave's posterior is a sum over every exchanged
  set it could have come with, each term a move generation. Exchange rows get
  their own bucket, `eval_nll_exchange`, against the uniform prior only.
- **Empty posterior:** for a pass, an empty bag or a bingo, the prior's NLL.
  These rows are already invalid.

**An FFI entry point** computes, per held-out row, the uniform NLL, the
inferrer NLL and the enumerated flag. Its `OppMoveObservation` comes from a
`.slog` helper built on `slog_position_simmer`'s replay, which already
recovers each position's opponent leave. The cost is about 0.1 ms per scored
leave, up to 0.4 s per row: a few minutes over a held-out set on a
28-thread machine, once per held-out hash.

## The win table: a side product

**Yes, the win table should be a side product.** The `lim` workload already
plays the games the table is fitted from, under the policy the LIM models.
deep_bot.md's generations pair them for that reason ("its games train the
next LIM and fit the next win table"). Each generation's table is fitted to
the games the matching LIM generation saw, so a consumer that pins one
generation gets a matched pair. The fit scans the window's outcomes, which
takes seconds of CPU against an epoch of training.

**The fit is also a standalone tool,** `py/scribblez/lim/win_table.py` with a
CLI over any set of `.slog` files. The workload calls the same function. So
the table does not wait on the LIM:

- it lands first, before any LIM code ([PRs](#prs));
- v0's first table can be fitted on an existing HastyBot hidden-leaves
  corpus;
- the PSM's survey-tool work, whose uniform-belief mode needs only the table,
  can start immediately.

**One feature definition.** The leaf is
`table(spread + v(own leave), tiles unseen)`, from the side to move.

- **One engine function, `win_table_feature(...)`,** computes that pair for
  the sim leaf. An FFI scan applies it to every turn of every game in a set
  of `.slog` files, through `BlockDecoder`'s game view and the replay, so the
  fit and the lookup cannot disagree.
- **"Own leave"** is `leave_from_replay` for the side to move: the tiles
  they kept at their last move. At a sim leaf, the side to move has always
  moved within the sim. In the fit, a player's first turn has no previous
  leave and is skipped.
- **Every turn, not just the eligible region.** Sims stop past the point
  where the bag empties, so the fit covers turns down to a few tiles unseen.
  With no random openings, every turn's outcome comes from agent play.
  Positions where the game is over are not fitted, since the lookup handles
  them exactly.

**The fit** bins turns by spread (1 point, clamped to ±300) and by tiles
unseen (0–93), and pools neighboring unseen counts while a cell is thin. It
makes each unseen column monotone in spread by isotonic regression, and then
smooths across unseen counts. The details are tuned against held-out log
loss, beside Macondo's table as the reference.

**The engine reader.** `WinPctTable::load` splits into a stream parser for
the grid and two file readers:

- `load_versioned` checks the header and parses the grid after it;
- the Macondo reader keeps reading Macondo's unversioned file for
  `macondo_default()`.

The simmers that consume a pinned table take it as a `const WinPctTable&` in
place of the `macondo_default()` singleton. That injection is the survey
tool's `--win-table` work (deep_bot_v0.md, step 7).

**Delivery.** The trainer writes each generation's table to
`TagPaths.win_table_path(gen)`, under `models/win_table/`. Files directly in
`models/` that are not model exports count as shared ONNX sidecars
(`TagPaths.onnx_sidecars`). Those are re-delivered every generation and
shipped with every match-eval model, so a per-generation table there would
be re-sent every generation. The subdirectory keeps the tables out, as the
evidence trainer keeps its companions out. A consumer (the `psm` workload)
pins a `(lim tag, generation)` pair through `finalize`, as `move_set_eval`
pins its teacher, and resolves the table next to `onnx_path(gen)`. The
`win_tables/vN.csv` files under the mount in deep_bot_v0.md are replaced by
these pinned outputs, and by the CLI's output for tables fitted outside a
tag.

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
| LIM ONNX | metadata `lim_format_version` = `kLimFormatVersion`, covering the encoder's inputs and the 29-float output layout. `LimSpec::kVersions` gates it and rejects a missing entry. Also stamped: `leave_values` (`HastyEquity`'s KLV hash), checked by `LimNet` | metadata `tag`, `generation`, `policy` (the `hasty_spec` string), `head` |
| LIM checkpoint | `lim_format_version` and `leave_values` in the checkpoint config. Resuming with a different value is refused, as the evidence trainer refuses a mismatched `move_encoding_version` | `GenerationalState` |
| Win table | a `#`-prefixed header ahead of the CSV grid: `format_version=`, then the provenance lines. The version covers the grid's shape and the feature definition, so a change to `win_table_feature` bumps it | header lines `tag` and `generation` (or the CLI's corpus), `policy`, `games`, `positions` |
| Held-out baseline cache | a `format_version` field, bumped when the baseline's computation changes | the held-out set's content hash |
| Leave table | none: built in memory by one function and never stored. Its content is fixed by the `leave_values` the LIM records | — |

`kLimFormatVersion` and `kWinTableFormatVersion` are C++ constants, exported
to Python through `format_layout()["constants"]` as the `.slog` and `.mset`
versions are. Python stamps the engine's value and never keeps its own copy.
`.slog` is unchanged.

## PRs

Each PR has its own tests.

1. **The win table.**
   - `leave_from_replay`, `win_table_feature` and its FFI scan.
   - `lim/win_table.py` with its CLI.
   - The versioned header, `kWinTableFormatVersion`, and `WinPctTable`'s
     stream parser with `load_versioned`.
   - v0's first table, fitted on an existing HastyBot hidden-leaves corpus.
   - *Reports:* held-out log loss against Macondo's table.
2. **Engine rows.**
   - A mechanical commit moves the shared encoder helpers into
     `board_planes.h`.
   - Then: `LimInput`/`LimHistory`, `LimInputEncoder`, the replay recorder
     and history transpose, `OppLeaveTarget` with validity, `LimTask`,
     `DecodeTask::kLim` with its FFI shapes, the task-aware `decode_rows`,
     and `kLimFormatVersion` in `format_layout`.
   - Tests:
     - the history slots line up with the replay;
     - pre-move cross-checks equal the board's cross-checks before the move;
     - hidden tiles cannot change a row;
     - transposed rows match transposed boards and moves, placement planes
       included;
     - missing history encodes as zeros;
     - the target is the true leave, and each invalid cause is flagged;
     - the position tasks' rows are byte-identical with and without the
       recorder change.
3. **The LIM trainer and workload.**
   - `LeaveTable` with its FFI export, and `HastyEquity`'s KLV hash.
   - The `lim` package with the tilted head, the streaming loss and its
     gradcheck.
   - The generational trainer, the regenerated held-out set, the eval with
     the uniform baseline and calibration, the checkpoint version gate, and
     the per-generation win-table refit and delivery.
   - The face-up refusal, the `generate` signature change, the workload spec
     and registration, and the dashboard tab and figure.
   - *Reports:* the loss pass's throughput by leave length and its device
     memory (this sets the batch size and the chunking), and the first
     NLL-versus-uniform curve.
4. **The comparison head:** `head = autoregressive`.
   - *Reports:* NLL for both heads, by leave length and bag band. This feeds
     PR 6's output layout.
5. **The `RackInferrer` baseline.**
   - `RackPosterior` lookup and `RackInferrer::log_posterior`.
   - The `.slog` observation helper, the FFI entry, the cache, and
     `eval_nll_vs_inferrer` and `eval_nll_exchange`.
6. **Serving.**
   - ONNX export with its metadata, `LimSpec`, `LimNet` with the
     `leave_values` check, and `LimSampler`.
   - Parity fixtures through `add_parity_fixture` (registered in
     `parity_fixtures_cleanup`): ONNX output against PyTorch, and
     `LimSampler`'s `log Z` and `log_prob` against the PyTorch table pass.
   - *Reports:* the CPU time per query for 6-tile leaves, which feeds the
     survey tool's cost estimate.

PR 1 depends on nothing else, and PR 2 needs only PR 1's `leave_from_replay`.
PRs 4 and 5 each need only PR 3, and land in either order. The PSM's
survey-tool work (deep_bot_v0.md, step 7) can start after PR 1; its `--lim`
flag needs PR 6.

*Measure, after PR 5:* NLL against the uniform prior and against
`RackInferrer`, by bag band and by leave length, for both heads. The gap
between the heads, concentrated where it is largest (6-tile leaves in
particular), says whether the 29-float output is enough for the agent's
beliefs or whether the tilt needs more terms before serving fixes the
layout.

## Open questions

- **Leave tables as files.** The plan builds them in memory, so they need no
  format version. If they are wanted on disk (for inspection, or for a
  consumer with no FFI), they get a header with a format version and the
  `leave_values` hash, as the win table does.
- **The input size.** 348 planes is about 310 KB a row in FP32. If loader
  bandwidth binds in PR 3's throughput report, the history's cross-checks
  can shrink to their delta against the current cross-checks, which is zero
  away from the moves played since. That is a `lim_format_version` bump.
