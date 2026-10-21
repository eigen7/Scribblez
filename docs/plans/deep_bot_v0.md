# DeepBot v0: the leave inference model and the policy-set model

**Status: proposed 2026-10-14, build plan added 2026-10-15; not built.** The
implementation plan for the first two models of [deep_bot.md](deep_bot.md),
each trained from HastyBot-vs-HastyBot games in standard Scrabble (hidden
leaves):

1. **[The leave inference model](deep_bot.md#the-leave-inference-model),**
   `I_0`.
2. **[The policy-set model](deep_bot.md#the-policy-set-model),** in
   both of its forms. Training both on the same data lets the measurements
   choose between them.

The design of each model is in [deep_bot.md](deep_bot.md). This document
covers what is specific to v0:
- the data source and the dashboard workloads;
- the engine and Python work, and where each piece lives;
- the win table and how it is fitted;
- how every artifact is versioned;
- the v0 choices of size and parameter;
- the measurements;
- the PRs.

Measurements inform the next step; no thresholds are fixed in advance.

The leave inference model comes first. Its labels are free (true leaves from
replay, no sims), so it is the quick way to prove the workload machinery, and
the policy-set model's label generation then draws opponent leaves from it.

Both models train on positions from the whole game except the endgame, with
no weighting by bag size.

## One workload per model

Each model is one dashboard workload that does everything: it generates data,
trains, slides its data window (deleting what falls out), and publishes eval
metrics after every generation. Both follow the generational shape of
[position_eval](../position_eval_workload.md) and reuse its machinery:

| Piece | Reused from |
|---|---|
| Spec and registration | `WorkloadSpec` / `RoleSpec` ([workloads/base.py](../../py/scribblez/workloads/base.py)), an entry in `WORKLOADS` |
| Self-play generation | `selfplay_gen.generate` with `face_up_leaves=False` |
| Staging, ingest into generations, gating | the generational scheduler and data home ([generational/scheduler.py](../../py/scribblez/generational/scheduler.py), [data_home.py](../../py/scribblez/generational/data_home.py)) |
| The window and eviction | `lifecycle.window_dirs`, `lifecycle.evict_beyond_window` ([generational/lifecycle.py](../../py/scribblez/generational/lifecycle.py)) |
| Records and metrics | `TrainRecorder` → `train_ingest` → `metrics` in `dashboard.db` ([generational/records.py](../../py/scribblez/generational/records.py)) |
| Dashboard | a `WORKLOAD_TABS` entry ([web/src/workloads.tsx](../../web/src/workloads.tsx)), plus one `FIGURES` builder per workload for its eval figure |

**What makes a tag generational.** The controller treats a tag as
generational when its spec has `scheduler=TICK_FOR_TASK` and its `train`
role sets `ingest=train_ingest:tick` (`workers._home_trainer`). The
scheduler and data home read `games_per_generation`, `open_ahead` and
`max_rows` off the params by name, so both workloads define them.

**Roles.**

- **`generate`:** parallel, local or ssh. Any number of workers.
- **`train`:** singleton, GPU. It hosts the data home, so the scheduler runs
  on its machine.

**The trainer's loop**, per generation:
1. wait for the generation;
2. train one epoch over the window;
3. evaluate on the held-out set;
4. export and checkpoint;
5. stage the record, delivering its outputs with the record last;
6. evict generations beyond the window.

**Held-out data** differs between the two workloads. Leave-inference games are
cheap, so the trainer regenerates its held-out set
([below](#held-out-data-regenerated)). Policy-set pools are expensive, so they
are held out as they are generated ([below](#held-out-pools)).

## Versions

Each artifact has a **format version**: an integer constant in the code that
writes and reads it. It is bumped whenever the artifact's layout or meaning
changes, for example when a feature is added to a model's input. A reader
accepts only its own version. The project keeps no backward compatibility,
so a bump makes older artifacts unloadable by design, and the error names
both versions.

A format version is distinct from the **generation**: which trainer step,
over which policy's games, produced the artifact. Generations are provenance,
recorded beside the format version but never checked as a format.

| Artifact | Format version | Provenance |
|---|---|---|
| Leave table (`LeaveValueArrays`) | none of its own: it is built in memory from the leave values and never stored. Its rows are fixed by the leave values (the KLV's content hash, which models record) and by its layout and order, which `leave_inference_format_version` covers | — |
| Win table | `kWinTableFormatVersion` in a `#`-prefixed header ahead of the CSV grid. It covers the grid's shape and the feature definition, so a change to `win_table_feature` bumps it | header lines: `tag` and `generation` (or the CLI's corpus), `policy`, `games`, `positions` |
| Leave inference ONNX | metadata `leave_inference_format_version`, covering the encoder's inputs and the 29-float output layout. `LeaveInferenceSpec::kVersions` gates it and rejects a missing entry. Also stamped: `leave_values`, the content hash of the KLV whose values it was trained against, checked at load; `leave_inference_public`, stamped by public-task exports (absent reads as 0) and refused at load when 1 | metadata `tag`, `generation`, `policy` (the `hasty_spec` string), `head` |
| Leave inference checkpoint | `leave_inference_format_version` and the `leave_values` hash in the checkpoint config. Resuming with a different value is refused, as the evidence trainer refuses a mismatched `move_encoding_version` | `GenerationalState` |
| Pool sidecar | `kPoolFormatVersion` in its header, after the magic | header: the win table's and the leave inference model's identities (tag, generation, format version), or uniform beliefs; `horizon_plies` |
| Policy-set ONNX and checkpoints | the existing `input_encoding_version` and `move_encoding_version`, plus `policy_set_format_version` for each form's set-conditioning stage and outputs | `tag`, `generation`, `form` |

Every format constant is a C++ constant, exported to Python through
`format_layout()["constants"]` as the `.slog` and `.mset` versions are.
Python stamps and checks the engine's value and never keeps its own copy.
`.slog` is unchanged.

## The win table

The policy-set model's label rollouts stop at a
[versioned win table](deep_bot.md#sims).

**It is a side product of the `leave_inference` workload.** That workload
already plays the games the table is fitted from, under the policy the leave
inference model learns. deep_bot.md's generations pair them for that reason:
"its games train the next leave inference model and fit the next win
table". The trainer refits the table from its window every generation and
delivers it beside that generation's ONNX. A consumer that pins one
generation gets a table and a model fitted on the same games. The fit scans
the window's outcomes, which takes seconds of CPU against an epoch of
training.

**The fit is also a standalone tool:** `win_table.py` in
`py/scribblez/leave_inference/`, with a CLI over any set of `.slog` files.
The workload calls the same function, so the table does not wait on the
model:

- It lands first ([Build order](#build-order)).
- v0's first table can be fitted on an existing HastyBot hidden-leaves
  corpus.
- The policy-set model's survey-tool work, whose uniform-belief mode needs
  only the table, can start immediately.

**One feature definition.** The leaf is
`table(spread + v(own leave), tiles unseen)`, from the side to move.

- **One engine function, `win_table_feature(...)`,** computes that pair for
  the sim leaf. An FFI scan applies it to every turn of every game in a set
  of `.slog` files, through `BlockDecoder`'s game view and the replay, so the
  fit and the lookup cannot disagree.
- **"Own leave"** is the tiles the side to move kept at their last move.
  `binlog::opp_leave_from_replay` generalizes to
  `leave_from_replay(g, t, player, rack_now)`, which both this feature and the
  leave inference target use, so the leave has one definition. At a sim
  leaf, the side to move has always moved within the sim. In the fit, a
  player's first turn has no previous leave and is skipped.
- **Every turn, not just the eligible region.** Sims stop past the point
  where the bag empties, so the fit covers turns down to a few tiles unseen.
  With no random openings, every turn's outcome comes from agent play.
  Positions where the game is over are not fitted, since the lookup handles
  them exactly.

**The fit** bins turns by spread (1 point, clamped to ±300) and by tiles
unseen (0–93), and pools neighboring unseen counts while a cell is thin. It
makes each unseen column monotone in spread by isotonic regression, and then
smooths across unseen counts. The details are tuned against held-out log
loss. Macondo's bundled table is the reference.

**The engine reader.** `WinPctTable::load` splits into a stream parser for
the grid and two file readers:
- `load_versioned` checks the header and parses the grid after it;
- the Macondo reader keeps reading Macondo's unversioned file for
  `macondo_default()`.

The simmers that consume a pinned table take it as a `const WinPctTable&` in
place of the `macondo_default()` singleton.

**Delivery.** The trainer writes each generation's table to
`TagPaths.win_table_path(gen)`, under `models/win_table/`. Files directly in
`models/` that are not model exports count as shared ONNX sidecars
(`TagPaths.onnx_sidecars`). Those are re-delivered every generation and
shipped with every match-eval model. The subdirectory keeps the tables out
of that set, as the evidence trainer keeps its companions out.

## Workload 1: `leave_inference`

### Rows

A row is one turn, seen by the player to move (the **observer**) before they
move. The target is the opponent's leave, `leave_from_replay` for the
opponent. `turns_per_game` sets how many turns of each game a pass draws, as
in position_eval. Its default, 5, trains each of a game's ~20 eligible turns
exactly once over the 4 generations it spends in the window. Unlike
position_eval, a game's rows have different targets (each turn's leave), so
drawing several per game does not invite memorizing a shared label; the
shuffle spreads them across batches.

- **Rows without information.** Some rows' leave distributions do not depend
  on the model, and they carry no loss:
  - turn 0, where the opponent has not moved;
  - after a bingo, where the leave is empty and so certain;
  - after a pass, where the leave is the whole rack, drawn from the prior
    alone (there is no 7-tile table).

  The loader cannot drop them, because it plans each batch's rows from the
  `.slog` metadata before decoding them. So the target carries a validity
  flag and the cause, and the loss masks invalid rows. Batch shapes stay
  fixed, which the compiled trainer needs. `max_rows` counts planned rows, as
  in position_eval. The eval reports the share of each cause.
- **No random openings.** The workload passes `random_opening_mean=0` to
  `selfplay_gen.generate` and has no param for it. A random ply in the
  history, or as the move that set the target leave, is not the policy the
  model learns. The eligible region then starts at turn 0, and every move a
  row reads was played by HastyBot.
- **Hidden leaves only.** The target is visible in face-up games, so the
  trainer refuses any `.slog` whose header has `kFlagFaceUpLeaves` set. The
  header test is factored out of `binlog::load_pending_slogs`, which already
  refuses face-up files for five tools, and is not written a third time.

### Engine work

#### `LeaveInferenceInputEncoder`

The encoder lives in `engine/include/encoding/leave_inference_input_encoder.h`.
It is a standalone encoder, like `MaxMovePerLaneInputEncoder`, so the
position encoder's `InputEncodingSpec` and its version are untouched.

**Its input is a value, not a replay.** `LeaveInferenceInput` holds:
- the current position: the board, the observer's rack, the scores and the
  bag count;
- a `LeaveHistory` of the opponent's last 3 moves, and the tiles the
  observer exchanged on each of their last two turns (empty if they did
  not). Each
  slot has a presence flag, the move, its score, the bag count before it, the
  cross-check planes as they stood before it was played, its forgone plays,
  and which tile kinds its leave could hold.

The encoder reads only this value. The agent will build the same value for a
query whose most recent move is Alice's candidate
([deep_bot.md](deep_bot.md#the-leave-inference-model)), without a `.slog`,
so serving needs no second encoder.

**Spatial planes (15 × 15):**

| Block | Planes |
|---|---|
| Current board (`BoardPlanes`) | 31 |
| Current cross-checks, horizontal and vertical | 52 |
| Per history move (×3): squares it placed tiles on | 1 |
| Per history move (×3): cross-checks before it was played | 52 |

That is 242 planes.

**Scalars:**

- **The current position:**
  - the observer's rack counts (27);
  - the unseen pool, as a thermometer;
  - the score differential and the bag count;
  - the length of the leave being predicted, one-hot over 1–6;
  - the tiles the observer exchanged on their previous turn, then on the
    turn before (27 counts each).
- **Per history move:**
  - a presence flag (missing early in the game, and then the move's other
    floats are zero);
  - its type (placement, exchange or pass);
  - the tiles it placed (27 counts);
  - the number of tiles it played or exchanged;
  - its score;
  - the bag count before it;
  - its [forgone plays](#forgone-plays), two floats per tile kind.

**Shared helpers.** The position encoder's helpers sit in an anonymous
namespace in `game_state_encoder.cpp`. A mechanical commit moves every one
this encoder calls into `encoding/board_planes.h`: the cross-check planes,
the placement plane, the rack counts, the unseen-pool thermometer and the
score-differential scalar. The two encoders then share one convention.

**History slots** are the opponent's last 3 moves, most recent first. The
observer's own moves are left out: the opponent's rack was drawn before each
of them, so a play or a pass by the observer tells nothing about it that the
board does not. An exchange does. On the observer's previous turn, the tiles
it returned are unseen but cannot be in the leave; that reaches the row
twice, as the exchanged tiles' counts and as the [leave
pool](#the-leave-pool). On the turn before, the opponent has drawn since, so
the leave holds at most as many of those tiles as they drew; that reaches
the row as the second block of counts.

**An opponent's exchange shows its count only.** The tiles they returned are
hidden from the observer, so for an opponent exchange the 27 tile counts stay
zero. The observer's own exchange shows its tiles. The opponent's drawn
tiles never reach the input. One encoder test checks this by encoding the
same public history with different hidden tiles and requiring identical
rows.

#### Producing `LeaveHistory` from a `.slog`

`GameStateEncoder` keeps only each player's last move, and a task's
`encode_row` sees only an `EncodeContext`, so the history is recorded during
the replay.

- **The recorder.** `PositionEncoder::replay_to_sampled` gains an optional
  recorder argument, passed only by the leave inference task, so the other
  tasks' replay is unchanged. Before applying each of the opponent's last 3
  moves, the recorder:
  - encodes the board's cross-check planes (after `ensure_movegen_caches`)
    straight into the slot;
  - stores the move, its score and the bag count.

  That is 3 cross-check computations per row, and no Board copies.
- **Symmetry.** When the DataLoader transposes the replayed state, the
  history is transposed with it. Each slot's move goes through
  `Move::transpose()`. Its cross-check planes get a spatial transpose that
  also swaps the horizontal and vertical halves. As elsewhere, no encoder
  knows about the augmentation.
- **`EncodeContext`** gains a pointer to the history, which only the leave
  inference task reads.

#### The leave pool

The leave was kept from a rack the opponent drew before the observer's
previous move. If that move was an exchange, its tiles went back to the bag
afterwards: they are unseen, but cannot be in the leave. So the leave pool,
the pool the prior is over (the target's `leave_pool`, which the loss and the
uniform baseline read), is the unseen pool less those tiles (`leave_pool` in
`leave_inference_input_encoder.h`). The model cannot make this correction
itself: its letter weights act on presence, and the multiplicities come from
the pool it is given.

- **The refill differs.** The opponent's draw after their move came from a
  bag that held the exchanged tiles, so a sampler draws the leave from the
  leave pool and the refill from the unseen pool less the leave.
- **Older exchanges cap rather than shrink.** After an exchange two of the
  observer's turns back, the opponent has drawn since, so its tiles cap the
  leave (at most as many as they drew) rather than leave its pool. A
  hypergeometric pool cannot express that, so the pool keeps them, and the
  row's second block of exchange counts leaves the cap to the network, which
  sees how many tiles each opponent move played. Exchanges further back are
  not in the row.

#### Forgone plays

For an opponent's play of `K` tiles (1 to 6) scoring `s`, and each tile kind
`τ` (blank included), the **forgone play** is the best-scoring legal play on
the board before the move that places a sub-multiset of `K + {τ}` and needs
`τ`: it holds one more `τ` than `K` does. A play `K` alone could make never
counts. The input is the margin, the forgone play's score minus `s`, clamped
to [−50, 127]; −50 also stands for "no such play". The margin is a lower
bound on what holding `τ` offered, since the leave's other tiles are unknown,
and the model learns how much to read into it. Exchanges, passes and bingos
have none.

- **Computation** ([belief/forgone_plays.h](../../engine/include/belief/forgone_plays.h)):
  one exhaustive generation from `K` plus a blank finds every `τ` at once.
  The plays that place the extra blank give `τ = blank` directly. A blank
  designated `L` stands for an extra `L` when the play also places all of
  `K`'s own `L`s, and is rescored with `L`'s value on its square. About
  0.4 ms per play, so about 6 ms per HastyBot game, against play_game's
  8.4 ms.
- **Stored, not replayed.** A `.forgone` sidecar beside each `.slog`
  ([data/forgone_play_log.h](../../engine/include/data/forgone_play_log.h))
  holds every move a row can read: turns `[0, eligible_end − 1)` of each
  game. Its header pins the lexicon and the `.slog`'s game table, and the
  DataLoader refuses a `.slog` whose sidecar is missing or does not match it
  ([architecture.md](../architecture.md#the-replay-reconstruction-invariant)).
- **No leak.** Which tile kinds get a margin depends on the board alone:
  `τ` gets none once the board after the move holds every copy. Excluding
  the tiles the mover could see would drop exactly the one-copy tiles they
  kept, and the gaps would give the leave away.
- **Applicability**, decided by the encoder: a margin enters the row only if
  that move's leave pool, as the observer saw it just after the move, holds
  `τ`. A rule from the current pool would be wrong after the opponent
  exchanged a tile the observer later drew.
- **Encoding:** per slot and tile kind, `[applies, margin / 100]`, zero when
  it does not apply. With the slots and the exchange counts, the row has 453
  scalars, and `kLeaveInferenceFormatVersion` is 3.
- **Generation:** the workload's generators run `forgone_play_tool` over each
  chunk before delivering it, and the trainer over its held-out games.

#### The target, the task and the DataLoader mode

- **`OppLeaveTarget`:** 7 sorted tile indices (27 letters with the blank at
  26, padded with −1), the length, the validity flag and the cause, then the
  [leave pool](#the-leave-pool) and the bag size.
- **`LeaveInferenceTask`:** `encode_row` writes the input's encoding and the
  target.
- **`DecodeTask::kLeaveInference`:** a third DataLoader task code, wired as
  `kMaxMovePerLane` was:
  - the `BlockDecoder` branch;
  - the row-size getters in `data_loader.h`;
  - the `dl_new` mapping, and shape queries in the FFI;
  - `"leave_inference": 2` in `NativeDataLoader._TASK_CODES`.
- **Addressable decode.** `scribblez_decode_rows` today hard-codes the
  position task's row width. It takes the task too, so the held-out eval can
  decode a fixed list of `(game, turn)` rows.

#### The leave table: `LeaveValueArrays`

`LeaveValueArrays` (`lexicon/leave_value_arrays.h`) holds the same leaves and
values as `LeaveValues`, laid out for a different query. `LeaveValues`
answers "what is this leave worth?" by lookup. `LeaveValueArrays` holds every
leave of each length 1–6 in one flat array, for the passes that scan a whole
length. The 6-tile array has 737,311 rows; the 4-tile array has 25,254.

- **Each length is a structure of arrays:** the values as floats, and, for
  each of the leave's k tile slots, a byte array of that tile's code, its
  letter × 6 plus how many copies of the letter precede it. A row is 4 + k
  bytes. A leave is the set of its codes, so the codes carry the
  multiplicities the pool prior weighs, and the letter's first-copy code marks
  its presence, which is all the model's letter weights read. A pass builds
  one 162-entry table per query, with each copy's step of the pool prior plus
  `θ_l` on the first-copy entries, and sums k lookups per row before adding
  the value tilt.
- **Built in memory** from `LeaveValues`, by iterating its entries in the
  leave values file's own order, so the order is fixed by the file. The
  constructor refuses leave values that lack a leave drawable from a full
  bag, since the model normalizes over these rows.
- **Python reaches it through the FFI**, through the `ffi.LeaveValueArrays`
  class, which copies a length's rows out. The trainer and the engine's
  sampler therefore see the same rows in the same order, which sampling's
  pairing and the parity tests depend on.
- **Its identity is the KLV's.** `LeaveValues` records the content hash of
  the KLV it loaded. The trainer stamps that hash, as `leave_values`, into
  every checkpoint and ONNX. A consumer that loads a model checks it against
  its own leave values. A model reads its tilt `a` and `b` on the values it
  was trained against, or not at all. A change to the row layout or order is
  a `leave_inference_format_version` bump.

#### Sampling: Algorithm 1, on the CPU

`belief::LeaveSampler` takes the 29 floats, the leave's length, the leave
pool, the observer's unseen pool, and the leave table.

- **One pass over the table** for that length gives each row's log weight,
  the CDF in table order, and `log Z`. Each row costs about 28 additions, as
  in the design.
- **`sample(u)`** binary-searches the CDF for the leave's row. The refill, a
  uniform draw from the unseen pool less the leave, is the sim's: SimRunner
  already seats a known leave and draws the rest of the rack from the bag.
- **`log_prob(row)`** returns a row's log probability, which the parity test
  uses.

v0's only engine consumer is the survey tool: one query per position, about
300 samples. The GPU-batched sampler and Algorithm 2 serve the agent's
~2,000 queries per turn, so they are built with Bob's subtree
([deep_bot.md](deep_bot.md#build-order), step 4). The CPU sampler is their
parity reference.

#### Serving

- **`LeaveInferenceSpec`** in `nn/model_specs.h`:
  - `kGraph = "leave_inference"`, and `kAcceptUntaggedGraph = false`;
  - one input row per position (`input_spatial`, `input_scalar`), and one
    output `leave_params` of 29 floats;
  - `kVersions = {{"leave_inference_format_version",
    kLeaveInferenceFormatVersion, -1}}`.
- **`LeaveInferenceNet`** drives `NeuralNet<LeaveInferenceSpec>` directly, as
  `move_proposal_nets` does. TrtEvalService's batching earns nothing at one
  call per position. At load, it checks the model's `leave_values` entry
  against the loaded leave values' KLV hash.
- **The export.** The trainer exports each generation's tilted-head model to
  `models/`, beside its win table, so a generation can be pinned.

### Python work: `py/scribblez/leave_inference/`

| File | Contents |
|---|---|
| `model.py` | The encoder (a conv tower over the 242 planes, an MLP over the scalars, fused into a context vector) and the heads |
| `loss.py` | The tilted-prior NLL ([below](#the-loss)) |
| `autoregressive.py` | The comparison head's decoder, step mask and sequence NLL ([below](#the-comparison-head)) |
| `heads.py` | The heads' names, free of torch for the workload's `head` param |
| `win_table.py` | The win-table fit; the engine writes and evaluates the tables through the FFI |
| `trainer.py` | The generational loop and the win-table refit |
| `train_loop.py` | One epoch, and the held-out eval |
| `holdout.py` | The held-out games |
| `onnx_export.py` | Export, stamped through `onnx_export_util` (build step 5) |

The conv tower reuses `spatial_trunk.py` if its constructor takes the plane
count. Otherwise it is a small local tower. The input channel count is the
only difference.

#### The loss

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
- **`base` per chunk.** The table stays on the device as `uint8` counts.
  Each chunk's counts are expanded to a one-hot over (letter, count) pairs,
  216 columns. `base` is then the product of that one-hot with the batch's
  `[B, 216]` matrix of `log C(n_l, c)` values. Counts the pool cannot supply
  take a large negative constant.
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

For v0 data the quality term fits especially well: HastyBot chooses its leave
by score plus exactly the leave value the tilt acts on.

#### The comparison head

`head = autoregressive` replaces the 29 floats with a decoder over the sorted
leave: a 27-way softmax per step, masked by the pool's remaining counts and by
sort order, on the same encoder. It is trained and evaluated in Python only,
and never exported. The workload selects it with the `head` param.

It lands before serving. The gap between the two heads says whether the 29
floats are enough or whether the tilt needs more terms, and that decides the
output layout that serving bakes into `LeaveInferenceSpec`, `LeaveSampler`
and the parity fixtures. The 29-float head stays the design's choice. The
comparison measures what it gives up; it does not compete to replace it.

### The workload

- **Roles:**
  - `generate`: `selfplay_gen.generate` with `hasty_spec`, then
    `forgone_play_tool` over each chunk (`selfplay_gen.ChunkSidecar`); each
    `.forgone` is delivered just before its `.slog`, and the scheduler moves
    the two together (`SchedulerConfig.chunk_sidecar`).
  - `train`: the trainer above, hosting the data home.
- **`face_up_leaves` is not a param.** `selfplay_gen.generate` takes it as a
  keyword argument instead of reading it off the params. `run_generate` (for
  `max_move_per_lane`) and `position_eval.run_generate` pass their own
  param, and this workload passes `False`.
- **Registration:**
  - `WORKLOADS` in `workloads/__init__.py`;
  - `END_PARAMS["leave_inference"] = {"max_rows"}` in `test_end_params.py`,
    which asserts that every workload declares its end params.
- **Dashboard:**
  - a `WORKLOAD_TABS.leave_inference` entry: `LossTab`, a training tab on
    `leave_inference_metrics`, and `InfoTab`;
  - a `FIGURES["leave_inference_metrics"]` builder in `dashboard/api.py`;
  - `LEAVE_INFERENCE_QUALITY` metric groups in `dashboard/plots.py`.
- **Outputs per generation:** the win table and the checkpoint, and from
  build step 5 the ONNX.

### Held-out data: regenerated

The held-out set is `holdout_games` games that the trainer plays itself at
startup. It runs `run_games` at the fixed seed `holdout.SEED`, HastyBot against
HastyBot, and writes to a scratch directory on its machine.

- **Reproducible.** `play_game` plays game `g` at seed `base + g`, and the
  bag is seeded from it. With the greedy policy, a fixed seed gives the same
  games on any home.
- **Disjoint from training.** Generators run with seed 0, which draws a fresh
  random base per chunk.
- **Nothing durable to lose.** A held-out directory in the data home would
  need adding to the controller's pull, seed and sweep sets, or it is lost
  when an ssh trainer moves or a spot instance dies. Regeneration avoids
  that, and needs no change to the scheduler.
- **Fixed for the run.** The same games every generation keep the curve
  comparable. v0's policy is fixed, so a fixed set loses nothing. A later
  model on a changing policy regenerates its set from its own policy.
- **Identified by a content hash** of its `.slog` files, recorded in the
  run's control events. A home whose regeneration differs, for example
  across CPU architectures, shows up there.

The eval reads the held-out rows through a `SlogDataset` without symmetry, in
a fixed seeded order.

### Eval metrics

Per generation, on the held-out positions:

| Metric | Meaning |
|---|---|
| `eval_nll` | mean NLL of the true leave over valid rows, in nats |
| `eval_nll_vs_uniform` | `eval_nll` minus the NLL under the hypergeometric prior over leaves from the leave pool ([belief/leave_prior.h](../../engine/include/belief/leave_prior.h)) |
| `eval_nll_bag_*`, `eval_nll_len_*` | `eval_nll` by bag band, and by leave length |
| `eval_invalid_*` | the share of rows without information, by cause |
| `eval_letter_calibration` | mean absolute gap between `E_p[mask]` and actual presence, per letter |
| `eval_value_calibration` | `E_p[v]` against the true leave's value, and the coverage of its implied spread |
| `eval_wintable_logloss`, `…_macondo` | the refitted win table's log loss on the held-out outcomes, and Macondo's bundled table's on the same features |

With `head = autoregressive`, the NLL metrics come from its decoder, and the
calibration metrics are left out.

### Parameters

- `games_per_generation`, `open_ahead` and `max_rows`;
- `hasty_temperature` (default 0, greedy) and `hasty_top_k`, which
  `hasty_spec` reads;
- `window` and `turns_per_game`;
- `holdout_games`;
- `head` (from the comparison head on);
- the network's and the optimizer's usual parameters, through profiles.

## Workload 2: `policy_set`

### Data: labeled pools

A generator worker's cycle:

1. **Play** a chunk of `games_per_chunk` (default 50) HastyBot-vs-HastyBot
   games, `face_up_leaves=False`, without random openings.
2. **Pick positions.** `positions_per_game` turns (default 2) of each game,
   uniform over the eligible region, seeded by the chunk's name.
3. **Survey each position** with `sim_candidate_survey_tool`'s pool recipe,
   screen stage only (no confirm stage), over the design's
   [candidate set](deep_bot.md#labels): `rollouts` (default 100) per
   candidate, stopped at the win table after `horizon_plies` (default 3: the
   candidate, then Bob, Alice, Bob). Racing stays on.
4. **Deliver the chunk's pool records** (below) as the `.slog`'s sidecar.

**Survey-tool changes.**

- **Candidates:** blank plays (with the design's 40-point prune), every
  distinct exchange, and the pass, added to the exhaustive recipe.
- **`--win-table <path>` and `--horizon-plies K`:** rollouts stop after `K`
  plies at the table's win probability, with the leave value of the side to
  move added to the spread. The table is passed to the simmer, in place of
  `macondo_default()`. The tool's endgame solves are not used.

**Opponent racks.** A new flag, `--leave-inference <ONNX path>`, sets the
opponent's leave belief:

- **With the flag,** the tool runs the leave inference model once per
  position, from the chooser's view. Its input is the training row itself,
  decoded from the `.slog` and its `.forgone` sidecar
  (`belief/slog_leave_beliefs.h`), so the generator runs `forgone_play_tool`
  first. It samples one leave per rollout with `LeaveSampler`, from a seeded
  variate, and the rollout draws the rest of the rack from the unseen pool.
  Where the leave is not modelled (turn 0, after a pass, an empty leave, or
  an empty bag), the whole rack is drawn uniformly.
- **Without it,** the opponent's whole rack is drawn uniformly from the
  unseen pool, which is today's hidden-leaves behavior.

Rollout `i` uses the same rack sample for every candidate, so the per-move
values stay paired. With the model, the generator role needs a GPU for its
TensorRT engine; one evaluation per position makes the need small.

**The pool record** is a new compact sidecar beside the `.slog`, not the
survey's JSON. Its header carries the magic and `kPoolFormatVersion`.

- **Per pool:** the identities of the win table and the leave inference
  model (tag, generation and format version), or uniform beliefs; and
  `horizon_plies`.
- **Per position:** the `.slog` game and turn, and the
  [opponent-rack input](#the-opponent-rack-input).
- **Per candidate:**
  - the `Move` (16 bytes);
  - mean win probability;
  - rollouts completed (racing stops some early);
  - the paired standard error against its comparator (below);
  - the static-equity rank;
  - `q(a)`.

v0 computes `q(a)` at generation time by the normal approximation to the
paired differences against the leader, since the per-rollout outcomes are not
kept. It treats the non-leader moves as independent of each other, an
approximation acceptable for v0. A candidate that racing stopped is compared
with the leader at the checkpoint that stopped it, over its own rollouts.
Compared with the final leader's same rollouts instead, stopped candidates
took 47–83% of a pool's `q` in a 20-rollout test: the final leader is picked
on all the rollouts, and among a thousand stopped candidates some beat its
early ones by chance.

**Generations** are `positions_per_generation` pools. The window and
eviction work as in `leave_inference`. Pools are expensive, so the window
default is large.

**The source tag.** `leave_inference_tag` and `leave_inference_generation`
pin a generation of a `leave_inference` tag through `finalize`, as
`move_set_eval` pins its teacher. The pin supplies the win table and, with
`leave_beliefs = inferred`, the model. With
`leave_beliefs = uniform`, it supplies the win table alone.

**Parameters:**
- the source: `leave_inference_tag`, `leave_inference_generation` and
  `leave_beliefs` (`inferred` or `uniform`);
- the survey: `horizon_plies`, `positions_per_game`, `rollouts` (the screen
  budget, default 100), `race` and `blank_prune_points`;
- self-play: `games_per_chunk` and the `hasty_*` policy;
- generations and held-out pools: `positions_per_generation`, `window`,
  `holdout_every` and `holdout_chunks`;
- training: `forms` and `max_rows`.

### The opponent-rack input

The policy-set model reads the chooser's belief about the opponent's rack
([design](deep_bot.md#the-policy-set-model)) as `OppRackSummary`
(`belief/opp_rack_summary.h`), which describes the rack the sims draw:

- **Presence, 27 floats:** per tile kind, the probability that the rack holds
  at least one. With a modelled leave, the rack lacks a kind exactly when
  both the leave and the refill do, and the refill's chance of missing it is
  hypergeometric: `C(U − k − u_t, 7 − k) / C(U − k, 7 − k)` for a `k`-tile
  leave, `U` unseen tiles and `u_t` unseen copies of `t`. The leave's
  marginals come from one more pass over the sampler's rows
  (`LeaveSampler::moments`). Without a modelled leave, the rack is a uniform
  draw of 7.
- **Whether the leave was modelled,** and the mean and standard deviation of
  its static value in points (0 when not): how strong the believed leave is.

The summary is at the rack level, not the leave's, so it is defined at every
position the sims cover: under uniform beliefs, and where the model does not
apply (turn 0, after a pass, an empty leave). It is what the sims drew from in
each case.

The survey writes it into each position of the pool record, rebuilding the
position's belief after its sims, since the simmer drops it. The agent
computes it from the same belief its sims sample.

### Held-out pools

Every staged chunk with `crc32(stem) % holdout_every == 0` goes to the
held-out set (`data/held_out/chunks`) instead of a generation, the convention
`pair_store.split_pair_stems` uses. The set takes the first `holdout_chunks`
of them; later held-out chunks are deleted. The trainer evaluates on the set.

- **The scheduler.** It routes held-out chunks before filling generations.
  Routed chunks still enter the ingest ledger, so they are not staged again.
  The set carries a generation's manifest and is marked complete once full,
  and it is never evicted.
- **The controller.** The set's parent joins its pull, seed and sweep sets
  beside the generations, so the pools survive a trainer moving between
  machines. Being one directory with a manifest, it travels exactly as a
  generation does: pulled once complete, seeded acknowledged.
- **The baseline.** Beside the forms it trains (steps 8a and 8b), the
  trainer records per generation the window's pool statistics and the
  static-equity top N's metrics on the held-out set, under the `equity_` key
  prefix.

### Inputs at train time

Inputs are rebuilt by replay, as `move_set_eval`'s trainer does. The pool's
`.slog` reference and its candidate moves go to `ffi.move_set_positions`,
which returns the board rows and per-move cross-check cells and letters. The
process runs in the hidden-leaves arm (`ffi.set_opp_leave_input`, adopted
from a header flag in the pool sidecar). No legal-move enumeration is needed,
because the pool lists the candidates. The
[opponent-rack input](#the-opponent-rack-input) comes from the pool record
itself (`pools.Pool`), not from replay.

### The two forms

The forms are defined in the [design](deep_bot.md#the-policy-set-model);
these are v0's choices. Both use the move set model's encoders (the board
trunk, `MoveEncoder`, `CrossCheckEncoder`), trained from scratch. Today's
move set checkpoints are face-up and differ in their input blocks.

**The equity input.** Each candidate's HastyBot static equity, less its
pool's best, joins its embedding, so a form starts from the static-equity
ranking and learns its correction. The equities are recomputed by the replay
that builds the board rows (`ffi.move_set_positions(..., equities=True)`).

**Form 1, `greedy`:** a set-conditioning stage plus a gain head.
- The stage is the evidence-fusion stage's set half with chosen-move tokens
  ([evidence_fusion.py](../../py/scribblez/evidence_fusion.py)): the tokens
  self-attend, the board tokens cross-attend into them, and zero-initialized
  projections write back, so an empty set leaves the plain model. The
  fusion's observation channels have no counterpart, since chosen moves are
  not simmed. The candidates then re-attend to the conditioned board through
  the encoder's own attention, and the gain head reads them.
- Rows come by subset assembly. Per position, draw subsets `S` of sizes 0 to
  `N_max − 1`, biased toward high-value moves as deployment sets are. The
  target for a held-out move `a` is `max(0, v(a) − max_S v)`. A batch of
  subsets per position shares one trunk pass, which is the training-side form
  of the cached steps.
- The loss is Huber on the gain.

**Form 2, `slot`:** a slot decoder with `N_max` slots (default 16).
- The slots cross-attend to the move embeddings and self-attend to each
  other, and each points to a move by softmax.
- The loss is the coverage loss over slot prefixes, with the pool's `q(a)` as
  the label distribution, summed over `N` in {1, 2, 4, 8, 16}.

### Training both forms

The `forms` parameter is `greedy`, `slot` or `both`.

With `both`, one trainer holds the two models, each on its own encoder, and
trains them on the same batches: one optimizer over both, the summed loss,
each model's gradient clipped on its own, which updates each as a separate
optimizer would. Records carry the form as a key prefix (`loss_greedy_gain`,
`slot_eval_recall_8`, ...), so the dashboard plots them side by side. Both
models are move-set-sized, so one GPU holds them.

A form per machine needs two trainer roles reading one data home. Eviction
would then wait for both trainers' pulls, an extension of the `pulled`
acknowledgment in `lifecycle`. That comes only if one GPU's throughput binds,
which it should not while label generation is the bottleneck.

### Eval metrics

Per generation and form, on held-out pools, for `N` in {1, 2, 4, 8, 16}:

| Metric | Meaning |
|---|---|
| `eval_recall_N` | how often the pool's sim-best move is in `S` |
| `eval_cover_N` | `Σ_{a∈S} q(a)`, the soft form of recall, P(best ∈ S) |
| `eval_regret_N` | mean `v(a*) − max_{a∈S} v(a)`, in win probability |
| `eval_tail_recall_N` | recall on pools whose sim-best move is outside the static-equity top 10 |
| `eval_set_ms` | time to build one batch of 256 sets at `N = 8` |

Each metric is reported beside the same metric for the static-equity top `N`,
the baseline every DeepBot set has to beat. The figure plots recall and
regret against `N` per form, over generations.

### Cost

From the blind_spots measurement, the exhaustive screen at 1,000 rollouts
played to the end of the game costs about 30 seconds of a 28-thread machine
per position.

- **Two changes cut that:** 300 rollouts instead of 1,000, and rollouts
  truncated at `K` plies instead of played out.
- **Blank plays push the other way,** by enlarging the candidate set.

A first measurement, at 300 rollouts and a 5-ply horizon on 24 threads of a
machine already busy with a `leave_inference` run, simmed 8 pools in 33 s:
about 4 s a pool, or roughly 900 pools per machine-hour. Pools ranged from 150
to 2,500 candidates. The real rate is measured on the first generation and
shown on the Stats tab.

Generation sets the pace and is embarrassingly parallel, so the `generate`
role is meant for a fleet: owned ssh machines and rented machines, as
blind_spots runs. With the leave inference model, those machines need a GPU.

## Build order

Each step is a PR with its own tests.

**The win table:**

1. **The standalone fit.**
   - `leave_from_replay`, `win_table_feature` and its FFI scan.
   - `win_table.py` with its CLI.
   - The versioned header, `kWinTableFormatVersion`, and `WinPctTable`'s
     stream parser with `load_versioned`.
   - v0's first table, fitted on an existing HastyBot hidden-leaves corpus.
   - *Reports:* held-out log loss against Macondo's table.

**The leave inference model:**

2. **Engine rows, and the leave table.**
   - A mechanical commit moves the shared encoder helpers into
     `board_planes.h`.
   - Then:
     - the encoder: `LeaveInferenceInput`, `LeaveHistory`,
       `LeaveInferenceInputEncoder`, and the replay recorder with the
       history transpose;
     - the rows: `OppLeaveTarget` with validity, `LeaveInferenceTask`,
       `DecodeTask::kLeaveInference` with its FFI shapes, the task-aware
       `decode_rows`, and `kLeaveInferenceFormatVersion` in `format_layout`;
     - the leave table: `LeaveValueArrays` built in memory, its FFI class,
       and the KLV hash on `LeaveValues`.
   - Tests:
     - the history slots line up with the replay;
     - pre-move cross-checks equal the board's cross-checks before the move;
     - hidden tiles cannot change a row;
     - transposed rows match transposed boards and moves, placement planes
       included;
     - missing history encodes as zeros;
     - the target is the true leave, and each invalid cause is flagged;
     - the position tasks' rows are byte-identical with and without the
       recorder change;
     - the leave table has the stated row counts, a fixed order, and
       `HastyEquity`'s values.
3. **The trainer and workload.**
   - The `leave_inference` package: the tilted head, the leave table's
     arrays, and the streaming loss with its gradcheck.
   - The trainer:
     - the generational loop and the regenerated held-out set;
     - the eval, with the uniform baseline and calibration;
     - the checkpoint version gate;
     - the per-generation win-table refit and its delivery.
   - The face-up refusal, the `generate` signature change, the workload spec
     and registration, and the dashboard tab and figure.
   - *Reports:* the loss pass's throughput by leave length and its device
     memory (this sets the batch size and the chunking), and the first
     NLL-versus-uniform curve.
4. **The comparison head:** `head = autoregressive`.
   - *Reports:* NLL for both heads, by leave length and bag band. This feeds
     step 5's output layout.
5. **Serving.**
   - ONNX export with its metadata, `LeaveInferenceSpec`,
     `LeaveInferenceNet` with the `leave_values` check, and `LeaveSampler`.
   - Parity checks: a TensorRT parity binary run by `py/tools/gpu_parity.py`
     (the served 29 floats against PyTorch), and a Python test of
     `LeaveSampler`'s log probabilities against the PyTorch table pass.
   - *Reports:* the CPU time per query for 6-tile leaves, which feeds the
     survey tool's cost estimate. Measured: 7.4 ms per query with 300 samples,
     on a contended machine.

*Measure:* the model's NLL against the uniform prior, by bag band and by
leave length, for both heads. The gap between the heads, concentrated where
it is largest (6-tile leaves in particular), says whether the 29-float
output is enough for the agent's beliefs or whether the tilt needs more terms
before serving fixes the layout.

**The policy-set model:**

6. **Engine:** in the survey tool:
   - `--win-table`, `--horizon-plies` and `--leave-inference`;
   - blank plays with the 40-point prune, exchanges and the pass in the
     candidate set;
   - the pool sidecar with `q(a)`, and `kPoolFormatVersion`.

   The uniform-belief path needs only step 1; the leave inference flags need
   step 5.
7a. **The generator side:** the `policy_set` generate runner (chunks of
   `games_per_chunk` games, each surveyed into a `.pool` before delivery),
   the source-tag pin and its role inputs, and the pool reader on
   `ffi.move_set_positions`.
7b. **The workload:** its registration, with a trainer that hosts the data
   home (the generation scheduler runs only beside a trainer), held-out
   pools through the scheduler and the controller's pull, seed and sweep,
   and the eval metrics for the static-equity baseline.
8a. **The shared encoder and the slot form:** pool batches, the move set
   encoder reading the opponent-rack input as a scalar block beside the
   board row's, the slot decoder and its coverage loss, trained under
   `forms`, its eval metrics beside the baseline, `kPolicySetFormatVersion`,
   and the figure.
8b. **The greedy form:** the set-conditioning stage and gain head, rows by
   subset assembly, and `forms = both`.

*Measure:* recall, cover, regret and tail recall against `N` for each form,
and against static equity; time per batch of sets. This decides the form.

**Later:** a trainer per form, if one GPU binds.

Step 1 depends on nothing else, and step 2 needs only step 1's
`leave_from_replay`. Step 6's uniform path can start after step 1.

## Open questions

- **`horizon_plies`.** Fewer plies are faster and less noisy, but lean harder
  on the win table. Macondo uses 5 plies with more than 14 tiles unseen.
- **`rollouts` against pool count.** Fewer rollouts per pool means noisier
  `q(a)` but more positions. The soft labels absorb some of the noise.
- **A CPU path for the leave inference model** in the generator, if renting
  GPU machines for the fleet proves costly. One evaluation per position is
  cheap enough for a CPU runtime, but the engine serves networks only
  through TensorRT today.
- **The input size.** 242 planes is about 220 KB a row in FP32. If loader
  bandwidth binds in step 3's throughput report, the history's cross-checks
  can shrink to their delta against the current cross-checks, which is zero
  away from the moves played since. That is a
  `leave_inference_format_version` bump.
