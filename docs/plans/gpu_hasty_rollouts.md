# Hasty rollouts on the GPU

**Status: feasibility analysis and build plan, 2026-10-23; a standalone demo
is built (see [Demo results](#demo-results)). The first integration, the
policy-set model's label generation, is planned as steps 9-11 of
[deep_bot_v0.md](deep_bot_v0.md#build-order); DeepBot's sims run on it
([deep_bot.md](deep_bot.md#sims)).** Whether
batches of HastyBot rollouts can run on the GPU with substantially better
throughput than the multithreaded CPU engine, and how to build it if so.

## The task

A position: a board `S`, our leave `L` (0 to 6 tiles), the scores, and the
unseen pool (the bag plus the opponent's rack). One rollout:

1. draw the opponent's rack (7 tiles) and our replenishment (`7 - |L|`
   tiles) uniformly without replacement from the unseen pool;
2. the opponent makes the top static-equity play (HastyBot: score plus the
   leave value, the pre-endgame and opening adjustments, and the endgame
   adjustment; exchanges included when the bag holds 7 or more);
3. we make the top static-equity play;
4. the outcome is the score differential, with the end-of-game rack
   adjustments if someone played out.

The task is `N` rollouts and their mean. This is the mechanical core of every
simulation the project runs: SimRunner's rollouts, the policy-set labels, and
DeepBot's subtrees (where the draws come from the leave inference model and the
rollout runs a few plies to a win table). The sampling and the leaf differ
there; the per-ply work does not.

## Verdict

Feasible, and worth building if a one-kernel microbenchmark (PR 2 below)
confirms the throughput model. The expected gain over the 28-thread CPU is an
order of magnitude or more; the model's break-even point is low enough that
the microbenchmark settles the question in a week of work, before the rest is
built.

The two ideas that make it work, and the two that the straightforward port
would get wrong:

- **No move list.** HastyBot needs an argmax, not the list of legal plays. On
  the GPU every candidate play is scored as it is found and reduced to the
  block's running best; nothing variable-sized is ever materialized, so there
  is no overflow path and no CPU fallback by move count.
- **No best-first search.** The CPU's speed comes from MAGPIE's shadow bounds:
  rank the word extents by an admissible equity bound, generate them in that
  order, stop when no remaining bound beats the best play. That is a serial
  dependence through the running best, and the bound itself is an
  order-dependent walk. The GPU instead enumerates every candidate extent with
  cheap necessary-condition filters and evaluates them all in parallel. It does
  several times the CPU's lookups per ply; each lookup costs a few dozen
  instructions and an L2 hit, and the parallelism is what the GPU has.
- **One thread block per rollout, lanes over work items,** with the rollout's
  board and cross-checks in shared memory. A thread-per-rollout layout (each
  lane running the CPU algorithm on its own position) is the wrong shape: the
  per-rollout state is ~3 KB, which does not fit registers or L1 at any useful
  occupancy, and every loop trip count differs across lanes.
- **The lexicon tables are L2-resident.** The word map (~15 MB, measured in PR
  2), the leave table (914,624 leaves; 16 MB as an 8-byte-slot hash table), and
  the KWG for cross-checks (4.7 MB) together fit this GPU's 64 MB L2, so the
  random probes that dominate the work are ~300-cycle L2 hits that occupancy
  hides. On a GPU with a small L2 (an A10G has 6 MB) they go to device memory;
  the bandwidth is still ample at a few thousand 32-byte probes per ply, so L2
  residency is a multiplier, not a requirement.

## The CPU baseline

Measured 2026-10-23 on the dev laptop (i7-13850HX, 8 P-cores + 12 E-cores, 28
threads, idle), hasty-vs-hasty `play_game` on NWL23, which plays each ply
through the production WordMap path (blank racks through the GADDAG) with the
bag draws and the game log:

| Threads | Plies/s | µs per ply (per thread) |
|---------|---------|-------------------------|
| 1       | 8,060   | 124                     |
| 20      | 106,000 | 189                     |
| 28      | 124,000 | 226                     |

A rollout is two plies plus the draw and the setup, so the CPU does roughly
50-60K rollouts/s on this machine with every core busy. That is the number
to beat. The bare movegen is 67 µs per ply (55 extents, 686 lookups); the rest
of a ply is the equity precompute (the subrack-and-leave DFS), the shadow walk,
applying the move and recomputing cross-checks, and the draw.

## The throughput model

Per ply, blank-free, mid-game board, exhaustive enumeration with the filters
below. The lookup count is anchored on a measurement: the repo's naive
full-board WordMap generator, which enumerates spans times subracks with the
same placeable-square filter, made 4,475 lookups per move before the shadow
work replaced it.

| Work item                     | Count per ply | Lane-instructions each | Total   |
|-------------------------------|---------------|------------------------|---------|
| Leave lookups (all subracks)  | 127           | ~30 + 1 probe          | 4K      |
| Span enumeration              | ~30 lanes     | ~1K per lane           | 30K     |
| Word-map probes               | 2,000-4,500   | ~30 + 1-2 probes       | 60-135K |
| Word validations and scoring  | 500-1,000     | ~80                    | 40-80K  |
| Argmax reduction, bookkeeping |               |                        | ~10K    |

Call it 150-250K lane-instructions per ply and ~0.5M per rollout with the
draw, the apply, and the cross-check updates. This GPU (76 SMs, 128 lanes
each, 1.68 GHz) issues 16 T lane-instructions/s at peak. Irregular integer
code with random table probes reaches a few percent to perhaps 20% of that:

| Fraction of peak | Rollouts/s | Over the 28-thread CPU |
|------------------|------------|------------------------|
| 0.2%             | 60K        | 1x (break-even)        |
| 3%               | 1M         | ~20x                   |
| 10%              | 3M         | ~60x                   |

The claim is not the 60x. It is that break-even sits at a fraction of peak so
low that the kernel would have to be badly wrong to miss it, and that 3% is a
conservative floor for this kind of kernel. The microbenchmark measures the
real fraction on the real tables; everything after it is engineering.

What the model does not capture, and the microbenchmark will:

- the lane utilization under divergence (word lists and span counts vary
  across lanes; the block-per-rollout layout keeps the control flow
  warp-uniform within a phase, not the trip counts);
- how much the subrack-versus-cross-mask prefilter cuts the probe count (88%
  of the CPU's probes find no word);
- shared-memory list sizes (spans per rollout; a cap forces a second pass,
  not a fallback).

Batching: one block of 128 threads per rollout with ~4 KB of shared memory
gives ~16 resident blocks per SM, about 1,200 rollouts in flight. A batch
should hold 10K+ rollouts to amortize the launch and the tail. DeepBot's shape
(candidates times sampled opponent racks times sampled own racks times a
policy set, of order 10^5-10^6 rollouts per turn) is a natural batch; a
SimRunner call of 30 candidates times 100 rollouts is a small one and runs in
a few milliseconds.

## Design

### Device-resident tables, uploaded once per process

- **Word map:** per length, the open-addressing slot table (key BitRack,
  word start, count) and the flat word array, copied from `WordMap` as built.
  Probing is the CPU's: Fibonacci hash of the two halves, linear probe,
  16-byte key compare.
- **Leave table:** every leave of 1-6 tiles (914,624 under the standard
  distribution) in a hash table keyed by the sorted tiles packed 5 bits each
  into 32 bits, value `float`. Built on the host from `LeaveValues`. The CPU's
  incremental KLV cursor is the wrong shape for lanes; one probe per subrack
  is.
- **KWG arcs** (the GADDAG), for cross-check recomputation after a ply, with
  `Board::cross_check_at`'s walk translated as is.
- The pre-endgame table, the premium grid, tile values, the letter
  distribution.

### Per batch, from the host

The position in the board's natural frame: squares, both views' cross-check
masks and scores and anchors (the host's caches, copied), scores, bag size,
our leave, the unseen pool as counts. The batch seed. Optionally, per-rollout
inputs that replace a draw: the opponent's leave from the leave inference
model, which stays a host-side network; the device draws the rest.

Precomputable here because the opponent's board is `S` for every rollout: the
opponent-ply span list (every span with 1-7 placed tiles, with its
playthrough key and per-square cross masks), so the opponent ply skips span
enumeration and applies only the rack-dependent filter. Our ply's board
differs per rollout, so it enumerates on the device.

### Per-rollout state, in shared memory

Squares (225 B), cross-check masks and scores for both views (~2.3 KB),
anchors, both racks as counts, the pool permutation (up to 93 B), scores, the
bag size, the two subrack leave arrays, the span list, the running best.

### The kernel pipeline, per batch

1. **Draw.** A counter-based generator (Philox) keyed by the batch seed and
   the rollout index drives a partial Fisher-Yates shuffle of the pool: the
   opponent's 7, then our refill. The CPU reference uses the same generator,
   so every rollout's racks, and so its outcome, can be compared exactly.
2. **Ply** (the opponent's, then ours; one routine):
   1. *Leaves:* 127 lanes each build one subrack of the mover's rack (a 7-bit
      mask over the sorted rack, as `TurnLeaves` does) and probe the leave
      table for the complementary leave. This array also prices every
      exchange (bag of 7 or more: the best kept subset) and the endgame
      adjustment.
   2. *Spans:* a warp per lane (2 views times 15 rows). It walks the lane's
      filled pattern and emits every `(start, length)` whose ends are not
      adjacent to tiles, that places 1 to `rack_tiles` tiles, that touches a
      tile (or the center on an empty board), and whose placed squares each
      admit some rack letter (cross mask and rack letter mask). Each span
      carries its playthrough BitRack, its placed-square positions and cross
      masks, and its multipliers.
   3. *Probes:* a work item is (span, subrack of size `placed`). The lane first
      checks that each placed square's cross mask admits a subrack letter;
      then probes the word map with `playthrough + subrack`; then, for each
      word returned, verifies the placed letters against the cross masks in
      order, scores the play (main word, cross words from the stored cross
      scores, bingo), and forms the equity: score, the subrack's leave, the
      pre-endgame term for `placed` tiles, the opening penalty on an empty
      board, the endgame adjustment when the bag is empty.
   4. *Argmax:* a block reduction under `hasty_move_better`'s total order
      (equity, then Macondo's tiebreaks: score, play before exchange before
      pass, start square, orientation, fewer tiles, shorter word, lower tile
      sequence), encoded as a sortable key so that the device picks the same
      move the CPU does. No play and no exchange means a pass.
3. **Apply.** Write the tiles, add the score, refill from the pool, update the
   bag size. Cross-checks change only at the empty squares just past each end
   of the perpendicular run through each placed tile, in each view
   (`Board::update_caches_after_place`): at most 4 squares per placed tile,
   each a KWG walk, one lane each. Anchors are re-evaluated around the placed
   tiles.
4. **Leaf.** The score differential with the play-out adjustments. The mean
   is a reduction across blocks. For DeepBot, a `K`-ply loop and the win
   table (a small table, device-resident) replace this step.

### Blanks

The word map is blank-free; on the CPU a blank rack takes the GADDAG path.
On the device, a subrack holding `b` blanks enumerates their designations,
restricted to letters the span's placed squares admit (the union of their
cross masks), and probes each. That is roughly 8x the work for the ~12% of
plies with a blank on the rack: an average cost near 2x, warp-uniform, and
everything stays on the device. MAGPIE's answer, precomputed blank tables
(keys with a blank count; ~200 MB), is the upgrade if that share matters.

### Precision and correctness

The equity is a sum of a score, a `float` leave value and a few table terms;
the device computes it in the CPU's `double` so ties resolve identically. The
oracle is exact: same seed, same racks, same moves, same outcome per rollout,
checked against HastyBot through the CPU reference below.

## Alternatives

- **The opponent-ply table.** The opponent's board is fixed across the batch,
  so the best play for any rack is `max` over the rack's subracks `s` of
  (the best score among `S`'s plays whose placed tiles are exactly `s`) plus
  the leave of the rest. One precompute per position (every play on `S` from
  the whole unseen pool, grouped by placed multiset, best score per group)
  turns the opponent ply into 127 probes. This speeds up the CPU as much as
  the GPU and so does not change the comparison. Its build cost (all plays
  from a 90-tile rack, with blank variants) sets a break-even `N` that must be
  measured; it is a later optimization for either backend, not part of the
  feasibility.
- **A faster CPU.** The production path is already at MAGPIE parity. SIMD
  across rollouts or across subracks is bounded at about 2x by the probes'
  random access; it is not a route to an order of magnitude.
- **Porting the shadow bounds to the device.** The walk is serial and
  order-dependent, and its purpose, stopping early, has no parallel
  equivalent. A one-pass, loose per-span bound against a cheap lower bound on
  the best equity (the best exchange, or the best single-tile play) can prune
  spans before probing; it is an optimization to try in PR 3, not a
  prerequisite.
- **Thread per rollout.** Rejected above: the state does not fit, and the
  control flow diverges everywhere.
- **Overflow to the CPU on large move sets.** Unnecessary once no list is
  built. The only candidate for a CPU route is the blank rack, and the
  designation enumeration keeps even that on the device.

## Where it lands

A new library, `scribblez_gpu_rollouts`, as the project's first CUDA
translation unit. Build notes, found while probing the device for this
document:

- The engine's CMake never enables the CUDA language (the TensorRT layer uses
  the runtime as a host library). This library enables it for itself only.
- nvcc 12.9 refuses gcc 16 as the host compiler; the image also ships gcc 13,
  so the CUDA sources set `CMAKE_CUDA_HOST_COMPILER` to it. The reflection
  flags the C++ sources use must not reach nvcc, which the TU boundary
  guarantees.
- The test suites hide the GPU, so the device tests are an explicit check in
  the mold of `gpu_parity.py`, not part of `run_tests.py`.

The interface is a batch call: a position, `N`, a seed, optional per-rollout
opponent leaves; back come the per-rollout outcomes. SimRunner and the survey
tool keep their CPU path; a backend selector routes batches to the device.

### PRs

1. **CPU reference and benchmark.** A `rollout_bench` tool that defines the
   task exactly (position, `N`, seed, threads): the Philox draw, HastyBot for
   both plies, the leaf, the mean, rollouts/s. Counters for the exhaustive
   span and probe counts and the blank-rack share, to replace the model's
   estimates with measurements. This is the oracle every later PR compares
   against.
2. **Device tables and the ply microbenchmark.** Upload the word map, the
   leave table and the KWG; implement the draw and the opponent ply only
   (leaves, spans, probes, argmax) on the fixed board `S`; compare the chosen
   move against HastyBot on the same racks over many positions; measure plies
   per second. This is the go/no-go measurement for the whole plan.
3. **The full rollout.** Apply, cross-check and anchor updates, our ply, the
   leaf, the batch mean; per-rollout parity with PR 1; rollouts/s. The span
   prefilter and the loose bound go here if the microbenchmark says the
   probes dominate.
4. **Blanks** by designation enumeration, with the parity check extended to
   blank racks; exchanges and the endgame adjustments if PR 3 left them out.
5. **Integration:** the backend selector in SimRunner and the survey tool.
   Batching across candidates and per-rollout opponent leaves are in place
   (see [Batches of candidates](#batches-of-candidates)); host-side leave
   inference to supply those leaves is not.

Deferred: the opponent-ply table. The rest of PR 5's integration, with the
leaf network on the device instead of a `K`-ply horizon to the win table, is
steps 9-11 of [deep_bot_v0.md](deep_bot_v0.md#build-order).

## Demo results

The tables in this section and the next were measured with the
single-position demo that preceded batches; [Batches of
candidates](#batches-of-candidates) describes the current one.

`gpu_rollout_demo` (engine/apps) is PRs 1-4 of the list above in one
standalone binary: it reaches a position by HastyBot self-play, runs the same
N rollouts through the CPU reference (`sim/hasty_rollout_task.h`, HastyBot on
every ply) and through the kernel (`src/gpu/rollout_kernel.cu`), and checks
them rollout by rollout. Both draw with the same counter-based generator, so
a rollout holds the same racks on either side and its two moves and its
outcome compare exactly. On every position tried, all 65,536 rollouts were
identical, blanks included.

Measured 2026-10-23 on the dev laptop (RTX 5000 Ada, 28 CPU threads), NWL23,
65,536 rollouts per run, three self-play positions after 10 plies:

| Position (our leave)        | Racks with a blank | CPU rollouts/s | GPU rollouts/s | GPU / CPU |
|-----------------------------|--------------------|----------------|----------------|-----------|
| seed 2 (no blanks unseen)   | 0%                 | 164,000        | 607,000        | 3.7x      |
| seed 3 (leave CIOT?)        | 50%                | 34,000         | 243,000        | 7.1x      |
| seed 1 (leave ENT?)         | 57%                | 20,000         | 153,000        | 7.5x      |

The CPU falls off on blank racks because they take the GADDAG path; the
device handles them through MAGPIE-style blank tables (a subrack with blanks
probes a table keyed by its letters and gets the words containing them plus
one or two more, each tagged with the blanks' letters), so its cost per ply
rises less.

What the demo settled about the design:

- **One probe per item, with the blanks in the table.** Enumerating a
  blank's designations at probe time (26 probes per one-blank subrack) was
  the first version, and it ran below CPU parity on blank-heavy positions.
- **A whole ply is one work queue.** Chunking the spans through shared memory
  (8 chunks per ply, a barrier each) left the block waiting on its slowest
  thread at every barrier. The spans now go to a per-block slot of global
  scratch (L2-resident while hot), the items of the ply are handed out
  through a shared counter, and a warp takes 32 blank items at once and
  streams their words across its lanes, so it keeps 32 loads in flight
  instead of walking one hit list at a time. Each of these steps was worth
  1.3-2x.
- **The blank-heavy plies are memory-bound.** The blank tables (~300 MB)
  live in device memory, not L2; a 65,536-rollout batch on seed 1 moves
  about 200 GB of 32-byte sectors in under half a second, close to this
  GPU's memory bandwidth. Packing each list entry as a 16-byte word (one
  sequential load per word instead of an index and a random fetch) was the
  last gain. Blank-free plies are not bandwidth-bound; their tables sit in
  L2.
- **Occupancy is register-bound** at 64 registers (`__launch_bounds__`),
  4 blocks of 256 threads per SM. Shared memory is 17 KB per block.
- A cheaper span validation (a per-span board pattern compared as vectors)
  cost more than it saved, through the larger span record. Instruction count
  is not the limit.
- `nvcc` 12.9 refuses gcc 16 as host compiler; the CUDA library compiles
  through the image's gcc 13, and the engine's C++26-only flags are scoped to
  CXX in the root CMakeLists.txt. The Nsight profilers are not usable in the
  container (no performance-counter permission), so the kernel carries its
  own per-phase cycle counters and work counters, which the demo prints.

### The leaf network on the device

The kernel leaves every rollout's end state (board, both players' leaves
before their draws, cross-checks, scores, bag) in device memory
(`rollout_abi::LeafState`). `GpuRolloutValues` (the demo's `--leaf-model`)
evaluates every leaf with a
[DeepBot leaf eval](../deep_bot_leaf_eval_results.md) export: a second kernel
builds each leaf's `LeafEvalInputEncoder` row straight into the TensorRT
net's input buffers (`NeuralNet::device_ptr`, `predict_device_inputs`), in
batches of the net's maximum rows, and nothing but the three win/draw/loss
floats per leaf comes back to the host. The device-built rows are checked
against the CPU encoder on the CPU reference's end states (`CpuLeaf`), and
they are identical, join planes included. The model is the conv trunk from
the `leaf-conv-nojoin` run (generation 77, exported by
`python -m scribblez.deep_bot_leaf_eval.onnx_export`).

Measured as above (65,536 rollouts, FP16):

| Position | Rollout kernel | Leaf encoder | Leaf model | Rollouts + leaves |
|----------|----------------|--------------|------------|-------------------|
| seed 2   | 540K/s         | 3.0M rows/s  | 272K/s     | 171K/s            |
| seed 1   | 138K/s         | 2.9M rows/s  | 267K/s     | 88K/s             |

The model, not the encoder, is the leaf's cost, near the 256K evals/s the
[results](../deep_bot_leaf_eval_results.md) measured for this trunk. The same
leaves through the host route, for comparison: the CPU encoder runs about 71K rows/s per thread (so about 1.5M/s over the machine),
and `predict()` with the 85 KB rows uploaded from host memory manages 48K
evals/s: the upload, not the network, bounds it. That is the case for keeping
the leaf on the device: at parity on the rollouts alone, the leaf network
would already run 5-6x faster fed from the kernel than from the host.

### Batches of candidates

A launch takes a batch (`sim/hasty_rollout_task.h`'s `RolloutBatch`):
groups, which are what one chooser's candidates at one position share (the
unseen pool, the seed, and the opponent's known leave per rollout index);
starts, one per candidate applied to the board; and tasks, one per (start,
rollout index), one block each. Rollout `i` of every start in a group draws
the same opponent rack and the same bag order, and a task's outcome depends
only on its start and rollout index, so a batch can mix any rounds of any
positions. An exchange's tiles go back into the bag after the chooser's
refill, and a candidate that goes out with the bag empty ends the game at the
start (exact, no plies). A rollout whose bag is empty after its two plies
keeps playing hasty moves to the game's end instead of stopping at a leaf:
the leaf network trains only on turns that began with tiles in the bag, and
with both racks known and nothing left to draw, the hasty playout is the
exact value it would only approximate. Two scoreless turns in a row end such
a game as six would, since the position can no longer change. At seed 6
after 16 plies, where most rollouts empty the bag, that took the kernel from
0.50 s to 0.77 s for 65,536 tasks; on positions with a fuller bag nothing
changes. `GpuRolloutValues` (`gpu/gpu_rollout_values.h`) runs a batch through
the kernel, the device encoder and the leaf network and returns each task's
win/draw/loss from the chooser's side (exact when the game ended);
`CpuRolloutValues` is the same interface over the CPU reference, with a leaf
function, for the test suites.

`gpu_rollout_demo` makes the mover's top 15 moves plus the best exchange
one group, and every other rollout seats a known opponent leave. Measured
2026-10-23 on the dev laptop at 4,096 rollouts per candidate (65,536 tasks),
with another process holding about 30% of the GPU, so the rates are a floor:

| Position (rack) | CPU rollouts/s | Kernel rollouts/s | With the leaf network |
|-----------------|----------------|-------------------|-----------------------|
| seed 1 (AAGLNOR) | 41,000        | 249,000           | 116,000               |
| seed 2 (AEILSTT) | 119,000       | 444,000           |                       |
| seed 3 (AHILUVY) | 77,000        | 367,000           |                       |

All 65,536 tasks matched the CPU reference on all three, and so did the
device's leaf rows and win values on the first 2,048. Rows per leaf network
call (512 to 8,192) made no difference beyond the contention's noise.

`py/tools/gpu_parity.py` runs a leaf eval TensorRT-versus-PyTorch family
(`test_leaf_eval_parity`) and the demo on 7 candidates x 256 rollouts with
that family's random model, rows and values compared.

Not done, in the order they look worth trying: an L2 persistence window for
the word map and leave table so the blank traffic does not evict them; the
opponent-ply table; a loose per-span equity bound against the best exchange
to skip spans before probing; dropping the two-blank table (rare, ~200 MB)
in favor of designation enumeration for those items; 128-thread blocks for
occupancy.

