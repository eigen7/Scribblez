# Hasty rollouts on the GPU

**Status: feasibility analysis and build plan, 2026-10-23; not built.** Whether
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
5. **Integration:** the backend selector in SimRunner and the survey tool,
   batching across candidates, host-side leave inference feeding per-rollout
   opponent leaves.

Deferred: the `K`-ply horizon with the win-table leaf (DeepBot's form), MAGPIE
blank tables, the opponent-ply table.
