# Endgame solver benchmark results

Measurements from `endgame_bench` (see `engine/apps/endgame_bench.cpp`) on the
development machine (Docker container, NWL23 lexicon, Release build). They
characterize the `hastybot-endgame` agent as a function of the solver's node
budget (default 220).

**Hardware**: Intel Core i7-13850HX (28 hardware threads), 64 GB RAM, inside
the project's Docker container, Release build.

## Methodology

We measure the endgame solver along two dimensions: **speed** and **skill**.

- **Speed**: how fast is the endgame solver, in terms of impact on self-play
  game throughput relative to a hasty-vs-hasty baseline?
- **Skill**: how much does enabling the endgame solver improve WDL stats
  against hasty?

Skill is measured by capturing each seeded hasty-vs-hasty game's first
bag-empty position, then sweeping a synthetic score margin m from -N to +N
(the first actor's point of view; scores are set to (m, 0)). Both agent
types decide off spread alone, so the margin fully defines the position, and
comparing the solver seat's win% against the hasty baseline's from the
identical seat isolates the solver's effect; HastyBot's moves are
score-independent, so one hasty playout per game fixes the baseline at every
margin.

Speed uses an operation-count model: modeled time = a x logical move
generations, with a calibrated single-threaded against real playouts, and
the model's mean per-run relative error reported alongside. All reported
numbers are deterministic, so per-game multithreading and the tool's reuse
optimizations (descending-budget reuse, the solver's move-generation memo)
cannot distort them.

The solver maintains node move lists incrementally along the search path
(PathMoveLists; see `EndgameSolver::set_incremental_movegen`), which changes
the real cost of a logical move generation but not its count nor any solve
result. `--incremental=0` disables the maintenance for A/B runs; every skill
and head-to-head table below is bit-identical under both settings, and the
speed sections quote both calibrations.

## Skill vs hasty

100 games, seed 1. y is the solver seat's win% minus the hasty baseline's
win% at the same margin -- e.g. "+10%" means the solver's win% is 10 points
above hasty's from the identical seat (a win counts 1, a draw 0.5). The
final column is the hasty baseline's own win% at that margin.

| margin | budget 100 | 200 | 400 | 800 | 1600 | hasty win% |
|---|---|---|---|---|---|---|
| -100 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 2.5% |
| -80 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 8.0% |
| -60 | +0.0% | +0.0% | +1.0% | +1.5% | +2.5% | 10.0% |
| -40 | +4.0% | +5.0% | +8.0% | +14.0% | +10.0% | 19.5% |
| -20 | +8.0% | +10.5% | +15.0% | +17.0% | +18.0% | 40.5% |
| 0 | +6.5% | +9.5% | +10.5% | +10.5% | +11.5% | 75.5% |
| 20 | +1.0% | +2.0% | +3.0% | +3.0% | +2.0% | 91.0% |
| 40 | +1.0% | +1.0% | +1.0% | +1.0% | +1.0% | 96.0% |
| 60 | +0.0% | +0.0% | +1.0% | +2.0% | +2.0% | 97.0% |
| 80 | +0.0% | +0.0% | +0.0% | +1.0% | +1.0% | 99.0% |
| 100 | +0.0% | +0.0% | +0.0% | +1.0% | +1.0% | 99.0% |

Readings:

- Skill concentrates at contested margins and vanishes at both extremes --
  decided games are converted equally by everyone.
- The peak sits at slightly-losing margins (+18.0% at margin -20, budget
  1600): the solver rescues games hasty loses more than it protects games
  hasty already wins, since hasty's baseline is already 91%+ at winning
  margins. (Budget 800 beating 1600 at margin -40 is 100-game sampling
  noise, not a real non-monotonicity.)
- Skill is meaningful even at the smallest budget (+8.0% at margin -20,
  budget 100).

## Modeled cost vs hasty

Same margin axis; modeled solver-seat endgame ms per playout (see
Methodology). Calibration for this run: a = 35.9 us/movegen with the
incremental move-list maintenance on (the production setting), 65.8 us with
it off (`--incremental=0`) -- a 1.83x per-generation saving. Mean per-run
relative error 83% -- the error reflects per-playout fixed overheads and the
root-vs-derived generation cost split that the one-term model folds into a,
so treat the table as relative structure more than absolute ms.

| margin | 100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|
| -100 | 0.40 | 1.16 | 4.08 | 13.26 | 28.00 |
| -80 | 0.40 | 1.11 | 3.97 | 12.85 | 27.17 |
| -60 | 0.40 | 1.11 | 3.85 | 12.64 | 26.53 |
| -40 | 0.40 | 0.95 | 3.17 | 11.03 | 23.81 |
| -20 | 0.46 | 0.79 | 2.62 | 6.65 | 13.44 |
| 0 | 0.38 | 0.87 | 2.28 | 4.76 | 8.31 |
| 20 | 0.33 | 0.70 | 2.49 | 6.28 | 12.79 |
| 40 | 0.32 | 0.69 | 2.72 | 7.28 | 15.20 |
| 60 | 0.32 | 0.69 | 2.88 | 7.61 | 16.31 |
| 80 | 0.31 | 0.73 | 3.01 | 7.91 | 16.83 |
| 100 | 0.32 | 0.73 | 2.92 | 7.72 | 16.44 |

Readings:

- Cost is lowest near the middle and at winning margins, and peaks at
  deep-losing margins (margin -100 vs +100 at budget 1600: 28 vs 16 ms).
- A win verdict rests on a single winning line, so the root scan cuts off
  the moment one is found; a loss proof must refute every root move, and at
  the endgame's first positions both racks are full, so the out-play
  futility sets are empty and nothing prunes the refutation.

## Whole-game throughput

`--mode=games --games=200 --seed=7 --threads=1` (projections respected, as
self-play generation runs) times endgame-vs-endgame self-play against the
hasty-vs-hasty baseline (~4.6 ms/game):

| budget | ratio (incremental on) | ratio (`--incremental=0`) |
|---|---|---|
| 100 | 1.07x | 1.08x |
| 200 | 1.18x | 1.29x |
| 400 | 1.88x | 2.58x |
| 800 | 4.01x | 6.33x |
| 1600 | 7.02x | 11.65x |

The incremental maintenance roughly halves the solver's per-game overhead
above the hasty baseline (e.g. at budget 1600: 27.5 vs 48.9 ms/game of
endgame overhead). The same mode's head-to-head table reports each budget's
win% and W/D/L record against plain hasty, bucketed by baseline bag-empty
spread; it is bit-identical under both settings (budgets 800 and 1600: 52.0%
overall, 54.2% in the 0-19 bucket).

## Proof certificates

Every class proof carries a certificate: the class-critical side's moves
come from fresh narrow-window re-proofs at each position of the walk, run
over the warm transposition table outside the node budget (tracked in
`EndgameResult::certificate_nodes`, negligible in practice), gated on a
terminal check that the walk lands on the proven class. This is enforced by
the test suite, plus curated GCG endgames under `engine/tests/data/` pinned
to their proven class and proof cost by `EndgameGcgCases`.

## Analyzing a single position

`endgame_tool --gcg FILE` solves one GCG endgame position (bag empty; the
mover's rack from a #RackN pragma, the opponent's derived from the board)
with a verbose trace of the machinery: the replier's out-play set, every root
move's block-or-outscore futility bound ("needs >= +X to reach a draw"), each
deepening iteration's verdict, the certificate walk, and the projected line.

## Reproducing

```
target/engine/endgame_bench --mode=endgames --games=100 --seed=1 \
    --budgets=100,200,400,800,1600 --margin-max=100 --margin-step=20 \
    --threads=8
target/engine/endgame_bench --mode=games --games=200 --seed=7 \
    --threads=1 --budgets=100,200,400,800,1600
target/engine/endgame_tool  --gcg engine/tests/data/FOE.gcg
```

Append `--incremental=0` to either endgame_bench command for the
scratch-generation A/B.
