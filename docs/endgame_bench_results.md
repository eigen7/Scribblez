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

## Skill vs hasty

100 games, seed 1. y is the solver seat's win% minus the hasty baseline's
win% at the same margin -- e.g. "+10%" means the solver's win% is 10 points
above hasty's from the identical seat (a win counts 1, a draw 0.5). The
final column is the hasty baseline's own win% at that margin.

| margin | budget 100 | 220 | 400 | 1600 | hasty win% |
|---|---|---|---|---|---|
| -100 | +0.0% | +0.0% | +0.0% | +0.0% | 2.5% |
| -80 | +0.0% | +0.0% | +0.0% | +0.0% | 8.0% |
| -60 | +0.0% | +0.0% | +0.0% | +2.5% | 10.0% |
| -40 | +0.0% | +1.0% | +1.0% | +9.0% | 19.5% |
| -20 | +0.0% | +0.0% | +0.5% | +10.0% | 40.5% |
| 0 | +0.5% | +1.0% | +1.0% | +8.0% | 75.5% |
| 20 | +0.0% | +0.0% | +0.0% | +3.0% | 91.0% |
| 40 | +0.0% | +0.0% | +0.0% | +1.0% | 96.0% |
| 60 | +0.0% | +0.0% | +0.0% | +2.0% | 97.0% |
| 80 | +0.0% | +0.0% | +0.0% | +1.0% | 99.0% |
| 100 | +0.0% | +0.0% | +0.0% | +1.0% | 99.0% |

Readings:

- Skill concentrates at contested margins and vanishes at both extremes --
  decided games are converted equally by everyone.
- The peak sits at slightly-losing margins (+10% at margin -20, budget
  1600): the solver rescues games hasty loses more than it protects games
  hasty already wins, since hasty's baseline is already 91%+ at winning
  margins.
- Low budgets add little anywhere: a solve is declined when the position has
  more root moves than the budget, so the budget gates engagement.

## Modeled cost vs hasty

Same margin axis; modeled solver-seat endgame ms per playout (see
Methodology). Calibration for this run: a = 109.6 us/movegen, mean per-run
relative error 116% -- the error reflects per-playout fixed overheads that
the one-term model folds into a, so treat the table as relative structure
more than absolute ms.

| margin | 100 | 220 | 400 | 1600 |
|---|---|---|---|---|
| -100 | 2.47 | 7.84 | 18.59 | 124.8 |
| -80 | 2.47 | 7.84 | 18.59 | 124.2 |
| -60 | 2.47 | 7.83 | 18.59 | 123.6 |
| -40 | 2.43 | 7.62 | 17.78 | 119.0 |
| -20 | 2.37 | 7.38 | 17.09 | 115.9 |
| 0 | 2.23 | 6.82 | 16.00 | 111.0 |
| 20 | 2.15 | 6.58 | 15.61 | 109.6 |
| 40 | 2.19 | 6.65 | 15.71 | 109.8 |
| 60 | 2.19 | 6.65 | 15.70 | 109.7 |
| 80 | 2.19 | 6.65 | 15.69 | 110.0 |
| 100 | 2.19 | 6.65 | 15.69 | 110.0 |

Readings:

- Cost falls toward decided margins: class proofs land sooner and short-
  circuit deepening.
- Proving the opponent's win is systematically dearer than proving our own
  (margin -100 vs +100 at budget 1600: 124.8 vs 110.0 ms, ~14%), because a
  loss proof must refute every root move while a win proof needs a single
  winning line.

## Whole-game throughput

`--mode=games` (projections respected, as self-play generation runs) times
endgame-vs-endgame self-play against the hasty-vs-hasty baseline (~5
ms/game): budget 100 -> 1.25x, 220 -> 2.11x, 400 -> 5.93x, 1600 -> 33.7x.
The same mode's head-to-head table reports each budget's win% and W/D/L
record against plain hasty (no spread column).

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
    --budgets=100,220,400,1600 --margin-max=100 --margin-step=20 \
    --threads=8
target/engine/endgame_bench --mode=games --games=200 --seed=7 \
    --threads=1 --budgets=100,220,400,1600
target/engine/endgame_tool  --gcg engine/tests/data/FOE.gcg
```
