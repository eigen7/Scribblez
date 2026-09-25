# Endgame solver benchmark results

How the `hastybot-endgame` agent's skill and cost vary with the endgame
solver's node budget, measured with `endgame_bench`
(`engine/apps/endgame_bench.cpp`).

**Summary.** Solving the endgame buys up to +18.8 points of win rate over
plain HastyBot from the same seat, concentrated at slightly-losing margins.
Skill saturates above budget 800, while cost keeps doubling with the budget.
Budgets 200 and 400 buy the most skill per unit of generation time, and **400
is the shipped default** (`EndgameSolver::Params::budget`): the last rung
before the returns collapse, at 2.00× the cost of a HastyBot-vs-HastyBot game.

**Setup.** Intel Core i7-13850HX (28 hardware threads), 64 GB RAM, inside the
project's Docker container; NWL23 lexicon; Release build.

## Methodology

The solver is measured along two dimensions, **skill** and **cost**, both from
one sweep. The sweep captures each seeded HastyBot-vs-HastyBot game's first
bag-empty position and replays it once per (margin, budget) cell:

- a synthetic score margin m sets the first mover's scores to (m, 0);
- an EndgameHastyBot takes that seat and a plain HastyBot replies, with
  projections respected as in self-play generation.

Both agent types decide off the spread alone, so the margin fully defines the
position, and comparing the solver seat's result against the HastyBot
baseline's from the identical seat isolates the solver's effect. HastyBot's
moves do not depend on the score, so one HastyBot playout per game fixes the
baseline at every margin.

The margin axis is asymmetric, [-80, +40], because HastyBot's own win rate is
not symmetric about zero: it is already 80% at margin 0 and 97% by +40. The
contested region sits below zero, so that is where the axis is spent.

**Skill** is the solver seat's win rate minus the HastyBot baseline's at the
same margin (a win counts 1, a draw 0.5). It uses win rate rather than spread
because under the production `spread_matters=false` setting the solver
optimizes the win/draw/loss class and stops at its proof; spread is not what
it is trying to move (only its fallback, when no proof lands, is). Skill is
deterministic, since greedy tie-breaks are fixed and each cell is a pure
function of position, margin and budget. It is averaged over all 1000 games.

**Cost** is measured, not modeled: `EndgameAgent<HastyBot>` times its own
`solve()` calls and the sweep sums them over a playout. It is quoted as a
multiple of a plain HastyBot-vs-HastyBot game timed under the same conditions
(4.14 ms/game here), because the question is what solving the endgame costs
self-play generation. The whole-game throughput section below uses the same
unit. 1.00× is a free endgame.

A wall-clock number is only meaningful when nothing else competes for cores,
caches and clock. So the first `--time-games` games and the baseline run
alone on a single thread, and only they feed the cost curve; the rest run
across all workers and contribute only their deterministic results. Cost is
the one reported number that is not reproducible bit-for-bit.

The sweep exploits one redundancy, the **budget-nesting skip**. At a fixed
margin, budgets run in descending order. Once a run's deepest solve stayed
under a smaller budget b', re-running at b' would be bit-identical, so its
result and measured time stand for b' as well. This fills about half the grid
(310,315 of 605,000 cells in the run below).

A second redundancy is left unused. A solve depends on the margin only through
the fixed ±1 class window, so neighbouring margins usually execute
identically: 88% of adjacent pairs in a 20-game probe at budget 1600, which
would collapse roughly 8 cells into 1. Exploiting it needs a certificate of
the margin interval a solve is invariant over, which means threading margin
sensitivity through every comparison in the search. That is not worth it
while the full sweep runs in 11 minutes.

## Skill vs. HastyBot

1000 games, seed 1. The y axis is the solver seat's win rate minus the
HastyBot baseline's at the same margin: "+10" means the solver wins 10
percentage points more often than HastyBot does from the identical seat. The
lower panel shows HastyBot's own win rate for context.

![Endgame solver skill vs start-of-endgame margin](images/endgame_skill_vs_margin.svg)

Readings:

- Skill concentrates where the game is contested and vanishes at both
  extremes, where every player converts a decided endgame.
- The peak sits at slightly-losing margins (+18.8 at margin -19, budget 1600),
  not at 0. The solver rescues games HastyBot loses more than it protects
  games HastyBot already wins, because HastyBot's baseline is already 80% at
  margin 0.
- Even the smallest budget buys real skill (+6.2 at margin -20, budget 100).
- Skill saturates above budget 800. 1600 leads across the losing half of the
  axis (+14.2 vs. +13.7 at margin -30, +18.8 vs. +18.2 at the peak), but the
  two curves are within 0.7 points of each other everywhere in [-17, +6],
  where 800 sometimes leads. Doubling past 800 buys a fraction of what
  doubling to it did.

**Why the spread fallback exists.** A class-only search that proves nothing
still has to return a move, and the move it has is ranked by a bound rather
than by points; deepening only reshuffles such moves. Playing them makes the
budget ladder run backwards: 1600 scored *below* 800 across the whole
contested band (+11.5 vs. +13.2 at margin -30). Of the 45 games the two
budgets disagreed on, 43 were in the class-unknown bucket, so the effect lies
entirely in the unproven path, not in how proven positions are handled. So
`EndgameSolver::solve_class_first` spends the reserved half of the budget on a
spread pass when no proof lands, as the `spread_matters` driver does. Measured
at margin -30, 1000 games:

| class-only driver | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|
| narrow-window move when unproven | +4.2 | +9.4 | +13.2 | +11.5 |
| spread fallback when unproven | +4.1 | +9.3 | +13.7 | **+14.2** |

Two variants ruled out the other candidate explanations, both concerning
proven losses rather than unproven positions. Conceding a proven loss through
its certificate instead of playing it out is worth +0.1 (`--projections=0`).
Handing lost positions to HastyBot instead of playing the solver's arbitrary
class-equal move is worth +0.2. Neither restored monotonicity; the fallback
did.

## Cost

What one self-play game costs when its endgame is solved, as a multiple of a
HastyBot-vs-HastyBot game (see Methodology), on a log scale. 1.00× is free.

![Endgame solver cost vs start-of-endgame margin](images/endgame_cost_vs_margin.svg)

Readings:

- Cost is lowest near the middle and at winning margins, and peaks at
  deep-losing margins: 7.88× at margin -80 vs. 3.88× at margin 0, budget 1600.
- The asymmetry is structural. A win verdict rests on a single winning line,
  so the root scan stops as soon as it finds one. A loss proof must refute
  every root move, and at the endgame's first positions both racks are full,
  so the out-play futility sets are empty and nothing prunes the refutation.
- The curve flattens below margin -40: from there down the position is lost
  whatever the solver does, so the search is the same full refutation every
  time.
- Cost grows far faster than skill above budget 400. At margin -20, budget
  800 buys 4.0 points of skill over 400 for 1.8× more game time; 1600 gives
  0.1 of that back for 3.3× more.
- Skill per unit of generation time picks out 200 and 400 together: +8.9 at
  1.20× and +14.6 at 2.00× are 7.4 and 7.3 points per multiple of a
  HastyBot-vs-HastyBot game, against 4.7 at budget 800 and 2.5 at 1600.
  **400 is the shipped default**: the last rung before the returns collapse,
  and of the two, the one that buys strength rather than games. A generator
  that optimizes win rate rather than score margin should reach the bag-empty
  point in contested positions more often than HastyBot does, putting more
  weight on the band where the solver earns its keep.

## Whole-game throughput

`--mode=games --games=200 --seed=7 --threads=1` (projections respected, as in
self-play generation) times EndgameHastyBot-vs-EndgameHastyBot self-play
against the HastyBot-vs-HastyBot baseline (~4.7 ms/game). The right column
disables the solver's incremental move-list maintenance (`PathMoveLists`) with
`--incremental=0`, which changes speed but no result: every skill number in
this document is bit-identical under both settings.

| budget | ratio (incremental on) | ratio (`--incremental=0`) |
|---|---|---|
| 100 | 1.05x | 1.10x |
| 200 | 1.20x | 1.41x |
| 400 | 2.00x | 3.08x |
| 800 | 3.99x | 7.15x |
| 1600 | 7.35x | 13.75x |

These are whole self-play games at their natural margins, so they are in the
same unit as the cost figure above and land near its mid-margin values.
Incremental maintenance roughly halves the solver's per-game overhead above
the HastyBot baseline (at budget 1600: 29.5 vs. 59.5 ms/game of endgame
overhead).

The same mode also reports each budget's win rate and W/D/L record against
plain HastyBot, bucketed by the baseline's bag-empty spread. It is
bit-identical under both settings, and it saturates where the margin sweep
says it should: budgets 800 and 1600 post the same 52.0% overall and the same
54.2% in the 0–19 bucket.

## Proof certificates

Every class proof carries a certificate: a line of play whose class-critical
moves come from fresh narrow-window re-proofs at each position of the walk.
The re-proofs run over the warm transposition table outside the node budget
(tracked in `EndgameResult::certificate_nodes`; negligible in practice), and a
terminal check confirms that the walk lands on the proven class. The test
suite enforces this, including curated GCG endgames under
`engine/tests/data/` pinned to their proven class and proof cost by
`EndgameGcgCases`.

## Analyzing a single position

`endgame_tool --gcg FILE` solves one GCG endgame position (bag empty; the
mover's rack from a `#RackN` pragma, the opponent's derived from the board)
and traces the machinery: the replier's out-play set, every root move's
block-or-outscore futility bound ("needs >= +X to reach a draw"), each
deepening iteration's verdict, the certificate walk, and the projected line.

## Reproducing

```
target/engine/endgame_bench --mode=endgames --games=1000 --seed=1 \
    --budgets=100,200,400,800,1600 --margin-min=-80 --margin-max=40 \
    --margin-step=1 --threads=24 --time-games=100 \
    > docs/data/endgame_margin_sweep.txt
py/tools/plot_endgame_bench.py docs/data/endgame_margin_sweep.txt
target/engine/endgame_bench --mode=games --games=200 --seed=7 \
    --threads=1 --budgets=100,200,400,800,1600
target/engine/endgame_tool  --gcg engine/tests/data/FOE.gcg
```

The `--projections=0` variant above is one margin of the same sweep:

```
target/engine/endgame_bench --mode=endgames --games=1000 --seed=1 \
    --budgets=200,400,800,1600 --margin-min=-30 --margin-max=-30 \
    --threads=24 --time-games=0
```

The sweep takes about 11 minutes, most of it the single-threaded timing
phase. `docs/data/endgame_margin_sweep.txt` is the captured run the two
figures are drawn from, and the plot script regenerates them from it. Append
`--incremental=0` to either `endgame_bench` command for the incremental
move-list A/B.
