# Endgame solver benchmark results

Measurements from `endgame_bench` (see `engine/apps/endgame_bench.cpp`) on the
development machine (Docker container, single-threaded runs, NWL23 lexicon,
Release build). They characterize the `hastybot-endgame` agent's cost/strength
tradeoff as a function of its `--endgame-nodes` budget and are the basis for
the shipped default of 220.

**Hardware**: Intel Core i7-13850HX (28 hardware threads), 64 GB RAM, inside
the project's Docker container, Release build.

## Methodology

- **Cost** (`--mode=games`): mean wall-time per game of
  endgame-vs-endgame self-play, as a ratio to the hasty-vs-hasty baseline
  (~5 ms/game, ~210 games/s single-threaded), same seeds across configs.
  Cost rows come from dedicated single runs on an otherwise idle machine
  (`--games=200 --seed=7 --threads=1`); wall-clock ratios from parallel or
  loaded runs are not comparable.
- **Strength** (`--mode=games` head-to-head): the endgame bot's mean final
  spread against a plain greedy HastyBot. Each seed is played twice with the
  seats mirrored and the pair averaged, which cancels per-seed tile-draw luck --
  the dominant variance source. Unpaired spread estimates swing by +/- 5+
  points below ~1000 games and are not usable.
- Strength rows below pool 6 independent 800-game seed shards per budget
  (4800 games per row); the quoted uncertainty is the standard error across
  shards.

**Why mean spread is the headline strength metric.** The default (full-window)
solver maximizes the exact final spread, and the training pipeline consumes
final scores as value targets (see docs/architecture.md), so points/game
measures both the objective the agent optimizes and the currency the project
cares about. It also accumulates signal in every game: in an endgame that is
already decided, a win/loss record saturates (greedy play converts a big lead
just fine) while optimal play still banks extra points. The W/D/L record is
kept as a sanity column -- exact spread-maximal play in a won position never
un-wins it, so a spread gain paired with a W-L loss would flag a bug -- and it
becomes the primary metric only for the first-win (`--wld`) mode, whose window
deliberately stops maximizing spread. Note the complementary blind spot: on the
self-play distribution most endgames are decided, so neither metric isolates
"conversion skill in close positions"; conditioning the paired analysis on the
bag-empty position (e.g. |spread at bag-empty| <= 20), not on the outcome,
would measure that without engineering artificial states.

## Cost and strength vs node budget

| `--endgame-nodes` | game-time cost | head-to-head vs greedy HastyBot |
|---|---|---|
| 50 | 1.04x | -- |
| 100 | 1.18x | +0.22 +/- 0.02 pts/game |
| 200 | 1.83x | +0.47 +/- 0.03 |
| **220 (default)** | **2.02x** | **+0.54 +/- 0.04** |
| 240 | 2.19x | -- |
| 300 | 2.98x | -- |
| 400 | 5.58x | +0.88 +/- 0.04 |
| 800 | 15.5x | -- |
| 1600 | 30.0x | +4.81 +/- 0.14 |
| 5000 | ~84x | +5.69 +/- 0.19 |
| 20000 | ~224x | +5.62 +/- 0.19 |

The default of 220 is the largest budget whose endgame-vs-endgame games stay
within ~2x the hasty-vs-hasty game time.

Strength saturates at roughly **+5.6 points/game -- the full value of exact
endgame play over greedy static-equity play** in hasty-level self-play. Budget
1600 captures ~85% of the ceiling at a small fraction of the cost.

Strength rises superlinearly through the low budgets because the budget gates
*engagement*, not search noise: a solve is declined up front when its root move
count exceeds the budget, and the agent falls back to HastyBot's static-equity
move unless the solver completed its first iteration. Raising the budget
admits richer (more consequential) endgames to a real search; it never makes
an admitted search noisier.

## Per-solve cost (`--mode=solves`, 80 captured bag-empty positions)

| budget | mean us | p50 us | p95 us | mean depth | % differ | % proven | exit save (nodes) |
|---|---|---|---|---|---|---|---|
| 100 | 433 | 178 | 1549 | 0.4 | 36% | 16% | 35% |
| 200 | 1109 | 321 | 4435 | 0.6 | 36% | 21% | 30% |
| 400 | 3761 | 1250 | 17488 | 0.8 | 39% | 24% | 24% |
| 1600 | 39493 | 17678 | 154802 | 1.6 | 41% | 36% | 20% |
| 5000 | 138489 | 59816 | 824729 | 2.3 | 45% | 51% | 24% |

Solves are movegen-bound (roughly 20-60k nodes/s; each node and each greedy
playout ply runs a full move generation), so per-solve cost tracks nodes spent
almost linearly once positions stop being declined. "Mean depth" reads low
because proven positions report the depth that proved them and stop there.
"Exit save" is the node share the proven short-circuit avoids versus letting
deepening run on the same positions -- note it overstates the wall-time
saving: the avoided nodes are the cheapest in the system (shallow
re-iterations over tiny terminal-heavy trees with a warm table), while the
unprovable rich positions own most of the wall time.

## Proven verdicts and outplay-futility pruning

Two search mechanisms shape where the budget goes:

- **Proven verdicts**: a result is proven when it rests entirely on real game
  ends (no greedy-playout leaf). Iterative deepening stops the moment an
  iteration returns a proven verdict -- the exact spread in the full window, a
  settled win/draw/loss class in the first-win window -- so small positions
  stop at the depth that proves them instead of iterating to the budget.
- **Opponent-outplay futility pruning**: a mover move that provably leaves an
  opponent out-play intact (halo geometry, conservative toward not firing) is
  bounded by the terminal spread of that reply; moves bounded below alpha are
  skipped without recursion, and the bound also demotes provably weak moves in
  the move ordering. The out-play sets are maintained incrementally down the
  search path (one extra move generation per solve, none per node); see
  endgame_solver.h and outplays.h. Sound (A/B-identical values over the
  randomized suites, with any divergent best move verified as an equal-valued
  tie) and proven-grade. Disabling it (`--no-futility` in solves mode) roughly
  doubles the nodes the fixed test batch spends (a 49% cut, per
  EndgameSolver.OutplayFutilityCutsNodes) and drops the % proven column
  substantially (36% -> 29% at budget 1600, 51% -> 30% at 5000).

Their measured effect on the cost/strength frontier is **small**, and the
reason is structural: both mechanisms fire hardest exactly where positions are
cheap (small, provable ones), while the positions that dominate wall time
cannot be proven within these budgets and still burn to the node cap. Since
strength at a fixed budget is engagement-gated (see above) and pruning does
not change which positions pass the decline gate, the head-to-head numbers at
a fixed budget are unchanged within noise; the node and proof gains instead
buy the frontier's price points (the ~2x point sits at budget 220) and the
structural benefits -- proofs make first-win exits sound and are the
foundation for proof caching across turns.

## First-win (WLD) mode

`--endgame-wld` (agent: `EndgameHastyBotAgent::Params.endgame_wld`) pins the
solver's root alpha-beta window to `(EndgameSolver::kFirstWinAlpha,
EndgameSolver::kFirstWinBeta)` = `(-1, +1)` instead of the full
`(-inf, +inf)` spread window. The search then only has to resolve which side
of an even final spread the position falls on -- win, draw, or loss -- rather
than the exact point margin, so proofs land with far less search effort per
ply of true difficulty. The tradeoff is that among winning root moves the
solver returns an arbitrary one instead of the spread-maximal one, and a
proven-lost position (`value <= kFirstWinAlpha`) is meaningless to compare
move-by-move, so `EndgameHastyBotAgent` discards it and plays HastyBot's
static-equity move instead -- final spread still matters to the game log even
in a lost position.

At a fixed `--endgame-nodes` budget, cost is governed by the budget itself
(iterative deepening spends the budget regardless of window width), so wld
mode's wall-time cost roughly tracks spread mode's at the same budget:

| `--endgame-nodes` | spread-mode cost | wld-mode cost |
|---|---|---|
| 50 | 1.04x | 1.08x |
| 100 | 1.18x | 1.26x |
| 200 | 1.83x | 1.91x |
| 400 | 5.58x | 5.92x |
| 800 | 15.5x | 16.4x |
| 1600 | 30.0x | 32.2x |

Strength, both modes on identical seeds: 6 shards x 800 games = 4800 games
per budget and mode, same paired-seat protocol as the spread-mode table above.
Under wld the mean spread shrinks by construction (the solver stops maximizing
margin once a win is proven), so the W/D/L record is the number that judges
wld strength:

| `--endgame-nodes` | mode | mean eg spread | W | D | L | W-L margin |
|---|---|---|---|---|---|---|
| 200 | spread | +0.47 +/- 0.03 | 2410 | 6 | 2384 | +0.54pp |
| 200 | wld | +0.25 +/- 0.01 | 2406 | 6 | 2388 | +0.38pp |
| 400 | spread | +0.88 +/- 0.04 | 2420 | 6 | 2374 | +0.96pp |
| 400 | wld | +0.49 +/- 0.04 | 2416 | 6 | 2378 | +0.79pp |
| 1600 | spread | +4.81 +/- 0.14 | 2497 | 14 | 2289 | +4.33pp |
| 1600 | wld | +2.60 +/- 0.13 | 2488 | 13 | 2299 | +3.94pp |

The comparison is one-sided: **at a fixed node budget, wld mode is dominated
by spread mode** -- equal-or-worse W/D/L at every budget while banking roughly
half the spread, at the same wall-time cost. This follows from the budget
semantics: a hard node cap spends the whole budget regardless of window width,
so the narrow window only redistributes nodes toward depth rather than saving
anything, and the deeper-but-margin-blind search does not convert into extra
wins against this opponent. First-win search pays off under *completion-based*
budgets, where a narrow window finishes its proof sooner and returns the saved
time (the regime pre-endgame solvers run endgames in); under this solver's
node cap it has no winning operating point, and spread mode is the right
default. The mode remains available for callers that need cheap win/loss
proofs and pair it with an early exit on proof.

## Reproducing

```
target/engine/endgame_bench --mode=games  --games=800 --seed=S --budgets=B --threads=1 [--wld]
target/engine/endgame_bench --mode=solves --games=30  --seed=42 --budgets=100,200,400,1600,5000
```

For strength numbers, run several `--seed` shards per budget and pool them;
single-shard head-to-head spreads carry a few points of standard error even
with seat mirroring. For cost numbers, run one config at a time on an idle
machine. `--wld` applies to `--mode=games` only; it makes both the
endgame-vs-endgame cost sweep and the endgame-vs-hasty head-to-head use the
first-win window.
