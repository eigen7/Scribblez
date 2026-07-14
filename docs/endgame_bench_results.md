# Endgame solver benchmark results

Measurements from `endgame_bench` (see `engine/apps/endgame_bench.cpp`) on the
development machine (Docker container, single-threaded runs, NWL23 lexicon,
Release build). They characterize the `hastybot-endgame` agent's cost/strength
tradeoff as a function of its `--endgame-nodes` budget and are the basis for
the shipped default of 200.

**Hardware**: Intel Core i7-13850HX (28 hardware threads), 64 GB RAM, inside
the project's Docker container, Release build.

## Methodology

- **Cost** (`--mode=games`): mean wall-time per game of
  endgame-vs-endgame self-play, as a ratio to the hasty-vs-hasty baseline
  (~5 ms/game, ~215 games/s single-threaded), same seeds across configs.
- **Strength** (`--mode=games` head-to-head): the endgame bot's mean final
  spread against a plain greedy HastyBot. Each seed is played twice with the
  seats mirrored and the pair averaged, which cancels per-seed tile-draw luck —
  the dominant variance source. Unpaired spread estimates swing by +/- 5+
  points below ~1000 games and are not usable.
- Aggregates below pool 6-8 independent seed shards per budget (900-4800 games
  per row); the quoted uncertainty is the standard error across shards.

## Cost and strength vs node budget

| `--endgame-nodes` | game-time cost | head-to-head vs greedy HastyBot |
|---|---|---|
| 100 | 1.19x | +0.15 +/- 0.02 pts/game |
| **200 (default)** | **1.85x** | **+0.40 +/- 0.04** |
| 400 | 5.7x | +0.97 +/- 0.05 |
| 1600 | 30x | +4.83 +/- 0.08 |
| 5000 | ~73x | +5.48 +/- 0.22 |
| 20000 | ~202x | +5.58 +/- 0.35 |

Strength saturates at roughly **+5.5 points/game — the full value of exact
endgame play over greedy static-equity play** in hasty-level self-play. Budget
1600 captures ~90% of the ceiling at a tenth of the cost of 20000.

Strength rises superlinearly through the low budgets because the budget gates
*engagement*, not search noise: a solve is declined up front when its root move
count exceeds the budget, and the agent falls back to HastyBot's static-equity
move unless the solver completed its first iteration. Raising the budget
admits richer (more consequential) endgames to a real search; it never makes
an admitted search noisier.

## Per-solve cost (`--mode=solves`, 80 captured bag-empty positions)

| budget | mean us | p50 us | p95 us | mean depth | % differ | % proven | exit save (nodes) |
|---|---|---|---|---|---|---|---|
| 100 | 406 | 161 | 1503 | 0.4 | 36% | 14% | 33% |
| 200 | 1054 | 304 | 4088 | 0.6 | 38% | 19% | 25% |
| 400 | 3590 | 1419 | 16691 | 0.8 | 39% | 22% | 22% |
| 1600 | 36348 | 18123 | 139361 | 1.5 | 45% | 30% | 16% |

Solves are movegen-bound (roughly 30-70k nodes/s; each node and each greedy
playout ply runs a full move generation), so per-solve cost tracks nodes spent
almost linearly once positions stop being declined. "Mean depth" reads low
because proven positions report the depth that proved them and stop there.
"Exit save" is the node share the proven short-circuit avoids versus letting
deepening run on the same positions -- note it overstates the wall-time
saving (~8% at budget 200): the avoided nodes are the cheapest in the system
(shallow re-iterations over tiny terminal-heavy trees with a warm table),
while the unprovable rich positions own most of the wall time.

## Proven verdicts and outplay-threat pruning

Two search mechanisms shape where the budget goes:

- **Proven verdicts**: a result is proven when it rests entirely on real game
  ends (no greedy-playout leaf). Iterative deepening stops the moment an
  iteration returns a proven verdict -- the exact spread in the full window, a
  settled win/draw/loss class in the first-win window -- so small positions
  stop at the depth that proves them instead of iterating to the budget. This
  is why the solves-mode "mean depth" column reads low: proven positions
  report their proving depth.
- **Outplay-threat pruning**: when the mover's leave keeps two out-plays that
  no single reply can block (halo geometry over rows/columns, conservative
  toward not firing), opponent replies scoring too little to beat the
  guaranteed out-line are skipped without recursion. Sound (A/B-identical
  values and best moves over the randomized suite) and proven-grade.

Measured effect at the budgets that matter: **small**. Cost at the 2x point
dropped from ~1.97x to ~1.85x and strength there is unchanged (+0.40); the
node cut from threat pruning is 2-4%. The mechanics fire exactly where
positions are cheap (small, provable ones); the positions that dominate cost
cannot be proven within these budgets and still burn to the cap. The
mechanisms matter structurally -- proofs make first-win exits and future
proof-caching sound -- but they do not move the cost/strength frontier by
themselves.

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
| 50 | 1.07x | 1.04x |
| 100 | 1.26x | 1.22x |
| 200 | 1.97x | 1.89x |
| 400 | 6.13x | 5.90x |
| 800 | 16.70x | 16.26x |
| 1600 | 29.72x | 31.33x |

(`--mode=games --games=200 --seed=7 --threads=1`, with and without `--wld`;
single-shard, illustrative rather than a strength-grade sample.)

Strength, both modes on identical seeds: 6 shards x 800 games = 4800 games
per budget and mode, same paired-seat protocol as the spread-mode table above.
Under wld the mean spread shrinks by construction (the solver stops maximizing
margin once a win is proven), so the W/D/L record is the number that judges
wld strength:

| `--endgame-nodes` | mode | mean eg spread | W | D | L | W-L margin |
|---|---|---|---|---|---|---|
| 200 | spread | +0.43 +/- 0.02 | 2399 | 17 | 2384 | +0.31pp |
| 200 | wld | +0.20 +/- 0.02 | 2398 | 17 | 2385 | +0.27pp |
| 400 | spread | +0.86 +/- 0.06 | 2400 | 24 | 2376 | +0.50pp |
| 400 | wld | +0.47 +/- 0.06 | 2399 | 24 | 2377 | +0.46pp |
| 1600 | spread | +4.61 +/- 0.11 | 2479 | 18 | 2303 | +3.67pp |
| 1600 | wld | +2.44 +/- 0.08 | 2471 | 17 | 2312 | +3.31pp |

The comparison is one-sided: **at a fixed node budget, wld mode is dominated
by spread mode** -- equal-or-worse W/D/L at every budget (well within noise at
200/400, slightly behind at 1600) while banking roughly half the spread, at
the same wall-time cost. This follows from the budget semantics: a hard node
cap spends the whole budget regardless of window width, so the narrow window
only redistributes nodes toward depth rather than saving anything, and the
deeper-but-margin-blind search does not convert into extra wins against this
opponent. First-win search pays off under *completion-based* budgets, where a
narrow window finishes its proof sooner and returns the saved time (the
regime pre-endgame solvers run endgames in); under this solver's node cap it
has no winning operating point, and spread mode is the right default. The
mode remains available for callers that need cheap win/loss proofs and pair
it with an early exit on proof.

## Reproducing

```
target/engine/endgame_bench --mode=games  --games=800 --seed=S --budgets=B --threads=1 [--wld]
target/engine/endgame_bench --mode=solves --games=30  --seed=42 --budgets=100,200,400,1600,5000,20000
```

For strength numbers, run several `--seed` shards per budget and pool them;
single-shard head-to-head spreads carry a few points of standard error even
with seat mirroring. `--wld` applies to `--mode=games` only; it makes both the
endgame-vs-endgame cost sweep and the endgame-vs-hasty head-to-head use the
first-win window.
