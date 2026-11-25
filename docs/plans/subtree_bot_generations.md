# Plan: one workload that advances SubtreeBot's generations

**Status: proposed 2026-11-13; steps 1 to 5 of its
[build order](#build-order) are built.** This is step 11 of SubtreeBot's
[build order](subtree_bot.md#build-order) as a dashboard workload, and step 3
of [the order of the remaining work](subtree_bot_roadmap.md). Its
measurements inform the next step; no pass or kill thresholds are fixed in
advance.

**Decision.** A new workload, `subtree_bot`, whose single tag is a line of
SubtreeBot generations. The operator creates the tag, adds a trainer and any
number of GPU generators, and adds or removes generator machines while it
runs.

- **Generators** play SubtreeBot against itself with the tag's newest models and
  deliver chunks that hold the games and everything the three trainers read.
- **One trainer** trains the leave inference model, the leaf network and the
  policy on each completed generation, and publishes the three exports
  together as a **release**.
- **Generators pick up each release** at their next chunk. Nothing is pinned
  at task creation except the seeds.
- **A measure role** plays each release against HastyBot and against release
  0, records its picks at the known positions, and charts the results by
  release. BestBot is too slow for the loop: a match against it is run on
  demand, against any kept release.

The tag reuses the generational machinery as it stands: one data home beside
one trainer, staged chunks, generation manifests and the `open_ahead` gate
([generational_training.md](../generational_training.md)). What is new is
the trainer that holds three models, the release, and the path that carries a
release to a running generator.

## Why one workload

Today the three models are three workloads (`leave_inference`,
`deep_bot_leaf_eval`, `policy_set`), each with its own HastyBot generators,
and `policy_set` pins one export of the other two when its tag is created. A
generation of SubtreeBot under that arrangement is three new tags made by hand
in the right order, and the games are played three times.

SubtreeBot's games serve all three trainers at once. One game record gives the
leave inference model its rows (replay recovers every leave), the leaf
network its rows (every turn, labelled by the game's result), and the policy
its pools. So the loop has one corpus, and the natural unit is one tag.

## Terms

- **Generation `g`:** a batch of `games_per_generation` self-play games in
  its own directory, as in every generational workload.
- **Release `k`:** the leave inference model `I_k`, the leaf network `L_k`
  and the policy `π_k`, exported for the engine, with a manifest that names
  the three files and their hashes. Release 0 is the seeds. Release `k`
  (`k ≥ 1`) has trained on generations `0 … k−1`.
- **The generating agent `A_k`:** SubtreeBot under the `hasty` reply model with
  the [screen](subtree_bot.md#the-screen-at-the-root), on `I_k` and `L_k`.

## The loop

1. A generator starts a chunk: it takes the newest release it holds and plays
   `games_per_chunk` games of `A_k` against itself.
2. It writes the chunk's sidecars and delivers the chunk to staging.
3. The data home's scheduler moves staged chunks into the open generation,
   and marks it complete at `games_per_generation` games.
4. The trainer, waiting on generation `g`, evaluates release `g` on it, then
   trains each model, exports release `g + 1` and advances its cursor.
5. The controller carries the new release to every running generator.

Steps 1 to 3 and the cursor are the existing protocol. Generators never stop
for the trainer: with `open_ahead = 1` the next generation fills while the
trainer works on the last.

## What a chunk holds

| File | Written by | Read by |
|---|---|---|
| `.slog` | `play_game` | all three trainers |
| `.forgone` | `forgone_play_tool`, on the CPU | the leave inference rows; the pools' beliefs |
| `.pool` | self-play itself (below) | the policy, value and continuation heads |
| `.survivors` | self-play itself (below) | the policy's target and the value head's, at the screen's survivors |
| `.group` | `policy_set_pool_tool`, off until the store (step 12) | the store input |

The chunk's `.pool` header already names the leaf network and leave
inference model its labels came from, by hash, tag and generation
(`PoolSources`, data/pool_log.h). That is the chunk's release stamp. The
trainer reads it for each generation's share of games played on the newest
release, so the mix is on record.

### Self-play writes the pools

The screen at SubtreeBot's root is a pool. It runs the pool labeler
(`PoolLabeler`) over `policy_set::pool_candidates` with the turn's belief, at
100 raced rollouts a move: the same code, candidates and budget as
`policy_set_pool_tool`, except that the screen leaves the pass out. Every
searched turn of a self-play game therefore computes a pool, and today throws
it away.

The plan is that `play_game` writes those pools as the chunk's `.pool`: the
agent hands each turn's labelled pool to the game runner, which knows the
game and turn index the record needs. Measured 2026-11-13 on the dev
laptop's GPU (RTX 5000 Ada):

| Source of pools | Games a second | Pools a second |
|---|---|---|
| `policy_set` tag `v1`: HastyBot games, 2 positions a game labelled by the tool | 2.7 | 5.4 |
| SubtreeBot self-play, 2 positions a game labelled again by the tool | about 0.27 | about 0.5 |
| SubtreeBot self-play, every searched turn's screen kept (about 20 a game) | 0.30 | about 6 |

The first row is the tag's generator stats (50 games and their 100 pools in
18.6 s). The self-play rate is 56 games in 184 s at 8 game threads, and 28
threads give 0.27, so the GPU bounds it. The second row adds the tool's 0.19
s a pool to each game. So with its pools kept, SubtreeBot self-play labels pools
as fast as the pool tool does on HastyBot games, and the games come with
them. Labelled a second time by the tool, the loop would produce a tenth of
the pools `policy_set` gets today.

The pools of one game are more alike than pools of 20 games, so a pool here
is worth less than one of `v1`'s. How much less is read from the policy's
held-out curve against pools trained, not assumed.

A pool is about 88 KB (`v1`'s files), most of it the rollouts' values, so a
game's pools are about 1.8 MB and a generation of 2,000 games about 3.5 GB.
`pool_positions_per_game` keeps a sample of each game's turns where a
generator's uplink cannot carry that.

**The pool carries the turn's sims too.** The agent gives the screen's 10
or 32 survivors 1,000 raced rollouts each, and values a survivor that empties
the bag by the endgame solver. Under the `policy` reply
([roadmap](subtree_bot_roadmap.md#the-policy-reply)) those sims are the ones in
which Bob replies by the policy. They are the agent's decision, and the
policy's target is the decision, so the record holds both stages: every
move's screen label, and each survivor's label from the turn's sims, or its
solved value.

- **The trainer combines them,** so the target can change without new
  games. The first form is the sims' `q` over the survivors and no mass on a
  move the screen dropped, which needs no comparison across the two rollout
  counts. The value head fits the sims' win probability for a survivor and
  the screen's for the rest.
- **Where a solved value meets the race** (7% of the turns measured
  below), the target follows the agent's own comparison: the race's winner
  beside each solved survivor, by value. Those contenders split the mass by
  the label's rule on independent estimates, the winner's standard error
  from its rollouts and a solved value's from its 100 endgames; the race's
  share then goes to the simmed survivors by their label
  (`policy_set/survivors.py`).
- **The value head's variance target and the regret metric** stay on the
  screen's rollout values, which under the `hasty` reply are the same sims
  on other racks. Under the `policy` reply they will not be, and both move
  to the survivors' values then.

On the 20 games measured below, the turn's labels against the screen's:

| | The screen's | The turn's |
|---|---|---|
| Mean label entropy (nats) | 1.64 | 0.41 |
| Mean mass on the likeliest best | 0.60 | 0.84 |
| Turns whose likeliest best is another move | | 35% |
| The screen's mass on moves it then dropped | 11% | |
| Mass on solved survivors, at the 30 turns that have them | | 0.69 |

**The engine half is built** (2026-11-16): `play_game
--pool-positions-per-game N` has the agents record each searched turn and
writes two sidecars beside each `.slog`.

- **The `.pool`** is the screen's, in the pool tool's format unchanged, so
  the `policy_set` corpus and its readers are untouched. A recorded screen
  sims the pool whole as the tool does: the pass too, which stays out of the
  candidates, and also at a turn with no more moves than candidates.
- **The `.survivors`** (data/survivor_log.h) holds, position for position,
  each survivor's index into the pool, its label from the turn's sims with
  their rollout values, or the solver's value alone. The values are kept:
  they are what any later target or paired comparison is computed from.
- **`N` turns of each game** are kept, drawn uniformly from the turns
  searched.

Measured on the dev laptop, 20 games of the generating agent on release 0's
exports, every searched turn kept:

| | |
|---|---|
| Games a second | 0.25 (79 s; a batch this small pays its last games' tail) |
| Pools | 445, 22 a game, 5.6 a second |
| Candidates a pool | 609 on average |
| Survivors a pool | 20 on average; 190 solved values over 30 turns |
| Rollout values kept for a pool's survivors | 6,300 of a budget of 20,300 |
| Bytes a game | 1.9 MB of pools, 0.57 MB of survivors |
| Turns where the sims' likeliest best is not the screen's | 39% |

## The trainer

One process, the tag's single `train` role, hosting the data home. The
controller's data plane assumes one trainer per tag (the first slot whose
role ingests is the home; `workers.py`), and one trainer is enough here:
training takes minutes where a generation takes hours.

| Model | Trainer speed | New rows in 2,000 games | One pass over them |
|---|---|---|---|
| Leave inference | 4,200 rows/s | 40,000 | 10 s |
| Leaf network | 7,800 rows/s | 40,000 | 5 s |
| Policy | 107 pools/s | 40,000 pools | 6 min |

Speeds are the three `v1` tags' trainer stats; a game has about 20 eligible
turns. On one laptop GPU 2,000 games take 1.9 hours.

**One trainer, three model units.** Each of the three existing trainers is
split into a model unit (build, train one epoch over a list of generation
directories, evaluate a list of them, export, save and load its state) and
the loop that drives it. The standalone workloads keep their loops over the
same units, so they still run, and their results stay bit-identical. The
`subtree_bot` trainer drives the three units from one cursor.

**Per generation `g`:**

1. **Evaluate before training.** Release `g`'s models are scored on
   generation `g`, which none of them has trained on and which the newest
   agent played. This replaces the fixed held-out sets (the leave and leaf
   trainers' self-played HastyBot games, `policy_set`'s held-out chunks),
   which would age as the policy moves, and it spends no games on
   evaluation.
2. **Train** each model one epoch over its own window of generations ending
   at `g` (`leave_window`, `leaf_window`, `policy_window`).
3. **Publish.** The three exports, their outputs at the known positions
   (`positions.json`, about 1 MB, for the dashboard's Positions tab), then
   the release manifest, then one
   checkpoint holding the three models' and optimizers' state, the cursor,
   and last the generation's record. A release exists once its manifest
   does, so a generator never reads two models of different releases. Every
   release's exports are kept, about 23 MB, so that any release can be
   played later. Every `keep_checkpoint_every`th release's checkpoint is
   kept as well, as a seed for a later tag. A trainer on an ssh machine
   hands each release and each kept checkpoint to the controller as a
   complete set, as it does its state pair, and keeps its own copy of the
   releases for the generators beside it.
4. **Evict** generations older than the longest window.

A series of scores, each on a different generation's games, mixes the
models' progress with the games' drift: the leave of a stronger player may
simply be harder to infer. So each generation is also scored under release
0, kept loaded, and every chart draws both. The gap is the loop's gain over
its seeds on today's games.

**Seeds and warm start.** `seed_leave_tag`, `seed_leaf_tag` and
`seed_policy_tag` name tags of the three standalone workloads. Each seed's
rolling checkpoint is copied into the new tag when it is created, so the tag
does not depend on its sources afterwards, and the controller pushes the
copies to a trainer on an ssh machine. A seed's architecture params must equal the tag's; the trainer
refuses a mismatch as `check_resumable` does. The models start from the
seeds' weights with fresh optimizers. No trainer warm-starts from another tag
today.

**The rows-clock is re-sized.** The seeds' schedules are sized for HastyBot's
data rate: the leave inference tag trains 400,000 rows a generation, warms up
over 200,000 and averages its deployed weights over 20 million. Here a
generation trains 40,000 rows. Carried over, the deployed weights would be
the average of every iterate since the tag began for its first 500
generations, and each release would trail the newest games by half the run.
The tag's defaults size the warmup and the averaging window in generations
of this workload's rows, and the fine-tuning learning rates are the first
thing the early generations tune.

**Windows.** The SubtreeBot plan said to train `I_k` on `P_k`'s games alone,
because a buffer that mixes policies converges to Bayes' rule under their
average. This plan relaxes that to a short window of SubtreeBot's own
generations. One generation is too few rows to train on alone without many
passes over them, and consecutive releases differ far less than HastyBot
differs from any of them, which is the gap that matters. `leave_window = 1`
is the strict rule, and the evaluation before training shows what each
setting costs: generation `g` is played by the agent the next release must
model.

The leaf network keeps its bound on reuse (each turn of a game trained
once; [generational_training.md](../generational_training.md)), since its
rows share their game's result. The policy's window shrinks from `v1`'s 10:
at 40,000 pools a generation, ten generations are an hour of training for
each release.

The win tables the leave and leaf trainers refit are not carried: nothing in
SubtreeBot reads them.

## The generator

A GPU role on the engine image, like `policy_set`'s. Each cycle:

1. It reads the newest release manifest it holds. A local generator reads
   the tag's `models/` in place; a remote one reads the copies the controller
   pushed.
2. It runs `play_game` with `subtreebot-endgame` in both seats under `--reply
   hasty`, on the release's leaf network and leave inference model, hidden
   leaves, no random openings.
3. It writes the `.forgone` sidecars and delivers the chunk.

The process is relaunched per chunk, as every generator is, so a new release
costs a model load. TensorRT refits its cached plan when only the weights
change ([generational_teacher.md](generational_teacher.md)), so a release
does not cost a plan build. The first chunk on a new machine does.

The self-play budget (`screen_rollouts`, `rollouts`, the candidate counts)
is the tag's params, at the agent's defaults. A cheaper budget plays more
games with noisier pools, and the GPU bounds the rate, so it is the lever on
games per day. It is tuned from the policy's and the match's curves, not
set here.

## Pacing

The loop is asynchronous. A generator always plays the newest release it
has, and a chunk goes into whichever generation is open when it arrives.

- Generation `g` is therefore mostly release `g`'s games, with a share from
  release `g − 1` played while the trainer worked. The share is about the
  training time over the generation time: small on one GPU, larger as
  generators are added. `games_per_generation` is the knob that restores it,
  and the record's share of games on the newest release charts it.
- `open_ahead = 1` lets the next generation fill during training and still
  parks the generators if the trainer falls a whole generation behind. A
  stopped trainer parks them at once, as today.

The alternative is synchronous: gate the generators while the trainer works
and discard chunks played under an older release. It would give generations
of one policy each, at the cost of an idle fleet during training and one
discarded chunk per generator per release. The purity is not worth that,
for the reason given under [Windows](#the-trainer).

## Measurements

Per generation, from the trainer, each also under release 0:

- **Leave inference:** log-likelihood of the true leave over the uniform
  draw, own and hidden views.
- **The same, by the static-equity rank of the move played** (rank 0, 1 to
  9, 10 and below; the rank is in the turn's pool). This is the belief
  measurement the loop exists to move, and it replaces the per-candidate
  table as the per-generation read. The [SubtreeBot plan's
  table](subtree_bot.md#generations) scores Bob's belief after each of Alice's
  candidates, most of which she does not play. After a candidate she would
  not play from her rack, a belief that follows Bayes' rule under her policy
  stays below the uniform draw, rightly: the racks that play it are not
  hers. Training on the agent's games closes the gap only where she plays
  the move, so that is where it is measured.
- **Leaf network:** held-out loss per head.
- **Policy:** how often its top move is the pool's best, against static
  equity's, and the regret of its top `N`.
- **The share of each generation's games played on the newest release.**

Per release, from the measure role (every `match_every_generations`):

- **A match against HastyBot** (`hastybot-endgame`) and **a match against
  `A_0`**, each `match_games` games, mirrored in pairs. See [the
  matches](#the-matches).
- **The picks at the known positions**
  ([positions/NWL23/known-positions](../../positions/NWL23/known-positions/README.md))
  on the release's models, under the
  generating agent and under `--reply subtree`: a handful of turns. These
  are the positions the project is judged on.
- **The picks at the blind spots**
  ([blind_spots.md](../blind_spots.md)) are read on demand, not per release
  (`scripts.deep_bot.blind_spot_picks`). Their verdicts come from hasty sims
  under the seeds' leaf network, so they do not rank releases finely.

### The matches

Measured 2026-11-14 on the dev laptop (28 threads, one RTX 5000 Ada), the
generating agent on the seeds' newest exports:

| Opponent | Games | Cost of a game | The agent's wins | Correlation of a pair's two results (win, spread) |
|---|---|---|---|---|
| HastyBot (`hastybot-endgame`) | 300 | 1.8 s of the GPU | 186 (62%) | −0.11, −0.30 |
| BestBot (`bestbot-endgame`) | 55 | 18 core-minutes | 28 | −0.47, −0.80 (27 pairs) |
| `A_0` (the agent itself) | | 3.3 s of the GPU, from self-play's rate | | |

BestBot's sims run on the CPU: a match fills every core for about 1.4 games
a minute, 12 hours for 1,000 games. So it is not in the loop.

**In the loop,** `match_games` starts at 200 against each opponent. On one
laptop that is 17 minutes of the GPU, about 15% of a 2,000-game generation's.
The win rate of 200 games has a standard error of about 3.4 points, which
does not resolve one release's gain; the chart draws each release's rate and
the mean over a trailing window of releases, and a day's releases together
come to about 1 point. Mirrored pairs gain little against HastyBot, whose
games part ways early.

The two opponents read differently:

- **HastyBot is a regression read.** Under the `hasty` reply the agent's
  model of its opponent is HastyBot exactly, so release 0 is the release
  best fitted to this opponent. Later releases learn their beliefs and leaf
  values from SubtreeBot's games, which describe HastyBot less well. The rate
  can stall or fall while the agent gets stronger; a fall is still worth
  seeing.
- **`A_0` is where the loop's own gain shows,** though only against its own
  line.

**On demand,** a match command takes the tag, a release and an opponent's
player spec, runs on any machine (the release's exports are all it needs),
and files its games under that release, where the chart draws them beside
the loop's. This is how a release is measured against BestBot, during the
run or after it.

**A luck correction, later.** Most of a match's variance is the tiles. The
correction (AIVAT's, for chance events) takes a value function fixed apart
from the game and, at each draw, subtracts the value of the state after the
tiles drawn less its mean over the draws the bag could have dealt. Whatever
the value function, the sum has mean zero, so the corrected outcome
estimates the same win rate; the better the value function, the less
variance is left. It reads nothing from the players, only each state before
and after a draw, so it can be computed from a game's log after the fact. The loop is built to let it arrive later:

- **Every match game's log is kept** (`.slog`: the opening racks, the moves
  and the tiles drawn after each), beside a row per game naming the release,
  the opponent, the seats, the seed and the scores. A game's log is about
  600 bytes.
- **The chart is computed from the per-game rows** by one estimator, today
  the mean. A correction adds a column to the rows and a second estimator;
  no match is replayed.

The natural value function is one batch of the agent's hasty rollouts with
release 0's leaf network per draw: 1,000 rollouts, what the agent gives one
of its ten candidates. Its worth is unmeasured. The floor is known: on
40,000 HastyBot self-play games, a correction linear in the counts of each
tile drawn removes 19% of the win variance and 28% of the spread's. Its
first output would be the variance it removes on the stored games, BestBot's
included.

## What the controller needs

| Need | Today | Change |
|---|---|---|
| One data home beside one trainer | built | none |
| Staging, relay to a remote home, pulls, the gate | built | none |
| Adding and removing generator slots on machines | built, by hand | none |
| A role's out-of-tag inputs (the seeds) | built, pushed when a container is created | none |
| A new release reaching a running remote slot | built (step 4): a role names the files of its tag it reads, and each reconcile pass pushes the ones a running slot lacks | the measure role names the same files |
| A remote trainer's releases reaching the controller | built (step 4): each is moved home as a complete set, like the state pair | none |
| Work dispatched per export, results ingested and charted | built (step 6): a release and release 0 are assigned to the measure slot, and each match's games come back with a row a game | none |
| A match on demand against a kept release | not built | a command that plays the match anywhere and files its games under the release |
| Charts | one model's metrics per tag | a grid per model from one record |
| Queue placement | `layout` and `gpu_need` hooks | measure the generator's and the trainer's GPU memory |

The tag queue places a tag on one machine. Generators on further machines
are added by hand, as `policy_set` tag `v1`'s are.

## Operating it

- **Launch:** create the tag, naming the three seed tags; add the trainer,
  then generators.
- **More or fewer machines:** add an ssh generator slot to a machine, or
  stop one. A new slot receives the newest release and starts. A stopped one
  loses its chunk in flight, at most `games_per_chunk` games.
- **The trainer's machine is lost:** the controller holds every pulled
  generation and the last state pair, and seeds a new home, as for any
  generational tag.
- **Training goes non-finite:** the trainer stops before publishing
  (`finite_guard`) and its slot is paused as failed, which parks the
  generators. The last release stays the newest.
- **A match against BestBot:** run the match command on a release, on a
  machine with idle cores. It fills them, so not on a generator's machine
  unless its threads are capped.
- **A release plays worse:** nothing gates a release, as nothing gates
  AlphaZero's. The match chart shows it, and the remedy is a new tag seeded
  from an earlier release's kept checkpoint.
- **The end:** the tag runs until paused. `max_generations` is its optional
  end condition, as `max_rows` is elsewhere.

## What the loop does not close

- **The policy is trained but not played.** The `hasty` generating agent
  reads `I_k` and `L_k`. `π_k` rides in the release so that the agent can
  take it up without a change here. The [order of the remaining
  work](subtree_bot_roadmap.md) has it played as Bob's reply in the sims, and
  the generating agent then switches to that reply.
- **Bob's subtree is not in the generating agent,** and waits: it costs
  seconds a turn where the `hasty` reply costs a fifth of one. So the loop
  logs no paired `Δ` and trains no root correction head.
- **Grouped pools** stay off until the store input is built. They are a
  sidecar the pool tool writes over the chunk, as in `policy_set`.

## Sizing

On one laptop GPU the loop plays about 26,000 games a day: 13 releases at
2,000 games each, with 520,000 leave and leaf rows and as many pools. The
matches take about 15% of that at `match_games` 200, leaving 11 releases a
day. The
seeds trained on 117 million rows (leave inference), 419 million (leaf) and
1.3 million pools (policy: about 140,000 pools, each about ten times).

So the policy gets several times its seed's corpus in a day. The leaf
network gets a thousandth of its seed's rows a day, and whether rows from
SubtreeBot's games move it at that rate is the open question for it: its curve
under release 0 against the newest release answers that in the first days.
The leave inference model sits between: its target is one policy's tells,
and 40,000 rows a generation is what the short window is for.

`games_per_generation` starts at 2,000 and scales with the fleet.

## Build order

Each is a pull request.

1. **The trainers' model units.** Built 2026-11-15: each of the three
   trainers is a loop around a unit (`leave_inference/unit.py`,
   `deep_bot_leaf_eval/unit.py`, `policy_set/unit.py`), with no change in
   behavior. *Measure:* each standalone trainer's outputs on a fixed seed,
   before and after. Each trainer's weights and record after one generation
   on a small corpus are the same bit for bit.
2. **The `subtree_bot` trainer:** seeds and warm start, the three units on one
   cursor, evaluation before training with release 0 beside it, releases,
   the combined checkpoint and record, the re-sized clocks. Built
   2026-11-15 (`scribblez/deep_bot/`, with the params in
   `workloads/deep_bot.py`), driven so far only by its tests: a unit's
   params are its seed's recorded config with this tag's window, turns per
   game, learning rate and clocks, so its architecture is the seed's by
   construction. Copying the seeds into a new tag and the charts come with
   the workload (step 3).
3. **The generator and the workload,** on the trainer's machine: SubtreeBot
   self-play on the newest release, pools from the pool tool at
   `pool_positions_per_game` for now, the charts. Built 2026-11-16
   (`workloads/deep_bot.py`): a tag can be created, and the loop runs end to
   end on the laptop. *Measure:* seconds a chunk spends loading models; the
   first releases' curves. On a two-generation tag of 40 games a generation:
   - a new release cost no TensorRT plan build, and its first 20-game chunk
     took 100 s where the chunk before took 102 s, so the load is within
     the noise of a chunk;
   - the pool tool labeled 7 to 9 pools a second beside the trainer, which
     at 20 pools a game is over 2 s a game on top of the 4 s or so the
     game takes, until self-play writes the pools (step 5);
   - release 0's leave inference model, which scores 0.94 nats better than
     the uniform draw on HastyBot's games, scored 0.85 worse and 0.09
     better on the two generations of SubtreeBot's. Forty games a generation
     settle nothing, but it is the direction the [parts'
     match](subtree_bot_roadmap.md#the-parts-measured) pointed.
4. **Releases to remote slots.** Built 2026-11-16: both roles run on ssh
   machines. The controller pushes the newest release to each generator
   that does not share the trainer's tree, and the seeds to a trainer
   elsewhere; a remote trainer hands its releases and kept checkpoints home
   (`RoleSpec.tag_files`, `DeliveredSet`). *Measure:* the share of games
   per release as generators are added, which each generation's record now
   carries. Checked live only in part, on a 20-game chunk: a generator on a
   second laptop received release 0 once it started, played it,
   and its chunk filled generation 0 here. Not yet seen live: a second
   release reaching a running generator, and a trainer on an ssh machine.
5. **Self-play writes the pools.** Built 2026-11-16, with its numbers
   [above](#self-play-writes-the-pools): the engine change with both stages
   of the turn in the record, the per-game sample, the workload taking its
   pools from self-play, the policy's target from the two stages, and the
   leave inference score by the played move's rank
   (`deep_bot/belief_by_rank.py`; in the 20 games above the agent played
   static equity's first move at 56% of its turns, a move ranked 1 to 9 at
   29% and one ranked 10 or lower at 14%). *Measure:* pools
   a second and bytes a game (above); and the policy's curve against pools
   trained, under the screen's labels and under the turn's, which is not
   yet measured: a tag's games hold both.
6. **The measure role:** the matches against HastyBot and `A_0` with their
   games kept, the match command, the known positions, the blind spots.
   *Measure:* the GPU time a match takes from a running generator.
   Built 2026-11-16, the matches: one measure slot plays each release as the
   generating agent against HastyBot and against release 0, on the
   controller's assignment, newest release first
   (`deep_bot/measure.py`, `measure_dispatch.py`). Every game comes home and
   is kept under the tag at `matches/release_<k>/<opponent>.json` (a row per
   game) and `.slog`; the Matches tab draws each release's win rate from the
   rows and its mean over the last 10 releases. The match command is
   `python3 -m scripts.deep_bot.match --tag T --release K --opponent NAME
   --player SPEC`, whose match is recorded like the role's and gets its own
   panel. The known positions are collected, with their answers, in
   `positions/NWL23/known-positions`, and `python3 -m
   scripts.deep_bot.known_position_picks --player SPEC` reads any agent's
   picks at them. The measure slot takes those picks for each release it
   is assigned, under the generating agent and under `--reply subtree`, and
   the Positions tab shows one position at a time under a release: the
   board, the picks beside the position's answer, and the release's policy
   and leave inference outputs there, which the trainer computes as it
   publishes the release. The blind spots' picks stay on demand
   (`scripts.deep_bot.blind_spot_picks`): the worked ones are among the
   known positions, and the tag's thousand live on the controller's mount.
7. **Queue placement:** the roles' measured GPU memory and the layout.

Not scheduled: the [luck correction](#the-matches), over the games step 6
keeps.

## Alternatives considered

- **A coordinator over three ordinary tags.** Each tag keeps its trainer and
  charts, and a fourth owns the games. The data plane has no way for a
  trainer to read another tag's generations, so the games would be copied
  three times or the plane extended; the operator would pause, resume and
  place four tags; and a release would have to wait on three cursors.
- **Three trainer roles in one tag.** The controller relays chunks to, and
  pulls generations from, the first ingesting slot only, and eviction would
  need an acknowledgment per trainer
  ([subtree_bot_v0.md](subtree_bot_v0.md) notes the same). It buys training in
  parallel, which is not the bottleneck.
- **Pools from the pool tool alone.** No engine change, and a tenth of the
  pools ([above](#self-play-writes-the-pools)). It is the first form built
  (step 3), not the end state.
- **A run-forever producer** that refreshes weights in place
  ([generational_training.md](../generational_training.md)). It saves a
  model load per chunk. Step 3 measures that load; `games_per_chunk` trades
  it against how soon a release is picked up.
