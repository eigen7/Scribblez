# Scribblez

A research codebase for building a superhuman Scrabble AI.

- [docs/design.md](docs/design.md) — the design document: why existing engines
  are beatable and the target architecture.
- [docs/roadmap.md](docs/roadmap.md) — the phased plan and current status.
- [docs/README.md](docs/README.md) — the full documentation index.

## Layout

```
py/build.py         one-shot build: compile the engine + install web deps
py/run_tests.py     full test suite: C++ (ctest) + Python (pytest) + web (vitest)
engine/             C++ core (CMake): GADDAG movegen, HastyBot, game loop,
                    .slog binary logs, training DataLoader, Monte-Carlo sim,
                    TensorRT neural agent, FFI for Python
web/                React + TypeScript front-ends (Vite): human play + the
                    training dashboard
py/scribblez/       Python library: FFI wrapper, torch dataset, models,
                    generational-training lifecycle, dashboard API
py/scripts/         entry points: trainers, data generation, evaluation
py/tools/           small utilities (lexicon tools, formatters)
docs/               documentation (see docs/README.md)
positions/          committed GCG evaluation datasets + Monte-Carlo ground truth
phonies/            the generated phony lexicon (word-validity experiments)
submodules/         git submodules (dev-environment machinery; see submodules/README.md)
```

## Build

Scribblez is developed and run inside a Docker container. All build tools,
Boost, Node, Python, CUDA-capable PyTorch, and the rest are baked into the
image; you don't install any of them on the host.

**One-time host setup** (`docker` and `go` are the only host requirements; the
wizard checks the rest):

```bash
./setup_wizard.py        # picks a mount dir, clones+builds Macondo, fetches lexica
./build_docker_image.py  # builds the local `scribblez` docker image
```

A plain `git clone` suffices: the `submodules/devenv_utils` submodule is
populated automatically the first time any of these scripts runs.

**Every dev session**, launch the container and build inside it:

```bash
./run_docker.py          # drop into a shell at /workspace/repo
# (inside the container)
py/build.py              # cmake + npm install
py/run_tests.py          # C++ + Python + web test suites
```

`./run_docker.py` bind-mounts the repo and your mount dir, so build artifacts
in `target/` and downloaded data both persist on the host between container
runs. Re-running it while a container is already up just `exec`s into the
running one.

Release builds use link-time optimization (LTO) where the toolchain supports it.

## Dictionary

The engine loads a lexicon from a **KWG** file (the wolges/Macondo binary
word-graph format). The default lexicon is **NWL23** (NASPA Word List 2023).
The NWL wordlist is copyrighted, so the `.kwg` binary is **not** committed,
and we don't bake it into the Docker image either. Instead, `setup_wizard.py`
downloads it on the host from the public Woogles/liwords URL into your mount
directory at `<mount>/lexica/NWL23.kwg`, which the container sees at
`/workspace/mount/lexica/NWL23.kwg`. The wizard's "install lexica" step is
re-runnable; it lists what you already have and prompts for any additional
lexica to fetch (CSW24, NSWL23, NWL20, etc.).

A KWG bundles both a forward DAWG (node 0) and a GADDAG (node 1). The move
generator uses the GADDAG (Gordon's algorithm); whole-word lookup and
cross-checks use the DAWG. `Dictionary::build_from_words` builds an equivalent
in-memory DAWG+GADDAG and is used by the unit tests.

## Play games

`target/engine/play_game` runs games between any pair of seats; `--help`
documents every flag and per-player option. Each `--player "--type=T [opts]"`
adds a seat (default: two greedy players). Player types:

- **`greedy`** — highest-scoring play, ties broken randomly.
- **`hastybot`** — in-process static-equity bot (score + leave value +
  adjustments), optionally softmax-sampling among its top-K
  (`--temperature`, `--top-k`). The self-play workhorse.
- **`neural`** — HastyBot move-gen + the position evaluation model: applies candidate
  plays and picks the one whose post-move state the model rates highest
  (`--model=<onnx>`, TensorRT under the hood).
- **`human`** — a browser UI seat (below).

Useful runner flags: `--games N --threads T` for parallel batches, `--seed`,
`--log-dir` for one `.gcg` per game, `--binary-log-dir` for batched `.slog`
training data, `--random-opening-mean` / `--random-handicap-max` for self-play
diversification.

```bash
./target/engine/play_game --player "--type=hastybot" --player "--type=hastybot" \
    --games 1000 --threads 8 --seed 42 --verbose
```

### Play against the AI (web UI)

A `human` player is driven through a small React web app. You don't run any npm
commands yourself: after `py/build.py` has installed the web dependencies,
`play_game` launches the front-end's Vite dev server itself, speaks WebSocket to
it, and opens your browser:

```bash
./target/engine/play_game --player "--type=human" --player "--type=hastybot"
```

The browser loads the UI from Vite, which proxies the WebSocket back to the
engine. The engine frees the ports if a previous run left something holding
them, and shuts the dev server down when the game ends.

Place tiles by clicking a square and typing (use the **arrow keys** to move the
highlighted square), or by dragging tiles from your rack; click a placed square
to remove it, click a cell twice to flip typing direction, and press a letter
onto a blank to choose its value. When your tiles form a legal play the "Play
Move" button appears; "Pass" is always available. Tick **Show legal moves** to
browse/preview the engine's move list (sorted by score, highest first). Swap the
`--player` order to take the second seat.

## Train

The position evaluation model trains in an open-ended generate→train loop
— self-play games are generated to disk per generation, trained over a sliding
window, and the run is restartable at any point (see
[docs/generational_training.md](docs/generational_training.md)):

```bash
./py/scripts/position_eval/train.py -t mytag
```

The trainer launches the per-tag React dashboard (loss curves, structural
probes, calibration, Monte-Carlo position comparison, live controls); launch it
standalone with `./py/scripts/dashboard.py`. Sibling trainers:
`py/scripts/max_move_per_lane/train.py` (the lexical representation probe) and
the lexical-NN experiment trainers under `py/scripts/word_validity/` and
`py/scripts/rack_best/`. `py/scripts/generate_data.py` generates standalone
train/test `.slog` splits; `py/scripts/position_eval/evaluate.py` runs the
eval suites on a checkpoint.

Data flow, formats, and the replay-reconstruction invariant are documented in
[docs/architecture.md](docs/architecture.md).

## Game log formats

Training data is stored as **`.slog`** — a compact binary format holding many
games per file as initial racks + move/draw sequences, replayed into training
rows at load time ([binary_log.h](engine/include/data/binary_log.h) is the
authoritative layout).

Human-readable logs use **GCG**, the de-facto standard Scrabble game-log format
(as written/read by Macondo and Quackle). Header pragmata define the players,
then one event line per turn:

```text
#character-encoding UTF-8
#player1 Alice Alice
#player2 Bob Bob
>Alice: ADNORRV H8 ANDRO +14 14
>Bob: AEIIMOQ 10F QA.I +34 34
>Bob: EEHNTW? B1 WrE.THEN +92 376
>Bob: (ADGUY) +10 386
>Alice: ADGUY (ADGUY) -10 428
```

Each event is `>nick: rack ...`:
* **play** — `POS WORD +score cumulative`. `POS` is the main word's first
  square (`8D` across, `D8` down). In `WORD`, a `.` is a tile already on the
  board (played through) and a lowercase letter is a designated blank.
* **exchange** — `-TILES +0 cumulative`.
* **pass** — `- +0 cumulative`.
* **end-of-game** — the player who goes out gains the opponent's leftover
  tiles, `(TILES) +points cumulative`; a player left holding tiles loses their
  value, `rack (TILES) -points cumulative`. Stalemate (6 consecutive zero-point
  turns) records a penalty line for each player.
