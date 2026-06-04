# Scribblez

A research codebase for building a superhuman Scrabble AI.
See [docs/Scribblez.pdf](docs/Scribblez.pdf) for the design document.

This snapshot contains the v0 game engine: a C++ Scrabble core (GADDAG move
generator + scoring + game loop) plus a Greedy baseline agent, a GCG game-log
writer, a browser UI for playing against the AI, and a Python `torch` dataset
that consumes the logs.

## Layout

```
build.py        one-shot build: compile the engine + install web deps
engine/         C++ core (CMake)
  include/scribblez/    public headers
  src/                  implementation (incl. web_server.cpp: WebSocket server +
                        Vite dev-server launcher; player_factory.cpp: --player CLI)
  apps/play_game.cpp    CLI to play one game (greedy/human in any combination)
  tests/                hand-rolled unit tests
web/            React + TypeScript front-end for human play (Vite)
python/scribblez/       torch Dataset + DataLoader over GCG logs
python/scripts/         small utilities
docs/                   design document
```

## Build

Requires CMake >= 3.16, a C++17 compiler, Boost (>= 1.83), and -- for
human-vs-AI web play -- Node.js/npm. The one-shot build script compiles the
engine and installs the web UI's npm dependencies:

```bash
./build.py            # cmake configure+build, then `npm ci` in web/
./build.py --debug    # debug build
./build.py --clean    # wipe build/ first
ctest --test-dir build   # runs engine unit tests
```

(You can still drive CMake directly -- `cmake -S . -B build && cmake --build
build -j` -- but then install the web deps yourself with `npm --prefix web ci`.)

This produces:

* `build/engine/play_game` -- run a single game.
* `build/engine/scribblez_tests` -- unit tests.

Release builds use link-time optimization (LTO) where the toolchain supports it.

## Dictionary

The engine loads a lexicon from a **KWG** file (the wolges/Macondo binary word-graph
format). The default lexicon is **NWL23** (NASPA Word List 2023). The wordlist
itself is copyrighted, so the `.kwg` binary -- which encodes it -- is **not
committed**; obtain it from a Macondo checkout (`data/lexica/gaddag/NWL23.kwg`)
and place or symlink it under `data/lexica/` here:

```bash
mkdir -p data/lexica
ln -s /path/to/macondo/data/lexica/gaddag/NWL23.kwg data/lexica/NWL23.kwg
```

A KWG bundles both a forward DAWG (node 0) and a GADDAG (node 1). The move
generator uses the GADDAG (Gordon's algorithm); whole-word lookup and cross-checks
use the DAWG. `Dictionary::build_from_words` builds an equivalent in-memory
DAWG+GADDAG and is used by the unit tests.

## Run a game

```bash
./build/engine/play_game \
    --seed 42 \
    --out /tmp/game.gcg \
    --verbose
```

Flags:
* `--player "--type=T"` -- add one seat; repeat once per seat, e.g.
  `--player "--type=human" --player "--type=greedy"`. Each spec is its own
  little option string: `--type` is `greedy` or `human` and an optional
  `--name=...` sets the display name. Defaults to two greedy players; at most
  one `human` is supported.
* `--kwg PATH` (alias `--dict`) -- lexicon to use. Defaults to
  `data/lexica/NWL23.kwg`.
* `--seed N` (default: hardware random)
* `--out PATH` (default: stdout)
* `--port N` -- engine WebSocket port for human play (default: 8080)
* `--vite-port N` -- browser UI (Vite) port for human play (default: 5173)
* `--web-dir DIR` -- front-end package directory (default: `web`)
* `--verbose` -- print final score and turn count to stderr

## Play against the AI (web UI)

A `human` player is driven through a small React web app. You don't run any npm
commands yourself: after `./build.py` has installed the web dependencies,
`play_game` launches the front-end's Vite dev server itself, speaks WebSocket to
it, and opens your browser:

```bash
./build/engine/play_game --player "--type=human" --player "--type=greedy"
# -> starts Vite, then: Open http://localhost:5173 in your browser to play.
```

The browser loads the UI from Vite (port 5173), which proxies the `/ws`
WebSocket back to the engine (port 8080). The engine frees those ports if a
previous run left something holding them, and shuts the dev server down when the
game ends.

Place tiles by clicking a square and typing (use the **arrow keys** to move the
highlighted square), or by dragging tiles from your rack; click a placed square
to remove it, click a cell twice to flip typing direction, and press a letter
onto a blank to choose its value. When your tiles form a legal play the "Play
Move" button appears; "Pass" is always available. Tick **Show legal moves** to
browse/preview the engine's move list (sorted by score, highest first). Swap the
`--player` order to take the second seat.

## Game log format (GCG)

Games are logged in **GCG**, the de-facto standard Scrabble game-log format
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
* **play** -- `POS WORD +score cumulative`. `POS` is the main word's first
  square (`8D` across, `D8` down). In `WORD`, a `.` is a tile already on the
  board (played through) and a lowercase letter is a designated blank.
* **exchange** -- `-TILES +0 cumulative`.
* **pass** -- `- +0 cumulative`.
* **end-of-game** -- the player who goes out gains the opponent's leftover
  tiles, `(TILES) +points cumulative`; a player left holding tiles loses their
  value, `rack (TILES) -points cumulative`. Stalemate (6 consecutive zero-point
  turns) records a penalty line for each player.

## Python: consume logs

```bash
pip install -r python/requirements.txt
PYTHONPATH=python python -m scripts.inspect_log /tmp/game.gcg
```

`scribblez.GameLogDataset` replays each game forward and yields one
`TurnSample` per turn (board state before the move, rack counts, scores,
bag size, the move taken, the eventual game outcome). `build_dataloader`
returns a `torch.utils.data.DataLoader` with a default collate that stacks
the tensor fields and keeps the variable-length `move` dicts as a list.

## Status / next steps

* v0: engine + Greedy agent + GCG logs + torch Dataset (this snapshot).
* Planned: binary log format + FFI loader; pybind11 bindings for in-process
  move generation; first neural network heads.
