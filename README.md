# Scribblez

A research codebase for building a superhuman Scrabble AI.
See [docs/Scribblez.pdf](docs/Scribblez.pdf) for the design document.

This snapshot contains the v0 game engine: a C++ Scrabble core (move
generator + scoring + game loop) plus a Greedy baseline agent, a JSON
game-log writer, and a Python `torch` dataset that consumes the logs.

## Layout

```
engine/         C++ core (CMake)
  include/scribblez/    public headers
  src/                  implementation
  apps/play_game.cpp    CLI to play one Greedy-vs-Greedy game
  tests/                hand-rolled unit tests
python/scribblez/       torch Dataset + DataLoader over JSON logs
python/scripts/         small utilities
docs/                   design document
```

## Build

Requires CMake >= 3.16 and a C++17 compiler.

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build         # runs engine unit tests
```

This produces:

* `build/engine/play_game` -- run a single game.
* `build/engine/scribblez_tests` -- unit tests.

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
    --out /tmp/game.json \
    --verbose
```

Flags:
* `--kwg PATH` (alias `--dict`) -- lexicon to use. Defaults to
  `data/lexica/NWL23.kwg`.
* `--seed N` (default: hardware random)
* `--out PATH` (default: stdout)
* `--verbose` -- print final score and turn count to stderr

## Game log JSON format

```jsonc
{
  "seed": 42,
  "players": [{"name": "Greedy"}, {"name": "Greedy"}],
  "turns": [
    {
      "player": 0,
      "rack_before": "ADNORRV",
      "bag_size_before": 86,
      "move": {
        "type": "play",                // or "exchange" / "pass"
        "horizontal": true,
        "start_row": 7, "start_col": 3,
        "main_word": "VONDA",
        "score": 26,
        "tiles": [
          {"row": 7, "col": 3, "letter": "V", "is_blank": false},
          ...
        ]
      },
      "score_delta": 26,
      "cumulative_scores": [26, 0],    // after the move
      "drawn": "TYSIA"                  // tiles drawn from the bag after the move
    },
    ...
  ],
  "final_scores": [463, 373],
  "end_reason": "out"                  // "out" | "stalemate" | "max_turns"
}
```

End-of-game scoring is included in `final_scores`:
* `"out"`: the going-out player adds opponent's remaining tile values to
  their score, and the opponent subtracts their own remaining tile values.
* `"stalemate"` (6 consecutive zero-point turns): each player subtracts
  their own remaining tile values.

## Python: consume logs

```bash
pip install -r python/requirements.txt
PYTHONPATH=python python -m scripts.inspect_log /tmp/game.json
```

`scribblez.GameLogDataset` replays each game forward and yields one
`TurnSample` per turn (board state before the move, rack counts, scores,
bag size, the move taken, the eventual game outcome). `build_dataloader`
returns a `torch.utils.data.DataLoader` with a default collate that stacks
the tensor fields and keeps the variable-length `move` dicts as a list.

## Status / next steps

* v0: engine + Greedy agent + JSON logs + torch Dataset (this snapshot).
* Planned: binary log format + FFI loader; pybind11 bindings for in-process
  move generation; first neural network heads.
