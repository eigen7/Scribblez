# Architecture: the training-data pipeline

A code-level map of how a self-play game becomes a training row, for anyone
changing the data format, the input encoding, or the labels. It names each
component and the file that owns it. The reasons behind the models live in
[roadmap.md](roadmap.md) and [design.md](design.md); the network wiring is in
[model_architectures.md](model_architectures.md).

## Pipeline at a glance

```
generate role / generate_data.py ─▶ play_game ─▶ GameRunner ─▶ GameEngine ─▶ Game
                                                                              │
                                                                          GameLog
                                                                              │
                                                                     BinaryLogWriter
                                                                              │
                                                                          .slog file
                                                                              │
                                                  DataLoader (via the FFI) ◀──┘
                                                          │
                                                    BlockDecoder
                                                          │
                                  PositionEncoder (replays the game through
                                                   GameStateEncoder)
                                                          │
                                        input row + AllTargets labels
                                                          │
                                                torch tensors ─▶ model
```

| Stage | Owner | Notes |
|-------|-------|-------|
| Launch self-play | [selfplay.py](../py/scribblez/selfplay.py), called by the workloads' generate role ([selfplay_gen.py](../py/scribblez/workloads/selfplay_gen.py)) and the one-shot [generate_data.py](../py/scripts/generate_data.py) | Shells out to the `play_game` binary with two HastyBot seats and `--binary-log-dir`. |
| Game loop / threading | [GameRunner](../engine/src/arena/game_runner.cpp) | Owns the run: the base seed, the win tally, the parallel game loop, and the `.slog` writer. |
| One game | [GameEngine](../engine/src/arena/game_engine.cpp), [Game](../engine/src/game/game.cpp) | `GameEngine` owns the per-thread agent pairs and sets up each game (seed, handicap, random opening); `Game` plays it and fills a [GameLog](../engine/include/game/game_log.h): initial racks, every move and draw, final scores. |
| Serialize | [BinaryLogWriter](../engine/src/data/binary_log.cpp) | Buffers finished games and writes one `.slog` per 1000 games (`kGamesPerFile` in `game_runner.cpp`). |
| On-disk format | [binary_log.h](../engine/include/data/binary_log.h) | The authoritative layout (see below). |
| Load | [DataLoader](../engine/src/data/data_loader.cpp) | Expands games into rows, shuffles, and fills batches on decoder threads. |
| Decode | [BlockDecoder](../engine/src/data/block_decoder.cpp) | Builds a `GameLog` view over a game's bytes in the file and hands it to the encoder. |
| Replay + encode | [PositionEncoder](../engine/src/encoding/position_encoder.cpp), [GameStateEncoder](../engine/src/encoding/game_state_encoder.cpp) | Replays the game to the requested turn and writes the input row and the labels. |
| Input layout | [input_encoder.h](../engine/include/encoding/input_encoder.h) | `InputEncodingSpec` and the block registry: block order, sizes, and offsets. 87 planes and 136 scalars; the open-leaves arm adds 27 scalars. |
| Label layout | [training_targets.h](../engine/include/training/training_targets.h) | The `AllTargets` registry, the single source of truth for the label heads. |
| Stream to Python | [scribblez_ffi.cpp](../engine/src/serve/scribblez_ffi.cpp) → [ffi.py](../py/scribblez/ffi.py) → [dataset.py](../py/scribblez/dataset.py) | A C ABI over the `DataLoader`; epoch-based batch streaming. |
| Train | [train.py](../py/scripts/position_eval/train.py), [position_eval/model.py](../py/scribblez/position_eval/model.py) | The generational generate→train loop ([generational_training.md](generational_training.md)). |

`PositionEncoder` is the one tensorization path. The `StreamingGameProducer`
([streaming_game_producer.h](../engine/include/arena/streaming_game_producer.h))
drives the same `GameEngine` and encodes live games straight into a ring
buffer without writing a `.slog`, and because it shares the encoder its rows
are byte-identical to decoded ones.

## The `.slog` format

A `.slog` file holds many games, each stored as the minimum needed to replay
every state: the initial racks plus the move sequence, each move bundled with
the tiles drawn right after it. This is about 20× smaller than fully expanded
per-position records, so far more games fit in the DataLoader's shuffle
buffer.

The byte layout (a `FileHeader`, a `GameMetadata` table, then per game an
`InitialRacks` and a `TurnBlob[]`) is specified in
[binary_log.h](../engine/include/data/binary_log.h) and versioned by
`kVersion`. The decoder and the FFI both reject a version mismatch, so a stale
file fails loudly instead of misparsing. This document deliberately does not
repeat the struct fields.

- **Write.** [`BinaryLogWriter::write_batch`](../engine/src/data/binary_log.cpp)
  records each game's **eligible** turns, the region
  `[eligible_begin, eligible_end)`. `eligible_end` ends the leading run of
  turns whose bag was non-empty; `eligible_begin` is the position right after
  the game's last random-opening ply (see below). The region widths sum into
  the `FileHeader`'s `num_sample_positions`, so a loader knows the epoch size
  without scanning the file. Games with an empty region are dropped. The
  writer also records a `sampled_turn`, one representative position per game
  used only by probes and position dumps.
- **Read.** The [`DataLoader`](../engine/src/data/data_loader.cpp) expands each
  game into one training row per eligible turn, so an epoch sees every
  position. [`BlockDecoder::decode_one`](../engine/src/data/block_decoder.cpp)
  builds each row. The `post_move` flag selects the snapshot: before the mover
  plays, or after the move but before the draw. The diagonal symmetry
  (`(r,c) → (c,r)`) is applied to a random half of the rows when the epoch's
  `apply_symmetry` is set; the replayed state is transposed as a whole before
  encoding, so no encoder or target knows about the augmentation.

## The replay-reconstruction invariant

**A training row is rebuilt by replaying moves, never read back from an
expanded record.** This decides where every value comes from:

- **Inputs are recomputed by the replay.** The board, the unseen-tile pool,
  the last-move metadata and the **score differential** are all rebuilt by
  applying moves through `GameStateEncoder`, which accumulates each play's
  score as it goes
  ([`apply_move`](../engine/src/encoding/game_state_encoder.cpp)). The
  score-differential input is whatever the running scores are at the sampled
  turn; nothing stores it per position.
- **Targets come from the stored final scores.** The WLD and score-diff labels
  are computed from the `GameLog`'s final scores, which the encoder copies into
  the `EncodeContext` ([encode_context.h](../engine/include/encoding/encode_context.h))
  the targets read. They are independent of the replay's running tally.

The consequence: any per-game state that must reach the input has to be
seedable into the replay. A starting-score handicap (`play_game
--random-handicap-max`) is the worked example. It is stored with the game and
seeds the `GameStateEncoder` at replay start, so the score-differential input
reflects it at every turn; it is also baked into the final scores, so the
targets stay consistent without further work. Default self-play uses no
handicap.

## Random openings

`play_game --random-opening-mean M` (with `M > 0`) plays each game's first
`K ~ round(Exp(M))` plies uniformly at random instead of by the seated agents
([`Game::set_random_opening`](../engine/src/game/game.cpp)). A random ply picks
among all legal placements and exchanges, and passes only when neither
exists. The point is off-policy coverage: it drives self-play into states,
especially unusual leaves, that agent play never visits.

Random moves corrupt the final-score targets of every position they precede,
so the eligible region starts right after the last random ply. That position
is the first whose remaining game is pure agent play, and is itself exactly
the kind of unusual state the mechanism exists to cover. Random moves are
ordinary `TurnBlob`s, so replay is unaffected. A game that ends during its
random opening has an empty eligible region and is dropped.

## Determinism and seeding

- [SeedProducer](../engine/src/util/seed_producer.cpp) is the global RNG
  source. `GameRunner` draws one base seed, and game *g* is played with seed
  `base + g`, or `base + g/2` under `--paired`, where games 2k and 2k+1 share a
  seed with the seats swapped so per-seed tile luck cancels. The game seed
  seeds that game's [Bag](../engine/src/game/bag.cpp), and derivations
  of it pick the handicap and the random-opening length.
- Nothing relies on a seed reproducing a specific bag, so auxiliary per-game
  randomness may draw from the game seed freely.
- The generate role runs every chunk with seed 0, which makes `play_game` draw
  a fresh seed per chunk: a fleet splitting a generation under any
  deterministic seed partition would duplicate games, so distributed corpora
  are deliberately not reproducible.
- The DataLoader's epoch shuffle and symmetry choices are seeded per epoch; see
  [data_loader.h](../engine/include/data/data_loader.h) for the epoch API
  (`epoch_start`, then repeated batch fills).
