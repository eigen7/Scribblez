# Architecture: the training-data pipeline

This is a code-level map of how a self-play game becomes a training row. It
complements [docs/roadmap.md](roadmap.md) (which covers the *why* and the model
roadmap) and [docs/design.md](design.md) (the design document) by
naming the components and the file that owns each one. For the *what each score
means to the model* story, see roadmap §1 (position evaluation model heads).

## Pipeline at a glance

```
generate_data.py ─▶ play_game ─▶ GameRunner ─▶ Game ─▶ GameLog
                                                          │
                                                BinaryLogWriter
                                                          │
                                                      .slog file
                                                          │
                                  NativeDataLoader (FFI) ─▶ DataLoader
                                                          │
                                                    BlockDecoder
                                                    (replays the game)
                                                          │
                              GameStateEncoder ─▶ input row + AllTargets labels
                                                          │
                                            torch tensors ─▶ model
```

| Stage | Owner | Notes |
|-------|-------|-------|
| Launch self-play | [py/scripts/generate_data.py](../py/scripts/generate_data.py) | Shells out to the `play_game` binary with two HastyBots and `--binary-log-dir`. |
| Game loop / threading | [GameRunner](../engine/src/selfplay/game_runner.cpp) | Owns agents, seeds (one per game), the win tally, and the parallel game loop. |
| One game | [Game](../engine/src/game/game.cpp) | Plays a single game and accumulates a [GameLog](../engine/include/game/game.h) (initial racks, every move + draw, final scores). |
| Serialize | [BinaryLogWriter](../engine/src/data/binary_log.cpp) | Batches finished games and flushes them to one `.slog` file per `--games-per-file`. |
| On-disk format | [binary_log.h](../engine/include/data/binary_log.h) | **Authoritative** layout (see below). |
| Load + replay | [BlockDecoder](../engine/src/data/block_decoder.cpp) | Replays a game forward to a given turn and emits one populated tensor row (the loader emits one per eligible turn). |
| State tracking + encoding | [GameStateEncoder](../engine/src/encoding/game_state_encoder.cpp) | Maintains board/scores/last-moves during replay and writes the model input. |
| Input layout | [input_encoder.h](../engine/include/encoding/input_encoder.h) | The `InputEncodingSpec` + block registry: block order/sizes and layout queries (full layout 88 planes + 992 scalars; base layout 85 + 936). |
| Label layout | [training_targets.h](../engine/include/training/training_targets.h) | The `AllTargets` registry — single source of truth for the label heads. |
| Stream to Python | [scribblez_ffi.cpp](../engine/src/serve/scribblez_ffi.cpp) → [py/scribblez/ffi.py](../py/scribblez/ffi.py) → [dataset.py](../py/scribblez/dataset.py) | C ABI over the `DataLoader`; epoch-based batch streaming. |
| Train | [py/scripts/position_eval/train.py](../py/scripts/position_eval/train.py), [py/scribblez/position_eval/model.py](../py/scribblez/position_eval/model.py) | Generational generate→train loop (see [docs/generational_training.md](generational_training.md)); ResNet trunk + the heads in `AllTargets`. |

## The `.slog` lifecycle

A `.slog` file holds many games as the *minimum* data needed to faithfully
replay every state — initial racks plus the move sequence (each move bundled
with the tiles drawn right after it). This is ~20× smaller than storing
fully-expanded per-position records, so far more games stay resident in the
DataLoader's shuffle buffer. The exact byte layout (FileHeader, the
`GameMetadata` table, then per-game `InitialRacks` + `TurnBlob[]`) lives in
[binary_log.h](../engine/include/data/binary_log.h) and is versioned by
`kVersion`; the decoder and FFI both reject a version mismatch, so stale files
fail loudly rather than misparse. Treat that header as the spec — this doc does
not duplicate the struct fields, so they cannot drift.

- **Write** — [`BinaryLogWriter::write_batch`](../engine/src/data/binary_log.cpp)
  records, per game, which turns are training-**eligible** (the region
  `[eligible_begin, eligible_end)`: `eligible_end` is the leading prefix of
  turns whose bag was non-empty, and `eligible_begin` is the position after the
  game's last random-opening ply — see below), and tallies the region widths
  into the `FileHeader`'s `num_sample_positions`. Games with an empty region
  are dropped. Per file: one `FileHeader`, a `GameMetadata` for every game,
  then each game's blobs. (`sampled_turn` is also recorded but is eval-only —
  a single representative position per game for probes / dumps.)
- **Read** — the [`DataLoader`](../engine/src/data/data_loader.cpp) expands
  each game into **one training row per eligible turn** (so an epoch sees every
  position, not one per game), and
  [`BlockDecoder::decode_one`](../engine/src/data/block_decoder.cpp)
  reconstructs each row: it replays the move sequence up to that turn through a
  `GameStateEncoder`, then encodes the input and the labels. The
  `post_move` flag selects the pre-move snapshot (active player about to play)
  vs. the post-move snapshot (just played, before drawing). Diagonal symmetry
  (`(r,c) → (c,r)`) is applied stochastically per row.

## The replay-reconstruction invariant

**A training row is reconstructed by replaying moves, not read back from an
expanded record.** This is the single most important thing to understand about
the pipeline, and it dictates where every value originates:

- **Model inputs are recomputed from the replay.** The board, the unseen-tile
  pool, last-move metadata, and the **score differential** are all rebuilt by
  applying moves through `GameStateEncoder`, which accumulates each play's score
  as it goes ([`apply_move`](../engine/src/encoding/game_state_encoder.cpp)).
  The score-diff input is therefore *derived*, not stored per position — it is
  whatever the running scores are at the sampled turn.
- **Targets come from the stored final scores.** The WLD and ScoreDiff heads are
  computed from `GameMetadata`'s final scores via the
  [GameLogView](../engine/include/training/training_targets.h) the decoder
  fills in — independent of the replay's running tally.

A practical consequence: any per-game state that must reach the input encoding
has to be seedable into the replay. A starting-score handicap is stored in
`GameMetadata` (`initial_score_p0/p1`) and used to seed the `GameStateEncoder`
at the top of `PositionEncoder::replay_to_sampled`, so the score-diff *input*
reflects it at every position; because the handicap is also baked into the
game's final scores, the *targets* stay consistent automatically. The default
self-play run requests no handicap (`--random-handicap-max 0`), so every game
starts 0-0 and the feature is dormant unless handicaps are requested.

## Random openings (off-policy state coverage)

With `play_game --random-opening-mean M` (> 0), each game's first K plies are
played **uniformly at random** — among all legal placements plus all legal
exchanges (every distinct non-empty sub-multiset of the rack, bag permitting),
passing only when neither exists — instead of by the seated agents, with K
drawn per game as `round(Exp(mean M))` from the game seed
([`SelfPlayEngine`](../engine/src/selfplay/self_play_engine.cpp) →
[`Game::set_random_opening`](../engine/src/game/game.cpp)). This drives
self-play into states — especially unusual rack leaves — that agent-vs-agent
play never visits, so the value model learns to evaluate them.

Random moves pollute the final-score *targets* of every position they follow,
so the eligible region starts at `K - 1`: the position right after the last
random ply is the first one whose remaining game is pure agent play (and is
itself exactly the kind of unusual state the mechanism exists to cover). The
random moves are ordinary `TurnBlob`s, so replay reconstruction is unaffected.
A game that terminates during its random opening (e.g. six consecutive
pass/exchange plies) has an empty eligible region and is dropped by the writer.
`position_eval/train.py` generates with `--random-opening-mean 2` by default.

## Determinism and seeding

- [SeedProducer](../engine/src/selfplay/seed_producer.cpp) is the global RNG
  source; `GameRunner` pulls one base seed and gives game *g* the seed
  `base + g`, which seeds that game's [Bag](../engine/src/game/bag.cpp).
- Seeds are **not** required to map to fixed bags — nothing in the system relies
  on reproducing a specific bag from a seed, so auxiliary per-game randomness
  (e.g. handicap selection) may draw from the game seed freely.
- The sampled-turn choice and the DataLoader's epoch shuffle are independently
  seeded; see [data_loader.h](../engine/include/data/data_loader.h) for the
  epoch API (`epoch_start` then repeated batch fills).
