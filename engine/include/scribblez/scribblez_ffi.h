// Plain C ABI for the scribblez training-data loader. Designed for
// consumption from Python via cffi.
//
// All entry points are thread-safe with respect to distinct
// DataLoaderHandle instances. A single DataLoaderHandle's methods are NOT
// re-entrant; the loader internally uses its own thread pools but each
// public call must be made from one Python thread at a time.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One named tensor in the row layout. `dims` points into a static array
// owned by the library; the caller must NOT free it. Returned tensor lists
// are terminated by a sentinel entry whose `name == NULL`.
typedef struct ScribblezShape {
  const char* name;
  const int* dims;
  int num_dims;
  int target_index;  // -1 for inputs; 0..N-1 for targets, in row layout order
} ScribblezShape;

// Static description of the input tensor(s). Layout is fixed by the C++
// engine: a single (32, 15, 15) spatial-plane tensor followed by an extra
// flat vector of scalar features (which here is concatenated into the
// "input" name as a 1D fallback -- callers that want the structured
// view can split using kInputSpatialFloats + kInputScalarFloats from a
// downstream Python constants file).
const ScribblezShape* scribblez_input_shapes(void);

// Static description of the target tensor(s) in row order:
//   target_index=0  "wld"                (3,)
//   target_index=1  "score_diff"         (1,)
//   target_index=2  "opp_next_placement" (15, 15)
const ScribblezShape* scribblez_target_shapes(void);

// Total floats per output row -- sum of input + all target sizes. Useful
// to size the output buffer without iterating the shape arrays in Python.
int scribblez_row_size_floats(void);

// Shapes / sizes for the LEXICAL task (the "highest-scoring move per lane"
// model), a sibling layout to the post-move shapes above. Input: a (31, 15, 15)
// board-plane tensor + a 27-float rack-count vector. Targets, in row order:
//   target_index=0  "lane_occupancy" (30, 15, 27)
//   target_index=1  "lane_score"     (30,)
//   target_index=2  "lane_mask"      (30,)
const ScribblezShape* scribblez_lexical_input_shapes(void);
const ScribblezShape* scribblez_lexical_target_shapes(void);
int scribblez_lexical_row_size_floats(void);
int scribblez_lexical_input_floats(void);

// Score-differential sweep encoder -- a sister to the DataLoader that reads a
// .slog and materializes input tensors with NO sampling or shuffling. It
// replays a sampled position and re-encodes it once per integer score
// differential in [diff_lo, diff_hi], sweeping ONLY the active player's score
// advantage while board, racks, and move history stay constant. Let
// R = diff_hi - diff_lo + 1.
//
// `path`       : .slog file to read.
// `game_idx`   : a single game (0 .. num_games-1) -> R tensors, OR < 0 for
//                EVERY game in the file -> num_games * R tensors, position-
//                major (game g occupies rows [g*R, (g+1)*R)).
// `post_move`  : encode the post-move snapshot (1) or pre-move (0).
// `out_inputs` : receives the input tensors, each scribblez_input_floats()
//                floats long, contiguous.
//
// Returns 0 on success, -1 on I/O error / bad header / out-of-range index.
int scribblez_encode_score_diff_sweep(const char* path, int64_t game_idx, int post_move,
                                      int diff_lo, int diff_hi, float* out_inputs);

// Floats in a single input tensor (spatial + scalar), i.e. the per-position
// stride of scribblez_encode_score_diff_sweep's output.
int scribblez_input_floats(void);

// Render an ASCII description of a sampled position (POV, scores, leave, last
// moves, board) into `out` (NUL-terminated, truncated to out_cap). Returns
// the full string length on success (which may exceed out_cap - 1, signaling
// the caller to retry with a larger buffer), or -1 on I/O / header error.
int scribblez_dump_position(const char* path, int64_t game_idx, int post_move, char* out,
                            int out_cap);

// Like scribblez_dump_position, but emits the web UI's GameState JSON (board,
// bonuses, rack, scores, tile_scores, ...) for the sampled position, suitable
// for driving the web-style image renderer. Same return/truncation contract.
int scribblez_dump_position_json(const char* path, int64_t game_idx, int post_move, char* out,
                                 int out_cap);

// Write a new .slog at `dst_path` containing the `num_picks` selected games,
// in order. `src_paths[i]` and `game_indices[i]` together identify the i-th
// game (a source .slog path and the game index within it). Games are copied
// verbatim; only the header and start offsets are recomputed. Returns 0 on
// success, -1 on any I/O / header / out-of-range error.
int scribblez_sample_slog(const char* dst_path, const char* const* src_paths,
                          const int64_t* game_indices, int num_picks);

// Peek at the FileHeader of a .slog file. Returns 0 on success and fills
// *out_num_positions / *out_file_size. Returns -1 on I/O failure or magic
// / version mismatch.
int scribblez_read_file_header(const char* path, int64_t* out_num_positions,
                               int64_t* out_file_size);

// Opaque DataLoader handle.
typedef struct DataLoaderHandle DataLoaderHandle;

DataLoaderHandle* scribblez_dl_new(int64_t memory_budget, int num_worker_threads,
                                   int num_prefetch_threads);

void scribblez_dl_delete(DataLoaderHandle* h);

// Register one .slog file (chronological, oldest-first). `num_positions`
// must match the value the FileHeader reports; `file_size` must match the
// on-disk size in bytes.
void scribblez_dl_add_file(DataLoaderHandle* h, const char* path, int64_t num_positions,
                           int64_t file_size);

int64_t scribblez_dl_num_positions(const DataLoaderHandle* h);

// Epoch-based streaming API. Shuffles files and positions within files
// deterministically based on `seed`. Returns the number of complete batches
// in the epoch (the last partial batch, if any, is also yielded).
//
// `turns_per_game` controls per-game turn subsampling: 0 trains on every
// eligible turn; k > 0 draws k turns per game per epoch, and `epoch_index`
// selects which turns so successive epochs cover distinct turns (see
// DataLoader::EpochConfig).
int scribblez_dl_epoch_start(DataLoaderHandle* h, int batch_size, int post_move, int apply_symmetry,
                             uint64_t seed, int turns_per_game, int epoch_index);

// Fill `output` with the next batch of the current epoch. Returns the
// number of rows written (0 = epoch exhausted). `output` must have capacity
// for at least batch_size * scribblez_row_size_floats() floats.
int scribblez_dl_load_batch(DataLoaderHandle* h, float* output);

// Query current resident memory in bytes (useful for testing eviction).
int64_t scribblez_dl_resident_bytes(const DataLoaderHandle* h);

// ===========================================================================
// Streaming self-play -> training pipeline
// ===========================================================================
//
// The Python trainer owns N row buffers ("slots"), each at least
// rows_per_slot * scribblez_row_size_floats() floats, and passes their
// addresses in. C++ producer threads play HastyBot self-play games and write
// each game's sampled training row directly into the current slot. When a slot
// fills, scribblez_stream_wait_full_slot returns its index; the trainer
// consumes it (copying out / moving to GPU) and calls
// scribblez_stream_release_slot to hand it back. With N=2 this double-buffers
// CPU game generation against GPU training. No game data touches disk.

// Throughput / backpressure snapshot. producer_blocked_ns growing means the
// producers waited for a free slot (the consumer/GPU is the bottleneck);
// consumer_blocked_ns growing means the consumer waited for a full slot (the
// producers/CPU are the bottleneck).
typedef struct ScribblezStreamStats {
  int64_t games_played;     // games whose sampled row was committed
  int64_t games_dropped;    // games with no eligible (bag-nonempty) turn
  int64_t rows_committed;   // == positions produced
  int64_t slots_published;  // full slots handed to the consumer
  int64_t producer_blocked_ns;
  int64_t consumer_blocked_ns;
} ScribblezStreamStats;

typedef struct StreamHandle StreamHandle;

// Create the streamer over `num_slots` caller-owned float buffers (each at least
// rows_per_slot * scribblez_row_size_floats() floats). `player_specs` is an
// array of `num_specs` NUL-terminated `--player` spec strings (typically two
// "--type=hastybot"). Does NOT start producing until scribblez_stream_start.
// Returns NULL on bad config / lexicon load failure.
StreamHandle* scribblez_stream_new(float* const* slot_ptrs, int num_slots, int rows_per_slot,
                                   int num_threads, int post_move, int apply_symmetry,
                                   uint64_t seed, int handicap_max, const char* const* player_specs,
                                   int num_specs);

// Like scribblez_stream_new, but streams LEXICAL-task rows
// (scribblez_lexical_row_size_floats() floats each): every slot row is a lexical
// input + per-lane labels, sampled uniformly over all turns (there is no
// post_move snapshot choice). Shares the rest of the streaming API
// (start/wait/release/stats/stop/delete) with the post-move streamer.
StreamHandle* scribblez_lexical_stream_new(float* const* slot_ptrs, int num_slots,
                                           int rows_per_slot, int num_threads, int apply_symmetry,
                                           uint64_t seed, int handicap_max,
                                           const char* const* player_specs, int num_specs);

// Spawn the producer threads. Idempotent.
void scribblez_stream_start(StreamHandle* h);

// Block until a full slot is ready; returns its index [0, num_slots), or -1 if
// the stream has stopped and no more slots will come. Releases the Python GIL
// (it is a plain C call invoked via ctypes), so the training thread runs while
// producers fill slots.
int scribblez_stream_wait_full_slot(StreamHandle* h);

// Return a consumed slot to the producers.
void scribblez_stream_release_slot(StreamHandle* h, int slot);

// Snapshot the throughput / backpressure counters.
void scribblez_stream_get_stats(StreamHandle* h, ScribblezStreamStats* out);

// Signal shutdown: wakes the consumer and producers and joins the producer
// threads. Idempotent.
void scribblez_stream_stop(StreamHandle* h);

// Destroy the streamer (stops + joins first if needed).
void scribblez_stream_delete(StreamHandle* h);

#ifdef __cplusplus
}
#endif
