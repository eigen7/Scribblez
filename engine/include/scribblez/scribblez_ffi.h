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

// Description of the session's input tensor(s): a spatial-plane tensor
// followed by a flat vector of scalar features. The shapes depend on the
// session's input-encoding spec (a contingent-features session reports the
// full layout; a baseline session the smaller base layout), so they hang off
// the session. Declared below the session typedef; see there.

// Static description of the target tensor(s) in row order:
//   target_index=0  "wld"                (3,)
//   target_index=1  "score_diff"         (1,)
//   target_index=2  "opp_next_placement"  (15, 15)
//   target_index=3  "self_next_placement" (15, 15)
//   target_index=4  "opp_win_placement"   (15, 15)
//   target_index=5  "self_win_placement"  (15, 15)
const ScribblezShape* scribblez_target_shapes(void);

// Shapes / sizes for the max-move-per-lane task (the "highest-scoring move per lane"
// model), a sibling layout to the post-move shapes above. Input: a (31, 15, 15)
// board-plane tensor + a 27-float rack-count vector. Targets, in row order:
//   target_index=0  "lane_occupancy" (30, 15, 27)
//   target_index=1  "lane_score"     (30,)
//   target_index=2  "lane_mask"      (30,)
const ScribblezShape* scribblez_max_move_per_lane_input_shapes(void);
const ScribblezShape* scribblez_max_move_per_lane_target_shapes(void);
int scribblez_max_move_per_lane_row_size_floats(void);
int scribblez_max_move_per_lane_input_floats(void);

// ===========================================================================
// Lexicon-bound session
// ===========================================================================
//
// Every entry point that needs the dictionary -- position encoding, GCG
// analysis, and DataLoader / stream construction -- is a method of a session,
// created once per process. Constructing the session loads
// <lexica-dir>/<lexicon_name>.kwg; a missing lexicon throws out of the
// constructor, which (uncaught across the C ABI) terminates the process.
// There is deliberately no failure signaling past that point: nothing useful
// can be done without a dictionary, so a live session pointer is proof the
// lexicon is loaded.
// `contingent_features` selects the process's experiment arm: nonzero encodes
// the full layout including the contingent-draw potential blocks; zero skips
// their move generation entirely and encodes the smaller base layout. The
// session's shape/size queries below report whichever layout it encodes, so
// callers never branch on the arm.
typedef struct ScribblezSession ScribblezSession;

ScribblezSession* scribblez_session_new(const char* lexicon_name, int contingent_features,
                                        int opp_leave_input);
void scribblez_session_delete(ScribblezSession* s);

// The session's input tensor shapes (see scribblez_target_shapes for the
// shape-entry conventions), total input floats, and total floats per training
// row (input + all target sizes -- sizes the output buffer without iterating
// the shape arrays in Python).
const ScribblezShape* scribblez_input_shapes(ScribblezSession* s);
int scribblez_input_floats(ScribblezSession* s);
int scribblez_row_size_floats(ScribblezSession* s);

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
// `out_inputs` : receives the input tensors, each scribblez_input_floats(s)
//                floats long, contiguous.
//
// Returns 0 on success, -1 on I/O error / bad header / out-of-range index.
int scribblez_encode_score_diff_sweep(ScribblezSession* s, const char* path, int64_t game_idx,
                                      int post_move, int diff_lo, int diff_hi, float* out_inputs);

// Sim evidence for a penultimate-bingo analysis GCG's final decision point:
// replays the GCG to the state BEFORE its final recorded move, ranks the
// mover's legal moves by HastyBot static equity, and runs SimRunner over the
// top-K (common random numbers, `rollouts` HastyBot rollouts each; with
// open_leaves != 0 every rollout starts the opponent from the leave their
// last recorded move retained, with their replenishments sampled; otherwise
// their whole rack is sampled uniformly from the unseen pool). Writes the
// evidence as
// consecutive SimObsRecord blobs (the .sobs record layout,
// sim_observation_log.h) into `out_records`, which must hold at least top_k
// records. Sets *played_rank to the index of the GCG's final (actually
// played) move within the returned candidates, or -1 if it fell outside the
// top-K. Returns the record count, or -1 on a parse error / an endgame
// position (the sim requires a non-empty bag at the decision point).
int scribblez_gcg_sim_evidence(ScribblezSession* s, const char* gcg_text, int top_k, int rollouts,
                               int threads, uint64_t seed, int open_leaves, char* out_records,
                               int* played_rank);

// Decode explicit training rows of one .slog file: row j is the position at
// (game_idx[j], turn_idx[j]), encoded exactly like a DataLoader training row
// (input floats followed by the label block) but with no symmetry flip.
// `post_move` selects the snapshot for every row. Writes n rows of
// scribblez_row_size_floats() floats to `out`. Returns 0 on success, -1 on an
// I/O / header error. Serves consumers that pair rows with per-position
// sidecar data (the .sobs sim observations) and so must address positions by
// identity rather than stream them shuffled.
int scribblez_decode_rows(ScribblezSession* s, const char* path, const int64_t* game_idx,
                          const int64_t* turn_idx, int64_t n, int post_move, float* out);

// Encode `n` candidate Moves into the pre-move value model's move-encoder input
// arrays (the single source of truth for that layout;
// scribblez/pre_move_value_move_encoder.h -- the training dataset reaches it
// here and the M_pre agent shares it at inference). `moves` points to n
// contiguous 16-byte serialized Moves (as stored in .slog/.pmt);
// `pre_move_score_diffs` is the mover's pre-move score advantage (points) per
// move, used to form the resultant post-move differential feature. Writes
// out_letters and out_squares as int32[n * max_placed], out_blanks and
// out_tile_mask as uint8[n * max_placed], and out_scalars as
// float[n * num_scalars], where max_placed / num_scalars come from
// scribblez_pre_move_value_move_dims. A pure function of its inputs: it needs
// no session or dictionary.
void scribblez_pre_move_value_encode_moves(const void* moves, int64_t n,
                                           const int32_t* pre_move_score_diffs,
                                           int32_t* out_letters, uint8_t* out_blanks,
                                           int32_t* out_squares, uint8_t* out_tile_mask,
                                           float* out_scalars);

// The move-encoder layout constants, so Python callers never hardcode them:
//   max_placed    letter/square array width (tiles per move slot)
//   num_scalars   per-move scalar-feature count
//   letter_vocab  letter-embedding vocabulary size (valid ids 0..letter_vocab-1;
//                 0 is the empty slot, 1..26 the letters)
//   cells         board-square embedding size (max square index + 1)
void scribblez_pre_move_value_move_dims(int32_t* max_placed, int32_t* num_scalars,
                                        int32_t* letter_vocab, int32_t* cells);

// The board input's score-differential scalar, so the pre-move dataset can read
// each position's pre-move differential straight out of the encoded row (rather
// than recomputing it): `scalar_index` is its index within the scalar input
// vector for this session's arm, and `scale` is the divisor the encoder applied
// (so points = input_scalar[scalar_index] * scale).
void scribblez_score_diff_input_layout(ScribblezSession* s, int32_t* scalar_index, float* scale);

// Render an ASCII description of a sampled position (POV, scores, leave, last
// moves, board) into `out` (NUL-terminated, truncated to out_cap). Returns
// the full string length on success (which may exceed out_cap - 1, signaling
// the caller to retry with a larger buffer), or -1 on I/O / header error.
int scribblez_dump_position(ScribblezSession* s, const char* path, int64_t game_idx, int post_move,
                            char* out, int out_cap);

// Like scribblez_dump_position, but emits the web UI's GameState JSON (board,
// bonuses, rack, scores, tile_scores, ...) for the sampled position, suitable
// for driving the web-style image renderer. Same return/truncation contract.
int scribblez_dump_position_json(ScribblezSession* s, const char* path, int64_t game_idx,
                                 int post_move, char* out, int out_cap);

// Parse a GCG file's text into its max-move-per-lane analysis position (the board
// after all recorded moves, the on-move player, and that player's #Rack), then:
//   - if `out_input` is non-null, fill it with the model input tensor
//     (MaxMovePerLaneInputEncoder, no flip): scribblez_max_move_per_lane_input_floats()
//     floats;
//   - emit the lane-analysis JSON (board + ground-truth per-lane targets and
//     maximal plays) into `out_json` (NUL-terminated, truncated to out_cap).
// Returns the full JSON length (same retry/truncation contract as the dump
// functions), or -1 on a parse error.
int scribblez_max_move_per_lane_analyze_gcg(ScribblezSession* s, const char* gcg_text,
                                            char* out_json, int out_cap, float* out_input);

// Parse a penultimate-bingo GCG's text into its post-move analysis position and fill
// `out_input` with the post-move value model's input tensor (scribblez_input_floats()
// floats), encoded from the POV of the player that made the final recorded move
// (whose leave is the encode-time rack). The encoding replays the recorded moves, so
// it is byte-identical to a training row's input for the same position. Returns
// scribblez_input_floats() on success, or -1 on a parse error / non-PLAY final move.
int scribblez_post_move_value_analyze_gcg(ScribblezSession* s, const char* gcg_text,
                                          float* out_input);

// Emit the web-render board bundle (GameState JSON: board / bonuses / rack /
// tile_scores, plus a "start_player" field) for a penultimate-bingo GCG's post-move
// analysis position, from the POV of the player that made the final move (its leave
// is the shown rack), into `out_json` (NUL-terminated, truncated to out_cap). Returns
// the full JSON length (same retry/truncation contract as the dump functions), or -1
// on a parse error / non-PLAY final move.
int scribblez_post_move_value_board_json(const char* gcg_text, char* out_json, int out_cap);

// Like scribblez_post_move_value_analyze_gcg, but encodes from an explicit alternate
// `leave_str` ('?' = a blank) -- a dashboard what-if -- instead of the GCG's recorded
// leave. The alternate leave must match the recorded leave's tile count and use only
// tiles available off the board. Returns scribblez_input_floats() on success; on
// failure returns -1 and writes a human-readable reason into `out_err` (NUL-
// terminated, truncated to err_cap).
int scribblez_post_move_value_analyze_gcg_leave(ScribblezSession* s, const char* gcg_text,
                                                const char* leave_str, float* out_input,
                                                char* out_err, int err_cap);

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

// Create a loader over the session's lexicon. `task` selects the training row it
// decodes: 0 = the post-move value row (expands each game over its bag-non-empty
// eligible-turn prefix), 1 = the max-move-per-lane row (expands over every turn).
// A task-1 loader emits scribblez_max_move_per_lane_row_size_floats() per row; a
// task-0 loader emits scribblez_row_size_floats().
DataLoaderHandle* scribblez_dl_new(ScribblezSession* s, int64_t memory_budget,
                                   int num_worker_threads, int num_prefetch_threads, int task);

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
// Returns NULL on bad config.
StreamHandle* scribblez_stream_new(ScribblezSession* s, float* const* slot_ptrs, int num_slots,
                                   int rows_per_slot, int num_threads, int post_move,
                                   int apply_symmetry, uint64_t seed, int handicap_max,
                                   const char* const* player_specs, int num_specs);

// Like scribblez_stream_new, but streams max-move-per-lane-task rows
// (scribblez_max_move_per_lane_row_size_floats() floats each): every slot row is a
// max-move-per-lane input + per-lane labels, sampled uniformly over all turns (there is no
// post_move snapshot choice). Shares the rest of the streaming API
// (start/wait/release/stats/stop/delete) with the post-move streamer.
StreamHandle* scribblez_max_move_per_lane_stream_new(ScribblezSession* s, float* const* slot_ptrs,
                                                     int num_slots, int rows_per_slot,
                                                     int num_threads, int apply_symmetry,
                                                     uint64_t seed, int handicap_max,
                                                     const char* const* player_specs,
                                                     int num_specs);

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
