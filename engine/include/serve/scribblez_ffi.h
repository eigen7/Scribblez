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

// The input shapes depend on the session's encoding spec, so they hang off the
// session; see scribblez_input_shapes below. The targets do not:
//   target_index=0  "wld"                (3,)
//   target_index=1  "score_diff"         (1,)
//   target_index=2  "opp_next_placement"  (15, 15)
//   target_index=3  "self_next_placement" (15, 15)
//   target_index=4  "opp_win_placement"   (15, 15)
//   target_index=5  "self_win_placement"  (15, 15)
const ScribblezShape* scribblez_target_shapes(void);

// The max-move-per-lane task's sibling layout. Input: a (31, 15, 15)
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
// `opp_leave_input` selects the process's experiment arm: nonzero encodes the
// open-leaves layout, whose extra block carries what the opponent kept from
// their last move. The session's shape/size queries below report whichever
// layout it encodes, so callers never branch on the arm.
typedef struct ScribblezSession ScribblezSession;

ScribblezSession* scribblez_session_new(const char* lexicon_name, int opp_leave_input);
void scribblez_session_delete(ScribblezSession* s);

// The session's input tensor shapes (see scribblez_target_shapes for the
// shape-entry conventions), total input floats, and total floats per training
// row (input + all target sizes -- sizes the output buffer without iterating
// the shape arrays in Python).
const ScribblezShape* scribblez_input_shapes(ScribblezSession* s);
int scribblez_input_floats(ScribblezSession* s);
int scribblez_row_size_floats(ScribblezSession* s);

// A sister to the DataLoader that reads a .slog with NO sampling or shuffling:
// it replays a sampled position and re-encodes it once per integer score
// differential in [diff_lo, diff_hi], sweeping ONLY the active player's score
// advantage while board, racks, and move history stay constant. Let
// R = diff_hi - diff_lo + 1.
//
// `game_idx`   : one game -> R tensors, or < 0 for EVERY game in the file ->
//                num_games * R tensors, game g occupying rows [g*R, (g+1)*R).
// `post_move`  : the post-move snapshot (1) or the pre-move one (0).
// `out_inputs` : contiguous, each tensor scribblez_input_floats(s) long.
//
// Returns 0 on success, -1 on I/O error / bad header / out-of-range index.
int scribblez_encode_score_diff_sweep(ScribblezSession* s, const char* path, int64_t game_idx,
                                      int post_move, int diff_lo, int diff_hi, float* out_inputs);

// Sim evidence for a penultimate-bingo analysis GCG's final decision point:
// replay to the state BEFORE its final recorded move, rank the mover's legal
// moves by HastyBot static equity, and run SimRunner over the top-K. With
// open_leaves != 0 every rollout starts the opponent from the leave their last
// recorded move retained; otherwise their whole rack is sampled.
//
// `out_records` must hold at least top_k consecutive SimObsRecord blobs (see
// sim_observation_log.h). *played_rank receives the index of the GCG's actually
// played move among the candidates, or -1 if it fell outside the top-K. Returns
// the record count, or -1 on a parse error or an endgame position, the sim
// requiring a non-empty bag.
int scribblez_gcg_sim_evidence(ScribblezSession* s, const char* gcg_text, int top_k, int rollouts,
                               int threads, uint64_t seed, int open_leaves, char* out_records,
                               int* played_rank);

// Row j is the position at (game_idx[j], turn_idx[j]), encoded exactly like a
// DataLoader training row but with no symmetry flip; `out` takes n rows of
// scribblez_row_size_floats(). Returns 0 on success, -1 on an I/O / header
// error.
//
// It exists for consumers that pair rows with per-position sidecar data (the
// .sobs sim observations) and so must address positions by identity rather than
// stream them shuffled.
int scribblez_decode_rows(ScribblezSession* s, const char* path, const int64_t* game_idx,
                          const int64_t* turn_idx, int64_t n, int post_move, float* out);

// Encode `n` candidate Moves into the move set evaluation model's move-encoder
// input arrays, whose layout training/move_set_encoder.h owns -- the training
// dataset reaches it here and the agent shares it at inference. `moves` points to n
// contiguous 16-byte serialized Moves (as stored in .slog/.mset);
// `pre_move_score_diffs` is the mover's pre-move score advantage (points) per
// move, used to form the resultant post-move differential feature. Writes
// out_letters and out_squares as int32[n * max_placed], out_blanks and
// out_tile_mask as uint8[n * max_placed], and out_scalars as
// float[n * num_scalars], where max_placed / num_scalars come from
// scribblez_move_set_move_dims. A pure function of its inputs: it needs
// no session or dictionary.
void scribblez_move_set_encode_moves(const void* moves, int64_t n,
                                     const int32_t* pre_move_score_diffs, int32_t* out_letters,
                                     uint8_t* out_blanks, int32_t* out_squares,
                                     uint8_t* out_tile_mask, float* out_scalars);

// So Python callers never hardcode them:
//   max_placed    letter/square array width (tiles per move slot)
//   num_scalars   per-move scalar-feature count
//   letter_vocab  letter-embedding vocabulary size (valid ids 0..letter_vocab-1;
//                 0 is the empty slot, 1..26 the letters)
//   cells         board-square embedding size (max square index + 1)
void scribblez_move_set_move_dims(int32_t* max_placed, int32_t* num_scalars, int32_t* letter_vocab,
                                  int32_t* cells);

// The move-feature semantics version (move_set_encoder.h kMoveEncodingVersion):
// recorded in checkpoints and stamped into mset ONNX exports, so a model can
// never silently run against an encoder whose rows it was not trained on.
int32_t scribblez_move_set_encoding_version(void);

// Where the board input's score-differential scalar sits, so the move set
// dataset can read a position's pre-move differential straight out of the
// encoded row instead of recomputing it: points =
// input_scalar[scalar_index] * scale.
void scribblez_score_diff_input_layout(ScribblezSession* s, int32_t* scalar_index, float* scale);

// An ASCII description of a sampled position (POV, scores, leave, last moves,
// board) into `out`, NUL-terminated and truncated to out_cap. Returns the full
// string length, which may exceed out_cap - 1 and so signal a retry with a
// larger buffer, or -1 on I/O / header error.
int scribblez_dump_position(ScribblezSession* s, const char* path, int64_t game_idx, int post_move,
                            char* out, int out_cap);

// scribblez_dump_position's output as the web UI's GameState JSON instead,
// for the web-style image renderer. Same return/truncation contract.
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

// Parse a dataset GCG's text into its post-move position and fill
// `out_input` with the position evaluation model's input tensor, encoded from
// the POV of the player that made the final recorded move (whose leave is the
// encode-time rack; the opponent-leave arm also reads what the opponent's last
// move retained). The encoding replays the recorded moves, so it is
// byte-identical to a training row's input for the same position. Encodes
// under an explicit arm (opp_leave_input) rather than the session's, so one
// dashboard process serves models of every arm (the session contributes only
// its dictionary); `input_cap` must equal the arm's
// input_floats, else -1 (the caller's model disagrees with the engine layout).
// Returns the floats written on success, or -1 with a reason in `out_err`
// (NUL-terminated, truncated to err_cap): a parse error, a non-PLAY final
// move, or the width mismatch.
int scribblez_position_eval_analyze_gcg(ScribblezSession* s, const char* gcg_text,
                                        int opp_leave_input, float* out_input, int input_cap,
                                        char* out_err, int err_cap);

// Emit the web-render board bundle (GameState JSON: board / bonuses / rack /
// tile_scores, plus "start_player", "last_move", and "opp_leave" fields) for
// a dataset GCG's post-move position, from the POV of the player that made the
// final move (its leave is the shown rack), into `out_json` (NUL-terminated,
// truncated to out_cap). Returns the full JSON length (same retry/truncation contract as
// the dump functions), or -1 on a parse error / non-PLAY final move.
int scribblez_position_eval_board_json(const char* gcg_text, char* out_json, int out_cap);

// Like scribblez_position_eval_analyze_gcg, but encodes from explicit
// alternate leaves ('?' = a blank) -- a dashboard what-if -- instead of the
// GCG's recorded ones: `leave_str` for the POV and, unless NULL, `opp_leave_str`
// for the opponent. Each alternate must match the tile count of the leave it
// replaces and the two together use only tiles available off the board; a
// violation is one more -1-with-reason, phrased for a person.
int scribblez_position_eval_analyze_gcg_leaves(ScribblezSession* s, const char* gcg_text,
                                               const char* leave_str, const char* opp_leave_str,
                                               int opp_leave_input, float* out_input, int input_cap,
                                               char* out_err, int err_cap);

// A position-set .gcg's decision point (read_gcg_position: final recorded
// state, side to move next, rack from its #RackN pragma) as the move set
// evaluation model's inputs -- what the dashboard's trajectory pane re-scores
// under a torch checkpoint (training/trajectory_position.h). Encodes under an
// explicit arm (opp_leave_input) rather than the session's, so one dashboard
// process serves models of every arm; the session contributes only its
// dictionary. opp_leave_input doubles as the information
// condition the position's sidecars were simmed under.
//   out_input      the mover's pre-move board row; `input_cap` must equal the
//                  arm's input_floats, else -1 (the caller's model config
//                  disagrees with the engine layout);
//   out_score_diff the mover's pre-move score differential (points);
//   out_moves      up to `moves_cap` 16-byte Moves: the FULL legal move list
//                  in the equity ranking the trajectory generator drew from.
// Returns the legal move count (which may exceed moves_cap -- retry with a
// larger buffer; the row and differential are written either way), or -1 with
// a reason in `out_err` (NUL-terminated, truncated to err_cap).
int scribblez_gcg_position_inputs(ScribblezSession* s, const char* gcg_text, int opp_leave_input,
                                  float* out_input, int input_cap, int32_t* out_score_diff,
                                  void* out_moves, int moves_cap, char* out_err, int err_cap);

// The pane's web-render bundle for the same decision point: the mover's-POV
// GameState JSON plus "mover", "opp_leave", "last_move", and "moves" -- every
// legal move's GCG notation in the order scribblez_gcg_position_inputs emits
// them under the same `open_leaves`. Returns the full JSON length (the dump
// functions' retry/truncation contract), or -1 on a parse error.
int scribblez_gcg_position_board_json(ScribblezSession* s, const char* gcg_text, int open_leaves,
                                      char* out_json, int out_cap);

// Write a new .slog at `dst_path` containing the `num_picks` selected games,
// in order. `src_paths[i]` and `game_indices[i]` together identify the i-th
// game (a source .slog path and the game index within it). Games are copied
// verbatim; only the header and start offsets are recomputed. Returns 0 on
// success, -1 on any I/O / header / out-of-range error.
int scribblez_sample_slog(const char* dst_path, const char* const* src_paths,
                          const int64_t* game_indices, int num_picks);

// Returns 0 having filled *out_num_positions / *out_file_size, or -1 on I/O
// failure or a magic / version mismatch.
int scribblez_read_file_header(const char* path, int64_t* out_num_positions,
                               int64_t* out_file_size);

// A JSON document describing the sidecar binary formats (.slog / .sobs /
// .mset): per struct the field names, offsets, and numpy dtype codes -- taken
// from the compiler, so it cannot drift from the structs -- plus magics,
// versions, flag bits, and the MoveType values (data/format_layout.h
// documents the shape). Python builds its numpy dtypes from this instead of
// hand-mirroring the packed structs. Static storage; never freed. Needs no
// session.
const char* scribblez_format_layout_json(void);

typedef struct DataLoaderHandle DataLoaderHandle;

// Create a loader over the session's lexicon. `task` selects the training row it
// decodes: 0 = the position evaluation row (expands each game over its bag-non-empty
// eligible-turn prefix), 1 = the max-move-per-lane row (expands over every turn).
// A task-1 loader emits scribblez_max_move_per_lane_row_size_floats() per row; a
// task-0 loader emits scribblez_row_size_floats().
DataLoaderHandle* scribblez_dl_new(ScribblezSession* s, int64_t memory_budget,
                                   int num_worker_threads, int num_prefetch_threads, int task);

void scribblez_dl_delete(DataLoaderHandle* h);

// Chronological, oldest-first. `num_positions` must match what the FileHeader
// reports and `file_size` the on-disk size.
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

// Returns the rows written, 0 once the epoch is exhausted. `output` needs room
// for batch_size * scribblez_row_size_floats() floats.
int scribblez_dl_load_batch(DataLoaderHandle* h, float* output);

// Resident bytes, for testing eviction.
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

// Idempotent.
void scribblez_stream_start(StreamHandle* h);

// Blocks; -1 once the stream has stopped and no more slots will come. Being a
// plain C call invoked via ctypes it releases the Python GIL, so the training
// thread runs while producers fill slots.
int scribblez_stream_wait_full_slot(StreamHandle* h);

void scribblez_stream_release_slot(StreamHandle* h, int slot);

void scribblez_stream_get_stats(StreamHandle* h, ScribblezStreamStats* out);

// Wakes the consumer and producers, then joins the producer threads.
// Idempotent.
void scribblez_stream_stop(StreamHandle* h);

// Stops and joins first if needed.
void scribblez_stream_delete(StreamHandle* h);

#ifdef __cplusplus
}
#endif
