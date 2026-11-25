// The engine's plain C ABI for Python (loaded with ctypes by
// py/scribblez/ffi.py): row layouts, the training DataLoader, and the position
// encoders and analysis helpers the dashboard and tools call.
//
// Conventions:
//   - Most entry points return -1 on failure; an `out_err` buffer, where one
//     is taken, receives a reason.
//   - String outputs are NUL-terminated and truncated to `out_cap`. The return
//     value is the full length, so a caller can retry with a larger buffer.
//   - Distinct DataLoader handles may be used concurrently, but calls on one
//     handle must come from one thread at a time.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One named tensor in the row layout. `dims` is owned by the library; do not
// free it. Lists of these end with a sentinel entry whose `name` is NULL.
typedef struct ScribblezShape {
  const char* name;
  const int* dims;
  int num_dims;
  int target_index;  // -1 for inputs; 0..N-1 for targets, in row layout order
} ScribblezShape;

// ===========================================================================
// Lexicon-bound session
// ===========================================================================
//
// Every entry point that needs the dictionary (position encoding, GCG
// analysis, DataLoader construction) takes a session, created once per
// process. Creating it loads <lexica-dir>/<lexicon_name>.kwg. A missing lexicon
// throws out of the constructor and, uncaught across the C ABI, terminates the
// process. That is deliberate: nothing useful can be done
// without a dictionary, so a live session is proof the lexicon is loaded and
// no later call needs to check.
//
// `opp_leave_input` selects the session's input-encoding arm. Nonzero adds the
// open-leaves block, which carries what the opponent kept from their last move.
// The shape and size queries report the session's layout, so callers never
// branch on the arm.
typedef struct ScribblezSession ScribblezSession;

ScribblezSession* scribblez_session_new(const char* lexicon_name, int opp_leave_input);
void scribblez_session_delete(ScribblezSession* s);

// The process's CPU budget for the library's parallel steps (util/work_pool.h);
// by default, every core available to the process.
void scribblez_set_worker_count(int n);

// Training tasks, by code: the row a DataLoader (scribblez_dl_new) or
// scribblez_decode_rows decodes.
//   0  position evaluation: the session's input arm (spatial + scalar), then
//      the targets of AllTargets in training/training_targets.h
//   1  leave inference: training/leave_inference_task.h's input; targets
//      "opp_leave_tiles" (7,), "opp_leave_length", "opp_leave_valid",
//      "opp_leave_cause" (1,) each, "leave_pool" (27,), "bag_size" (1,)
//   2  public leave inference: task 1's tensors, the input in its public view
//   3  DeepBot leaf evaluation: training/leaf_eval_task.h's input; targets
//      "wld" (3,), "score_diff" (1,). Loads HastyBot's leave values.
// The shape lists are in row order and owned by the library. Each query
// returns NULL or -1 for an unknown task code.
const ScribblezShape* scribblez_task_input_shapes(ScribblezSession* s, int task);
const ScribblezShape* scribblez_task_target_shapes(ScribblezSession* s, int task);
int scribblez_task_row_floats(ScribblezSession* s, int task);

// Training rows of `task` for specific positions: row j is (game_idx[j],
// turn_idx[j]), encoded exactly as the DataLoader would, minus the symmetry
// transpose. `out` takes n rows of scribblez_task_row_floats(s, task). Returns 0
// on success, -1 on an I/O or header error, an unknown task, or, for leave
// inference, a missing or mismatched .forgone sidecar or a turn past the
// game's eligible region (the reason goes to stderr).
//
// For consumers that pair rows with per-position data (such as a held-out
// set's cached results) and so must address positions directly. `hide_private` hides a leave
// inference row's observer's private history (LeaveHistory::hide_private); other tasks ignore it.
int scribblez_decode_rows(ScribblezSession* s, int task, const char* path, const int64_t* game_idx,
                          const int64_t* turn_idx, int64_t n, int post_move, int hide_private,
                          float* out);

// The win table's fitting samples from one .slog (sim/win_table_feature.h):
// one WinTableSample record per fittable pre-move position of every game, its
// layout from scribblez_format_layout_json. Writes the first min(count, cap)
// records and returns the count, so a call with cap 0 sizes the buffer.
// Returns -1 on an I/O or header error, and -2 for a face-up-leaves file: win
// tables model hidden-leaves play only.
int64_t scribblez_win_table_samples(ScribblezSession* s, const char* path, void* out_samples,
                                    int64_t cap);

// The session lexicon's HastyBot leave values laid out for whole-length scans
// (lexicon/leave_value_arrays.h): every leave of each length 1..6 drawable
// from a full bag, in the leave values file's order. Free with
// scribblez_leave_value_arrays_delete.
typedef struct LeaveValueArraysHandle LeaveValueArraysHandle;
LeaveValueArraysHandle* scribblez_leave_value_arrays_new(ScribblezSession* s);
void scribblez_leave_value_arrays_delete(LeaveValueArraysHandle* a);

// The number of length-k leaves, or -1 for k outside 1..6.
int64_t scribblez_leave_value_arrays_rows(const LeaveValueArraysHandle* a, int k);

// Copies the length-k leaves, in order: their tiles' codes
// (LeaveValueArrays::code, letter * 6 + copy) slot by slot into out_codes,
// k arrays of rows bytes, and their values, in points, into out_values.
// Returns 0, or -1 for k outside 1..6.
int scribblez_leave_value_arrays_copy(const LeaveValueArraysHandle* a, int k, uint8_t* out_codes,
                                      float* out_values);

// The content hash of the .klv2 the values came from: the identity models
// trained against them record. Owned by the handle.
const char* scribblez_leave_value_arrays_content_hash(const LeaveValueArraysHandle* a);

// One LeaveSampler query (belief/leave_sampler.h) over the handle's length-k
// leaves: `params` the model's 29 floats (belief/leave_params.h), and
// `leave_pool` the 27 tile counts the leave is drawn from. Writes log Z to
// out_log_z, the log probability of each of the n_rows `rows` to
// out_log_probs, and the row drawn at each of the n_quantiles `quantiles` to
// out_samples. Returns 0, or -1 with a message in `out_err`.
int scribblez_leave_sampler_query(const LeaveValueArraysHandle* a, const float* params, int k,
                                  const int32_t* leave_pool, const int64_t* rows, int64_t n_rows,
                                  const double* quantiles, int64_t n_quantiles, double* out_log_z,
                                  double* out_log_probs, int64_t* out_samples, char* out_err,
                                  int err_cap);

// Writes a scribblez win table (sim/win_pct_table.h) from `grid`, the
// fitter's kRows x kCols floats in file order, and `header`, its provenance as
// "key=value" lines separated by '\n'. Returns 0, or -1 with a message in
// `out_err`.
int scribblez_win_table_write(const char* path, const float* grid, const char* header,
                              char* out_err, int err_cap);

// P(win) for each of `n` WinTableFeature records under the table at `path`:
// a scribblez table, or Macondo's winpct.csv when `macondo_format` is nonzero.
// Returns 0, or -1 with a message in `out_err` (a malformed file, or a
// scribblez table of another format version).
int scribblez_win_table_eval(const char* path, int macondo_format, const void* features, int64_t n,
                             float* out, char* out_err, int err_cap);

// The position-evaluation input for a dataset GCG's post-move position, from
// the POV of the player who made the final recorded move (their leave is the
// rack). The encoding replays the recorded moves, so it matches a training
// row's input for the same position exactly.
//
// Encodes under the given arm rather than the session's, so one dashboard
// process can serve models of every arm; the session supplies only the
// dictionary. `input_cap` must equal that arm's input width, or the caller's
// model disagrees with the engine's layout. Returns the floats written, or -1
// with a reason in `out_err`: a parse error, a non-PLAY final move, or a width
// mismatch.
int scribblez_position_eval_analyze_gcg(ScribblezSession* s, const char* gcg_text,
                                        int opp_leave_input, float* out_input, int input_cap,
                                        char* out_err, int err_cap);

// Collapse a model's placement logits for a dataset GCG's post-move position
// into per-cell occupancy marginals. `raw` holds the four placement heads'
// footprint logits (kPlacementHeads * kFootprintClasses); `out` receives four
// 15x15 planes (kPlacementHeads * 225 floats), computed with the same
// mask, softmax, and scatter the training targets use. Returns the floats
// written, or -1 with a reason in `out_err`: a parse error, a non-PLAY final
// move, or a buffer too small.
int scribblez_position_eval_collapse_placement(ScribblezSession* s, const char* gcg_text,
                                               const float* raw, int raw_cap, float* out,
                                               int out_cap, char* out_err, int err_cap);

// The same masked footprint distributions, per class rather than collapsed to
// cells: `out` receives kPlacementHeads * kFootprintClasses floats, illegal
// footprints at zero. Same inputs and errors as the collapse above. For
// measuring the sparsity and fidelity of the distilled placement target.
int scribblez_position_eval_masked_placement(ScribblezSession* s, const char* gcg_text,
                                             const float* raw, int raw_cap, float* out, int out_cap,
                                             char* out_err, int err_cap);

// The four placement heads' per-cell legality, as kPlacementHeads * 225 floats
// of 1 or 0. Returns the floats written, or -1 with a reason in `out_err`.
int scribblez_position_eval_legal_placement(ScribblezSession* s, const char* gcg_text, float* out,
                                            int out_cap, char* out_err, int err_cap);

// The board-rendering JSON for a dataset GCG's post-move position: GameState
// plus "start_player", "last_move", and "opp_leave", from the POV of the player
// who made the final move (their leave is the rack shown). Returns the full
// length, or -1 on a parse error or a non-PLAY final move.
int scribblez_position_eval_board_json(const char* gcg_text, char* out_json, int out_cap);

// scribblez_position_eval_analyze_gcg with alternate leaves in place of the
// recorded ones, for dashboard what-ifs: `leave_str` for the POV player and,
// unless NULL, `opp_leave_str` for the opponent ('?' is a blank). Each must
// have the tile count of the leave it replaces, and together they may use only
// tiles not on the board; a violation returns -1 with a readable reason.
int scribblez_position_eval_analyze_gcg_leaves(ScribblezSession* s, const char* gcg_text,
                                               const char* leave_str, const char* opp_leave_str,
                                               int opp_leave_input, float* out_input, int input_cap,
                                               char* out_err, int err_cap);

// A .slog file's game count and on-disk size, the arguments
// scribblez_dl_add_file needs. Returns 0, or -1 on an I/O failure or a magic
// or version mismatch.
int scribblez_read_file_header(const char* path, int64_t* out_num_games, int64_t* out_file_size);

// A JSON description of the binary file formats (.slog and its sidecars): each
// struct's field names, offsets, and numpy dtype codes, taken from the
// compiler so they cannot drift, plus magics, versions, flag bits, and
// MoveType values (schema in data/format_layout.h). Python builds its numpy
// dtypes from this rather than mirroring the structs by hand. Static storage;
// do not free. Needs no session.
const char* scribblez_format_layout_json(void);

typedef struct DataLoaderHandle DataLoaderHandle;

// `task` selects the training row the loader decodes (see
// scribblez_task_input_shapes); each game's turns while the bag is non-empty
// become rows.
// Returns NULL for an unknown task, or one whose data (e.g. leave values)
// fails to load.
DataLoaderHandle* scribblez_dl_new(ScribblezSession* s, int64_t memory_budget,
                                   int num_worker_threads, int num_prefetch_threads, int task);

void scribblez_dl_delete(DataLoaderHandle* h);

// Add files oldest first. `num_games` and `file_size` are what
// scribblez_read_file_header reports. Returns 0, or -1 with the reason in
// `out_err` (NUL-terminated, truncated to err_cap) when the task's rows read a
// .forgone sidecar that is missing or does not match the file.
int scribblez_dl_add_file(DataLoaderHandle* h, const char* path, int64_t num_games,
                          int64_t file_size, char* out_err, int err_cap);

int64_t scribblez_dl_num_positions(const DataLoaderHandle* h);

// Start an epoch, shuffling files and the positions within them
// deterministically from `seed`. Returns the number of complete batches; a
// trailing partial batch is also yielded.
//
// `turns_per_game` subsamples turns: 0 uses every eligible turn, and k > 0
// draws k per game, with `epoch_index` choosing which so that successive
// epochs cover different turns (DataLoader::EpochConfig). A leave inference
// loader hides the observer's private history in each row with probability
// `hide_private_fraction`.
int scribblez_dl_epoch_start(DataLoaderHandle* h, int batch_size, int post_move, int apply_symmetry,
                             uint64_t seed, int turns_per_game, int epoch_index,
                             double hide_private_fraction);

// Returns the rows written, 0 once the epoch is exhausted, or -1 if a file
// became unreadable mid-epoch. `output` needs room for batch_size rows.
int scribblez_dl_load_batch(DataLoaderHandle* h, float* output);

#ifdef __cplusplus
}
#endif
