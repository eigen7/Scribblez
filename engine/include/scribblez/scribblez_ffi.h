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
//   target_index=1  "score_diff"         (801,)
//   target_index=2  "opp_next_placement" (15, 15)
const ScribblezShape* scribblez_target_shapes(void);

// Total floats per output row -- sum of input + all target sizes. Useful
// to size the output buffer without iterating the shape arrays in Python.
int scribblez_row_size_floats(void);

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
int scribblez_dl_epoch_start(DataLoaderHandle* h, int batch_size, int post_move, int apply_symmetry,
                             uint64_t seed);

// Fill `output` with the next batch of the current epoch. Returns the
// number of rows written (0 = epoch exhausted). `output` must have capacity
// for at least batch_size * scribblez_row_size_floats() floats.
int scribblez_dl_load_batch(DataLoaderHandle* h, float* output);

// Query current resident memory in bytes (useful for testing eviction).
int64_t scribblez_dl_resident_bytes(const DataLoaderHandle* h);

#ifdef __cplusplus
}
#endif
