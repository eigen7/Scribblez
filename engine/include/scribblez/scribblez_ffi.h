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

// Decode rows [start, stop) into `output`, sequentially in chronological
// order. `output` must have capacity for at least (stop - start) *
// scribblez_row_size_floats() floats. `post_move != 0` selects the
// post-move snapshot at each game's sampled turn (default 0 = pre-move).
// `apply_symmetry != 0` enables per-row random diagonal flips.
void scribblez_dl_load(DataLoaderHandle* h, int64_t start, int64_t stop, int post_move,
                       int apply_symmetry, float* output);

// V2 epoch-based streaming API. Shuffles files and positions within files
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
