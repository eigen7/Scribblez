#pragma once

#include "nn/trt_util.h"

#include <cstdint>
#include <memory>
#include <string>

// A thin, synchronous wrapper around a TensorRT engine specialized to the
// Scribblez move set evaluation model: one position's board inputs
// ("input_spatial", "input_scalar", both a single row) plus five per-move
// feature tensors describing M candidates, scored in one pass into "wld" and
// "score_diff" rows. input_encoder.h owns the board layout and
// move_set_encoder.h the move-feature layout;
// py/scribblez/move_set_eval/onnx_export.py exports the graph these tensor
// names and dtypes come from.
//
// M is the only dynamic axis. Amortizing one board encode across a whole
// candidate set is the architecture's reason to exist, so a call is a position
// and its candidates, never a batch of positions -- the exported graph is the
// P=1 specialization.
//
// One MoveSetNet drives one engine from one thread: predict() blocks until the
// outputs are back, with no cross-thread batching and no async pipeline.

namespace scribblez {
namespace nn {

struct MoveSetNetParams {
  std::string onnx_path;  // exported ONNX model load() builds from
  int cuda_device_id = 0;

  // Upper bound on the candidates one predict() call scores, and the profile's
  // maximum, so it keys the engine-plan cache. MoveSetEvalService splits a
  // larger candidate set into chunks of this size.
  //
  // Size it to the realistic move-set ceiling rather than trimming it: every
  // chunk past the first re-pays a full synchronous round trip and another
  // board trunk pass -- the per-position work this architecture exists to pay
  // once -- while buying back almost no memory. Measured on the parity fixture
  // at M=4000: 0.37 ms in one chunk, 0.94 ms at max_moves=1024, 1.98 ms at 512,
  // against device memory of 46 MiB at 4096 and 42 MiB at 256.
  int max_moves = 4096;

  Precision precision = Precision::kFP16;
  uint64_t workspace_bytes = uint64_t{1} << 30;  // 1 GiB TensorRT scratch
  std::string mount_root = "/workspace/mount";   // root of the engine-plan cache

  // Builder optimization level 0, as in NeuralNetParams: a far shorter build
  // for slower inference, and cached separately from full-optimization plans.
  bool fast_build = false;
};

class MoveSetNet {
 public:
  explicit MoveSetNet(const MoveSetNetParams& params);
  ~MoveSetNet();

  MoveSetNet(const MoveSetNet&) = delete;
  MoveSetNet& operator=(const MoveSetNet&) = delete;

  // Build the engine from params.onnx_path, or deserialize the cached plan
  // built from this very file -- the mset cache is keyed on model content, so
  // a hit needs no refit (see the .cpp for why this graph cannot use the
  // position runtime's architecture-keyed, refitted cache). Exactly once,
  // before predict().
  //
  // Throws unless the model declares the move-set graph and the very
  // move-feature encoding version this build's encoder emits. Nothing
  // structural would catch that skew: an encoder change that alters what a
  // Move encodes to leaves every tensor shape intact, so a stale checkpoint
  // would silently be fed rows from a distribution it never trained on.
  void load();

  int max_moves() const;

  // Valid after load(): the board-row widths the served model consumes.
  int spatial_planes() const;
  int scalar_floats() const;

  // The model's input-encoding arm, from the ONNX metadata_props the exporter
  // stamps. Valid after load(); consumers cross-check it against the input
  // widths through input_encoder.h's registry.
  bool contingent_features() const;
  bool opp_leave_input() const;

  // Staging buffers for one call: the board row (written once per position) and
  // max_moves() rows of each move tensor, of which the caller fills the first
  // num_moves. The dtypes are move_set_encoder.h's own, so an encoded candidate
  // set copies in without conversion.
  float* input_spatial_host();  // spatial_planes() * 225
  float* input_scalar_host();   // scalar_floats()
  int32_t* move_letters_host();
  uint8_t* move_blanks_host();
  int32_t* move_squares_host();
  uint8_t* move_tile_mask_host();
  float* move_scalars_host();

  // Blocks until the outputs are back. Requires 1 <= num_moves <= max_moves().
  void predict(int num_moves);

  // Valid after predict(), for its first num_moves rows. wld is raw logits.
  const float* wld_host() const;         // num_moves x kWldFloats
  const float* score_diff_host() const;  // num_moves x [mean, std], std positive

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nn
}  // namespace scribblez
