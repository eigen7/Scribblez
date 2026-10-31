#pragma once

// The boundary between the engine (C++26, g++) and the cross-check delta
// kernel's CUDA translation unit (nvcc): flat arrays in, entries out. Only
// the ABI headers cross it.

#include "gpu/cross_check_abi.h"
#include "gpu/rollout_abi.h"

#include <cstddef>
#include <cstdint>

namespace scribblez::cross_check_gpu {

// The position every candidate of a call starts from, in the board's frame:
// its glyph codes, and its cross-check masks per view as Board keeps them
// (view 1 transposed, in view coordinates).
struct PositionAbi {
  uint8_t squares[rollout_abi::kSquares] = {};
  uint32_t cross_mask[2][rollout_abi::kSquares] = {};
};

struct DeviceState;

// The KWG `arcs` (Dictionary::arcs) on the device, with a stream and the
// buffers calls reuse. Throws std::runtime_error on a CUDA failure, as run
// does.
DeviceState* create_state(const uint32_t* arcs, size_t num_arcs, uint32_t root);
void free_state(DeviceState* state);

// out[i], the entries moves[i] changes at `position`, for i < n. Not
// thread-safe: one call at a time per state.
void run(DeviceState* state, const PositionAbi& position, const rollout_abi::PackedMove* moves,
         int n, cross_check_abi::MoveDeltas* out);

}  // namespace scribblez::cross_check_gpu
