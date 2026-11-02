// The cross-check delta kernel: one warp per candidate. The warp lays the
// candidate's tiles over the position's board in shared memory, recomputes
// the cross-checks of the squares Board::update_caches_after_place would
// (cross_check_device.cuh, the rollout kernel's own), and keeps those whose
// mask differs from the position's, in cross_check_abi.h's order.

#include "cross_check_delta_kernel.h"
#include "cross_check_device.cuh"
#include "cuda_buffers.cuh"

#include <cstring>
#include <cuda_runtime.h>

namespace scribblez::cross_check_gpu {

using cross_check_abi::MoveDeltas;
using rollout_abi::kBoardSize;
using rollout_abi::kRackSize;
using rollout_abi::kSquares;
using rollout_abi::PackedMove;

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRunEnds = 4 * kRackSize;

// A warp's post-move board, natural frame.
struct WarpSquares {
  const uint8_t* sq;
  __device__ uint8_t operator()(int view, int r, int c) const {
    return sq[view ? c * kBoardSize + r : r * kBoardSize + c];
  }
};

// The deltas compare masks only, so the cross-check scores go unused.
struct NoScore {
  __device__ int operator()(uint8_t) const { return 0; }
};

// Board::apply's placed squares of play `m`, in lane order (the order of its
// glyphs). Returns how many.
__device__ int placed_squares(const PackedMove& m, uint8_t* rows, uint8_t* cols) {
  int n = 0;
  for (int along = 0; along < kBoardSize; ++along) {
    if (((m.square_mask >> along) & 1u) == 0) continue;
    rows[n] = uint8_t(m.horizontal ? m.start : along);
    cols[n] = uint8_t(m.horizontal ? along : m.start);
    ++n;
  }
  return n;
}

// The board-frame square of view square (r, c).
__device__ inline uint16_t board_square(int view, int r, int c) {
  return uint16_t(view ? c * kBoardSize + r : r * kBoardSize + c);
}

// Whether `key` is among seen[0, n), adding it if not. The ends of the play's
// own word come up once per placed tile.
__device__ bool first_visit(uint16_t* seen, int& n, uint16_t key) {
  for (int i = 0; i < n; ++i)
    if (seen[i] == key) return false;
  seen[n++] = key;
  return true;
}

// Every lane of the warp runs the loop, which the cross-check ballot needs,
// and lane 0 writes the result.
__global__ void __launch_bounds__(kThreads)
  delta_kernel(const PositionAbi* pos, const PackedMove* moves, int n, gpu::DeviceDawg dawg,
               MoveDeltas* out) {
  __shared__ uint8_t boards[kWarps][kSquares];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int i = blockIdx.x * kWarps + warp;
  if (i >= n) return;
  const PackedMove m = moves[i];
  uint8_t* sq = boards[warp];
  for (int k = lane; k < kSquares; k += 32) sq[k] = pos->squares[k];
  uint8_t rows[kRackSize], cols[kRackSize];
  const int placed = m.type == 0 ? placed_squares(m, rows, cols) : 0;
  __syncwarp();
  if (lane == 0) {
    for (int k = 0; k < placed; ++k) sq[rows[k] * kBoardSize + cols[k]] = m.glyphs[k];
  }
  __syncwarp();

  const WarpSquares board{sq};
  MoveDeltas d;
  uint16_t seen[kMaxRunEnds];
  int nseen = 0;
  for (int t = 0; t < 4 * placed; ++t) {
    const gpu::RunEndTask task = gpu::run_end_task(t, placed);
    const int vr = task.view ? cols[task.placed] : rows[task.placed];
    const int vc = task.view ? rows[task.placed] : cols[task.placed];
    const int r = gpu::run_end_row(board, task.view, vr, vc, task.end);
    if (r < 0) continue;
    const int idx = r * kBoardSize + vc;
    if (!first_visit(seen, nseen, uint16_t(task.view * kSquares + idx))) continue;
    const uint32_t mask = gpu::cross_check_at(board, NoScore{}, dawg, task.view, r, vc, lane).mask;
    const uint32_t old = pos->cross_mask[task.view][idx];
    if (mask != old) {
      cross_check_abi::insert_sorted(
        d, {old, mask, board_square(task.view, r, vc), uint8_t(task.view), 0});
    }
  }
  if (lane == 0) out[i] = d;
}

}  // namespace

struct DeviceState {
  gpu::DeviceBuffer<uint32_t> arcs;
  uint32_t root = 0;
  cudaStream_t stream = nullptr;
  gpu::PinnedBuffer<PositionAbi> host_position;
  gpu::PinnedBuffer<PackedMove> host_moves;
  gpu::PinnedBuffer<MoveDeltas> host_out;
  gpu::DeviceBuffer<PositionAbi> position;
  gpu::DeviceBuffer<PackedMove> moves;
  gpu::DeviceBuffer<MoveDeltas> out;
};

DeviceState* create_state(const uint32_t* arcs, size_t num_arcs, uint32_t root) {
  auto* s = new DeviceState;
  s->arcs.upload(arcs, num_arcs);
  s->root = root;
  // Non-blocking: TensorRT's streams on other threads never wait on it.
  gpu::cuda_check(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking), "stream");
  s->host_position.reserve(1);
  s->position.reserve(1);
  return s;
}

void free_state(DeviceState* s) {
  if (!s) return;
  cudaStreamDestroy(s->stream);
  delete s;
}

void run(DeviceState* s, const PositionAbi& position, const PackedMove* moves, int n,
         MoveDeltas* out) {
  if (n <= 0) return;
  const size_t count = size_t(n);
  s->host_moves.reserve(count);
  s->host_out.reserve(count);
  s->moves.reserve(count);
  s->out.reserve(count);
  *s->host_position.data = position;
  std::memcpy(s->host_moves.data, moves, sizeof(PackedMove) * count);
  gpu::cuda_check(cudaMemcpyAsync(s->position.data, s->host_position.data, sizeof(PositionAbi),
                                  cudaMemcpyHostToDevice, s->stream),
                  "position upload");
  gpu::cuda_check(cudaMemcpyAsync(s->moves.data, s->host_moves.data, sizeof(PackedMove) * count,
                                  cudaMemcpyHostToDevice, s->stream),
                  "moves upload");
  const unsigned blocks = unsigned((n + kWarps - 1) / kWarps);
  delta_kernel<<<blocks, kThreads, 0, s->stream>>>(s->position.data, s->moves.data, n,
                                                   {s->arcs.data, s->root}, s->out.data);
  gpu::cuda_check(cudaGetLastError(), "launch");
  gpu::cuda_check(cudaMemcpyAsync(s->host_out.data, s->out.data, sizeof(MoveDeltas) * count,
                                  cudaMemcpyDeviceToHost, s->stream),
                  "deltas download");
  gpu::cuda_check(cudaStreamSynchronize(s->stream), "kernel");
  std::memcpy(out, s->host_out.data, sizeof(MoveDeltas) * count);
}

}  // namespace scribblez::cross_check_gpu
