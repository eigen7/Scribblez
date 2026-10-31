#include "gpu/cross_check_deltas_gpu.h"

#include "cross_check_delta_kernel.h"
#include "lexicon/dictionary.h"
#include "nn/shared_registry.h"
#include "sim/hasty_rollout_task.h"
#include "util/exception.h"

#include <vector>

namespace scribblez {

std::shared_ptr<GpuCrossCheckDeltas> GpuCrossCheckDeltas::shared(const Dictionary& dict) {
  static nn::SharedRegistry<const Dictionary*, GpuCrossCheckDeltas> registry;
  return registry.get_or_create(&dict, [&] { return std::make_shared<GpuCrossCheckDeltas>(dict); });
}

GpuCrossCheckDeltas::GpuCrossCheckDeltas(const Dictionary& dict)
    : dict_(dict),
      state_(cross_check_gpu::create_state(dict.arcs().data(), dict.arcs().size(), dict.root())) {}

GpuCrossCheckDeltas::~GpuCrossCheckDeltas() { cross_check_gpu::free_state(state_); }

void GpuCrossCheckDeltas::deltas(const Board& board, const Dictionary& dict, const Move* moves,
                                 int64_t n, cross_check_abi::MoveDeltas* out) {
  if (&dict != &dict_)
    throw util::Exception("GpuCrossCheckDeltas: called with a dictionary it did not upload");
  cross_check_gpu::PositionAbi position;
  board_abi(board, dict, position.squares, position.cross_mask);
  std::vector<rollout_abi::PackedMove> packed(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) packed[size_t(i)] = pack_move(moves[i]);
  std::lock_guard<std::mutex> lock(mutex_);
  cross_check_gpu::run(state_, position, packed.data(), int(n), out);
}

}  // namespace scribblez
