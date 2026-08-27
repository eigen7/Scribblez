#include "nn/fp32_pinning.h"

#include "util/exception.h"

#include <deque>
#include <set>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace scribblez {
namespace nn {

namespace {

// The runaway backstop: a healthy region reaches a renormalizing layer well
// within this. Tripping it means the substrings no longer match the graph.
constexpr int kMaxPinnedLayers = 64;

// Pin `layer` and, unless it is terminal (a renormalizing layer whose output is
// back in range), force FP32 output storage and push its outputs onto the
// frontier. Idempotent: a layer reached twice is pinned once.
void pin_layer(Fp32PinGraph& g, int layer, bool terminal, std::set<int>* pinned,
               std::deque<int>* frontier) {
  if (!pinned->insert(layer).second) return;
  g.pin_layer_fp32(layer);
  if (terminal) return;
  for (int out = 0; out < g.num_outputs(layer); ++out) {
    g.pin_output_fp32(layer, out);
    frontier->push_back(g.output_tensor(layer, out));
  }
}

}  // namespace

int pin_fp32_region(Fp32PinGraph& g, std::span<const char* const> substrings) {
  // Tensor id -> the layers consuming it, so the walk can step forward.
  std::unordered_map<int, std::vector<int>> consumers;
  for (int layer = 0; layer < g.num_layers(); ++layer)
    for (int in = 0; in < g.num_inputs(layer); ++in)
      consumers[g.input_tensor(layer, in)].push_back(layer);

  std::set<int> pinned;
  std::deque<int> frontier;

  for (const char* sub : substrings) {
    bool matched = false;
    for (int layer = 0; layer < g.num_layers(); ++layer) {
      if (g.layer_kind(layer) == PinLayerKind::kConstant) continue;
      if (g.layer_name(layer).find(sub) != std::string_view::npos) {
        pin_layer(g, layer, /*terminal=*/false, &pinned, &frontier);
        matched = true;
      }
    }
    // A substring that matches nothing means the exported architecture renamed
    // the region the spec believes is overflow-prone: pinning would silently
    // vanish and the NaNs would return. Fail loudly instead.
    if (!matched) {
      throw util::Exception(
        "FP32 pinning: no layer name contains \"{}\"; the exported architecture and the spec's "
        "overflow-prone-layer list (model_specs.h) have drifted apart",
        sub);
    }
  }

  while (!frontier.empty()) {
    const int tensor = frontier.front();
    frontier.pop_front();
    const auto it = consumers.find(tensor);
    if (it != consumers.end()) {
      for (const int c : it->second) {
        if (g.layer_kind(c) == PinLayerKind::kConstant) continue;
        pin_layer(g, c, g.layer_kind(c) == PinLayerKind::kRenormalizing, &pinned, &frontier);
      }
    }
    if (int(pinned.size()) > kMaxPinnedLayers) {
      throw util::Exception(
        "FP32 pinning walked {} layers without renormalizing; the spec's substrings no longer "
        "match the exported architecture",
        pinned.size());
    }
  }
  return int(pinned.size());
}

}  // namespace nn
}  // namespace scribblez
