#pragma once

#include "nn/eval_service.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

// A deterministic, CPU-only stand-in for a served model, for tests. It exists
// so the Python suite can drive the offline generators end to end (formats,
// sidecars, readers, training plumbing) without a GPU: a generator given
// `--model=synthetic:<variant>` serves this instead of building a TensorRT
// engine. It lives in production binaries only because those binaries are what
// the tests run. Its readouts mean nothing about Scrabble; never point a real
// run at it.
//
// The readouts are a fixed function of each row: a few hashed-weight linear
// projections of the inputs, squashed into valid head values. So different
// positions and candidates get different, finite, in-range readouts, the same
// input always gets the same readout, and nothing is ever loaded.

namespace scribblez {
namespace nn {

// What a synthetic model declares, parsed from its `synthetic:<variant>` name:
//   - hidden: the hidden-leaves input arm (opp_leave_input false)
//   - open-leaves: the open-leaves input arm (opp_leave_input true)
//   - non-finite: the hidden arm, every readout NaN, for exercising the
//     generators' non-finite-readout guards
struct SyntheticModel {
  bool opp_leave_input = false;
  bool non_finite = false;
};

// The synthetic model `model` names, or nullopt for anything else (a real
// model path). Throws util::CleanException for a `synthetic:` name with an
// unknown variant.
std::optional<SyntheticModel> parse_synthetic_model(const std::string& model);

// The content hash a generator stamps into its outputs for `model`: the file's
// bytes for a real model, the name itself for a synthetic one.
std::string model_content_hash(const std::string& model);

template <typename Spec>
class SyntheticEvalService : public EvalService<Spec> {
 public:
  using SpecBatch = Spec::Batch;

  explicit SyntheticEvalService(const SyntheticModel& model);

  bool opp_leave_input() const override { return model_.opp_leave_input; }
  int spatial_planes() const override;
  int scalar_floats() const override;

 protected:
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override;

 private:
  SyntheticModel model_;
  // The hashed weights of each projection, back to back, over one row's
  // inputs: the board row, then (move-set family) one candidate's features.
  std::vector<float> weights_;
};

extern template class SyntheticEvalService<PositionEvaluationSpec>;
extern template class SyntheticEvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez
