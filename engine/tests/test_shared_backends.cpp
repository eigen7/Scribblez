// The heavy-resource factories hand out one instance per distinct params while
// anyone holds it (nn::SharedRegistry), on the synthetic stand-ins the suites
// use in place of TensorRT and CUDA.

#include "gpu/rollout_backend.h"
#include "lexicon/dictionary.h"
#include "nn/eval_service.h"
#include "nn/neural_net.h"

#include <gtest/gtest.h>

#include <fstream>
#include <memory>

using namespace scribblez;

namespace {

RolloutBackend cpu_backend() {
  RolloutBackend b;
  b.backend = "cpu";
  b.leaf_model = RolloutBackend::kSyntheticLeaf;
  return b;
}

nn::NeuralNetParams<nn::MoveSetEvaluationSpec> synthetic_move_set(int cuda_device_id = 0) {
  nn::NeuralNetParams<nn::MoveSetEvaluationSpec> p;
  p.onnx_path = "synthetic:hidden";
  p.cuda_device_id = cuda_device_id;
  return p;
}

}  // namespace

TEST(SharedBackends, RolloutBackendIsSharedForEqualSettings) {
  if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) GTEST_SKIP() << "no lexicon";
  const Dictionary dict = Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG);
  const std::shared_ptr<RolloutValues> a = cpu_backend().shared(dict, 2);
  EXPECT_EQ(a, cpu_backend().shared(dict, 2));
  EXPECT_NE(a, cpu_backend().shared(dict, 3));
}

TEST(SharedBackends, RolloutBackendIsRebuiltOnceEveryHolderDropsIt) {
  if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) GTEST_SKIP() << "no lexicon";
  const Dictionary dict = Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG);
  std::shared_ptr<RolloutValues> first = cpu_backend().shared(dict, 2);
  const std::weak_ptr<RolloutValues> weak = first;
  first.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_NE(cpu_backend().shared(dict, 2), nullptr);
}

TEST(SharedBackends, MoveSetServiceIsSharedForEqualParams) {
  const std::shared_ptr<nn::MoveSetEvalService> a =
    nn::MoveSetEvalService::create(synthetic_move_set());
  EXPECT_EQ(a, nn::MoveSetEvalService::create(synthetic_move_set()));
  EXPECT_NE(a, nn::MoveSetEvalService::create(synthetic_move_set(/*cuda_device_id=*/1)));
}
