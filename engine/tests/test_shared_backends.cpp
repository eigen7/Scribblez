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

RolloutBackend cpu_backend(int leaf_rows = RolloutBackend{}.leaf_rows) {
  RolloutBackend b;
  b.backend = "cpu";
  b.leaf_model = RolloutBackend::kSyntheticLeaf;
  b.leaf_rows = leaf_rows;
  return b;
}

class SharedRolloutBackend : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!std::ifstream(SCRIBBLEZ_DEFAULT_KWG).good()) GTEST_SKIP() << "no lexicon";
    dict_ = std::make_unique<Dictionary>(Dictionary::load_kwg(SCRIBBLEZ_DEFAULT_KWG));
  }

  std::unique_ptr<Dictionary> dict_;
};

}  // namespace

TEST_F(SharedRolloutBackend, IsSharedForEqualSettings) {
  const std::shared_ptr<RolloutValues> a = cpu_backend().shared(*dict_);
  EXPECT_EQ(a, cpu_backend().shared(*dict_));
  EXPECT_NE(a, cpu_backend(cpu_backend().leaf_rows + 1).shared(*dict_));
}

TEST_F(SharedRolloutBackend, IsRebuiltOnceEveryHolderDropsIt) {
  std::shared_ptr<RolloutValues> first = cpu_backend().shared(*dict_);
  const std::weak_ptr<RolloutValues> weak = first;
  first.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_NE(cpu_backend().shared(*dict_), nullptr);
}
