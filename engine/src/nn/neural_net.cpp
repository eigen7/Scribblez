#include "nn/neural_net.h"

#include "encoding/input_encoder.h"
#include "nn/cuda_util.h"
#include "nn/onnx_metadata.h"
#include "training/training_targets.h"

#include <boost/program_options.hpp>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace scribblez {
namespace nn {

namespace {

// Names of the engine's I/O tensors. These match the input_names / output_names
// passed to torch.onnx.export in py/scribblez/position_eval/onnx_export.py.
constexpr const char* kInputSpatial = "input_spatial";
constexpr const char* kInputScalar = "input_scalar";
constexpr const char* kOutputWld = "wld";
constexpr const char* kOutputScoreDiff = "score_diff";
// Mask-head order (kNumMaskHeads entries; see neural_net.h).
constexpr const char* kOutputMasks[kNumMaskHeads] = {
  "opp_next_placement",
  "self_next_placement",
  "opp_win_placement",
  "self_win_placement",
};
static_assert(kOppNextPlacementFloats == kBoardCells && kSelfNextPlacementFloats == kBoardCells &&
                kOppWinPlacementFloats == kBoardCells && kSelfWinPlacementFloats == kBoardCells,
              "every placement mask is one float per board cell");

// Drops anything below a warning, so the build logs stay readable.
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};

nvinfer1::Dims spatial_dims(int rows, int planes) {
  nvinfer1::Dims d;
  d.nbDims = 4;
  d.d[0] = rows;
  d.d[1] = planes;
  d.d[2] = kBoardSide;
  d.d[3] = kBoardSide;
  return d;
}

nvinfer1::Dims scalar_dims(int rows, int floats) {
  nvinfer1::Dims d;
  d.nbDims = 2;
  d.d[0] = rows;
  d.d[1] = floats;
  return d;
}

}  // namespace

void NeuralNetParams::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()  //
    ("model", po::value<std::string>(&onnx_path)->required(),
     "exported ONNX model to build the TensorRT engine from")  //
    ("batch-size", po::value<int>(&max_batch_size)->default_value(max_batch_size),
     "maximum TensorRT batch size")  //
    ("fast-build", po::bool_switch(&fast_build),
     "TensorRT builder optimization level 0 (fast engine build, slower inference); "
     "for tests and smoke runs");
}

struct NeuralNet::Impl {
  explicit Impl(const NeuralNetParams& p) : params(p) {}

  int spatial_floats(int rows) const { return rows * spatial_planes * kBoardCells; }
  int scalar_size(int rows) const { return rows * scalar_floats; }

  // Read off the deserialized engine's declared tensor shapes and the model's
  // own ONNX metadata_props.
  int spatial_planes = 0;
  int scalar_floats = 0;
  bool contingent_features = false;
  bool opp_leave_input = false;
  ~Impl();

  void deserialize_engine(const std::vector<char>& plan);

  // Builds the plan in memory; touches no disk.
  std::vector<char> build_plan(const std::vector<char>& onnx_bytes);

  // For after deserializing a cached plan, which shares the architecture but
  // holds whatever same-architecture checkpoint first populated the cache.
  void refit_engine(const std::vector<char>& onnx_bytes);

  // Context, stream, and host/device buffers, once the engine exists.
  void allocate_buffers();

  NeuralNetParams params;
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  stream_t stream = nullptr;

  // TODO(Refactor): hold the engine's tensors in one list instead of a named
  // pointer pair per tensor. Adding a head (say 'ownership') currently means
  // hand-writing device/host pointers, allocations, and copies in five places;
  // the engine can enumerate its own I/O tensors (names, modes, and shapes), so
  // allocation, binding, and the copies could all be loops over that list.
  void* d_input_spatial = nullptr;
  void* d_input_scalar = nullptr;
  void* d_wld = nullptr;
  void* d_score_diff = nullptr;
  void* d_mask[kNumMaskHeads] = {};

  float* h_input_spatial = nullptr;
  float* h_input_scalar = nullptr;
  float* h_wld = nullptr;
  float* h_score_diff = nullptr;
  // The mask outputs always occupy device buffers (they must stay bound for
  // enqueueV3), but get host buffers -- and copies back -- only under
  // params.copy_masks; agent inference never reads them.
  float* h_mask[kNumMaskHeads] = {};

  int last_rows = -1;
};

NeuralNet::Impl::~Impl() {
  if (stream) {
    if (d_input_spatial) device_free(d_input_spatial);
    if (d_input_scalar) device_free(d_input_scalar);
    if (d_wld) device_free(d_wld);
    if (d_score_diff) device_free(d_score_diff);
    for (void* d : d_mask)
      if (d) device_free(d);
    if (h_input_spatial) host_free(h_input_spatial);
    if (h_input_scalar) host_free(h_input_scalar);
    if (h_wld) host_free(h_wld);
    if (h_score_diff) host_free(h_score_diff);
    for (float* h : h_mask)
      if (h) host_free(h);
    destroy_stream(stream);
  }
}

void NeuralNet::Impl::deserialize_engine(const std::vector<char>& plan) {
  engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine) throw std::runtime_error("Failed to deserialize TensorRT engine");
}

std::vector<char> NeuralNet::Impl::build_plan(const std::vector<char>& onnx_bytes) {
  std::cerr << "[TRT] Building engine from ONNX (one-time; cached per architecture afterward)...\n";

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  // The model path makes external-data references (e.g. an externalized frozen
  // lexicon blob beside the .onnx) resolve against the model's own directory;
  // without it TensorRT resolves them against the process CWD.
  if (!parser->parse(onnx_bytes.data(), onnx_bytes.size(), params.onnx_path.c_str())) {
    std::string msg = "Failed to parse ONNX model";
    if (parser->getNbErrors() > 0) msg += std::string(": ") + parser->getError(0)->desc();
    throw std::runtime_error(msg);
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, params.workspace_bytes);
  if (params.precision == Precision::kFP16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
  // The cache is keyed on model architecture, so a cached plan generally holds
  // a different same-architecture checkpoint's weights; every plan must be
  // refittable so a cache hit can swap in the loaded model's weights.
  config->setFlag(nvinfer1::BuilderFlag::kREFIT);
  // Level 0 takes the first working tactic per layer instead of timing the full
  // tactic set, trading inference speed for a far shorter build.
  if (params.fast_build) config->setBuilderOptimizationLevel(0);

  // The model's own declared input widths drive the profile: the ONNX file
  // says whether it consumes the full or the base input layout.
  const int planes = network->getInput(0)->getDimensions().d[1];
  const int scalars = network->getInput(1)->getDimensions().d[1];
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  int b = params.max_batch_size;
  profile->setDimensions(kInputSpatial, nvinfer1::OptProfileSelector::kMIN,
                         spatial_dims(1, planes));
  profile->setDimensions(kInputSpatial, nvinfer1::OptProfileSelector::kOPT,
                         spatial_dims(b, planes));
  profile->setDimensions(kInputSpatial, nvinfer1::OptProfileSelector::kMAX,
                         spatial_dims(b, planes));
  profile->setDimensions(kInputScalar, nvinfer1::OptProfileSelector::kMIN, scalar_dims(1, scalars));
  profile->setDimensions(kInputScalar, nvinfer1::OptProfileSelector::kOPT, scalar_dims(b, scalars));
  profile->setDimensions(kInputScalar, nvinfer1::OptProfileSelector::kMAX, scalar_dims(b, scalars));
  config->addOptimizationProfile(profile);

  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) throw std::runtime_error("TensorRT engine build failed");
  const char* data = static_cast<const char*>(plan->data());
  return std::vector<char>(data, data + plan->size());
}

void NeuralNet::Impl::refit_engine(const std::vector<char>& onnx_bytes) {
  std::unique_ptr<nvinfer1::IRefitter> refitter(nvinfer1::createInferRefitter(*engine, logger));
  std::unique_ptr<nvonnxparser::IParserRefitter> parser_refitter(
    nvonnxparser::createParserRefitter(*refitter, logger));
  // Model path for external-data resolution, as in build_plan().
  if (!parser_refitter->refitFromBytes(onnx_bytes.data(), onnx_bytes.size(),
                                       params.onnx_path.c_str())) {
    std::string msg = "Failed to read refit weights from ONNX model";
    if (parser_refitter->getNbErrors() > 0)
      msg += std::string(": ") + parser_refitter->getError(0)->desc();
    throw std::runtime_error(msg);
  }
  if (!refitter->refitCudaEngine()) throw std::runtime_error("Failed to refit TensorRT engine");
}

void NeuralNet::Impl::allocate_buffers() {
  context.reset(engine->createExecutionContext());
  if (!context) throw std::runtime_error("Failed to create TensorRT execution context");
  stream = create_stream();

  spatial_planes = engine->getTensorShape(kInputSpatial).d[1];
  scalar_floats = engine->getTensorShape(kInputScalar).d[1];

  int b = params.max_batch_size;
  auto dev = [](int floats) { return device_malloc(sizeof(float) * floats); };
  auto host = [](int floats) { return static_cast<float*>(host_malloc(sizeof(float) * floats)); };

  d_input_spatial = dev(spatial_floats(b));
  d_input_scalar = dev(scalar_size(b));
  d_wld = dev(b * kWldFloats);
  d_score_diff = dev(b * kScoreDiffOutputFloats);
  for (int h = 0; h < kNumMaskHeads; ++h) d_mask[h] = dev(b * kBoardCells);

  h_input_spatial = host(spatial_floats(b));
  h_input_scalar = host(scalar_size(b));
  h_wld = host(b * kWldFloats);
  h_score_diff = host(b * kScoreDiffOutputFloats);
  if (params.copy_masks) {
    for (int h = 0; h < kNumMaskHeads; ++h) h_mask[h] = host(b * kBoardCells);
  }

  context->setTensorAddress(kInputSpatial, d_input_spatial);
  context->setTensorAddress(kInputScalar, d_input_scalar);
  context->setTensorAddress(kOutputWld, d_wld);
  context->setTensorAddress(kOutputScoreDiff, d_score_diff);
  for (int h = 0; h < kNumMaskHeads; ++h) context->setTensorAddress(kOutputMasks[h], d_mask[h]);
}

// ---------------------------------------------------------------------------

NeuralNet::NeuralNet(const NeuralNetParams& params) : impl_(std::make_unique<Impl>(params)) {
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
}

NeuralNet::~NeuralNet() = default;

void NeuralNet::load() {
  set_device(impl_->params.cuda_device_id);

  std::vector<char> onnx_bytes = read_file_bytes(impl_->params.onnx_path);
  OnnxMetadata meta = parse_onnx_metadata(onnx_bytes);
  // A model that names its graph must name this one. An export predating the
  // entry names nothing and is accepted: the position graph is the only one
  // that existed then, and this runtime's tensor names would reject anything
  // else at bind time anyway.
  if (!meta.graph.empty() && meta.graph != kGraphPositionEval) {
    throw std::runtime_error("NeuralNet serves the " + std::string(kGraphPositionEval) +
                             " graph, but " + impl_->params.onnx_path + " declares graph '" +
                             meta.graph + "'");
  }
  impl_->contingent_features = meta.contingent_features;
  impl_->opp_leave_input = meta.opp_leave_input;
  std::string cache_path =
    engine_plan_cache_path(meta.architecture_signature, impl_->params.precision,
                           "batch_" + std::to_string(impl_->params.max_batch_size),
                           impl_->params.fast_build, impl_->params.mount_root);

  // The cache is keyed on the model's architecture signature, so a hit yields
  // a plan with the right structure but (in general) another checkpoint's
  // weights; refitting swaps in this model's weights, which is far cheaper
  // than an engine build.
  if (std::filesystem::exists(cache_path)) {
    impl_->deserialize_engine(read_file_bytes(cache_path));
    impl_->refit_engine(onnx_bytes);
  } else {
    std::vector<char> plan = impl_->build_plan(onnx_bytes);
    write_file_bytes(cache_path, plan.data(), plan.size());
    impl_->deserialize_engine(plan);
  }
  impl_->allocate_buffers();
}

int NeuralNet::max_batch_size() const { return impl_->params.max_batch_size; }
int NeuralNet::spatial_planes() const { return impl_->spatial_planes; }
int NeuralNet::scalar_floats() const { return impl_->scalar_floats; }
bool NeuralNet::contingent_features() const { return impl_->contingent_features; }
bool NeuralNet::opp_leave_input() const { return impl_->opp_leave_input; }

float* NeuralNet::input_spatial_host() { return impl_->h_input_spatial; }
float* NeuralNet::input_scalar_host() { return impl_->h_input_scalar; }
const float* NeuralNet::wld_host() const { return impl_->h_wld; }
const float* NeuralNet::score_diff_host() const { return impl_->h_score_diff; }
const float* NeuralNet::mask_host(int head) const { return impl_->h_mask[head]; }

void NeuralNet::predict(int num_rows) {
  Impl& m = *impl_;
  if (num_rows < 1 || num_rows > m.params.max_batch_size) {
    throw std::runtime_error("NeuralNet::predict: num_rows out of range");
  }

  if (num_rows != m.last_rows) {
    m.context->setInputShape(kInputSpatial, spatial_dims(num_rows, m.spatial_planes));
    m.context->setInputShape(kInputScalar, scalar_dims(num_rows, m.scalar_floats));
    m.last_rows = num_rows;
  }

  host_to_device_async(m.stream, m.d_input_spatial, m.h_input_spatial,
                       sizeof(float) * m.spatial_floats(num_rows));
  host_to_device_async(m.stream, m.d_input_scalar, m.h_input_scalar,
                       sizeof(float) * m.scalar_size(num_rows));

  if (!m.context->enqueueV3(m.stream)) throw std::runtime_error("TensorRT inference failed");

  device_to_host_async(m.stream, m.h_wld, m.d_wld, sizeof(float) * num_rows * kWldFloats);
  device_to_host_async(m.stream, m.h_score_diff, m.d_score_diff,
                       sizeof(float) * num_rows * kScoreDiffOutputFloats);
  if (m.params.copy_masks) {
    for (int h = 0; h < kNumMaskHeads; ++h) {
      device_to_host_async(m.stream, m.h_mask[h], m.d_mask[h],
                           sizeof(float) * num_rows * kBoardCells);
    }
  }

  synchronize_stream(m.stream);
}

}  // namespace nn
}  // namespace scribblez
