#include "nn/move_set_net.h"

#include "nn/cuda_util.h"
#include "nn/onnx_metadata.h"
#include "training/move_set_encoder.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scribblez {
namespace nn {

namespace {

// Names of the engine's I/O tensors, matching the input_names / output_names
// passed to torch.onnx.export in py/scribblez/move_set_eval/onnx_export.py.
// Only the board inputs are named here: everything else is reached through the
// binding table, and these two because the served model's board-row widths are
// read off them.
constexpr const char* kInputSpatial = "input_spatial";
constexpr const char* kInputScalar = "input_scalar";
constexpr const char* kMoveLetters = "move_letters";
constexpr const char* kMoveBlanks = "move_blanks";
constexpr const char* kMoveSquares = "move_squares";
constexpr const char* kMoveTileMask = "move_tile_mask";
constexpr const char* kMoveScalars = "move_scalars";
constexpr const char* kOutputWld = "wld";
constexpr const char* kOutputScoreDiff = "score_diff";

// The candidate count the plan is tuned for. Bounds are 1 and params.max_moves;
// this is the middle of the profile, near a typical full move set.
constexpr int kOptMoves = 512;

// This file duplicates NeuralNet's logger, refit, and load-or-build flow rather
// than sharing them. That is deliberate and temporary: the shared TensorRT
// graph abstraction neural_net.cpp's own TODO asks for is worth extracting from
// two runtimes that have both been proven against real engines, not designed
// against one. See docs/roadmap.md A4 (PR 4b).
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};

size_t element_size(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
      return 1;
    default:
      throw std::runtime_error("MoveSetNet: unsupported engine I/O data type");
  }
}

// Elements in one row of a tensor, i.e. the product of every dimension after
// the leading one.
int row_elements(const nvinfer1::Dims& dims) {
  int n = 1;
  for (int i = 1; i < dims.nbDims; ++i) n *= static_cast<int>(dims.d[i]);
  return n;
}

}  // namespace

// One of the engine's I/O tensors, with the host and device buffers bound to
// it. Seven inputs across three dtypes, and a per-call row count on all but the
// board pair, leave no room for the pointer-per-tensor style: allocation,
// binding, shape updates, and the copies are all loops over these.
struct Binding {
  std::string name;
  size_t elem_size = 0;
  int elems_per_row = 0;
  bool per_move = false;  // rows = the call's move count; the board inputs are one row
  bool input = false;
  void* device = nullptr;
  void* host = nullptr;

  size_t bytes(int num_moves) const {
    return elem_size * static_cast<size_t>(elems_per_row) * (per_move ? num_moves : 1);
  }
};

struct MoveSetNet::Impl {
  explicit Impl(const MoveSetNetParams& p) : params(p) {}
  ~Impl();

  void deserialize_engine(const std::vector<char>& plan);

  // Builds the plan in memory; touches no disk.
  std::vector<char> build_plan(const std::vector<char>& onnx_bytes);

  // For after deserializing a cached plan, which shares the architecture but
  // holds whatever same-architecture checkpoint first populated the cache.
  void refit_engine(const std::vector<char>& onnx_bytes);

  // Context, stream, and the binding table, once the engine exists.
  void allocate_buffers();

  // The binding for `name`, which the engine is guaranteed to expose (a model
  // missing a tensor this runtime feeds fails here rather than at inference).
  // A linear scan over nine entries, run a handful of times per position.
  Binding& binding(const char* name);

  MoveSetNetParams params;
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  stream_t stream = nullptr;
  std::vector<Binding> bindings;

  // Read off the deserialized engine's declared board input shapes and the
  // model's own ONNX metadata_props.
  int spatial_planes = 0;
  int scalar_floats = 0;
  bool contingent_features = false;
  bool opp_leave_input = false;

  int last_moves = -1;
};

MoveSetNet::Impl::~Impl() {
  if (!stream) return;
  for (Binding& b : bindings) {
    if (b.device) device_free(b.device);
    if (b.host) host_free(b.host);
  }
  destroy_stream(stream);
}

Binding& MoveSetNet::Impl::binding(const char* name) {
  for (Binding& b : bindings) {
    if (b.name == name) return b;
  }
  throw std::runtime_error("MoveSetNet: model has no tensor named '" + std::string(name) + "'");
}

void MoveSetNet::Impl::deserialize_engine(const std::vector<char>& plan) {
  engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine) throw std::runtime_error("Failed to deserialize TensorRT engine");
}

std::vector<char> MoveSetNet::Impl::build_plan(const std::vector<char>& onnx_bytes) {
  std::cerr << "[TRT] Building move set engine from ONNX (one-time; cached per architecture "
               "afterward)...\n";

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  // The model path makes external-data references resolve against the model's
  // own directory, as in NeuralNet::Impl::build_plan.
  if (!parser->parse(onnx_bytes.data(), onnx_bytes.size(), params.onnx_path.c_str())) {
    std::string msg = "Failed to parse ONNX model";
    if (parser->getNbErrors() > 0) msg += std::string(": ") + parser->getError(0)->desc();
    throw std::runtime_error(msg);
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, params.workspace_bytes);
  if (params.precision == Precision::kFP16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
  // Every plan must be refittable so a cache hit can swap in the loaded
  // checkpoint's weights (see load()).
  config->setFlag(nvinfer1::BuilderFlag::kREFIT);
  if (params.fast_build) config->setBuilderOptimizationLevel(0);

  // Only the move tensors carry the dynamic candidate axis; the board inputs
  // are the P=1 graph's single row, fully static, and need no profile entry.
  // Reading which is which off the parsed network keeps the graph itself the
  // authority on its own shapes.
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  for (int i = 0; i < network->getNbInputs(); ++i) {
    nvinfer1::ITensor* tensor = network->getInput(i);
    nvinfer1::Dims dims = tensor->getDimensions();
    if (dims.d[0] != -1) continue;
    auto with_moves = [&dims](int moves) {
      nvinfer1::Dims d = dims;
      d.d[0] = moves;
      return d;
    };
    const char* name = tensor->getName();
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, with_moves(1));
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT,
                           with_moves(std::min(kOptMoves, params.max_moves)));
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, with_moves(params.max_moves));
  }
  config->addOptimizationProfile(profile);

  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) throw std::runtime_error("TensorRT engine build failed");
  const char* data = static_cast<const char*>(plan->data());
  return std::vector<char>(data, data + plan->size());
}

void MoveSetNet::Impl::refit_engine(const std::vector<char>& onnx_bytes) {
  std::unique_ptr<nvinfer1::IRefitter> refitter(nvinfer1::createInferRefitter(*engine, logger));
  std::unique_ptr<nvonnxparser::IParserRefitter> parser_refitter(
    nvonnxparser::createParserRefitter(*refitter, logger));
  // Model path for external-data resolution, as in build_plan().
  //
  // The boolean this returns is deliberately not the verdict. TensorRT 10.11
  // counts one more ONNX-side weight than the engine exposes a slot for -- an
  // anonymous fusion product -- and fails its own strict count on a refit that
  // left nothing unmapped. getMissingWeights() is the question that matters and
  // the one the A4 refit gate (py/scripts/move_set_eval/trt_refit_probe.py)
  // validated against ground-truth outputs.
  parser_refitter->refitFromBytes(onnx_bytes.data(), onnx_bytes.size(), params.onnx_path.c_str());
  const int missing = refitter->getMissingWeights(0, nullptr);
  if (missing != 0) {
    std::string msg = "TensorRT refit left " + std::to_string(missing) + " weights unmapped";
    if (parser_refitter->getNbErrors() > 0)
      msg += std::string(": ") + parser_refitter->getError(0)->desc();
    throw std::runtime_error(msg);
  }
  if (!refitter->refitCudaEngine()) throw std::runtime_error("Failed to refit TensorRT engine");
}

void MoveSetNet::Impl::allocate_buffers() {
  context.reset(engine->createExecutionContext());
  if (!context) throw std::runtime_error("Failed to create TensorRT execution context");
  stream = create_stream();

  // Enumerate the engine's own I/O tensors: their names, modes, dtypes, and
  // shapes are all it takes to size and bind every buffer.
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    Binding b;
    b.name = engine->getIOTensorName(i);
    const nvinfer1::Dims dims = engine->getTensorShape(b.name.c_str());
    b.elem_size = element_size(engine->getTensorDataType(b.name.c_str()));
    b.elems_per_row = row_elements(dims);
    b.per_move = dims.d[0] == -1;
    b.input = engine->getTensorIOMode(b.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    const size_t capacity = b.bytes(params.max_moves);
    b.device = device_malloc(capacity);
    b.host = host_malloc(capacity);
    context->setTensorAddress(b.name.c_str(), b.device);
    bindings.push_back(b);
  }

  spatial_planes = engine->getTensorShape(kInputSpatial).d[1];
  scalar_floats = engine->getTensorShape(kInputScalar).d[1];
}

// ---------------------------------------------------------------------------

MoveSetNet::MoveSetNet(const MoveSetNetParams& params) : impl_(std::make_unique<Impl>(params)) {
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
}

MoveSetNet::~MoveSetNet() = default;

void MoveSetNet::load() {
  if (impl_->params.max_moves < 1) throw std::runtime_error("MoveSetNet: max_moves must be >= 1");
  set_device(impl_->params.cuda_device_id);

  std::vector<char> onnx_bytes = read_file_bytes(impl_->params.onnx_path);
  OnnxMetadata meta = parse_onnx_metadata(onnx_bytes);
  if (meta.graph != kGraphMoveSetEval) {
    throw std::runtime_error("MoveSetNet serves the " + std::string(kGraphMoveSetEval) +
                             " graph, but " + impl_->params.onnx_path + " declares graph '" +
                             meta.graph + "'");
  }
  if (meta.move_encoding_version != move_set::kMoveEncodingVersion) {
    throw std::runtime_error("Move-feature encoding version mismatch: " + impl_->params.onnx_path +
                             " was trained on v" + std::to_string(meta.move_encoding_version) +
                             ", this engine encodes v" +
                             std::to_string(move_set::kMoveEncodingVersion) +
                             " (retrain, or check out the matching model)");
  }
  impl_->contingent_features = meta.contingent_features;
  impl_->opp_leave_input = meta.opp_leave_input;

  std::string cache_path =
    engine_plan_cache_path(meta.architecture_signature, impl_->params.precision,
                           "moves_" + std::to_string(impl_->params.max_moves),
                           impl_->params.fast_build, impl_->params.mount_root);

  // A cache hit yields a plan with the right structure but, in general, another
  // checkpoint's weights; refitting swaps in this model's, far more cheaply
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

int MoveSetNet::max_moves() const { return impl_->params.max_moves; }
int MoveSetNet::spatial_planes() const { return impl_->spatial_planes; }
int MoveSetNet::scalar_floats() const { return impl_->scalar_floats; }
bool MoveSetNet::contingent_features() const { return impl_->contingent_features; }
bool MoveSetNet::opp_leave_input() const { return impl_->opp_leave_input; }

float* MoveSetNet::input_spatial_host() {
  return static_cast<float*>(impl_->binding(kInputSpatial).host);
}
float* MoveSetNet::input_scalar_host() {
  return static_cast<float*>(impl_->binding(kInputScalar).host);
}
int32_t* MoveSetNet::move_letters_host() {
  return static_cast<int32_t*>(impl_->binding(kMoveLetters).host);
}
uint8_t* MoveSetNet::move_blanks_host() {
  return static_cast<uint8_t*>(impl_->binding(kMoveBlanks).host);
}
int32_t* MoveSetNet::move_squares_host() {
  return static_cast<int32_t*>(impl_->binding(kMoveSquares).host);
}
uint8_t* MoveSetNet::move_tile_mask_host() {
  return static_cast<uint8_t*>(impl_->binding(kMoveTileMask).host);
}
float* MoveSetNet::move_scalars_host() {
  return static_cast<float*>(impl_->binding(kMoveScalars).host);
}
const float* MoveSetNet::wld_host() const {
  return static_cast<const float*>(impl_->binding(kOutputWld).host);
}
const float* MoveSetNet::score_diff_host() const {
  return static_cast<const float*>(impl_->binding(kOutputScoreDiff).host);
}

void MoveSetNet::predict(int num_moves) {
  Impl& m = *impl_;
  if (num_moves < 1 || num_moves > m.params.max_moves) {
    throw std::runtime_error("MoveSetNet::predict: num_moves out of range");
  }

  if (num_moves != m.last_moves) {
    for (const Binding& b : m.bindings) {
      if (!b.input || !b.per_move) continue;
      nvinfer1::Dims dims = m.engine->getTensorShape(b.name.c_str());
      dims.d[0] = num_moves;
      m.context->setInputShape(b.name.c_str(), dims);
    }
    m.last_moves = num_moves;
  }

  for (const Binding& b : m.bindings) {
    if (b.input) host_to_device_async(m.stream, b.device, b.host, b.bytes(num_moves));
  }

  if (!m.context->enqueueV3(m.stream)) throw std::runtime_error("TensorRT inference failed");

  for (const Binding& b : m.bindings) {
    if (!b.input) device_to_host_async(m.stream, b.host, b.device, b.bytes(num_moves));
  }

  synchronize_stream(m.stream);
}

}  // namespace nn
}  // namespace scribblez
