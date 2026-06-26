#include "scribblez/nn/neural_net.h"

#include "scribblez/input_encoder.h"
#include "scribblez/nn/cuda_util.h"
#include "scribblez/training_targets.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace scribblez {
namespace nn {

namespace {

using binlog::kBoardSide;
using binlog::kOppNextPlacementFloats;
using binlog::kScalarFloats;
using binlog::kScoreDiffOutputFloats;
using binlog::kSpatialFloats;
using binlog::kSpatialPlanes;
using binlog::kWldFloats;

// Names of the engine's I/O tensors. These match the input_names / output_names
// passed to torch.onnx.export in py/scribblez/onnx_export.py.
constexpr const char* kInputSpatial = "input_spatial";
constexpr const char* kInputScalar = "input_scalar";
constexpr const char* kOutputWld = "wld";
constexpr const char* kOutputScoreDiff = "score_diff";
constexpr const char* kOutputOpp = "opp_next_placement";

// Routes TensorRT's internal diagnostics to stderr, dropping anything below a
// warning so the build logs stay readable.
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};

std::vector<char> read_file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Failed to open file: " + path);
  std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(size));
  f.read(bytes.data(), size);
  return bytes;
}

// Write `bytes` to `path` atomically (tmp file + rename), creating parents. The
// temp name carries the pid and a random suffix so two processes building the
// same plan concurrently (multiple self-play workers share one cache directory)
// never write to the same temp file and corrupt each other's rename.
void write_file_bytes(const std::string& path, const char* bytes, size_t size) {
  std::filesystem::path p(path);
  std::filesystem::create_directories(p.parent_path());
  std::string tmp =
    path + ".tmp." + std::to_string(::getpid()) + "." + std::to_string(std::random_device{}());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open temp file: " + tmp);
    f.write(bytes, static_cast<std::streamsize>(size));
  }
  std::filesystem::rename(tmp, p);
}

nvinfer1::Dims spatial_dims(int rows) {
  nvinfer1::Dims d;
  d.nbDims = 4;
  d.d[0] = rows;
  d.d[1] = kSpatialPlanes;
  d.d[2] = kBoardSide;
  d.d[3] = kBoardSide;
  return d;
}

nvinfer1::Dims scalar_dims(int rows) {
  nvinfer1::Dims d;
  d.nbDims = 2;
  d.d[0] = rows;
  d.d[1] = kScalarFloats;
  return d;
}

}  // namespace

struct NeuralNet::Impl {
  explicit Impl(const NeuralNetParams& p) : params(p) {}
  ~Impl();

  // Set engine from a serialized plan blob.
  void deserialize_engine(const std::vector<char>& plan);

  // Build a serialized engine plan from the ONNX bytes (does not touch disk).
  std::vector<char> build_plan(const std::vector<char>& onnx_bytes);

  // Allocate context, stream, and host/device buffers once the engine exists.
  void allocate_buffers();

  NeuralNetParams params;
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  stream_t stream = nullptr;

  // TODO(Refactor): Decouple C++ inference buffers from the strict neural net architecture.
  // Currently, adding a new head requires manually hardcoding new 
  // device/host pointers, cudaMallocs, and cudaMemcpys across different places.
  //
  // Implementation Plan:
  // 1. Define a `TensorBuffer` struct (name, size_bytes, is_input, d_ptr, h_ptr).
  // 2. Query the engine dynamically during `allocate_buffers`:
  //    - num_tensors = engine->getNbIOTensors()
  //    - name = engine->getIOTensorName(i)
  //    - mode = engine->getTensorIOMode(name)  // Check for kINPUT vs kOUTPUT
  //    - shape = engine->getTensorShape(name)  // Calculate size_bytes
  // 3. Store buffers in `std::vector<TensorBuffer> inputs/outputs`.
  // 4. Maintain a flat `std::vector<void*> bindings` to pass cleanly to `enqueueV3()`.
  // 5. Replace manual cudaMemcpy calls with loops over the `inputs` and `outputs` vectors.
  void* d_input_spatial = nullptr;
  void* d_input_scalar = nullptr;
  void* d_wld = nullptr;
  void* d_score_diff = nullptr;
  void* d_opp = nullptr;

  float* h_input_spatial = nullptr;
  float* h_input_scalar = nullptr;
  float* h_wld = nullptr;
  float* h_score_diff = nullptr;
  // No host buffer for the opp_next_placement output: the engine still produces
  // it (d_opp must stay bound for enqueueV3), but no inference consumer reads
  // it, so it is never copied back to the host.

  int last_rows = -1;
};

NeuralNet::Impl::~Impl() {
  if (stream) {
    if (d_input_spatial) device_free(d_input_spatial);
    if (d_input_scalar) device_free(d_input_scalar);
    if (d_wld) device_free(d_wld);
    if (d_score_diff) device_free(d_score_diff);
    if (d_opp) device_free(d_opp);
    if (h_input_spatial) host_free(h_input_spatial);
    if (h_input_scalar) host_free(h_input_scalar);
    if (h_wld) host_free(h_wld);
    if (h_score_diff) host_free(h_score_diff);
    destroy_stream(stream);
  }
}

void NeuralNet::Impl::deserialize_engine(const std::vector<char>& plan) {
  engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine) throw std::runtime_error("Failed to deserialize TensorRT engine");
}

std::vector<char> NeuralNet::Impl::build_plan(const std::vector<char>& onnx_bytes) {
  std::cerr << "[TRT] Building engine from ONNX (one-time; cached afterward)...\n";

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  if (!parser->parse(onnx_bytes.data(), onnx_bytes.size())) {
    std::string msg = "Failed to parse ONNX model";
    if (parser->getNbErrors() > 0) msg += std::string(": ") + parser->getError(0)->desc();
    throw std::runtime_error(msg);
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, params.workspace_bytes);
  if (params.precision == Precision::kFP16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  int b = params.max_batch_size;
  profile->setDimensions(kInputSpatial, nvinfer1::OptProfileSelector::kMIN, spatial_dims(1));
  profile->setDimensions(kInputSpatial, nvinfer1::OptProfileSelector::kOPT, spatial_dims(b));
  profile->setDimensions(kInputSpatial, nvinfer1::OptProfileSelector::kMAX, spatial_dims(b));
  profile->setDimensions(kInputScalar, nvinfer1::OptProfileSelector::kMIN, scalar_dims(1));
  profile->setDimensions(kInputScalar, nvinfer1::OptProfileSelector::kOPT, scalar_dims(b));
  profile->setDimensions(kInputScalar, nvinfer1::OptProfileSelector::kMAX, scalar_dims(b));
  config->addOptimizationProfile(profile);

  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) throw std::runtime_error("TensorRT engine build failed");
  const char* data = static_cast<const char*>(plan->data());
  return std::vector<char>(data, data + plan->size());
}

void NeuralNet::Impl::allocate_buffers() {
  context.reset(engine->createExecutionContext());
  if (!context) throw std::runtime_error("Failed to create TensorRT execution context");
  stream = create_stream();

  int b = params.max_batch_size;
  auto dev = [](int floats) { return device_malloc(sizeof(float) * floats); };
  auto host = [](int floats) { return static_cast<float*>(host_malloc(sizeof(float) * floats)); };

  d_input_spatial = dev(b * kSpatialFloats);
  d_input_scalar = dev(b * kScalarFloats);
  d_wld = dev(b * kWldFloats);
  d_score_diff = dev(b * kScoreDiffOutputFloats);
  d_opp = dev(b * kOppNextPlacementFloats);

  h_input_spatial = host(b * kSpatialFloats);
  h_input_scalar = host(b * kScalarFloats);
  h_wld = host(b * kWldFloats);
  h_score_diff = host(b * kScoreDiffOutputFloats);

  context->setTensorAddress(kInputSpatial, d_input_spatial);
  context->setTensorAddress(kInputScalar, d_input_scalar);
  context->setTensorAddress(kOutputWld, d_wld);
  context->setTensorAddress(kOutputScoreDiff, d_score_diff);
  context->setTensorAddress(kOutputOpp, d_opp);
}

// ---------------------------------------------------------------------------

NeuralNet::NeuralNet(const NeuralNetParams& params) : impl_(std::make_unique<Impl>(params)) {
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
}

NeuralNet::~NeuralNet() = default;

void NeuralNet::load() {
  set_device(impl_->params.cuda_device_id);

  std::vector<char> onnx_bytes = read_file_bytes(impl_->params.onnx_path);
  std::string hash = content_hash(onnx_bytes);
  std::string cache_path = engine_plan_cache_path(
    hash, impl_->params.precision, impl_->params.max_batch_size, impl_->params.mount_root);

  if (std::filesystem::exists(cache_path)) {
    impl_->deserialize_engine(read_file_bytes(cache_path));
  } else {
    std::vector<char> plan = impl_->build_plan(onnx_bytes);
    write_file_bytes(cache_path, plan.data(), plan.size());
    impl_->deserialize_engine(plan);
  }
  impl_->allocate_buffers();
}

int NeuralNet::max_batch_size() const { return impl_->params.max_batch_size; }

float* NeuralNet::input_spatial_host() { return impl_->h_input_spatial; }
float* NeuralNet::input_scalar_host() { return impl_->h_input_scalar; }
const float* NeuralNet::wld_host() const { return impl_->h_wld; }
const float* NeuralNet::score_diff_host() const { return impl_->h_score_diff; }

void NeuralNet::predict(int num_rows) {
  Impl& m = *impl_;
  if (num_rows < 1 || num_rows > m.params.max_batch_size) {
    throw std::runtime_error("NeuralNet::predict: num_rows out of range");
  }

  if (num_rows != m.last_rows) {
    m.context->setInputShape(kInputSpatial, spatial_dims(num_rows));
    m.context->setInputShape(kInputScalar, scalar_dims(num_rows));
    m.last_rows = num_rows;
  }

  host_to_device_async(m.stream, m.d_input_spatial, m.h_input_spatial,
                       sizeof(float) * num_rows * kSpatialFloats);
  host_to_device_async(m.stream, m.d_input_scalar, m.h_input_scalar,
                       sizeof(float) * num_rows * kScalarFloats);

  if (!m.context->enqueueV3(m.stream)) throw std::runtime_error("TensorRT inference failed");

  // Only the wld and score_diff outputs are copied back; opp_next_placement is
  // produced into d_opp but has no inference consumer, so it stays on the GPU.
  device_to_host_async(m.stream, m.h_wld, m.d_wld, sizeof(float) * num_rows * kWldFloats);
  device_to_host_async(m.stream, m.h_score_diff, m.d_score_diff,
                       sizeof(float) * num_rows * kScoreDiffOutputFloats);

  synchronize_stream(m.stream);
}

}  // namespace nn
}  // namespace scribblez
