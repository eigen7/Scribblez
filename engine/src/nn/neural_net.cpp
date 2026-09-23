#include "nn/neural_net.h"

#include "nn/cuda_util.h"
#include "nn/onnx_metadata.h"
#include "util/exception.h"

#include <boost/program_options.hpp>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {
namespace nn {

namespace {

// Drops everything below a warning, so build logs stay readable.
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
      throw util::Exception("NeuralNet: unsupported engine I/O data type");
  }
}

// The product of every dimension after the leading (row) one.
int row_elements(const nvinfer1::Dims& dims) {
  int n = 1;
  for (int i = 1; i < dims.nbDims; ++i) n *= int(dims.d[i]);
  return n;
}

nvinfer1::Dims dims_with_rows(nvinfer1::Dims dims, int rows) {
  dims.d[0] = rows;
  return dims;
}

// The layout facts check_required_layout compares, as observed on either a
// parsed network or a built engine.
struct ObservedTensor {
  size_t elem_size;
  int elems_per_row;
  bool dynamic;
};

// Check every entry of the spec's tensor table against what `lookup` observes:
// dtype, row width, and whether the tensor rides the dynamic axis. This guard
// is essential. Buffers are sized from the engine's own declarations, but the
// staging and decode loops are written at the table's constants, so a model
// that disagrees would overrun those buffers rather than fail. Nor do the
// metadata gates catch it: they cover encoding versions, not shapes or dtypes.
//
// It runs twice: on the parsed network before a build (NetworkTensorLookup),
// so a bad model fails before wasting a build, and on the built engine
// (Impl::BindingLookup), which is the only check a plan-cache hit gets. Each
// lookup throws for a tensor the model does not declare at all.
template <typename Lookup>
void check_required_layout(const std::string& onnx_path, const Lookup& lookup,
                           std::span<const TensorSpec> tensors) {
  for (const TensorSpec& want : tensors) {
    const ObservedTensor got = lookup(want.name);
    if (got.elem_size != want.elem_size) {
      throw util::CleanException(
        "Tensor dtype mismatch: {} declares {} at {} bytes per element, this engine reads it at {}",
        onnx_path, want.name, got.elem_size, want.elem_size);
    }
    if (want.elems_per_row != 0 && got.elems_per_row != want.elems_per_row) {
      throw util::CleanException(
        "Tensor width mismatch: {} declares {} {} wide, this engine encodes {}", onnx_path,
        want.name, got.elems_per_row, want.elems_per_row);
    }
    if (got.dynamic != want.dynamic) {
      throw util::CleanException(
        "Tensor axis mismatch: {} declares {} {}, this engine stages it {}", onnx_path, want.name,
        got.dynamic ? "dynamic" : "static", want.dynamic ? "dynamic" : "static");
    }
  }
}

// check_required_layout's lookup on a parsed, not yet built, network.
class NetworkTensorLookup {
 public:
  explicit NetworkTensorLookup(const nvinfer1::INetworkDefinition& network) : network_(&network) {}

  ObservedTensor operator()(const char* name) const {
    for (int i = 0; i < network_->getNbInputs(); ++i) {
      if (ObservedTensor got; observe(network_->getInput(i), name, &got)) return got;
    }
    for (int i = 0; i < network_->getNbOutputs(); ++i) {
      if (ObservedTensor got; observe(network_->getOutput(i), name, &got)) return got;
    }
    throw util::CleanException("NeuralNet: model has no tensor named '{}'", name);
  }

 private:
  static bool observe(nvinfer1::ITensor* tensor, const char* name, ObservedTensor* out) {
    if (std::string(tensor->getName()) != name) return false;
    const nvinfer1::Dims dims = tensor->getDimensions();
    *out = {element_size(tensor->getType()), row_elements(dims), dims.d[0] == -1};
    return true;
  }

  const nvinfer1::INetworkDefinition* network_;
};

// One of the engine's I/O tensors, with the host and device buffers bound to
// it. With up to seven inputs across three dtypes, a member per tensor would
// not scale, so allocation, binding, shape updates, and copies all loop over a
// table of these.
struct Binding {
  std::string name;
  size_t elem_size = 0;
  int elems_per_row = 0;
  bool dynamic = false;  // rows = the call's row count; a static tensor is one row
  bool input = false;
  bool aux = false;  // host buffer and device-to-host copy only under copy_aux
  void* device = nullptr;
  void* host = nullptr;

  size_t bytes(int num_rows) const {
    return elem_size * size_t(elems_per_row) * (dynamic ? num_rows : 1);
  }
};

}  // namespace

void NeuralNetParamsBase::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()  //
    ("model", po::value<std::string>(&onnx_path)->required(),
     "exported ONNX model to build the TensorRT engine from")  //
    ("batch-size", po::value<int>(&max_rows)->default_value(max_rows),
     "maximum rows (batch of positions, or candidates of one) per TensorRT call")  //
    ("fast-build", po::bool_switch(&fast_build),
     "TensorRT builder optimization level 0 (fast engine build, slower inference); "
     "for tests and smoke runs");
}

struct NeuralNetBase::Impl {
  Impl(const NeuralNetParamsBase& p, const RuntimeSpec& s) : params(p), spec(s) {}
  ~Impl();

  void deserialize_engine(const std::vector<char>& plan);

  // Builds a serialized plan in memory; touches no disk.
  std::vector<char> build_plan(const std::vector<char>& onnx_bytes);

  // Swaps this model's weights into a deserialized cached plan, which holds the
  // weights of whichever same-architecture checkpoint first populated it.
  void refit_engine(const std::vector<char>& onnx_bytes);

  // Context, stream, and the binding table, once the engine exists.
  void allocate_buffers();

  // The binding for `name`; throws if the engine has no such tensor. A linear
  // scan over about a dozen entries, run a handful of times per call.
  Binding& binding(const char* name);
  const Binding& binding(const char* name) const;

  // check_required_layout's lookup on a built engine's bindings.
  class BindingLookup {
   public:
    explicit BindingLookup(const Impl& impl) : impl_(&impl) {}

    ObservedTensor operator()(const char* name) const {
      const Binding& b = impl_->binding(name);
      return {b.elem_size, b.elems_per_row, b.dynamic};
    }

   private:
    const Impl* impl_;
  };

  NeuralNetParamsBase params;
  RuntimeSpec spec;
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  stream_t stream = nullptr;
  std::vector<Binding> bindings;

  // Read off the engine's tensor shapes and the model's ONNX metadata at load.
  int spatial_planes = 0;
  int scalar_floats = 0;
  int channels = 0;
  bool opp_leave_input = false;

  int last_rows = -1;

  // binding() without the throw, for tensors only some specs declare.
  const Binding* find_binding(const char* name) const {
    for (const Binding& b : bindings) {
      if (b.name == name) return &b;
    }
    return nullptr;
  }
};

NeuralNetBase::Impl::~Impl() {
  if (!stream) return;
  for (Binding& b : bindings) {
    if (b.device) device_free(b.device);
    if (b.host) host_free(b.host);
  }
  destroy_stream(stream);
}

Binding& NeuralNetBase::Impl::binding(const char* name) {
  return const_cast<Binding&>(std::as_const(*this).binding(name));
}

const Binding& NeuralNetBase::Impl::binding(const char* name) const {
  for (const Binding& b : bindings) {
    if (b.name == name) return b;
  }
  throw util::CleanException("NeuralNet: model has no tensor named '{}'", name);
}

void NeuralNetBase::Impl::deserialize_engine(const std::vector<char>& plan) {
  engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine) throw util::Exception("Failed to deserialize TensorRT engine");
}

std::vector<char> NeuralNetBase::Impl::build_plan(const std::vector<char>& onnx_bytes) {
  std::cerr << "[TRT] Building " << spec.graph
            << " engine from ONNX (one-time; cached per architecture afterward)...\n";

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  // Passing the model path makes external-data references (such as a frozen
  // lexicon blob stored beside the .onnx) resolve against the model's own
  // directory rather than the process's working directory.
  if (!parser->parse(onnx_bytes.data(), onnx_bytes.size(), params.onnx_path.c_str())) {
    if (parser->getNbErrors() > 0) {
      throw util::CleanException("Failed to parse ONNX model: {}", parser->getError(0)->desc());
    }
    throw util::CleanException("Failed to parse ONNX model");
  }

  // TensorRT happily builds a valid graph with a narrowed or re-typed tensor,
  // so check the layout now rather than after a wasted build.
  check_required_layout(params.onnx_path, NetworkTensorLookup(*network), spec.tensors);

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, params.workspace_bytes);
  if (params.precision == Precision::kFP16) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (params.precision == Precision::kBF16) {
    // BF16 tensor cores need Ampere or newer (SM80+). On older hardware
    // TensorRT accepts the flag but silently falls back to FP32 tactics, so fail
    // loudly instead and name FP16 as the alternative.
    if (compute_capability_major() < 8) {
      throw util::CleanException(
        "BF16 serving requires an Ampere-or-newer GPU (compute capability >= 8.0); this device is "
        "SM {}. Re-run with --precision FP16.",
        sm_tag());
    }
    config->setFlag(nvinfer1::BuilderFlag::kBF16);
  }
  // A cached plan is shared by every checkpoint of one architecture, so it
  // must be refittable for a cache hit to swap in the loaded model's weights.
  config->setFlag(nvinfer1::BuilderFlag::kREFIT);
  if (params.fast_build) config->setBuilderOptimizationLevel(0);

  // Only inputs on the dynamic row axis get a profile entry. Which ones those
  // are is read off the parsed network, which the layout check above has
  // already pinned to the spec.
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  const int opt_rows = std::min(spec.opt_rows, params.max_rows);
  for (int i = 0; i < network->getNbInputs(); ++i) {
    nvinfer1::ITensor* tensor = network->getInput(i);
    const nvinfer1::Dims dims = tensor->getDimensions();
    if (dims.d[0] != -1) continue;
    const char* name = tensor->getName();
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dims_with_rows(dims, 1));
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT,
                           dims_with_rows(dims, opt_rows));
    profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX,
                           dims_with_rows(dims, params.max_rows));
  }
  config->addOptimizationProfile(profile);

  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) throw util::Exception("TensorRT engine build failed");
  const char* data = static_cast<const char*>(plan->data());
  return std::vector<char>(data, data + plan->size());
}

void NeuralNetBase::Impl::refit_engine(const std::vector<char>& onnx_bytes) {
  std::unique_ptr<nvinfer1::IRefitter> refitter(nvinfer1::createInferRefitter(*engine, logger));
  std::unique_ptr<nvonnxparser::IParserRefitter> parser_refitter(
    nvonnxparser::createParserRefitter(*refitter, logger));
  // The model path resolves external-data references, as in build_plan().
  const bool clean =
    parser_refitter->refitFromBytes(onnx_bytes.data(), onnx_bytes.size(), params.onnx_path.c_str());
  // TensorRT 10.11's parser-refitter can count one more ONNX weight than the
  // engine has a slot for (an anonymous fusion product; the move-set graph has
  // one), and then fails its own strict count on a refit that missed nothing.
  // So a false return is tolerated when the engine reports no missing weights.
  //
  // Neither signal proves a refit complete: a deserialized plan already holds a
  // value for every weight, so none is ever reported missing. The real check is
  // the parity tests, which compare a refitted engine's outputs against each
  // checkpoint's own reference outputs (py/scripts/move_set_eval/
  // trt_refit_probe.py reached the same conclusion).
  if (!clean && refitter->getMissingWeights(0, nullptr) != 0) {
    if (parser_refitter->getNbErrors() > 0) {
      throw util::Exception("Failed to read refit weights from ONNX model: {}",
                            parser_refitter->getError(0)->desc());
    }
    throw util::Exception("Failed to read refit weights from ONNX model");
  }
  if (!refitter->refitCudaEngine()) throw util::Exception("Failed to refit TensorRT engine");
}

void NeuralNetBase::Impl::allocate_buffers() {
  context.reset(engine->createExecutionContext());
  if (!context) throw util::Exception("Failed to create TensorRT execution context");
  stream = create_stream();

  // The engine's own tensor declarations size and bind every buffer. Only the
  // aux flag comes from the spec, since the engine cannot know which outputs
  // callers opt into copying back.
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    Binding b;
    b.name = engine->getIOTensorName(i);
    const nvinfer1::Dims dims = engine->getTensorShape(b.name.c_str());
    b.elem_size = element_size(engine->getTensorDataType(b.name.c_str()));
    b.elems_per_row = row_elements(dims);
    b.dynamic = dims.d[0] == -1;
    b.input = engine->getTensorIOMode(b.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    for (const TensorSpec& t : spec.tensors) {
      if (b.name == t.name) b.aux = t.aux;
    }
    const size_t capacity = b.bytes(params.max_rows);
    b.device = device_malloc(capacity);
    // Every output needs a device buffer to stay bound for enqueueV3, but aux
    // outputs get a host buffer, and a copy back, only under copy_aux.
    if (!b.aux || params.copy_aux) b.host = host_malloc(capacity);
    context->setTensorAddress(b.name.c_str(), b.device);
    bindings.push_back(b);
  }

  // Repeated here for a cache hit, which skipped build_plan()'s check.
  check_required_layout(params.onnx_path, BindingLookup(*this), spec.tensors);

  // The move-proposal step graph has no board inputs; its board arrives
  // pre-encoded as a handoff tensor.
  if (find_binding(SpatialInput::kName)) {
    spatial_planes = engine->getTensorShape(SpatialInput::kName).d[1];
    scalar_floats = engine->getTensorShape(ScalarInput::kName).d[1];
  }
  // The channels tensor is (rows, C), so its per-row width is C.
  if (spec.channels_tensor) channels = binding(spec.channels_tensor).elems_per_row;
}

// ---------------------------------------------------------------------------

namespace {

// The model must declare exactly the spec's graph, or, if the spec allows it,
// no graph at all.
void check_graph(const RuntimeSpec& spec, const OnnxMetadata& meta, const std::string& onnx_path) {
  if (meta.graph.empty() && spec.accept_untagged_graph) return;
  if (meta.graph != spec.graph) {
    throw util::CleanException("NeuralNet serves the {} graph, but {} declares graph '{}'",
                               spec.graph, onnx_path, meta.graph);
  }
}

// The model must declare exactly each encoding version the spec requires
// (see VersionRequirement in model_specs.h).
void check_versions(const RuntimeSpec& spec, const OnnxMetadata& meta,
                    const std::string& onnx_path) {
  for (const VersionRequirement& v : spec.versions) {
    const int declared = meta.int_entry(v.key, v.absent_value);
    if (declared != v.required) {
      throw util::CleanException(
        "Encoding version mismatch: {} declares {} v{}, this engine "
        "encodes v{} (retrain, or check out the matching model)",
        onnx_path, v.key, declared, v.required);
    }
  }
}

}  // namespace

NeuralNetBase::NeuralNetBase(const NeuralNetParamsBase& params, const RuntimeSpec& spec)
    : impl_(std::make_unique<Impl>(params, spec)) {
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
}

NeuralNetBase::~NeuralNetBase() = default;

void NeuralNetBase::load() {
  Impl& m = *impl_;
  if (m.params.max_rows < 1) throw util::CleanException("NeuralNet: max_rows must be >= 1");
  set_device(m.params.cuda_device_id);

  std::vector<char> onnx_bytes = read_file_bytes(m.params.onnx_path);
  OnnxMetadata meta = parse_onnx_metadata(onnx_bytes);
  check_graph(m.spec, meta, m.params.onnx_path);
  check_versions(m.spec, meta, m.params.onnx_path);
  m.opp_leave_input = meta.opp_leave_input;

  std::string cache_path =
    engine_plan_cache_path(meta.architecture_signature, m.params.precision,
                           std::format("{}_{}", m.spec.axis_tag, m.params.max_rows),
                           m.params.fast_build, m.params.mount_root);

  // The cache is keyed on the architecture signature, so a hit yields the
  // right structure with, in general, another checkpoint's weights. Refitting
  // swaps in this model's weights at a fraction of the cost of a build.
  if (std::filesystem::exists(cache_path)) {
    m.deserialize_engine(read_file_bytes(cache_path));
    m.refit_engine(onnx_bytes);
  } else {
    std::vector<char> plan = m.build_plan(onnx_bytes);
    write_file_bytes(cache_path, plan.data(), plan.size());
    m.deserialize_engine(plan);
  }
  m.allocate_buffers();
}

int NeuralNetBase::max_rows() const { return impl_->params.max_rows; }
int NeuralNetBase::spatial_planes() const { return impl_->spatial_planes; }
int NeuralNetBase::scalar_floats() const { return impl_->scalar_floats; }
int NeuralNetBase::channels() const { return impl_->channels; }
bool NeuralNetBase::opp_leave_input() const { return impl_->opp_leave_input; }

void* NeuralNetBase::host_ptr(const char* name) const { return impl_->binding(name).host; }

void NeuralNetBase::predict(int num_rows) {
  Impl& m = *impl_;
  if (num_rows < 1 || num_rows > m.params.max_rows) {
    throw util::Exception("NeuralNet::predict: num_rows out of range");
  }

  if (num_rows != m.last_rows) {
    for (const Binding& b : m.bindings) {
      if (!b.input || !b.dynamic) continue;
      m.context->setInputShape(b.name.c_str(),
                               dims_with_rows(m.engine->getTensorShape(b.name.c_str()), num_rows));
    }
    m.last_rows = num_rows;
  }

  for (const Binding& b : m.bindings) {
    if (b.input) host_to_device_async(m.stream, b.device, b.host, b.bytes(num_rows));
  }

  if (!m.context->enqueueV3(m.stream)) throw util::Exception("TensorRT inference failed");

  for (const Binding& b : m.bindings) {
    if (b.input || !b.host) continue;
    device_to_host_async(m.stream, b.host, b.device, b.bytes(num_rows));
  }

  synchronize_stream(m.stream);
}

}  // namespace nn
}  // namespace scribblez
