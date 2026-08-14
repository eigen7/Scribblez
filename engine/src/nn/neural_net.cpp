#include "nn/neural_net.h"

#include "nn/cuda_util.h"
#include "nn/onnx_metadata.h"

#include <boost/program_options.hpp>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {
namespace nn {

namespace {

// Drops anything below a warning, so the build logs stay readable.
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
      throw std::runtime_error("NeuralNet: unsupported engine I/O data type");
  }
}

// Elements in one row of a tensor, i.e. the product of every dimension after
// the leading one.
int row_elements(const nvinfer1::Dims& dims) {
  int n = 1;
  for (int i = 1; i < dims.nbDims; ++i) n *= static_cast<int>(dims.d[i]);
  return n;
}

// `dims` with the leading (row) dimension replaced.
nvinfer1::Dims dims_with_rows(nvinfer1::Dims dims, int rows) {
  dims.d[0] = rows;
  return dims;
}

// What the spec's TensorSpec table expects of one tensor, as read off either a
// pre-build network definition or a built engine's bindings -- the two places
// check_required_layout is called from.
struct ObservedTensor {
  size_t elem_size;
  int elems_per_row;
  bool dynamic;
};

// Every entry of the spec's tensor table against what `lookup` observes, so
// the same dtype/width/axis guard can run before a build (on the parsed
// network, cheaply rejecting a model that would otherwise waste a full build)
// and after one (on the built engine, the only check a cache hit gets since it
// skips the parse). The staging and decode loops are written at the table's
// constants against buffers the engine's own declarations size, so a model
// that disagrees would overrun those buffers rather than fail -- and neither
// shapes nor dtypes are part of the metadata the loader gates on. `lookup`
// throws its own "no tensor named" error for a tensor the model doesn't
// declare at all; NetworkTensorLookup, below, and Impl::BindingLookup are its
// two implementations.
template <typename Lookup>
void check_required_layout(const std::string& onnx_path, const Lookup& lookup,
                           std::span<const TensorSpec> tensors) {
  for (const TensorSpec& want : tensors) {
    const ObservedTensor got = lookup(want.name);
    if (got.elem_size != want.elem_size) {
      throw std::runtime_error("Tensor dtype mismatch: " + onnx_path + " declares " + want.name +
                               " at " + std::to_string(got.elem_size) +
                               " bytes per element, this engine reads it at " +
                               std::to_string(want.elem_size));
    }
    if (want.elems_per_row != 0 && got.elems_per_row != want.elems_per_row) {
      throw std::runtime_error("Tensor width mismatch: " + onnx_path + " declares " + want.name +
                               " " + std::to_string(got.elems_per_row) + " wide, this engine " +
                               "encodes " + std::to_string(want.elems_per_row));
    }
    if (got.dynamic != want.dynamic) {
      throw std::runtime_error("Tensor axis mismatch: " + onnx_path + " declares " + want.name +
                               (got.dynamic ? " dynamic" : " static") + ", this engine stages it " +
                               (want.dynamic ? "dynamic" : "static"));
    }
  }
}

// check_required_layout's lookup on a parsed-but-not-yet-built network, so
// build_plan() can reject a bad layout before spending a build on it.
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
    throw std::runtime_error("NeuralNet: model has no tensor named '" + std::string(name) + "'");
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
// it. Up to seven inputs across three dtypes, and a per-call row count on the
// dynamic ones, leave no room for a pointer-per-tensor style: allocation,
// binding, shape updates, and the copies are all loops over these.
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
    return elem_size * static_cast<size_t>(elems_per_row) * (dynamic ? num_rows : 1);
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

  // Builds the plan in memory; touches no disk.
  std::vector<char> build_plan(const std::vector<char>& onnx_bytes);

  // For after deserializing a cached plan, which shares the architecture but
  // holds whatever same-architecture checkpoint first populated the cache.
  void refit_engine(const std::vector<char>& onnx_bytes);

  // Context, stream, and the binding table, once the engine exists.
  void allocate_buffers();

  // The binding for `name`, which the engine is guaranteed to expose (a model
  // missing a tensor this runtime feeds fails here rather than at inference).
  // A linear scan over at most eleven entries, run a handful of times per call.
  Binding& binding(const char* name);
  const Binding& binding(const char* name) const;

  // check_required_layout's lookup on a built engine's bindings -- the cheap
  // guard a cache hit still gets, since it skips build_plan()'s own pre-build
  // check (NetworkTensorLookup, above).
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

  // Read off the deserialized engine's declared board input shapes and the
  // model's own ONNX metadata_props.
  int spatial_planes = 0;
  int scalar_floats = 0;
  bool contingent_features = false;
  bool opp_leave_input = false;

  int last_rows = -1;
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
  throw std::runtime_error("NeuralNet: model has no tensor named '" + std::string(name) + "'");
}

void NeuralNetBase::Impl::deserialize_engine(const std::vector<char>& plan) {
  engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine) throw std::runtime_error("Failed to deserialize TensorRT engine");
}

std::vector<char> NeuralNetBase::Impl::build_plan(const std::vector<char>& onnx_bytes) {
  std::cerr << "[TRT] Building " << spec.graph
            << " engine from ONNX (one-time; cached per architecture afterward)...\n";

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

  // Reject a model with the wrong tensor layout off the parsed graph, before
  // spending a build on it: TensorRT happily builds an engine for, say, a
  // narrowed or re-typed tensor, since the graph is still valid, and only
  // allocate_buffers()'s post-build check (kept as the cheap guard a cache hit
  // still gets, skipping this parse) would otherwise have caught the mismatch.
  check_required_layout(params.onnx_path, NetworkTensorLookup(*network), spec.tensors);

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

  // Only tensors carrying the dynamic row axis get a profile entry; a static
  // tensor (the move-set graph's board inputs) needs none. Reading which is
  // which off the parsed network keeps the graph itself the authority on its
  // own shapes -- the layout check above has already pinned it to the spec.
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
  if (!plan) throw std::runtime_error("TensorRT engine build failed");
  const char* data = static_cast<const char*>(plan->data());
  return std::vector<char>(data, data + plan->size());
}

void NeuralNetBase::Impl::refit_engine(const std::vector<char>& onnx_bytes) {
  std::unique_ptr<nvinfer1::IRefitter> refitter(nvinfer1::createInferRefitter(*engine, logger));
  std::unique_ptr<nvonnxparser::IParserRefitter> parser_refitter(
    nvonnxparser::createParserRefitter(*refitter, logger));
  // Model path for external-data resolution, as in build_plan().
  const bool clean =
    parser_refitter->refitFromBytes(onnx_bytes.data(), onnx_bytes.size(), params.onnx_path.c_str());
  // TensorRT 10.11's parser-refitter can count one more ONNX-side weight than
  // the engine exposes a slot for (an anonymous fusion product; the move-set
  // graph has one) and fail its own strict count on a refit that left nothing
  // behind, so a false return is tolerated when the engine reports no missing
  // weights. Neither signal can prove a refit complete -- a deserialized plan
  // carries a value for every weight, so nothing is ever "missing" -- which is
  // why the parity tests compare a refitted engine's outputs against each
  // checkpoint's own reference: the verification the API cannot provide
  // (py/scripts/move_set_eval/trt_refit_probe.py reached the same verdict).
  if (!clean && refitter->getMissingWeights(0, nullptr) != 0) {
    std::string msg = "Failed to read refit weights from ONNX model";
    if (parser_refitter->getNbErrors() > 0)
      msg += std::string(": ") + parser_refitter->getError(0)->desc();
    throw std::runtime_error(msg);
  }
  if (!refitter->refitCudaEngine()) throw std::runtime_error("Failed to refit TensorRT engine");
}

void NeuralNetBase::Impl::allocate_buffers() {
  context.reset(engine->createExecutionContext());
  if (!context) throw std::runtime_error("Failed to create TensorRT execution context");
  stream = create_stream();

  // Enumerate the engine's own I/O tensors: their names, modes, dtypes, and
  // shapes are all it takes to size and bind every buffer. Only the spec's aux
  // flag comes from the table -- the engine cannot know which outputs callers
  // opt into copying back.
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
    // Aux outputs always occupy device buffers (they must stay bound for
    // enqueueV3), but get host buffers -- and copies back -- only under
    // copy_aux; agent inference never reads them.
    if (!b.aux || params.copy_aux) b.host = host_malloc(capacity);
    context->setTensorAddress(b.name.c_str(), b.device);
    bindings.push_back(b);
  }

  // The cheap guard a cache hit still gets: build_plan()'s own check runs
  // only on a fresh build, so this is what catches a stale plan cached
  // before an encoder layout change.
  check_required_layout(params.onnx_path, BindingLookup(*this), spec.tensors);

  spatial_planes = engine->getTensorShape(SpatialInput::kName).d[1];
  scalar_floats = engine->getTensorShape(ScalarInput::kName).d[1];
}

// ---------------------------------------------------------------------------

namespace {

// The spec's graph, exactly -- which model family a file belongs to is the
// first thing its metadata declares -- with the spec deciding whether an
// export predating the entry is accepted (model_specs.h).
void check_graph(const RuntimeSpec& spec, const OnnxMetadata& meta, const std::string& onnx_path) {
  if (meta.graph.empty() && spec.accept_untagged_graph) return;
  if (meta.graph != spec.graph) {
    throw std::runtime_error("NeuralNet serves the " + std::string(spec.graph) + " graph, but " +
                             onnx_path + " declares graph '" + meta.graph + "'");
  }
}

// Every encoding version the spec requires, exactly (model_specs.h explains
// what a version gate protects against).
void check_versions(const RuntimeSpec& spec, const OnnxMetadata& meta,
                    const std::string& onnx_path) {
  for (const VersionRequirement& v : spec.versions) {
    const int declared = meta.int_entry(v.key, v.absent_value);
    if (declared != v.required) {
      throw std::runtime_error("Encoding version mismatch: " + onnx_path + " declares " + v.key +
                               " v" + std::to_string(declared) + ", this engine encodes v" +
                               std::to_string(v.required) +
                               " (retrain, or check out the matching model)");
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
  if (m.params.max_rows < 1) throw std::runtime_error("NeuralNet: max_rows must be >= 1");
  set_device(m.params.cuda_device_id);

  std::vector<char> onnx_bytes = read_file_bytes(m.params.onnx_path);
  OnnxMetadata meta = parse_onnx_metadata(onnx_bytes);
  check_graph(m.spec, meta, m.params.onnx_path);
  check_versions(m.spec, meta, m.params.onnx_path);
  m.contingent_features = meta.contingent_features;
  m.opp_leave_input = meta.opp_leave_input;

  std::string cache_path =
    engine_plan_cache_path(meta.architecture_signature, m.params.precision,
                           std::string(m.spec.axis_tag) + "_" + std::to_string(m.params.max_rows),
                           m.params.fast_build, m.params.mount_root);

  // The cache is keyed on the model's architecture signature, so a hit yields
  // a plan with the right structure but (in general) another checkpoint's
  // weights; refitting swaps in this model's weights, which is far cheaper
  // than an engine build. The parity tests are what verify a refit actually
  // served the right checkpoint: TensorRT 10.11 reports a refit that mapped
  // every weight and one that left a weight behind identically, so the
  // numeric check against each checkpoint's own reference outputs is the
  // guard the API cannot provide.
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
bool NeuralNetBase::contingent_features() const { return impl_->contingent_features; }
bool NeuralNetBase::opp_leave_input() const { return impl_->opp_leave_input; }

void* NeuralNetBase::host_ptr(const char* name) const { return impl_->binding(name).host; }

void NeuralNetBase::predict(int num_rows) {
  Impl& m = *impl_;
  if (num_rows < 1 || num_rows > m.params.max_rows) {
    throw std::runtime_error("NeuralNet::predict: num_rows out of range");
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

  if (!m.context->enqueueV3(m.stream)) throw std::runtime_error("TensorRT inference failed");

  for (const Binding& b : m.bindings) {
    if (b.input || !b.host) continue;
    device_to_host_async(m.stream, b.host, b.device, b.bytes(num_rows));
  }

  synchronize_stream(m.stream);
}

}  // namespace nn
}  // namespace scribblez
