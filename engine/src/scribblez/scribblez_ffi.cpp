#include "scribblez/scribblez_ffi.h"

#include "scribblez/binary_log.h"
#include "scribblez/data_loader.h"
#include "scribblez/input_encoder.h"
#include "scribblez/training_targets.h"

#include <cstdio>
#include <filesystem>

using scribblez::binlog::DataLoader;
using scribblez::binlog::FileHeader;
using scribblez::binlog::kMagic;
using scribblez::binlog::kVersion;

namespace {

// Static dim arrays referenced by the ScribblezShape entries below. These
// have static storage duration; the addresses returned to the caller stay
// valid for the lifetime of the process.
constexpr int kInputSpatialDims[3] = {scribblez::binlog::kSpatialPlanes,
                                      scribblez::binlog::kBoardSide, scribblez::binlog::kBoardSide};
constexpr int kInputScalarDims[1] = {scribblez::binlog::kScalarFloats};

const ScribblezShape kInputShapes[] = {
  {"input_spatial", kInputSpatialDims, 3, -1},
  {"input_scalar", kInputScalarDims, 1, -1},
  {nullptr, nullptr, 0, 0},
};

// Build the (null-terminated) target shape table at compile time directly
// from scribblez::binlog::AllTargets, so adding/removing a target struct
// in training_targets.h automatically updates the FFI advertisement with
// no edits here.
template <typename List>
struct TargetShapeTable;

template <typename... Ts>
struct TargetShapeTable<scribblez::binlog::TargetList<Ts...>> {
  static constexpr std::size_t kCount = sizeof...(Ts);
  static constexpr std::array<ScribblezShape, kCount + 1> kValue = []() {
    std::array<ScribblezShape, kCount + 1> a{};
    std::size_t i = 0;
    (void)std::initializer_list<int>{
      (a[i] = ScribblezShape{Ts::kName, Ts::kDims, static_cast<int>(std::size(Ts::kDims)),
                             static_cast<int>(i)},
       ++i, 0)...};
    a[kCount] = ScribblezShape{nullptr, nullptr, 0, 0};
    return a;
  }();
};

constexpr auto kTargetShapesArr = TargetShapeTable<scribblez::binlog::AllTargets>::kValue;

}  // namespace

extern "C" {

const ScribblezShape* scribblez_input_shapes(void) { return kInputShapes; }
const ScribblezShape* scribblez_target_shapes(void) { return kTargetShapesArr.data(); }

int scribblez_row_size_floats(void) { return DataLoader::row_size_floats(); }

int scribblez_read_file_header(const char* path, int64_t* out_num_positions,
                               int64_t* out_file_size) {
  if (!path || !out_num_positions || !out_file_size) return -1;
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return -1;
  FileHeader hdr{};
  const size_t n = std::fread(&hdr, sizeof(hdr), 1, f);
  std::fclose(f);
  if (n != 1) return -1;
  if (hdr.magic != kMagic) return -1;
  if (hdr.version != kVersion) return -1;
  std::error_code ec;
  const auto fsz = std::filesystem::file_size(path, ec);
  if (ec) return -1;
  *out_num_positions = static_cast<int64_t>(hdr.num_games);
  *out_file_size = static_cast<int64_t>(fsz);
  return 0;
}

struct DataLoaderHandle {
  DataLoader loader;
  explicit DataLoaderHandle(const DataLoader::Params& p) : loader(p) {}
};

DataLoaderHandle* scribblez_dl_new(int64_t memory_budget, int num_worker_threads,
                                   int num_prefetch_threads) {
  DataLoader::Params p;
  p.memory_budget = memory_budget;
  p.num_worker_threads = num_worker_threads;
  p.num_prefetch_threads = num_prefetch_threads;
  return new DataLoaderHandle(p);
}

void scribblez_dl_delete(DataLoaderHandle* h) { delete h; }

void scribblez_dl_add_file(DataLoaderHandle* h, const char* path, int64_t num_positions,
                           int64_t file_size) {
  if (!h || !path) return;
  h->loader.add_file(path, num_positions, file_size);
}

int64_t scribblez_dl_num_positions(const DataLoaderHandle* h) {
  if (!h) return 0;
  return h->loader.num_positions();
}

void scribblez_dl_load(DataLoaderHandle* h, int64_t start, int64_t stop, int post_move,
                       int apply_symmetry, float* output) {
  if (!h || !output) return;
  h->loader.load(start, stop, post_move != 0, apply_symmetry != 0, output);
}

int scribblez_dl_epoch_start(DataLoaderHandle* h, int batch_size, int post_move, int apply_symmetry,
                             uint64_t seed) {
  if (!h) return 0;
  DataLoader::EpochConfig cfg;
  cfg.batch_size = batch_size;
  cfg.post_move = post_move != 0;
  cfg.apply_symmetry = apply_symmetry != 0;
  cfg.seed = seed;
  return h->loader.epoch_start(cfg);
}

int scribblez_dl_load_batch(DataLoaderHandle* h, float* output) {
  if (!h || !output) return 0;
  return h->loader.load_batch(output);
}

int64_t scribblez_dl_resident_bytes(const DataLoaderHandle* h) {
  if (!h) return 0;
  return h->loader.resident_bytes();
}

}  // extern "C"
