#include "encoding/input_encoder.h"

#include <cstdio>
#include <cstdlib>

namespace scribblez {

namespace {

[[noreturn]] void fail_block_lookup(const char* kind) {
  std::fprintf(stderr, "input layout: %s block not included by the spec\n", kind);
  std::abort();
}

}  // namespace

int spatial_planes(const InputEncodingSpec& spec) {
  int planes = 0;
  for (const SpatialBlockDef& def : kSpatialBlocks) {
    if (def.contingent_only && !spec.contingent_features) continue;
    planes += def.planes;
  }
  return planes;
}

int scalar_floats(const InputEncodingSpec& spec) {
  int floats = 0;
  for (const ScalarBlockDef& def : kScalarBlocks) {
    if (!scalar_block_included(def, spec)) continue;
    floats += def.floats;
  }
  return floats;
}

int spatial_floats(const InputEncodingSpec& spec) { return spatial_planes(spec) * kBoardCells; }

int input_floats(const InputEncodingSpec& spec) {
  return spatial_floats(spec) + scalar_floats(spec);
}

int spatial_block_plane0(const InputEncodingSpec& spec, SpatialBlockId id) {
  int plane = 0;
  for (const SpatialBlockDef& def : kSpatialBlocks) {
    if (def.contingent_only && !spec.contingent_features) continue;
    if (def.id == id) return plane;
    plane += def.planes;
  }
  fail_block_lookup("spatial");
}

int scalar_block_offset(const InputEncodingSpec& spec, ScalarBlockId id) {
  int offset = 0;
  for (const ScalarBlockDef& def : kScalarBlocks) {
    if (!scalar_block_included(def, spec)) continue;
    if (def.id == id) return offset;
    offset += def.floats;
  }
  fail_block_lookup("scalar");
}

}  // namespace scribblez
