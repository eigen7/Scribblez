#pragma once

// One JSON document describing the binary sidecar formats (.slog, .sobs,
// .mset) for cross-language readers: per struct the field names, byte
// offsets, and numpy dtype codes -- offsets and sizes taken from the compiler
// via offsetof/sizeof, so the document cannot drift from the structs -- plus
// the formats' magics, versions, flag bits, and the Move/Glyph code tables.
//
// The FFI serves it to Python (scribblez_format_layout_json), where the
// numpy dtypes are built from it instead of hand-mirroring the packed
// structs. Document shape:
//
//   {"structs": {<Name>: {"itemsize": N, "fields": [
//        {"name": ..., "offset": ...,
//         "dtype": <numpy code> | {"struct": <Name>},   // nested reference
//         "shape": [...]?}]}},                          // subarray fields
//    "constants": {"board_size", "slog": {...}, "sobs": {...},
//                  "mset": {...}, "move_type": {...}, "glyph": {...},
//                  "placement_head_names": [...]}}

#include <string>

namespace scribblez {

const std::string& format_layout_json();

}  // namespace scribblez
