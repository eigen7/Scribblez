#pragma once

// A JSON description of the binary file formats (.slog, .sobs, .mset) and the
// constants that go with them, so Python readers build their numpy dtypes from
// the C++ structs rather than mirroring them by hand. Field lists come from
// reflection over the structs, so the document cannot drift from them. The FFI
// serves it as scribblez_format_layout_json.
//
// Document shape:
//
//   {"structs": {<Name>: {"itemsize": N, "fields": [
//        {"name": ..., "offset": ...,
//         "dtype": <numpy code> | {"struct": <Name>},  // nested struct
//         "shape": [...]?}]}},                         // std::array fields
//    "constants": {"board_size", "input_encoding_version",
//                  "slog": {...}, "sobs": {...}, "mset": {...},
//                  "move_type": {...}, "placement_head_names": [...],
//                  "placement_mask_names": [...], "footprint": {...}}}

#include <string>

namespace scribblez {

const std::string& format_layout_json();

}  // namespace scribblez
