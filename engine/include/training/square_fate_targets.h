#pragma once

// Row targets from EncodeContext::square_fates (encoding/square_fates.h). A
// task using them declares kRecordsSquareFates, so the replay computes them.

#include "encoding/encode_context.h"
#include "encoding/input_encoder.h"

namespace scribblez {

// Per-square ownership class (SquareFate as a float, -1 for a filled square),
// the label for a masked softmax over kSquareFateClasses.
struct OwnershipTarget {
  static constexpr const char* kName = "ownership";
  static constexpr int kDims[] = {kBoardSide, kBoardSide};
  static void encode(const EncodeContext& v, float* out);
};

// Per-square signed points (SquareFates::points), the label for the points
// mean and variance heads, masked like OwnershipTarget.
struct SquarePointsTarget {
  static constexpr const char* kName = "square_points";
  static constexpr int kDims[] = {kBoardSide, kBoardSide};
  static void encode(const EncodeContext& v, float* out);
};

}  // namespace scribblez
