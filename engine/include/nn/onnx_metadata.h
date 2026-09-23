#pragma once

#include <map>
#include <string>
#include <vector>

// Reads back the metadata_props entries our ONNX exporters stamp into every
// model. They record what a model file cannot say structurally: which optional
// input blocks it takes, which graph it is, and which encoding versions it was
// trained on. An entry an older exporter did not write reads as a documented
// "absent" value, so older models stay loadable where that value is accepted.
//
// The writers, all under py/scribblez/ and all through onnx_export_util.py:
// position_eval/onnx_export.py, move_set_eval/onnx_export.py, and
// move_set_eval/proposal_export.py.

namespace scribblez {
namespace nn {

// The `graph` entry's values: which model family, and so which spec, a file
// belongs to.
inline constexpr const char* kGraphPositionEval = "position_eval";
inline constexpr const char* kGraphMoveSetEval = "move_set_eval";
// The move proposal model's two graphs: the per-turn cache graph and the
// per-evidence-iteration step graph. Must match proposal_export.py's
// GRAPH_CACHE and GRAPH_STEP.
inline constexpr const char* kGraphMoveProposalCache = "move_proposal_cache";
inline constexpr const char* kGraphMoveProposalStep = "move_proposal_step";

struct OnnxMetadata {
  // The input-encoding arm: which optional block the board row carries.
  bool opp_leave_input = false;

  // Keys the engine-plan cache, so every checkpoint of one architecture shares
  // a plan. Required: parsing a model without it throws.
  std::string architecture_signature;

  // kGraph* above. Empty for an export predating the entry.
  std::string graph;

  // Every other entry, verbatim. The encoding-version gates read these through
  // int_entry().
  std::map<std::string, std::string> entries;

  // The integer entry at `key`, or `absent_value` for an export predating the
  // entry. Throws on a non-integer value.
  int int_entry(const std::string& key, int absent_value) const;
};

// Throws on unparseable bytes, or on a model with no architecture signature.
OnnxMetadata parse_onnx_metadata(const std::vector<char>& onnx_bytes);

}  // namespace nn
}  // namespace scribblez
