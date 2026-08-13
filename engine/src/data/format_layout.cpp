#include "data/format_layout.h"

#include "data/binary_log.h"
#include "data/sim_observation_log.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "sim/sim_runner.h"
#include "training/move_set_eval_target_log.h"
#include "training/training_targets.h"

#include <boost/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <meta>
#include <string>

namespace scribblez {

namespace {

namespace json = boost::json;

// Each struct's field list is enumerated with P2996 reflection, so a member
// added, removed, renamed, retyped, or reordered in a struct re-derives the
// document with no restatement to keep in sync. What remains hand-written is
// exactly the policy: the type -> dtype-code table below, and which structs
// are served (build_structs).

struct FieldInfo {
  const char* name;  // reader-facing (trailing '_' stripped)
  std::size_t offset;
  const char* dtype;       // numpy code, or null when struct_ref is set
  const char* struct_ref;  // referenced struct's document key, or null
  std::size_t count;       // subarray element count; 0 = scalar
};

// The numpy dtype code of one scalar member type -- one mapping per line,
// little-endian to match the on-disk packed structs. One-byte enums and
// wrappers (MoveType, Glyph, bool) serialize as their underlying byte.
consteval const char* scalar_code(std::meta::info t) {
  if (t == std::meta::dealias(^^uint8_t) || t == ^^Glyph || t == ^^MoveType || t == ^^bool)
    return "u1";
  if (t == std::meta::dealias(^^int8_t)) return "i1";
  if (t == std::meta::dealias(^^uint16_t)) return "<u2";
  if (t == std::meta::dealias(^^int16_t)) return "<i2";
  if (t == std::meta::dealias(^^uint32_t)) return "<u4";
  if (t == std::meta::dealias(^^int32_t)) return "<i4";
  if (t == std::meta::dealias(^^uint64_t)) return "<u8";
  if (t == std::meta::dealias(^^int64_t)) return "<i8";
  if (t == std::meta::dealias(^^float)) return "<f4";
  return nullptr;
}

// Member types that are themselves served structs become nested references,
// keyed by the type's own identifier (build_structs serves them under it).
consteval bool is_served_struct(std::meta::info t) { return t == ^^Move || t == ^^SimObservation; }

// std::to_string is not yet constexpr in this libstdc++.
consteval std::string dec(std::size_t v) {
  std::string s;
  do {
    s.insert(s.begin(), static_cast<char>('0' + v % 10));
    v /= 10;
  } while (v != 0);
  return s;
}

// The dtype half of one field: a nested reference, a text field (char array
// -> numpy fixed bytes), an opaque aggregate readers skip over (Rack -> numpy
// void), a subarray, or a scalar. A member type none of those cover is a
// compile-time error naming the gap.
consteval void describe_type(FieldInfo* f, std::meta::info type, std::size_t member_size) {
  std::meta::info t = std::meta::dealias(type);
  if (is_served_struct(t)) {
    f->struct_ref = std::define_static_string(std::meta::identifier_of(t));
    return;
  }
  if (std::meta::is_array_type(t)) {
    if (std::meta::remove_all_extents(t) != ^^char) throw "non-char array member in format layout";
    f->dtype = std::define_static_string("S" + dec(member_size));
    return;
  }
  if (t == ^^Rack) {
    f->dtype = std::define_static_string("V" + dec(member_size));
    return;
  }
  if (std::meta::has_template_arguments(t) && std::meta::template_of(t) == ^^std::array) {
    const auto args = std::meta::template_arguments_of(t);
    f->count = std::meta::extract<std::size_t>(args[1]);
    t = std::meta::dealias(args[0]);
  }
  f->dtype = scalar_code(t);
  if (!f->dtype) throw "unmapped member type in format layout";
}

// Move's private members end in '_'; the on-disk field names do not.
consteval const char* clean_name(std::string_view raw) {
  std::string n(raw);
  if (!n.empty() && n.back() == '_') n.pop_back();
  return std::define_static_string(n);
}

template <typename T>
consteval std::size_t num_members() {
  return std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()).size();
}

template <typename T>
consteval auto reflect_fields() {
  std::array<FieldInfo, num_members<T>()> out{};
  const auto members =
    std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked());
  for (std::size_t i = 0; i < members.size(); ++i) {
    const auto m = members[i];
    out[i].name = clean_name(std::meta::identifier_of(m));
    out[i].offset = static_cast<std::size_t>(std::meta::offset_of(m).bytes);
    describe_type(&out[i], std::meta::type_of(m), std::meta::size_of(m));
  }
  return out;
}

template <typename T>
json::object struct_json() {
  static constexpr auto fields = reflect_fields<T>();
  json::array out;
  for (const FieldInfo& fi : fields) {
    json::object f;
    f["name"] = fi.name;
    f["offset"] = fi.offset;
    if (fi.struct_ref) {
      f["dtype"] = json::object{{"struct", fi.struct_ref}};
    } else {
      f["dtype"] = fi.dtype;
    }
    if (fi.count > 0) f["shape"] = json::array{fi.count};
    out.push_back(std::move(f));
  }
  return {{"itemsize", sizeof(T)}, {"fields", std::move(out)}};
}

json::object build_structs() {
  json::object s;
  s["Move"] = struct_json<Move>();
  s["SimObservation"] = struct_json<SimObservation>();
  s["SobsFileHeader"] = struct_json<SimObsFileHeader>();
  s["SobsPositionHeader"] = struct_json<SimObsPositionHeader>();
  s["SobsRecord"] = struct_json<SimObsRecord>();
  s["MsetFileHeader"] = struct_json<move_set_eval::TargetFileHeader>();
  s["MsetPositionHeader"] = struct_json<move_set_eval::TargetPositionHeader>();
  s["SlogFileHeader"] = struct_json<binlog::FileHeader>();
  s["SlogGameMetadata"] = struct_json<binlog::GameMetadata>();
  s["SlogInitialRacks"] = struct_json<binlog::InitialRacks>();
  s["SlogTurnBlob"] = struct_json<binlog::TurnBlob>();
  return s;
}

json::object build_constants() {
  json::object c;
  c["board_size"] = BOARD_SIZE;
  c["slog"] = {{"magic", binlog::kMagic},
               {"version", binlog::kVersion},
               {"flag_face_up_leaves", binlog::kFlagFaceUpLeaves}};
  c["sobs"] = {{"magic", kSimObsMagic},
               {"version", kSimObsVersion},
               {"flag_retired_open_rack", 1},
               {"flag_open_leaves", kSimObsFlagOpenLeaves}};
  {
    json::array target_names;
    for (const char* name : move_set_eval::kTargetNamesV1) target_names.emplace_back(name);
    c["mset"] = {{"magic", move_set_eval::kTargetMagic},
                 {"version", move_set_eval::kTargetVersion},
                 {"flag_open_leaves", move_set_eval::kTargetFlagOpenLeaves},
                 {"flag_full_sweep", move_set_eval::kTargetFlagFullSweep},
                 {"target_names_v1", std::move(target_names)},
                 {"planes", move_set_eval::kTargetPlanes},
                 {"plane_cells", move_set_eval::kPlaneCells}};
  }
  c["move_type"] = {{"play", static_cast<int>(MoveType::PLAY)},
                    {"exchange", static_cast<int>(MoveType::EXCHANGE)},
                    {"pass", static_cast<int>(MoveType::PASS)}};
  // The four placement heads in declaration order (training_targets.h) --
  // also the .mset plane order and the SimObservation count-plane order.
  c["placement_head_names"] = {OppNextPlacementTarget::kName, SelfNextPlacementTarget::kName,
                               OppWinPlacementTarget::kName, SelfWinPlacementTarget::kName};
  return c;
}

}  // namespace

const std::string& format_layout_json() {
  static const std::string doc =
    json::serialize(json::value{{"structs", build_structs()}, {"constants", build_constants()}});
  return doc;
}

}  // namespace scribblez
