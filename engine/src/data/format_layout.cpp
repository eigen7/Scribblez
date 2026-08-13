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
#include <string>
#include <type_traits>

namespace scribblez {

namespace {

namespace json = boost::json;

// The numpy dtype code of one scalar member type, little-endian to match the
// on-disk packed structs. One-byte enums and wrappers (MoveType, Glyph, bool)
// serialize as their underlying byte; Rack is opaque to readers and maps to a
// numpy void of its size.
template <typename T>
constexpr const char* dtype_code() {
  if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, Glyph> ||
                std::is_same_v<T, MoveType> || std::is_same_v<T, bool>) {
    return "u1";
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return "i1";
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return "<u2";
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return "<i2";
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return "<u4";
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return "<i4";
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return "<u8";
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return "<i8";
  } else if constexpr (std::is_same_v<T, float>) {
    return "<f4";
  } else {
    static_assert(std::is_same_v<T, Rack>, "member type without a dtype code");
    return nullptr;  // handled by size below
  }
}

template <typename T>
struct MemberDtype {
  static void describe(json::object* f) {
    if constexpr (std::is_same_v<T, Rack>) {
      (*f)["dtype"] = "V" + std::to_string(sizeof(Rack));
    } else {
      (*f)["dtype"] = dtype_code<T>();
    }
  }
};

// char[N] is text (the model-hash field): numpy fixed bytes.
template <std::size_t N>
struct MemberDtype<char[N]> {
  static void describe(json::object* f) { (*f)["dtype"] = "S" + std::to_string(N); }
};

template <typename E, std::size_t N>
struct MemberDtype<std::array<E, N>> {
  static void describe(json::object* f) {
    (*f)["dtype"] = dtype_code<E>();
    (*f)["shape"] = json::array{N};
  }
};

template <typename T>
json::object field(const char* name, std::size_t offset) {
  json::object f;
  f["name"] = name;
  f["offset"] = offset;
  MemberDtype<T>::describe(&f);
  return f;
}

// A field whose type is another described struct; the reader resolves the
// reference into a nested dtype.
json::object struct_field(const char* name, std::size_t offset, const char* struct_name) {
  json::object f;
  f["name"] = name;
  f["offset"] = offset;
  f["dtype"] = json::object{{"struct", struct_name}};
  return f;
}

// The stringized member is the reader-facing field name, so it must be the
// clean on-disk name -- use FIELD only on structs whose members carry no
// trailing underscore; FIELD_AS names the field explicitly.
#define FIELD(Struct, member) field<decltype(Struct::member)>(#member, offsetof(Struct, member))
#define FIELD_AS(Struct, member, name) \
  field<decltype(Struct::member)>(name, offsetof(Struct, member))

json::object struct_json(std::size_t itemsize, json::array fields) {
  return {{"itemsize", itemsize}, {"fields", std::move(fields)}};
}

json::object glyph_constants() {
  // Computed through the Glyph API rather than restated, so the code table
  // stays wherever glyph.h puts it. reserved_/padding bytes are not fields.
  const Tile a = Tile::from_char('A');
  const Tile z = Tile::from_char('Z');
  return {{"empty", Glyph::empty().code()},
          {"letter_min", Glyph::of(a).code()},
          {"letter_max", Glyph::of(z).code()},
          {"designated_blank_min", Glyph::of_blank(a).code()},
          {"designated_blank_max", Glyph::of_blank(z).code()},
          {"blank", Glyph::blank().code()}};
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
  c["glyph"] = glyph_constants();
  // The four placement heads in declaration order (training_targets.h) --
  // also the .mset plane order and the SimObservation count-plane order.
  c["placement_head_names"] = {OppNextPlacementTarget::kName, SelfNextPlacementTarget::kName,
                               OppWinPlacementTarget::kName, SelfWinPlacementTarget::kName};
  return c;
}

json::object build_structs();  // needs FormatFieldAccess, defined after it

json::value build() { return {{"structs", build_structs()}, {"constants", build_constants()}}; }

}  // namespace

// Move's members are private; this friend (declared in move.h) is the one
// place outside the class that reads their offsets.
struct FormatFieldAccess {
  static json::array move_fields() {
    return {FIELD_AS(Move, type_, "type"),     FIELD_AS(Move, horizontal_, "horizontal"),
            FIELD_AS(Move, start_, "start"),   FIELD_AS(Move, num_played_, "num_played"),
            FIELD_AS(Move, glyphs_, "glyphs"), FIELD_AS(Move, square_mask_, "square_mask"),
            FIELD_AS(Move, score_, "score")};
  }
};

namespace {

json::object build_structs() {
  using binlog::FileHeader;
  using binlog::GameMetadata;
  using binlog::InitialRacks;
  using binlog::TurnBlob;
  using move_set_eval::TargetFileHeader;
  using move_set_eval::TargetPositionHeader;

  json::object s;
  s["Move"] = struct_json(sizeof(Move), FormatFieldAccess::move_fields());
  s["SimObservation"] =
    struct_json(sizeof(SimObservation),
                {FIELD(SimObservation, n), FIELD(SimObservation, wins),
                 FIELD(SimObservation, draws), FIELD(SimObservation, losses),
                 FIELD(SimObservation, delta_sum), FIELD(SimObservation, delta_sq_sum),
                 FIELD(SimObservation, opp_next_count), FIELD(SimObservation, self_next_count),
                 FIELD(SimObservation, opp_win_count), FIELD(SimObservation, self_win_count)});
  s["SobsFileHeader"] =
    struct_json(sizeof(SimObsFileHeader),
                {FIELD(SimObsFileHeader, magic), FIELD(SimObsFileHeader, version),
                 FIELD(SimObsFileHeader, reserved), FIELD(SimObsFileHeader, num_positions),
                 FIELD(SimObsFileHeader, flags)});
  s["SobsPositionHeader"] =
    struct_json(sizeof(SimObsPositionHeader),
                {FIELD(SimObsPositionHeader, game_index), FIELD(SimObsPositionHeader, turn_index),
                 FIELD(SimObsPositionHeader, num_candidates), FIELD(SimObsPositionHeader, rollouts),
                 FIELD(SimObsPositionHeader, base_seed)});
  s["SobsRecord"] = struct_json(
    sizeof(SimObsRecord), {struct_field("move", offsetof(SimObsRecord, move), "Move"),
                           struct_field("obs", offsetof(SimObsRecord, obs), "SimObservation")});
  s["MsetFileHeader"] =
    struct_json(sizeof(TargetFileHeader),
                {FIELD(TargetFileHeader, magic), FIELD(TargetFileHeader, version),
                 FIELD(TargetFileHeader, reserved), FIELD(TargetFileHeader, num_positions),
                 FIELD(TargetFileHeader, record_floats), FIELD(TargetFileHeader, record_planes),
                 FIELD(TargetFileHeader, flags), FIELD(TargetFileHeader, model_hash)});
  s["MsetPositionHeader"] = struct_json(
    sizeof(TargetPositionHeader),
    {FIELD(TargetPositionHeader, game_index), FIELD(TargetPositionHeader, turn_index),
     FIELD(TargetPositionHeader, num_candidates), FIELD(TargetPositionHeader, num_legal_moves)});
  s["SlogFileHeader"] =
    struct_json(sizeof(FileHeader),
                {FIELD(FileHeader, magic), FIELD(FileHeader, version), FIELD(FileHeader, flags),
                 FIELD(FileHeader, num_games), FIELD(FileHeader, num_sample_positions)});
  s["SlogGameMetadata"] =
    struct_json(sizeof(GameMetadata),
                {FIELD(GameMetadata, start_offset), FIELD(GameMetadata, num_turns),
                 FIELD(GameMetadata, sampled_turn), FIELD(GameMetadata, final_score_p0),
                 FIELD(GameMetadata, final_score_p1), FIELD(GameMetadata, initial_score_p0),
                 FIELD(GameMetadata, initial_score_p1), FIELD(GameMetadata, eligible_begin),
                 FIELD(GameMetadata, eligible_end)});
  s["SlogInitialRacks"] =
    struct_json(sizeof(InitialRacks), {FIELD(InitialRacks, p0), FIELD(InitialRacks, p1)});
  s["SlogTurnBlob"] =
    struct_json(sizeof(TurnBlob),
                {struct_field("move", offsetof(TurnBlob, move), "Move"), FIELD(TurnBlob, drawn)});
  return s;
}

}  // namespace

const std::string& format_layout_json() {
  static const std::string doc = json::serialize(build());
  return doc;
}

}  // namespace scribblez
