add_library(scribblez_core STATIC
  src/scribblez/tile.cpp
  src/scribblez/tile_counts.cpp
  src/scribblez/rack.cpp
  src/scribblez/bag.cpp
  src/scribblez/board.cpp
  src/scribblez/move.cpp
  src/scribblez/dictionary.cpp
  src/scribblez/movegen.cpp
  src/scribblez/game.cpp
  src/scribblez/gcg_writer.cpp
  src/scribblez/binary_log.cpp
  src/scribblez/game_state_encoder.cpp
  src/scribblez/training_targets.cpp
  src/scribblez/position_encoder.cpp
  src/scribblez/streaming_row_buffer.cpp
  src/scribblez/streaming_game_producer.cpp
  src/scribblez/block_decoder.cpp
  src/scribblez/slog_subset.cpp
  src/scribblez/position_json.cpp
  src/scribblez/data_loader.cpp
  src/scribblez/unique_id.cpp
  src/scribblez/agent.cpp
  src/scribblez/cli.cpp
  src/scribblez/self_play_engine.cpp
  src/scribblez/game_runner.cpp
  src/scribblez/hasty_equity.cpp
  src/scribblez/human_web_agent.cpp
  src/scribblez/leave_values.cpp
  src/scribblez/lexicon.cpp
  src/scribblez/macondo_bot.cpp
  src/scribblez/player_factory.cpp
  src/scribblez/seed_producer.cpp
  src/scribblez/web_server.cpp
)
# Compiled with PIC so the shared `scribblez_ffi` library can statically link
# in the core engine on Linux.
set_target_properties(scribblez_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(scribblez_core PUBLIC
  include
  # Engine root: lets headers do `#include "inlines/scribblez/foo.inl"` for
  # out-of-class inline definitions kept out of the public header.
  ${CMAKE_CURRENT_SOURCE_DIR}
)
# position_json serializes the GameState schema with Boost.Json; expose it to
# every consumer of the core library (FFI, web server, tests).
target_link_libraries(scribblez_core PUBLIC Boost::json)

# Plain C ABI for Python (cffi) consumption. Statically links scribblez_core
# so the resulting libscribblez_ffi.so is a single self-contained artifact
# that exposes only the symbols declared in scribblez/scribblez_ffi.h.
# The streaming pipeline plays self-play games inside the library, so it pulls
# in the agent/runner code that parses --player specs via Boost.ProgramOptions.
add_library(scribblez_ffi SHARED src/scribblez/scribblez_ffi.cpp)
target_link_libraries(scribblez_ffi PRIVATE
  scribblez_core
  Boost::program_options
  Boost::filesystem
  Boost::system
  Threads::Threads)

add_executable(play_game apps/play_game.cpp)
target_link_libraries(play_game PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

add_executable(manual_gcg_tool apps/manual_gcg_tool.cpp)
target_link_libraries(manual_gcg_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

add_executable(scribblez_tests
  tests/tests_main.cpp
  src/scribblez/hasty_equity.cpp
  src/scribblez/leave_values.cpp)
target_link_libraries(scribblez_tests PRIVATE scribblez_core Boost::json)
# Point the tests at the conventional local lexicon path so they cross-validate
# against the real GADDAG when one is present (the .kwg is not committed; the
# test skips gracefully if absent).
target_compile_definitions(scribblez_tests PRIVATE
  SCRIBBLEZ_DEFAULT_KWG="${CMAKE_SOURCE_DIR}/data/lexica/NWL23.kwg")
add_test(NAME scribblez_tests COMMAND scribblez_tests)

# Standalone ad-hoc driver for the binary-log DataLoader. Lets us point it at
# a directory of .slog files and verify load() produces well-formed rows.
add_executable(dataloader_smoke apps/dataloader_smoke.cpp)
target_link_libraries(dataloader_smoke PRIVATE scribblez_core Threads::Threads)

# Test helper: writes .slog files using an in-memory dictionary for integration
# testing (Python FFI tests, etc.).
add_executable(test_slog_writer apps/test_slog_writer.cpp)
target_link_libraries(test_slog_writer PRIVATE scribblez_core Threads::Threads)
