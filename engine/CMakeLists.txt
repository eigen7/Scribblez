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
)
target_include_directories(scribblez_core PUBLIC
  include
  # Engine root: lets headers do `#include "inlines/scribblez/foo.inl"` for
  # out-of-class inline definitions kept out of the public header.
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Concrete Agent implementations live in play_game (not scribblez_core) because
# they pull in Boost (program_options for from_spec(), boost::json/process for
# the web human agent and macondo bot). scribblez_core only sees the Agent ABC.
add_executable(play_game
  apps/play_game.cpp
  src/scribblez/agent.cpp
  src/scribblez/cli.cpp
  src/scribblez/game_runner.cpp
  src/scribblez/human_web_agent.cpp
  src/scribblez/macondo_oracle.cpp
  src/scribblez/macondo_oracle_pool.cpp
  src/scribblez/macondo_bot.cpp
  src/scribblez/player_factory.cpp
  src/scribblez/seed_producer.cpp
  src/scribblez/unique_id.cpp
  src/scribblez/web_server.cpp)
target_link_libraries(play_game PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

add_executable(scribblez_tests tests/tests_main.cpp)
target_link_libraries(scribblez_tests PRIVATE scribblez_core)
# Point the tests at the conventional local lexicon path so they cross-validate
# against the real GADDAG when one is present (the .kwg is not committed; the
# test skips gracefully if absent).
target_compile_definitions(scribblez_tests PRIVATE
  SCRIBBLEZ_DEFAULT_KWG="${CMAKE_SOURCE_DIR}/data/lexica/NWL23.kwg")
add_test(NAME scribblez_tests COMMAND scribblez_tests)
