add_library(scribblez_core STATIC
  src/tile.cpp
  src/tile_counts.cpp
  src/rack.cpp
  src/bag.cpp
  src/board.cpp
  src/move.cpp
  src/dictionary.cpp
  src/movegen.cpp
  src/seed_producer.cpp
  src/game.cpp
  src/gcg_writer.cpp
)
target_include_directories(scribblez_core PUBLIC include)

# Concrete Agent implementations live in play_game (not scribblez_core) because
# they pull in Boost (program_options for from_spec(), boost::json/process for
# the web human agent and macondo bot). scribblez_core only sees the Agent ABC.
add_executable(play_game
  apps/play_game.cpp
  src/agent.cpp
  src/human_web_agent.cpp
  src/macondo_bot.cpp
  src/player_factory.cpp
  src/web_server.cpp)
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
