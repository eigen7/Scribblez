add_library(scribblez_core STATIC
  src/game/tile.cpp
  src/game/tile_counts.cpp
  src/game/rack.cpp
  src/game/bag.cpp
  src/game/board.cpp
  src/game/move.cpp
  src/lexicon/dictionary.cpp
  src/game/movegen.cpp
  src/endgame/endgame_solver.cpp
  src/endgame/outplays.cpp
  src/endgame/path_move_lists.cpp
  src/training/lane_targets.cpp
  src/training/lane_analysis.cpp
  src/encoding/contingent_map.cpp
  src/encoding/input_encoder.cpp
  src/encoding/board_planes.cpp
  src/training/max_move_per_lane_input_encoder.cpp
  src/training/max_move_per_lane_task.cpp
  src/encoding/row_encoder.cpp
  src/game/game.cpp
  src/data/gcg_reader.cpp
  src/data/gcg_writer.cpp
  src/data/binary_log.cpp
  src/encoding/game_state_encoder.cpp
  src/training/training_targets.cpp
  src/lexicon/word_map.cpp
  src/training/training_task.cpp
  src/encoding/position_encoder.cpp
  src/data/streaming_row_buffer.cpp
  src/selfplay/streaming_game_producer.cpp
  src/data/block_decoder.cpp
  src/data/slog_subset.cpp
  src/serve/position_json.cpp
  src/data/data_loader.cpp
  src/agent/agent.cpp
  src/agent/agent_options.cpp
  src/selfplay/self_play_engine.cpp
  src/selfplay/game_runner.cpp
  src/lexicon/hasty_equity.cpp
  src/agent/human_web_agent.cpp
  src/serve/service_url.cpp
  src/lexicon/leave_values.cpp
  src/lexicon/lexicon.cpp
  src/agent/macondo_bot.cpp
  src/agent/endgame_hasty_bot.cpp
  src/selfplay/monte_carlo_sim.cpp
  src/selfplay/sim_runner.cpp
  src/selfplay/sim_observation_log.cpp
  src/training/move_set_eval_target_log.cpp
  src/training/move_set_encoder.cpp
  src/agent/player_factory.cpp
  src/training/position_eval_analysis.cpp
  src/selfplay/seed_producer.cpp
  src/serve/web_server.cpp
  src/util/string.cpp
  src/util/io.cpp
  src/util/math.cpp
  src/util/misc.cpp
  src/util/progress.cpp
)
# Compiled with PIC so the shared `scribblez_ffi` library can statically link
# in the core engine on Linux.
set_target_properties(scribblez_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(scribblez_core PUBLIC
  include
  # Engine root: lets headers do `#include "inlines/<category>/foo.inl"` for
  # out-of-class inline definitions kept out of the public header.
  ${CMAKE_CURRENT_SOURCE_DIR}
)
# position_json serializes the GameState schema with Boost.Json; expose it to
# every consumer of the core library (FFI, web server, tests).
target_link_libraries(scribblez_core PUBLIC Boost::json)
target_link_libraries(scribblez_core PUBLIC Eigen3::Eigen)

# Plain C ABI for Python (cffi) consumption. Statically links scribblez_core
# so the resulting libscribblez_ffi.so is a single self-contained artifact
# that exposes only the symbols declared in serve/scribblez_ffi.h.
# The streaming pipeline plays self-play games inside the library, so it pulls
# in the agent/runner code that parses --player specs via Boost.ProgramOptions.
# player_factory.cpp (in scribblez_core) dispatches to NeuralAgent::from_spec,
# so the neural agent TUs and scribblez_nn must also be linked here.
add_library(scribblez_ffi SHARED
  src/serve/scribblez_ffi.cpp
  src/agent/neural_agent.cpp
  src/agent/neural_agent_factory.cpp)
target_link_libraries(scribblez_ffi PRIVATE
  scribblez_core
  scribblez_nn
  Boost::program_options
  Boost::filesystem
  Boost::system
  Threads::Threads)

# --- Neural-net inference (TensorRT) -----------------------------------------
# The position evaluation model runs on the GPU through TensorRT. Discover
# CUDA and TensorRT assuming the standard NVIDIA TensorRT container layout (the
# project always runs inside that container -- see CLAUDE.md). The CUDA runtime
# is used purely as a host library, so no CUDA language needs to be enabled
# (and we avoid find_package(CUDAToolkit), which would require CMake >= 3.17).
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
  HINTS /usr/local/cuda/include /usr/include)
find_library(CUDA_CUDART cudart
  HINTS /usr/local/cuda/lib64 /usr/local/cuda/lib /usr/lib/x86_64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include)
find_library(TENSORRT_NVINFER nvinfer
  HINTS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)
find_library(TENSORRT_NVONNXPARSER nvonnxparser
  HINTS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)
if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_CUDART)
  message(FATAL_ERROR "CUDA runtime not found (need cuda_runtime_api.h, libcudart).")
endif()
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER OR NOT TENSORRT_NVONNXPARSER)
  message(FATAL_ERROR
    "TensorRT not found (need NvInfer.h, libnvinfer, libnvonnxparser). "
    "This project builds inside the NVIDIA TensorRT container.")
endif()

# A self-contained TensorRT inference layer: the NeuralNet engine wrapper and
# the synchronous NNEvaluationService. Kept out of scribblez_core so the core
# game library carries no CUDA/TensorRT dependency. The NvInfer headers are
# confined to this library's .cpp files (the public headers are pimpl/free of
# them), so consumers link it without needing the TensorRT include path.
add_library(scribblez_nn STATIC
  src/nn/cuda_util.cpp
  src/nn/trt_util.cpp
  src/nn/neural_net.cpp
  src/nn/nn_evaluation_service.cpp)
set_target_properties(scribblez_nn PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(scribblez_nn
  PUBLIC include ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CUDA_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(scribblez_nn PUBLIC
  ${CUDA_CUDART} ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER})
target_link_libraries(scribblez_nn PUBLIC Eigen3::Eigen Boost::program_options)
# Reading the exported model's metadata_props (the input-encoding arm) parses
# the ONNX protobuf directly. The system onnx package ships the ML variant of
# the generated headers, selected via ONNX_ML.
find_package(Protobuf REQUIRED)
find_library(ONNX_PROTO_LIB onnx_proto REQUIRED)
target_compile_definitions(scribblez_nn PRIVATE ONNX_ML=1)
target_link_libraries(scribblez_nn PUBLIC ${ONNX_PROTO_LIB} protobuf::libprotobuf)

# The NeuralAgent TUs are the only agent sources not already in scribblez_core;
# they are kept out of the core library so it carries no CUDA/TensorRT
# dependency. Compile them directly into play_game and link scribblez_nn, which
# supplies the NNEvaluationService the agent drives. Every other agent
# (HastyBot, human, greedy) and the runner come from scribblez_core.
add_executable(play_game
  apps/play_game.cpp
  src/agent/neural_agent.cpp
  src/agent/neural_agent_factory.cpp)
target_link_libraries(play_game PRIVATE
  scribblez_core
  scribblez_nn
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

add_executable(board_tool apps/board_tool.cpp)
target_link_libraries(board_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Solve one GCG endgame position with a verbose trace of the solver's
# reasoning (out-play futility view, iteration verdicts, certificate walk).
add_executable(endgame_tool apps/endgame_tool.cpp)
target_link_libraries(endgame_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Endgame-solver cost / whole-game-throughput benchmark (docs: phase-B endgame
# solver). Pure CPU: no neural path, so it does not link scribblez_nn.
add_executable(endgame_bench apps/endgame_bench.cpp)
target_link_libraries(endgame_bench PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# One-off Monte-Carlo ground-truth generator for the position-evaluation test dataset.
add_executable(monte_carlo_sim_tool apps/monte_carlo_sim_tool.cpp)
target_link_libraries(monte_carlo_sim_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Sim-observation (.sobs) generator for .slog data: HastyBot-equity top-K
# candidates per sampled position, simmed by SimRunner (docs/sim_residual_feedback.md).
add_executable(sim_obs_tool apps/sim_obs_tool.cpp)
target_link_libraries(sim_obs_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

add_executable(harvest_bingo_positions_tool apps/harvest_bingo_positions_tool.cpp)
target_link_libraries(harvest_bingo_positions_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# gtest_discover_tests registers each TEST() in a binary as its own ctest
# entry, so ctest reports and filters at single-test granularity.
include(GoogleTest)

add_executable(scribblez_tests
  tests/tests_main.cpp
  src/lexicon/hasty_equity.cpp
  src/lexicon/leave_values.cpp)
target_link_libraries(scribblez_tests PRIVATE scribblez_core Boost::json GTest::gtest_main)
# Point the tests at the container's lexicon mount (populated by
# setup_wizard.py) so they cross-validate against the real GADDAG when it is
# installed (the .kwg is not committed; the tests skip gracefully if absent).
target_compile_definitions(scribblez_tests PRIVATE
  SCRIBBLEZ_DEFAULT_KWG="/workspace/mount/lexica/NWL23.kwg")
gtest_discover_tests(scribblez_tests)

# GCG reader tests: rack-pragma parsing and the replay/snapshot reconstruction
# that reopens a recorded game.
add_executable(test_gcg_reader tests/test_gcg_reader.cpp)
target_link_libraries(test_gcg_reader PRIVATE scribblez_core Boost::json GTest::gtest_main)
gtest_discover_tests(test_gcg_reader)

# Endgame solver tests: Board make/unmake round-trips and EndgameSolver
# correctness (brute-force differential, oracle positions, determinism, node
# budget, cross-turn transposition-table reuse). Gated lexicon cases read the
# same container mount as scribblez_tests and skip gracefully when it is absent.
add_executable(test_endgame tests/test_endgame.cpp tests/test_endgame_agent.cpp)
target_link_libraries(test_endgame PRIVATE
  scribblez_core Boost::program_options GTest::gtest_main)
target_compile_definitions(test_endgame PRIVATE
  SCRIBBLEZ_DEFAULT_KWG="/workspace/mount/lexica/NWL23.kwg"
  SCRIBBLEZ_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
gtest_discover_tests(test_endgame)

# Agent unit tests: NeuralAgent selection in both top-K and all-moves modes
# (with a scripted EvalService stub), chunked evaluation, and encode_candidate
# parity. Compiles the agent's selection TU (not the factory) directly. The
# factory TU holds the production constructor and make_service(), the only
# references to NNEvaluationService, so this binary links neither scribblez_nn
# nor CUDA/TensorRT: the tests use the injecting constructor and never touch the
# GPU.
add_executable(test_neural_agent
  tests/test_neural_agent.cpp
  src/agent/neural_agent.cpp
  src/lexicon/hasty_equity.cpp
  src/lexicon/leave_values.cpp)
target_link_libraries(test_neural_agent PRIVATE
  scribblez_core Boost::json Threads::Threads GTest::gtest_main)
gtest_discover_tests(test_neural_agent)

# TensorRT-vs-PyTorch inference parity (Suite 1 hop B). Run with no arguments:
# it self-generates a fixture by invoking the Python generator at SCRIBBLEZ_PY_DIR
# (skips cleanly if torch/onnx is absent) and needs a GPU at run time. A fixture
# directory may be passed as the first non-gtest argument to reuse one instead,
# so this binary carries its own main() and links GTest::gtest, not gtest_main.
add_executable(test_nn_inference_parity tests/test_nn_inference_parity.cpp)
target_link_libraries(test_nn_inference_parity PRIVATE
  scribblez_nn scribblez_core Threads::Threads GTest::gtest)
target_compile_definitions(test_nn_inference_parity PRIVATE
  SCRIBBLEZ_PY_DIR="${CMAKE_SOURCE_DIR}/py")
gtest_discover_tests(test_nn_inference_parity)

# Standalone ad-hoc driver for the binary-log DataLoader. Lets us point it at
# a directory of .slog files and verify load() produces well-formed rows.
add_executable(dataloader_smoke apps/dataloader_smoke.cpp)
target_link_libraries(dataloader_smoke PRIVATE scribblez_core Threads::Threads)

# Test helper: writes .slog files using an in-memory dictionary for integration
# testing (Python FFI tests, etc.).
add_executable(test_slog_writer apps/test_slog_writer.cpp)
target_link_libraries(test_slog_writer PRIVATE scribblez_core Threads::Threads)

# Standalone TensorRT inference sanity check: load an ONNX model, run a batch of
# zero inputs, print WLD + ScoreDiff outputs. Exercises the engine path without
# any game logic.
add_executable(nn_infer_smoke apps/nn_infer_smoke.cpp)
target_link_libraries(nn_infer_smoke PRIVATE scribblez_nn Threads::Threads)

# Move set evaluation model distillation-target generator: stratified
# candidate sampling + batched teacher (position evaluation model) inference
# over .slog positions (docs/roadmap2.md track A).
add_executable(move_set_eval_target_generator apps/move_set_eval_target_generator.cpp)
target_link_libraries(move_set_eval_target_generator PRIVATE
  scribblez_nn
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)
