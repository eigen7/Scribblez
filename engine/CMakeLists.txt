add_library(scribblez_core STATIC
  src/scribblez/tile.cpp
  src/scribblez/tile_counts.cpp
  src/scribblez/rack.cpp
  src/scribblez/bag.cpp
  src/scribblez/board.cpp
  src/scribblez/move.cpp
  src/scribblez/dictionary.cpp
  src/scribblez/movegen.cpp
  src/scribblez/lane_targets.cpp
  src/scribblez/board_planes.cpp
  src/scribblez/lexical_input_encoder.cpp
  src/scribblez/lexical_task.cpp
  src/scribblez/game.cpp
  src/scribblez/gcg_reader.cpp
  src/scribblez/gcg_writer.cpp
  src/scribblez/binary_log.cpp
  src/scribblez/game_state_encoder.cpp
  src/scribblez/training_targets.cpp
  src/scribblez/word_map.cpp
  src/scribblez/training_task.cpp
  src/scribblez/position_encoder.cpp
  src/scribblez/streaming_row_buffer.cpp
  src/scribblez/streaming_game_producer.cpp
  src/scribblez/block_decoder.cpp
  src/scribblez/slog_subset.cpp
  src/scribblez/position_json.cpp
  src/scribblez/data_loader.cpp
  src/scribblez/unique_id.cpp
  src/scribblez/agent.cpp
  src/scribblez/agent_options.cpp
  src/scribblez/cli.cpp
  src/scribblez/self_play_engine.cpp
  src/scribblez/game_runner.cpp
  src/scribblez/hasty_equity.cpp
  src/scribblez/human_web_agent.cpp
  src/scribblez/instance_ports.cpp
  src/scribblez/leave_values.cpp
  src/scribblez/lexicon.cpp
  src/scribblez/macondo_bot.cpp
  src/scribblez/player_factory.cpp
  src/scribblez/seed_producer.cpp
  src/scribblez/web_server.cpp
  src/util/encoding.cpp
  src/util/crypto.cpp
  src/util/io.cpp
  src/util/math.cpp
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
target_link_libraries(scribblez_core PUBLIC Eigen3::Eigen)

# Plain C ABI for Python (cffi) consumption. Statically links scribblez_core
# so the resulting libscribblez_ffi.so is a single self-contained artifact
# that exposes only the symbols declared in scribblez/scribblez_ffi.h.
# The streaming pipeline plays self-play games inside the library, so it pulls
# in the agent/runner code that parses --player specs via Boost.ProgramOptions.
# player_factory.cpp (in scribblez_core) dispatches to NeuralAgent::from_spec,
# so the neural agent TUs and scribblez_nn must also be linked here.
add_library(scribblez_ffi SHARED
  src/scribblez/scribblez_ffi.cpp
  src/scribblez/neural_agent.cpp
  src/scribblez/neural_agent_factory.cpp)
target_link_libraries(scribblez_ffi PRIVATE
  scribblez_core
  scribblez_nn
  Boost::program_options
  Boost::filesystem
  Boost::system
  Threads::Threads)

# --- Neural-net inference (TensorRT) -----------------------------------------
# The post-move value model (M_post) runs on the GPU through TensorRT. Discover
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
  src/scribblez/nn/cuda_util.cpp
  src/scribblez/nn/trt_util.cpp
  src/scribblez/nn/neural_net.cpp
  src/scribblez/nn/nn_evaluation_service.cpp)
set_target_properties(scribblez_nn PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(scribblez_nn
  PUBLIC include ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CUDA_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(scribblez_nn PUBLIC
  ${CUDA_CUDART} ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER})
target_link_libraries(scribblez_nn PUBLIC Eigen3::Eigen)

# The NeuralAgent TUs are the only agent sources not already in scribblez_core;
# they are kept out of the core library so it carries no CUDA/TensorRT
# dependency. Compile them directly into play_game and link scribblez_nn, which
# supplies the NNEvaluationService the agent drives. Every other agent
# (HastyBot, human, greedy) and the runner come from scribblez_core.
add_executable(play_game
  apps/play_game.cpp
  src/scribblez/neural_agent.cpp
  src/scribblez/neural_agent_factory.cpp)
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

# Agent unit tests: NeuralAgent selection in both top-K and all-moves modes
# (with a scripted EvalService stub), chunked evaluation, and encode_candidate
# parity. Compiles the agent's selection TU (not the factory) directly. The
# factory TU holds the production constructor and make_service(), the only
# references to NNEvaluationService, so this binary links neither scribblez_nn
# nor CUDA/TensorRT: the tests use the injecting constructor and never touch the
# GPU.
add_executable(test_neural_agent
  tests/test_neural_agent.cpp
  src/scribblez/neural_agent.cpp
  src/scribblez/hasty_equity.cpp
  src/scribblez/leave_values.cpp)
target_link_libraries(test_neural_agent PRIVATE
  scribblez_core Boost::json Threads::Threads)
add_test(NAME test_neural_agent COMMAND test_neural_agent)

# TensorRT-vs-PyTorch inference parity (Suite 1 hop B). Run with no arguments:
# it self-generates a fixture by invoking the Python generator at SCRIBBLEZ_PY_DIR
# (skips cleanly if torch/onnx is absent) and needs a GPU at run time. A fixture
# directory may be passed as argv[1] to reuse one instead.
add_executable(test_nn_inference_parity tests/test_nn_inference_parity.cpp)
target_link_libraries(test_nn_inference_parity PRIVATE
  scribblez_nn scribblez_core Threads::Threads)
target_compile_definitions(test_nn_inference_parity PRIVATE
  SCRIBBLEZ_PY_DIR="${CMAKE_SOURCE_DIR}/py")
add_test(NAME test_nn_inference_parity COMMAND test_nn_inference_parity)

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
