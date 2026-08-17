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
  src/game/game_log.cpp
  src/data/gcg_reader.cpp
  src/data/gcg_writer.cpp
  src/data/binary_log.cpp
  src/encoding/game_state_encoder.cpp
  src/training/training_targets.cpp
  src/lexicon/word_map.cpp
  src/training/training_task.cpp
  src/encoding/position_encoder.cpp
  src/data/streaming_row_buffer.cpp
  src/arena/streaming_game_producer.cpp
  src/data/block_decoder.cpp
  src/data/slog_subset.cpp
  src/serve/position_json.cpp
  src/data/data_loader.cpp
  src/agent/agent.cpp
  src/agent/agent_options.cpp
  src/agent/greedy_agent.cpp
  src/arena/game_engine.cpp
  src/arena/game_runner.cpp
  src/lexicon/hasty_equity.cpp
  src/agent/human_web_agent.cpp
  src/serve/service_url.cpp
  src/lexicon/leave_values.cpp
  src/lexicon/lexicon.cpp
  src/agent/macondo_bot.cpp
  src/agent/endgame_turn_policy.cpp
  src/agent/endgame_hasty_bot.cpp
  src/sim/gcg_decision.cpp
  src/sim/monte_carlo_sim.cpp
  src/sim/sim_runner.cpp
  src/belief/leave_prior.cpp
  src/belief/move_likelihood.cpp
  src/belief/rack_inference.cpp
  src/data/sim_observation_log.cpp
  src/data/slog_sampling.cpp
  src/data/format_layout.cpp
  src/training/move_set_eval_target_log.cpp
  src/training/move_set_eval_candidates.cpp
  src/training/move_set_encoder.cpp
  src/agent/sim_agent.cpp
  src/agent/player_factory.cpp
  src/training/position_eval_analysis.cpp
  src/util/seed_producer.cpp
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
# misc.cpp/lexicon.cpp/the agent factories parse options with
# Boost.ProgramOptions, and web_server.cpp drives npm through Boost.Process
# (-> Boost.Filesystem/System). Declared here, where the code lives: consumers
# that only linked these transitively-by-luck survived Release LTO stripping,
# but -O0 test links surfaced the missing deps.
target_link_libraries(scribblez_core PUBLIC Boost::program_options Boost::filesystem
                                            Boost::system)

# Plain C ABI for Python (cffi) consumption. Statically links scribblez_core
# so the resulting libscribblez_ffi.so is a single self-contained artifact
# that exposes only the symbols declared in serve/scribblez_ffi.h.
# The streaming pipeline plays self-play games inside the library, so it pulls
# in the agent/runner code that parses --player specs via Boost.ProgramOptions.
# player_factory.cpp (in scribblez_core) dispatches to the model-driven agents'
# from_spec, so scribblez_neural_agents must also be linked here.
add_library(scribblez_ffi SHARED
  src/serve/scribblez_ffi.cpp)
target_link_libraries(scribblez_ffi PRIVATE
  scribblez_core
  scribblez_neural_agents
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

# A self-contained TensorRT inference layer: the NeuralNet<Spec> engine
# wrapper and the synchronous TrtEvalService<Spec>. Kept out of scribblez_core so the core
# game library carries no CUDA/TensorRT dependency. The NvInfer headers are
# confined to this library's .cpp files (the public headers are pimpl/free of
# them), so consumers link it without needing the TensorRT include path.
add_library(scribblez_nn STATIC
  src/nn/cuda_util.cpp
  src/nn/trt_util.cpp
  src/nn/onnx_metadata.cpp
  src/nn/neural_net.cpp
  src/nn/trt_eval_service.cpp)
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

# The model-driven agents (NeuralAgent, NeuralSimAgent, MsetSimAgent) and their
# shared evaluation core -- the only agent sources not in scribblez_core, kept out so
# the core game library carries no CUDA/TensorRT dependency. Every consumer
# that builds --player specs (play_game, scribblez_ffi, test_game_runner)
# links this; scribblez_nn supplies the TrtEvalService the agents drive.
# Every other agent (HastyBot, human, greedy) and the runner come from
# scribblez_core. The GPU-free unit-test binaries instead compile the
# selection TUs directly, factories excluded (see their comments below).
add_library(scribblez_neural_agents STATIC
  src/agent/candidate_evaluator.cpp
  src/agent/mset_sim_agent.cpp
  src/agent/mset_sim_agent_factory.cpp
  src/agent/neural_agent.cpp
  src/agent/neural_agent_factory.cpp
  src/agent/neural_service_options.cpp
  src/agent/neural_sim_agent.cpp
  src/agent/neural_sim_agent_factory.cpp)
set_target_properties(scribblez_neural_agents PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(scribblez_neural_agents PUBLIC
  scribblez_core
  scribblez_nn
  Boost::program_options)

add_executable(play_game
  apps/play_game.cpp)
target_link_libraries(play_game PRIVATE
  scribblez_core
  scribblez_neural_agents
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

if(SCRIBBLEZ_BUILD_TESTS)
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
  # parity. Compiles the selection TUs (not the factory) directly. The factory
  # TU holds the production constructor, whose nn::make_loaded_service() call
  # (defined in scribblez_nn's trt_eval_service.cpp) is where the concrete
  # service gets built -- by not compiling the factory, this binary links
  # neither scribblez_nn nor CUDA/TensorRT: the tests use the injecting
  # constructor and never touch the GPU.
  add_executable(test_neural_agent
    tests/test_neural_agent.cpp
    src/agent/candidate_evaluator.cpp
    src/agent/neural_agent.cpp
    src/lexicon/hasty_equity.cpp
    src/lexicon/leave_values.cpp)
  target_link_libraries(test_neural_agent PRIVATE
    scribblez_core Boost::json Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_neural_agent)

  # Rack inference: the hypergeometric prior and its sampler, the observations
  # that carry no information, posterior shape and reproducibility, and that
  # the leave the opponent actually held gains weight over its prior. Runs off
  # a synthetic leave table, so it needs no lexicon or Macondo data mount.
  add_executable(test_rack_inference tests/test_rack_inference.cpp)
  target_link_libraries(test_rack_inference PRIVATE
    scribblez_core Boost::json GTest::gtest_main)
  gtest_discover_tests(test_rack_inference)

  # SimAgent: that it plays the candidate its own rollouts rank best (checked by
  # replaying the decision through SimRunner), that a seed reproduces a game,
  # and that a bag-empty turn falls back to static equity. Runs off a synthetic
  # leave table, so it needs no lexicon or Macondo data mount.
  add_executable(test_sim_agent tests/test_sim_agent.cpp)
  target_link_libraries(test_sim_agent PRIVATE
    scribblez_core Boost::json Boost::program_options Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_sim_agent)

  # NeuralSimAgent, the position-evaluation-top-K agent: that the sim set is
  # the scripted model's top K (not static equity's), that the rollouts decide
  # among them, that drop-best-prob excludes the model's favourite, and that a
  # bag-empty turn falls back to static equity. Compiles the selection TUs (not
  # the factory), so like test_neural_agent it links neither scribblez_nn nor
  # CUDA/TensorRT: the tests inject a scripted EvalService stub.
  add_executable(test_neural_sim_agent
    tests/test_neural_sim_agent.cpp
    src/agent/candidate_evaluator.cpp
    src/agent/neural_sim_agent.cpp)
  target_link_libraries(test_neural_sim_agent PRIVATE
    scribblez_core Boost::json Boost::program_options Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_neural_sim_agent)

  # MsetSimAgent, the move-set-evaluation agent: that the sim set is the
  # scripted model's top K (not static equity's), that the rollouts decide among
  # them, that the whole candidate set reaches the model in one pass carrying the
  # shared encoder's own move features, and that the pre-move board row matches
  # the training decoder's. Compiles the selection TUs (not the factory), so like
  # test_neural_sim_agent it links neither scribblez_nn nor CUDA/TensorRT: the
  # tests inject a scripted MoveSetEvalService stub.
  add_executable(test_mset_sim_agent
    tests/test_mset_sim_agent.cpp
    src/agent/candidate_evaluator.cpp
    src/agent/mset_sim_agent.cpp)
  target_link_libraries(test_mset_sim_agent PRIVATE
    scribblez_core Boost::json Boost::program_options Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_mset_sim_agent)

  # GameRunner match-eval plumbing: --paired seat/seed mirroring and the
  # --results-file per-game JSON stream. Builds real agents through
  # PlayerFactory, whose dispatch references every agent type, so it links
  # scribblez_neural_agents the same way play_game does.
  add_executable(test_game_runner
    tests/test_game_runner.cpp)
  target_link_libraries(test_game_runner PRIVATE
    scribblez_core
    scribblez_neural_agents
    Boost::program_options
    Boost::json
    Boost::filesystem
    Boost::system
    Threads::Threads
    GTest::gtest_main)
  gtest_discover_tests(test_game_runner)

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

  # the move-set runtime against the PyTorch move set evaluation model, including the
  # plan-cache refit and the service's chunking. Same shape as the parity test
  # above: self-generates its fixture through SCRIBBLEZ_PY_DIR (skipping cleanly
  # if torch/onnx is absent), needs a GPU at run time, and takes an optional
  # fixture directory as its first argument, so it carries its own main().
  add_executable(test_mset_inference_parity tests/test_mset_inference_parity.cpp)
  target_link_libraries(test_mset_inference_parity PRIVATE
    scribblez_nn scribblez_core Threads::Threads GTest::gtest)
  target_compile_definitions(test_mset_inference_parity PRIVATE
    SCRIBBLEZ_PY_DIR="${CMAKE_SOURCE_DIR}/py")
  gtest_discover_tests(test_mset_inference_parity)

  # Test helper: writes .slog files using an in-memory dictionary for integration
  # testing (Python FFI tests, etc.).
  add_executable(test_slog_writer apps/test_slog_writer.cpp)
  target_link_libraries(test_slog_writer PRIVATE scribblez_core Threads::Threads)
endif()

# Standalone ad-hoc driver for the binary-log DataLoader. Lets us point it at
# a directory of .slog files and verify load() produces well-formed rows.
add_executable(dataloader_smoke apps/dataloader_smoke.cpp)
target_link_libraries(dataloader_smoke PRIVATE scribblez_core Threads::Threads)

# Standalone TensorRT inference sanity check: load an ONNX model, run a batch of
# zero inputs, print WLD + ScoreDiff outputs. Exercises the engine path without
# any game logic.
add_executable(nn_infer_smoke apps/nn_infer_smoke.cpp)
target_link_libraries(nn_infer_smoke PRIVATE scribblez_nn scribblez_core Threads::Threads)

# The same for the move set evaluation model: load an ONNX model, score a
# synthetic candidate set, print the decoded Evals. Exercises the move-set
# path on a real checkpoint without any game logic.
add_executable(mset_infer_smoke apps/mset_infer_smoke.cpp)
target_link_libraries(mset_infer_smoke PRIVATE scribblez_nn scribblez_core Threads::Threads)

# Move set evaluation model distillation-target generator: stratified
# candidate sampling + batched teacher (position evaluation model) inference
# over .slog positions (docs/roadmap.md track A).
add_executable(move_set_eval_target_generator apps/move_set_eval_target_generator.cpp)
target_link_libraries(move_set_eval_target_generator PRIVATE
  scribblez_nn
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Evidence trajectories (docs/roadmap.md item 4): the deployment schedule's
# candidate selection (anchor, student-model proposals, uniform tail) simmed
# per decision point. The library is the position-level recipe over the
# student model; the generator supplies decision points from .slog self-play
# data or .gcg position sets and writes trajectory .sobs sidecars.
add_library(scribblez_evidence STATIC src/training/evidence_trajectory.cpp)
set_target_properties(scribblez_evidence PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(scribblez_evidence PUBLIC scribblez_core scribblez_nn)

add_executable(evidence_trajectory_generator apps/evidence_trajectory_generator.cpp)
target_link_libraries(evidence_trajectory_generator PRIVATE
  scribblez_evidence
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)
