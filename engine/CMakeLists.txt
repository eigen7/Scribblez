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
  src/encoding/input_encoder.cpp
  src/encoding/board_planes.cpp
  src/training/max_move_per_lane_input_encoder.cpp
  src/training/max_move_per_lane_task.cpp
  src/game/game.cpp
  src/game/game_log.cpp
  src/data/gcg_reader.cpp
  src/data/gcg_post_move.cpp
  src/data/gcg_writer.cpp
  src/data/binary_log.cpp
  src/encoding/game_state_encoder.cpp
  src/training/training_targets.cpp
  src/training/footprint.cpp
  src/training/footprint_mask.cpp
  src/training/footprint_collapse.cpp
  src/lexicon/word_map.cpp
  src/training/training_task.cpp
  src/encoding/position_encoder.cpp
  src/data/block_decoder.cpp
  src/serve/position_json.cpp
  src/serve/client_message.cpp
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
  src/agent/hasty_bot.cpp
  src/agent/endgame_turn_policy.cpp
  src/agent/endgame_hasty_bot.cpp
  src/sim/monte_carlo_sim.cpp
  src/sim/sim_runner.cpp
  src/sim/slog_position_simmer.cpp
  src/sim/setup_plays.cpp
  src/sim/rollout_summary.cpp
  src/agent/candidate_evaluator.cpp
  # A GPU-free decorator over the EvalService interface, so it lives here and is
  # unit-tested against a scripted stub.
  src/nn/batching_position_eval_service.cpp
  src/agent/evidence_staging.cpp
  src/belief/leave_prior.cpp
  src/belief/move_likelihood.cpp
  src/belief/rack_inference.cpp
  src/data/sim_observation_log.cpp
  src/data/slog_sampling.cpp
  src/data/format_layout.cpp
  src/training/move_set_eval_target_log.cpp
  src/training/move_set_eval_candidates.cpp
  src/training/move_set_encoder.cpp
  src/training/cross_check_delta.cpp
  src/agent/sim_agent.cpp
  src/agent/weird_bot.cpp
  src/agent/player_factory.cpp
  src/training/position_eval_analysis.cpp
  src/training/trajectory_position.cpp
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
# lexicon.cpp and the agent option parsers use Boost.ProgramOptions;
# web_server.cpp drives npm through Boost.Process, which needs Boost.Filesystem
# and Boost.System. Declaring them on the library that uses them matters: a
# Release LTO link can strip the references and hide a missing dependency that
# an -O0 link then trips over.
target_link_libraries(scribblez_core PUBLIC Boost::program_options Boost::filesystem
                                            Boost::system)

# Plain C ABI for Python (cffi). Statically links the engine so
# libscribblez_ffi.so is one self-contained artifact exposing only the symbols
# declared in serve/scribblez_ffi.h.
add_library(scribblez_ffi SHARED
  src/serve/scribblez_ffi.cpp)
target_link_libraries(scribblez_ffi PRIVATE
  scribblez_core
  Boost::program_options
  Boost::filesystem
  Boost::system
  Threads::Threads)

# --- Neural-net inference (TensorRT) -----------------------------------------
# The models run on the GPU through TensorRT. CUDA and TensorRT are located at
# the standard NVIDIA TensorRT container paths, since the project always builds
# inside that container. The CUDA runtime is used only as a host library, so
# the CUDA language is never enabled.
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

# The TensorRT inference layer: the NeuralNet<Spec> engine wrapper and
# TrtEvalService<Spec>. It is kept out of scribblez_core so the game library
# has no CUDA/TensorRT dependency. NvInfer headers appear only in this
# library's .cpp files, so consumers need no TensorRT include path.
add_library(scribblez_nn STATIC
  src/nn/cuda_util.cpp
  src/nn/trt_util.cpp
  src/nn/onnx_metadata.cpp
  src/nn/neural_net.cpp
  src/nn/trt_eval_service.cpp
  # The move proposal model's two-graph (cache + step) runtime; see
  # docs/roadmap.md item 3. It drives two NeuralNets, so it lives here.
  src/agent/move_proposal_nets.cpp
  src/agent/move_proposal_session.cpp)
set_target_properties(scribblez_nn PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(scribblez_nn
  PUBLIC include ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CUDA_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(scribblez_nn PUBLIC
  ${CUDA_CUDART} ${TENSORRT_NVINFER} ${TENSORRT_NVONNXPARSER})
target_link_libraries(scribblez_nn PUBLIC Eigen3::Eigen Boost::program_options)
# For the game types, utilities, and the evidence staging the move-proposal
# runtime marshals.
target_link_libraries(scribblez_nn PUBLIC scribblez_core)
# onnx_metadata.cpp reads the exported model's metadata_props by parsing the
# ONNX protobuf directly. The system onnx package ships the ML variant of the
# generated headers, which ONNX_ML selects.
find_package(Protobuf REQUIRED)
find_library(ONNX_PROTO_LIB onnx_proto REQUIRED)
target_compile_definitions(scribblez_nn PRIVATE ONNX_ML=1)
target_link_libraries(scribblez_nn PUBLIC ${ONNX_PROTO_LIB} protobuf::libprotobuf)

# The model-driven agents (NeuralAgent, NeuralSimAgent, MsetSimAgent,
# UltimateBotAgent) plus SimAgent's factory, which can load a leaf model: every
# agent source that reaches TensorRT, kept out of scribblez_core for the same
# reason as scribblez_nn. Every consumer that parses --player specs links it.
# The factories are the only TUs that construct a TensorRT service; the agent
# unit tests below compile the other TUs directly and inject stubs, so they
# never link TensorRT.
add_library(scribblez_neural_agents STATIC
  src/agent/sim_agent_factory.cpp
  src/agent/mset_sim_agent.cpp
  src/agent/mset_sim_agent_factory.cpp
  src/agent/neural_agent.cpp
  src/agent/neural_agent_factory.cpp
  src/agent/neural_service_options.cpp
  src/agent/neural_sim_agent.cpp
  src/agent/neural_sim_agent_factory.cpp
  src/agent/ultimate_bot_agent.cpp
  src/agent/ultimate_bot_agent_factory.cpp)
set_target_properties(scribblez_neural_agents PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(scribblez_neural_agents PUBLIC
  scribblez_core
  scribblez_nn
  scribblez_evidence_select
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

# Solve one GCG endgame position with a verbose trace of the solver.
add_executable(endgame_tool apps/endgame_tool.cpp)
target_link_libraries(endgame_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Endgame solver cost and whole-game throughput benchmark
# (docs/endgame_bench_results.md). CPU-only.
add_executable(endgame_bench apps/endgame_bench.cpp)
target_link_libraries(endgame_bench PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Monte-Carlo ground truth for the position-evaluation test positions.
add_executable(monte_carlo_sim_tool apps/monte_carlo_sim_tool.cpp)
target_link_libraries(monte_carlo_sim_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Sim-observation (.sobs) sidecars for .slog data
# (docs/plans/sim_residual_feedback.md).
add_executable(sim_obs_tool apps/sim_obs_tool.cpp)
target_link_libraries(sim_obs_tool PRIVATE
  scribblez_core
  scribblez_nn
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# The sim-labeled-candidates survey (docs/plans/sim_labeled_candidates.md).
add_executable(sim_candidate_survey_tool apps/sim_candidate_survey_tool.cpp)
target_link_libraries(sim_candidate_survey_tool PRIVATE
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Rank every legal placement at one GCG position by the position evaluation
# model, scored the way NeuralAgent scores them.
add_executable(neural_rank_tool apps/neural_rank_tool.cpp)
target_link_libraries(neural_rank_tool PRIVATE
  scribblez_core
  scribblez_neural_agents
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

add_executable(harvest_positions_tool apps/harvest_positions_tool.cpp)
target_link_libraries(harvest_positions_tool PRIVATE
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
    tests/test_footprint.cpp
    tests/test_web_server.cpp)
  target_link_libraries(scribblez_tests PRIVATE
    scribblez_core scribblez_evidence_select Boost::json GTest::gtest_main)
  # The lexicon mount is populated by setup_wizard.py (the .kwg is never
  # committed); lexicon-dependent cases skip when it is absent. tests/data is
  # the frozen fixture corpus every data-reading test uses. The other
  # lexicon-gated binaries below get the same two definitions.
  target_compile_definitions(scribblez_tests PRIVATE
    SCRIBBLEZ_DEFAULT_KWG="/workspace/mount/lexica/NWL23.kwg"
    SCRIBBLEZ_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  gtest_discover_tests(scribblez_tests)

  add_executable(test_gcg_reader tests/test_gcg_reader.cpp)
  target_link_libraries(test_gcg_reader PRIVATE scribblez_core Boost::json GTest::gtest_main)
  gtest_discover_tests(test_gcg_reader)

  add_executable(test_monte_carlo_sim tests/test_monte_carlo_sim.cpp)
  target_link_libraries(test_monte_carlo_sim PRIVATE scribblez_core Boost::json GTest::gtest_main)
  target_compile_definitions(test_monte_carlo_sim PRIVATE
    SCRIBBLEZ_DEFAULT_KWG="/workspace/mount/lexica/NWL23.kwg"
    SCRIBBLEZ_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  gtest_discover_tests(test_monte_carlo_sim)

  add_executable(test_endgame tests/test_endgame.cpp tests/test_endgame_agent.cpp)
  target_link_libraries(test_endgame PRIVATE
    scribblez_core Boost::program_options GTest::gtest_main)
  target_compile_definitions(test_endgame PRIVATE
    SCRIBBLEZ_DEFAULT_KWG="/workspace/mount/lexica/NWL23.kwg"
    SCRIBBLEZ_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  gtest_discover_tests(test_endgame)

  # The model-driven agent tests compile the agent's selection TU without its
  # factory. The factory is what builds the TensorRT service
  # (nn::PositionEvalService::create() and friends, defined in scribblez_nn), so
  # leaving it out lets these binaries skip scribblez_nn and CUDA/TensorRT
  # entirely: the tests inject scripted stub services and never touch the GPU.
  add_executable(test_neural_agent
    tests/test_neural_agent.cpp
    src/agent/neural_agent.cpp)
  target_link_libraries(test_neural_agent PRIVATE
    scribblez_core Boost::json Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_neural_agent)

  add_executable(test_batching_eval_service tests/test_batching_eval_service.cpp)
  target_link_libraries(test_batching_eval_service PRIVATE
    scribblez_core Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_batching_eval_service)

  # Runs off a synthetic leave table, so it needs no lexicon or Macondo data.
  add_executable(test_rack_inference tests/test_rack_inference.cpp)
  target_link_libraries(test_rack_inference PRIVATE
    scribblez_core Boost::json GTest::gtest_main)
  gtest_discover_tests(test_rack_inference)

  # Runs off a synthetic leave table, so it needs no lexicon or Macondo data.
  add_executable(test_sim_agent tests/test_sim_agent.cpp)
  target_link_libraries(test_sim_agent PRIVATE
    scribblez_core Boost::json Boost::program_options Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_sim_agent)

  # Factory excluded, like test_neural_agent.
  add_executable(test_neural_sim_agent
    tests/test_neural_sim_agent.cpp
    src/agent/neural_sim_agent.cpp)
  target_link_libraries(test_neural_sim_agent PRIVATE
    scribblez_core Boost::json Boost::program_options Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_neural_sim_agent)

  # Factory excluded, like test_neural_agent.
  add_executable(test_mset_sim_agent
    tests/test_mset_sim_agent.cpp
    src/agent/mset_sim_agent.cpp)
  target_link_libraries(test_mset_sim_agent PRIVATE
    scribblez_core Boost::json Boost::program_options Threads::Threads GTest::gtest_main)
  gtest_discover_tests(test_mset_sim_agent)

  # Covers the evidence loop and UltimateBotAgent. Factory excluded, like
  # test_neural_agent.
  add_executable(test_ultimate_bot_agent
    tests/test_ultimate_bot_agent.cpp
    src/agent/ultimate_bot_agent.cpp)
  target_link_libraries(test_ultimate_bot_agent PRIVATE
    scribblez_core scribblez_evidence_select Boost::json Boost::program_options Threads::Threads
    GTest::gtest_main)
  gtest_discover_tests(test_ultimate_bot_agent)

  add_executable(test_weird_bot tests/test_weird_bot.cpp)
  target_link_libraries(test_weird_bot PRIVATE
    scribblez_core Boost::program_options GTest::gtest_main)
  target_compile_definitions(test_weird_bot PRIVATE
    SCRIBBLEZ_DEFAULT_KWG="/workspace/mount/lexica/NWL23.kwg"
    SCRIBBLEZ_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  gtest_discover_tests(test_weird_bot)

  # Builds real agents through PlayerFactory, whose dispatch references every
  # agent type, so it links scribblez_neural_agents like play_game does.
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

  # TensorRT-vs-PyTorch parity binaries. Each compares against a fixture built
  # by a Python generator (torch import + ONNX export, ~1.7 s). Since
  # gtest_discover_tests runs every TEST() in its own process, a ctest
  # FIXTURES_SETUP step generates each fixture once per suite run into
  # PARITY_FIXTURE_ROOT, and EXTRA_ARGS hands that directory to every case. The
  # fixtures are deterministic (fixed seed), so one copy serves all cases. Run
  # directly with no directory argument, a binary generates its own fixture;
  # that argument is why these binaries carry their own main() and link
  # GTest::gtest rather than gtest_main. All of them need a GPU at run time.
  #
  # Without torch/onnx the setup step leaves the directory empty and the cases
  # skip. When the imports are present, a generator failure fails the setup
  # and every case that depends on it, so a broken fixture cannot hide behind
  # a skip.
  set(PARITY_FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/parity_fixtures")

  function(add_parity_fixture fixture_name out_dir generator_module)
    string(JOIN " " extra_args ${ARGN})  # generator flags, as one space-joined string
    add_test(NAME ${fixture_name}_setup
      COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}/py"
        bash -c "if python3 -c 'import torch, onnx' 2>/dev/null; then exec python3 -m ${generator_module} --out-dir '${out_dir}' ${extra_args}; else echo 'torch/onnx unavailable; ${fixture_name} not generated (its cases will skip)'; fi")
    set_tests_properties(${fixture_name}_setup PROPERTIES
      FIXTURES_SETUP ${fixture_name}
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/py")
  endfunction()

  # Removes the generated fixtures once every dependent case has finished.
  add_test(NAME parity_fixtures_cleanup
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${PARITY_FIXTURE_ROOT}")
  set_tests_properties(parity_fixtures_cleanup PROPERTIES
    FIXTURES_CLEANUP "nn_parity_fixture;mset_parity_fixture;proposal_parity_fixture")

  # The position evaluation model.
  add_executable(test_nn_inference_parity tests/test_nn_inference_parity.cpp)
  target_link_libraries(test_nn_inference_parity PRIVATE
    scribblez_nn scribblez_core Threads::Threads GTest::gtest)
  target_compile_definitions(test_nn_inference_parity PRIVATE
    SCRIBBLEZ_PY_DIR="${CMAKE_SOURCE_DIR}/py")
  add_parity_fixture(nn_parity_fixture "${PARITY_FIXTURE_ROOT}/nn"
    scripts.position_eval.gen_parity_fixture --num-rows 8)
  gtest_discover_tests(test_nn_inference_parity
    EXTRA_ARGS "${PARITY_FIXTURE_ROOT}/nn"
    PROPERTIES FIXTURES_REQUIRED nn_parity_fixture)

  # The move set evaluation model, including the plan-cache refit and the
  # service's chunking.
  add_executable(test_mset_inference_parity tests/test_mset_inference_parity.cpp)
  target_link_libraries(test_mset_inference_parity PRIVATE
    scribblez_nn scribblez_core Threads::Threads GTest::gtest)
  target_compile_definitions(test_mset_inference_parity PRIVATE
    SCRIBBLEZ_PY_DIR="${CMAKE_SOURCE_DIR}/py")
  add_parity_fixture(mset_parity_fixture "${PARITY_FIXTURE_ROOT}/mset"
    scripts.move_set_eval.gen_mset_parity_fixture)
  gtest_discover_tests(test_mset_inference_parity
    EXTRA_ARGS "${PARITY_FIXTURE_ROOT}/mset"
    PROPERTIES FIXTURES_REQUIRED mset_parity_fixture)

  # The move proposal model's two-graph runtime: cache, evidence staging, then
  # step, over several evidence cases.
  add_executable(test_proposal_inference_parity tests/test_proposal_inference_parity.cpp)
  target_link_libraries(test_proposal_inference_parity PRIVATE
    scribblez_nn scribblez_core Threads::Threads GTest::gtest)
  target_compile_definitions(test_proposal_inference_parity PRIVATE
    SCRIBBLEZ_PY_DIR="${CMAKE_SOURCE_DIR}/py")
  add_parity_fixture(proposal_parity_fixture "${PARITY_FIXTURE_ROOT}/proposal"
    scripts.move_set_eval.gen_proposal_parity_fixture)
  gtest_discover_tests(test_proposal_inference_parity
    EXTRA_ARGS "${PARITY_FIXTURE_ROOT}/proposal"
    PROPERTIES FIXTURES_REQUIRED proposal_parity_fixture)

  # Writes small .slog files from an in-memory dictionary, for the Python FFI
  # tests.
  add_executable(test_slog_writer apps/test_slog_writer.cpp)
  target_link_libraries(test_slog_writer PRIVATE scribblez_core Threads::Threads)
endif()

# Ad-hoc check that DataLoader produces well-formed rows from a .slog directory.
add_executable(dataloader_smoke apps/dataloader_smoke.cpp)
target_link_libraries(dataloader_smoke PRIVATE scribblez_core Threads::Threads)

# TensorRT smoke tests, one per model family: load an ONNX export, run a
# synthetic batch, and print the outputs. They check the GPU path on a real
# checkpoint without any game logic.
add_executable(nn_infer_smoke apps/nn_infer_smoke.cpp)
target_link_libraries(nn_infer_smoke PRIVATE scribblez_nn scribblez_core Threads::Threads)

add_executable(mset_infer_smoke apps/mset_infer_smoke.cpp apps/synthetic_candidates.cpp)
target_link_libraries(mset_infer_smoke PRIVATE scribblez_nn scribblez_core Threads::Threads)

add_executable(proposal_infer_smoke apps/proposal_infer_smoke.cpp apps/synthetic_candidates.cpp)
target_link_libraries(proposal_infer_smoke PRIVATE scribblez_nn scribblez_core Threads::Threads)

# Distillation targets for the move set evaluation model: the position
# evaluation model (the teacher) scores sampled candidates at .slog positions.
add_executable(move_set_eval_target_generator apps/move_set_eval_target_generator.cpp)
target_link_libraries(move_set_eval_target_generator PRIVATE
  scribblez_nn
  scribblez_core
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)

# Evidence selection: which candidates get simmed, and in what order. Holds
# both the trajectory generator's one-shot recipe and the sequential evidence
# loop UltimateBot plays. It has no TensorRT dependency, so both are
# unit-tested with scripted stubs.
add_library(scribblez_evidence_select STATIC
  src/training/evidence_trajectory_select.cpp
  src/agent/evidence_loop.cpp)
set_target_properties(scribblez_evidence_select PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(scribblez_evidence_select PUBLIC scribblez_core)

# Evidence trajectories (docs/roadmap.md item 4). The library runs the
# per-position recipe against the move set evaluation model; the generator
# feeds it positions and writes trajectory .sobs sidecars.
add_library(scribblez_evidence STATIC src/training/evidence_trajectory.cpp)
set_target_properties(scribblez_evidence PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(scribblez_evidence PUBLIC scribblez_evidence_select scribblez_core scribblez_nn)

add_executable(evidence_trajectory_generator apps/evidence_trajectory_generator.cpp)
target_link_libraries(evidence_trajectory_generator PRIVATE
  scribblez_evidence
  Boost::program_options
  Boost::json
  Boost::filesystem
  Boost::system
  Threads::Threads)
