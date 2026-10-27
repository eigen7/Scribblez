# Project

Scribblez is a Scrabble AI: a C++ engine (move generation, self-play,
training-data logging, TensorRT inference) in engine/, Python training and
tooling in py/, and a React web UI in web/. The long-term goal is an engine
that beats existing AIs by replacing their context-blind static evaluation
with learned, belief-aware evaluation.

Read docs when the task touches their subject, not up front:

- docs/design.md -- the north-star design; read it for engine/model strategy
  work.
- docs/architecture.md -- how self-play games become encoded training tensors:
  the component chain, the `.slog` format, and the replay-reconstruction
  invariant (inputs are recomputed by replaying moves; targets come from
  stored final scores).
- docs/roadmap.md -- the model roadmap and rationale.
- docs/README.md -- the index of all documentation.

Routine build/refactor/tooling tasks rarely need any of them.

# Vendored subtrees

`subtrees/<dir>/` holds vendored git subtrees: read-only copies of repos within
our complete control — notably `devenv_utils`, which provides the dev-container
setup and the worktree/PR tooling. They are never edited in place (a pre-commit
hook enforces this): a change is authored in the source repo's working clone at
/workspace/mount/devenv_utils and lands through that repo's own PR, after which
this repo pulls the update. The rules — updating the copy, the authoring
workflow, coordinated changes — live in subtrees/devenv_utils/SUBTREES.md; read
it before touching anything under subtrees/.

# Worktrees and PR review

Unless told otherwise, never make changes directly in /workspace/repo: work in a
git worktree and land it through a GitHub pull request. The full workflow —
worktree → PR (`subtrees/devenv_utils/pr_flow.py`) → review/merge on GitHub —
is documented in **subtrees/devenv_utils/WORKFLOW.md**; follow it, and relay
the review/merge handoff `pr_flow.py create` prints.

Scribblez specifics for that workflow:
- For C++ work, run py/build.py in the worktree before trusting IDE diagnostics:
  .clangd resolves the compile database at the checkout's own target/, so clangd
  flags every include as missing in a worktree that has never been built.
- Build with `py/build.py --test` while building and testing a change: Release
  optimization minus link-time optimization, whose link step otherwise
  dominates every incremental rebuild. Each build type has its own build tree,
  and target/engine points at whichever was built last.
- For Python work, `export PYTHONPATH=<worktree>/py` before running Python or
  tests: the Docker image makes py/ importable via a site-packages .pth file
  baked with the absolute path /workspace/repo/py (see "Python code" below),
  and that path doesn't follow you into a worktree -- without the override,
  Python silently imports the main checkout's py/ instead of the worktree's.
- A Python-only change can borrow the main checkout's build rather than
  building its own: `ln -s /workspace/repo/target <worktree>/target`
  (py/scribblez/paths.py finds the engine at the checkout's target/engine).
  This holds only while the branch's engine sources match what main's build was
  built from; a test that cannot find an engine binary or FFI symbol, or whose
  engine fails TensorRT's CUDA initialization (an engine older than the
  synthetic models falls through to TensorRT, and the suite hides the GPU),
  means main's build is stale. Remove the link before running py/build.py in the worktree, which
  would otherwise build into main's tree through it.
- Builds are warning-free: a warning from py/build.py is fixed in the code, not
  left for later or silenced with a flag. Where gcc is provably wrong, rewrite so
  the compiler sees it (a comment says why); a `#pragma GCC diagnostic` is the
  last resort. Check the build output for `warning:` before opening a PR.
- Before opening a PR: the engine must build, the affected suites must pass
  (py/run_tests.py --cpp-only for C++ changes, --python-only for Python;
  py/tools/gpu_parity.py too for changes that reach TensorRT inference, see
  "Tests" below), and changed files must be clang-format/ruff clean. Say what
  was run in the PR body.

# Human-review-friendly Pull Requests

Try to organize changes into commits that can be reviewed easily by humans. For example,
instead of bundling logical changes with mechanical bulk renaming changes into one commit,
separate them into two: one commit containing only the logical changes, to be reviewed
carefully by a human, and then a follow-up commit containing the mechanical bulk renaming
changes, which can just be skimmed by a human.

# Environment

You can assume unless otherwise told that you are inside of a Docker container
launched by `run_docker.py`, and that the one-time setup `setup_wizard.py` was
run beforehand. You can always assume that the machine has GPU/NVIDIA/CUDA
availability.

Layout -- where data lives and what provisions it:

- /workspace/repo -- this repo (bind mount).
  - target/archs/\<arch\>/\<build type\>/ -- one CMake build tree per CPU
    microarchitecture and build type (created by py/build.py); target/engine
    is a symlink to this host's arch build of the most recent type built.
- /workspace/mount -- large + persistent data (bind mount; survives container
  relaunches):
  - lexica/ -- .kwg lexicon files, installed by setup_wizard.py from the
    public Woogles/liwords repo (copyrighted wordlists; never committed).
    SCRIBBLEZ_DEFAULT_KWG points at NWL23.kwg here.
  - ccache/ -- the compiler cache every checkout's engine build shares
    (CCACHE_DIR, set by the dev image).
  - macondo/ -- Macondo checkout, cloned at a pinned tag by py/build.py. It
    bundles the leave values (data/strategy/\<lexicon\>/leaves.klv2) and
    pre-endgame table that `HastyEquity::default_leaves_path()` /
    `default_peg_path()` read.
  - devenv_utils/ -- working clone of the devenv_utils repo, where changes to
    the vendored subtrees/devenv_utils copy are authored (see
    subtrees/devenv_utils/SUBTREES.md).
  - worktrees/<project>/ -- per-project working worktrees (see the PR
    workflow above).

# Common commands

- Full build: `py/build.py` (host arch, Release; `--debug`, `--test`, `--clean`, `-j N`).
- Rebuild one target:
  `cmake --build "$(dirname "$(readlink -f target/engine)")" --target <name> -j`
- All tests: `py/run_tests.py [--cpp-only | --python-only | --web-only]`
  (C++ tests run through ctest).
- One C++ test case: `./target/engine/<binary> --gtest_filter=<Suite>.<Name>`, where
  the binary is core_tests, search_tests, agent_tests or game_runner_tests
  (engine/CMakeLists.txt says which sources each holds).
- Stop a process by its PID, never by a `pkill -f`/`pgrep -f` pattern: the
  dashboard's live workers run as the same user with the same command lines as
  ad-hoc runs, and a pattern can also match your own shell.

# Tests

Agents run the suites after nearly every change, often several at once beside
experiments, so tests stay cheap:

- **No TensorRT building.** No test builds a TensorRT plan, and a model a
  test needs is tiny. An engine binary a Python test drives serves a
  synthetic model instead (`--model=synthetic:<variant>`,
  engine/include/nn/synthetic_eval_service.h); C++ tests inject stub
  services. The suites that have no use for the GPU hide it
  (CUDA_VISIBLE_DEVICES empty: ctest per shard, py/tests/conftest.py before
  torch loads), so a stray path into TensorRT fails rather than skipping or
  building. The single exception is ctest's `gpu_smoke`, which checks that a
  TensorRT plan builds and runs at all.
- **GPU kernels are tested on the GPU.** A CUDA kernel is checked against its
  CPU reference in ctest's `gpu_kernel_tests`
  (engine/tests/test_gpu_kernels.cpp), which sees the GPU and links the
  kernels without TensorRT; the CPU reference alone can't catch a bug that
  lives only on the device. A change to a kernel, its host side or the ABI
  between them (engine/src/gpu, engine/include/gpu) adds a case there.
- **TensorRT parity is a separate check:** `py/tools/gpu_parity.py` (~15 s)
  compares the engine's TensorRT outputs with PyTorch for every model family.
  Run it when a change reaches TensorRT inference or what feeds it: engine/
  src/nn or engine/include/nn, the agents' model sessions, ONNX export
  (py/scribblez/onnx_export_util.py, */onnx_export.py, proposal_export.py),
  a model definition, or the input/move encodings a model consumes.
- **Small configurations.** Search budgets, game counts, epochs and corpus
  sizes are the smallest that still exercise what the test asserts; a case
  over ~0.5 s needs a reason.

# Recording gotchas

When you discover a gotcha that (1) would have been cheaper to read in
documentation than to find by trial and error, and (2) seems likely to recur in
other sessions working on unrelated tasks, tell the user and ask whether they
want a standalone PR that records it for future sessions. Record it where the
next session will meet it: an enforcing check or a comment at the code
involved, docs/ or this file for repo-wide knowledge, devenv_utils for the
shared workflow. Findings specific to the task belong in the task's own PR or
plan instead.

# Comments and documentation

Follow the shared doctrine in subtrees/devenv_utils/COMMENTS.md.

# Macondo repo

It is checked out at /workspace/mount/macondo/

If you are asked questions regarding Macondo, please look there.

# Python code

Note that the Docker image makes `py/` importable by baking the absolute
path `/workspace/repo/py` into a site-packages `.pth` file at build time (not
a `PYTHONPATH` env var, which would conflict with pytest's rootdir
resolution). That path is fixed to the main checkout, so it does not follow
you into a worktree -- see "Worktrees and PR review" above.

Do not add `import` statements inside of functions without good reason. By default, they should go
atop the file.

Minor style note: if a function's return type is `None`, don't bother with the ` -> None`
type annotation.

## Ruff

After editing any file, make sure to sanitize it with ruff. Alternatively, you can run
`py/tools/ruff_format_all_py_files.py`.

# C++ Code

## Building

Build by running py/build.py

## Inline methods

- **Multiline methods** cannot be defined inside of a class definition. They must be moved into a
  corresponding `.cpp` file, or into a corresponding `.inl` file that is `#include`'d at the
  bottom of the header file.
- **Single-line methods** (single-expression bodies) may be defined inline
  directly in the header.
- Every `.inl` file `#include`s its own header at the top and is `#include`d
  at the bottom of that header.

## Style

- Functions should be short. They should obey the "one thing and one thing only" principle. After
  writing a function, you should review it to see if it can be shortened by pulling some part of it
  out into a helper that has a clear semantic meaning with a clear API boundary. If so, do it.
- Classes similarly should have a clear purpose - no "god" classes.
- If code starts to look similar in multiple spots, determine whether it can be refactored
  so that common components can be shared. If this can be done reasonably, be aggressive in
  achieving it.
- Don't define structs or lambdas within functions, unless there is a very good reason. This is
  almost always a violation of the above principles. Instead, define them outside the function.
- Aim for high modularity. You want small, self-contained components with well-defined API
  boundaries, which can be reasoned about and optimized in isolation. Then, you want higher-level
  components built on top of them. If you ever spot code that would permit this sort of separation,
  suggest a change.

## Heavy resources are process singletons

TensorRT services, GPU rollout backends and large tables exist once per
process and are shared by every game thread, never built per agent or thread.
Expose them through a `create()`/`shared()` factory over `nn::SharedRegistry`
(engine/include/nn/shared_registry.h); the raw constructors are
factory-internal, called directly only by tools that build one instance per
process. (EndgameSolver is the exception: its transposition table is
thread-exclusive by design.)

## Parallel work runs on the process's work pool

Parallel loops in agents and libraries go through `util::parallel_for` /
`util::parallel_lanes` (engine/include/util/work_pool.h), never their own
threads, and take no thread-count parameter: the pool is the process's CPU
budget, set once from play_game's or a tool's `--threads`, so an agent uses
every core when it plays alone and its own thread when every core is busy with
other games. Per-thread state keys off the OS thread (`thread_local`), not an
agent's `thread_id` or a lane index. Tools may keep dedicated threads for
pipeline stages (a GPU feeder, a progress monitor).

## Clang-format

After editing any file, make sure to sanitize it with clang-format. Alternatively, you can run
`py/tools/clang_format_all_cpp_files.py`.

## Backwards compatibility

This project is completely self-contained, and there are thus no backwards-compatibility
requirements. Never compromise on interface for the sake of backwards-compatibility.
