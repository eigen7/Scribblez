# Scribblez development & runtime image.
#
# Base: NVIDIA's TensorRT image so we can use PyTorch + GPU once we add
# self-play / model training. The Scrabble engine itself doesn't need CUDA.
#
# This image is pure tooling: the repo (with build artifacts) is bind-mounted
# at /workspace/repo, and large data (Macondo checkout, .kwg lexicon files,
# future self-play data) lives at /workspace/mount. Both are populated on the
# host by setup_wizard.py and mounted in by run_docker.py.
#
# We deliberately do NOT bake the Macondo binary or any .kwg files into the
# image:
#   - Macondo: lives in the user's mount dir so it can be updated independently
#     and so a single image works across users with different macondo versions.
#   - .kwg files encode a copyrighted wordlist and are fetched from the public
#     Woogles/liwords URL at setup time -- not redistributed by us.

FROM nvcr.io/nvidia/tensorrt:25.06-py3
WORKDIR /workspace/repo

# 1) Locale & timezone
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales tzdata && \
    locale-gen en_US.UTF-8 && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# 2) Core toolchain + Boost (for the C++ engine). libonnx-dev/libprotobuf-dev
#    supply the generated ONNX message headers and libprotobuf.so that
#    scribblez_nn uses to read the input-encoding arm out of an exported
#    model's metadata_props from C++. cmake is installed separately in 2d,
#    from a repo new enough to know the C++26 flags for gcc-16.
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo gosu curl wget rsync git vim tmux ack gdb lsof ca-certificates \
      build-essential ninja-build pkg-config \
      gcc-13 g++-13 clang-19 clangd-19 clang-format-19 clang-tidy-19 \
      libgtest-dev \
      libboost-program-options-dev libboost-filesystem-dev libboost-json-dev \
      libboost-system-dev \
      libeigen3-dev libprotobuf-dev sqlite3 \
      python3 python3-pip python3-dev python-is-python3 \
      golang-go libonnx-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang        clang        /usr/bin/clang-19        100 && \
    update-alternatives --install /usr/bin/clang++      clang++      /usr/bin/clang++-19      100 && \
    update-alternatives --install /usr/bin/clangd       clangd       /usr/bin/clangd-19       100 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 100 && \
    update-alternatives --install /usr/bin/clang-tidy   clang-tidy   /usr/bin/clang-tidy-19   100

# 2b) Upgrade the default gcc/g++ to GCC 16, for C++26 reflection
#     (-std=c++26 -freflection, experimental as of GCC 16.1). Ubuntu 24.04's
#     own archive tops out at gcc-14, so this pulls from the
#     ubuntu-toolchain-r/test PPA.
#     /usr/bin/gcc and /usr/bin/g++ ship as plain symlinks (not
#     update-alternatives slaves) owned by the gcc/g++ packages that
#     build-essential pulls in, so registering both versions with
#     update-alternatives and selecting gcc-16 is what flips the default;
#     cc/c++ already resolve through /usr/bin/gcc and /usr/bin/g++, so they
#     follow automatically. gcc-13/g++-13 stay installed (build-essential
#     depends on them directly) and remain selectable via
#     `update-alternatives --config gcc` as a rollback.
RUN apt-get update && apt-get install -y --no-install-recommends gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5" \
      | gpg --dearmor -o /etc/apt/keyrings/ubuntu-toolchain-r.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/ubuntu-toolchain-r.gpg] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu noble main" \
      > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-16 g++-16 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
      --slave /usr/bin/g++ g++ /usr/bin/g++-13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-16 160 \
      --slave /usr/bin/g++ g++ /usr/bin/g++-16 && \
    update-alternatives --set gcc /usr/bin/gcc-16

# 2c) Upgrade clangd to 21 (apt.llvm.org; noble's archive tops out at 19), so
#     the IDE can cope with the gcc-16 / C++26 setup from 2b. clangd-19's
#     __has_builtin misreports __reference_{constructs,converts}_from_temporary
#     (llvm/llvm-project#111477), so gcc-16's libstdc++ disables
#     std::reference_constructs_from_temporary_v -- which its C++26
#     optional<T&> code then uses unconditionally, breaking every TU that
#     includes <optional>. clangd-21 parses those headers cleanly. Note no
#     upstream clangd understands -freflection or P2996 syntax (gcc-16-only
#     so far); .clangd strips the flag, and the few TUs using ^^ / std::meta
#     stay unparseable in the IDE. clang/clang-format/clang-tidy stay at 19:
#     a clang-format bump could reflow the whole codebase.
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" \
      > /etc/apt/sources.list.d/llvm-21.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends clangd-21 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-21 210

# 2d) Upgrade cmake past what Ubuntu 24.04 ships (3.28.3), from Kitware's own
#     apt repo: CMake only maps CXX_STANDARD 26 to -std=c++26 for GCC >= 14
#     as of CMake 3.30, well past what noble carries. This also deletes the
#     cmake/cpack/ctest binaries the TensorRT base image bakes into
#     /usr/local/bin (untracked by dpkg, so apt can't replace them) -- they
#     precede /usr/bin on PATH and would otherwise keep shadowing the
#     upgraded, apt-managed cmake.
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /etc/apt/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main" \
      > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/* \
      /usr/local/bin/cmake /usr/local/bin/cpack /usr/local/bin/ctest

# 3) Node.js 22 (NodeSource), for the Vite-based web UI, and the CLI coding
#    agents (Claude Code, Codex) that run globally in the container.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g @anthropic-ai/claude-code @openai/codex

# 4) Python deps. Install PyTorch (CUDA-12.9 build, lined up with the TensorRT
#    base) so future training/self-play work has it ready. We use the stable
#    index rather than nightly: the nightly cu129 channel currently ships
#    aarch64 wheels only, while stable has x86_64.
RUN pip3 install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu129 \
      torch torchvision torchaudio
RUN pip3 install --no-cache-dir \
      numpy scipy pandas matplotlib tqdm pytest natsort ipython onnx onnxscript onnxruntime \
      ruff pypdf
RUN pip3 install --no-cache-dir bokeh==3.9.1  # must match pinned BokehJS version

# 5) Playwright's Chromium (+ its system libraries) for headless rendering of
#    board positions into web-styled PNGs (py/scripts/sample_test_subset.py).
#    The browser is installed to a world-readable shared path and the version
#    is pinned to the web/package.json `playwright` devDependency so the
#    runtime-installed package finds this exact build.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
RUN npx --yes playwright@1.61.0 install --with-deps chromium && \
    chmod -R a+rx /opt/ms-playwright && \
    rm -rf /var/lib/apt/lists/*

# 6) rclone, for moving data to/from the R2 results bucket (py/cloud tooling:
#     credential checks, result syncing, binary pushes). Installed from
#     rclone.org: Ubuntu's packaged rclone (1.60) predates the Cloudflare-R2
#     provider accommodations and mishandles R2 uploads.
RUN curl -fsSL https://downloads.rclone.org/v1.74.3/rclone-v1.74.3-linux-amd64.zip \
      -o /tmp/rclone.zip && \
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/rclone.zip').extractall('/tmp')" && \
    install -m 755 /tmp/rclone-v1.74.3-linux-amd64/rclone /usr/local/bin/rclone && \
    rm -rf /tmp/rclone.zip /tmp/rclone-v1.74.3-linux-amd64

# 7) Make py/ importable for any python in the container, without going
#    through PYTHONPATH (which conflicts with pytest's rootdir resolution).
RUN python3 -c "import site; \
    open(site.getsitepackages()[0]+'/scribblez.pth','w').write('/workspace/repo/py\n')"


# 8) Macondo's own data-path env var, so its child process finds the kwg/leaves
#    under <mount>/macondo/data when the engine launches it as a subprocess.
#    The engine itself defaults to /workspace/mount/lexica and
#    /workspace/mount/macondo/bin/shell directly (overridable via --lexica-dir
#    and --macondo on the command line).
ENV MACONDO_CONFIG_DATA_PATH=/workspace/mount/macondo/data

COPY devuser-setup.sh /usr/local/bin/devuser-setup.sh
COPY entrypoint.sh    /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/devuser-setup.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash"]
