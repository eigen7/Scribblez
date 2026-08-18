# Scribblez development & runtime image (base: NVIDIA TensorRT, for
# PyTorch + GPU). Pure tooling: the repo is bind-mounted at /workspace/repo
# and large data at /workspace/mount, both populated by setup_wizard.py.
# Macondo and .kwg lexica (copyrighted wordlists) are deliberately not baked
# into the image.

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

# 2) Core toolchain + libraries. libonnx-dev/libprotobuf-dev: scribblez_nn
#    reads ONNX model metadata from C++. cmake comes separately in 2d.
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

# 2b) gcc-16 (ubuntu-toolchain-r/test PPA) as the default gcc/g++, for C++26
#     reflection. gcc-13 stays installed as a rollback
#     (`update-alternatives --config gcc`).
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

# 2c) clangd-21 for (partial) c++26 support
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" \
      > /etc/apt/sources.list.d/llvm-21.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends clangd-21 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-21 210

# 2d) cmake >= 3.30 (Kitware repo) for CXX_STANDARD 26 with gcc. Also deletes
#     the base image's untracked /usr/local/bin cmake binaries, which would
#     otherwise shadow the upgrade.
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /etc/apt/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main" \
      > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/* \
      /usr/local/bin/cmake /usr/local/bin/cpack /usr/local/bin/ctest

# 3) Node.js 22 (NodeSource), for the web UI and the CLI coding agents.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g @anthropic-ai/claude-code @openai/codex

# 4) Python deps. PyTorch from the stable cu129 index (nightly lacks x86_64
#    wheels), matching the TensorRT base's CUDA.
RUN pip3 install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu129 \
      torch torchvision torchaudio
RUN pip3 install --no-cache-dir \
      numpy scipy pandas matplotlib tqdm pytest natsort ipython onnx onnxscript onnxruntime \
      ruff pypdf
RUN pip3 install --no-cache-dir bokeh==3.9.1  # must match pinned BokehJS version

# 5) Playwright's Chromium for headless board-position rendering, at a
#    world-readable path; version pinned to web/package.json's `playwright`.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
RUN npx --yes playwright@1.61.0 install --with-deps chromium && \
    chmod -R a+rx /opt/ms-playwright && \
    rm -rf /var/lib/apt/lists/*

# 6) rclone for the R2 results bucket, from rclone.org (Ubuntu's 1.60
#    mishandles R2 uploads).
RUN curl -fsSL https://downloads.rclone.org/v1.74.3/rclone-v1.74.3-linux-amd64.zip \
      -o /tmp/rclone.zip && \
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/rclone.zip').extractall('/tmp')" && \
    install -m 755 /tmp/rclone-v1.74.3-linux-amd64/rclone /usr/local/bin/rclone && \
    rm -rf /tmp/rclone.zip /tmp/rclone-v1.74.3-linux-amd64

# 7) Make py/ importable without PYTHONPATH (which conflicts with pytest's
#    rootdir resolution).
RUN python3 -c "import site; \
    open(site.getsitepackages()[0]+'/scribblez.pth','w').write('/workspace/repo/py\n')"


# 8) Temporary home for install targets that belong above. Putting new stuff
#    here helps to make image rebuilds faster thanks to Docker caching
#    mechanics. These items should occasionally be moved above.
RUN pip3 install --no-cache-dir schedulefree

# 9) Macondo's data-path env var, so it finds its kwg/leaves when the engine
#    launches it as a subprocess.
ENV MACONDO_CONFIG_DATA_PATH=/workspace/mount/macondo/data

COPY devuser-setup.sh /usr/local/bin/devuser-setup.sh
COPY entrypoint.sh    /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/devuser-setup.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash"]
