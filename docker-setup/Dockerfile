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

# 2) Core toolchain + Boost (for the C++ engine).
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo gosu curl wget rsync git vim tmux ack gdb lsof ca-certificates \
      build-essential cmake ninja-build pkg-config \
      gcc-13 g++-13 clang-19 clangd-19 clang-format-19 clang-tidy-19 \
      libgtest-dev \
      libboost-program-options-dev libboost-filesystem-dev libboost-json-dev \
      libboost-system-dev \
      python3 python3-pip python3-dev python-is-python3 \
      golang-go && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang        clang        /usr/bin/clang-19        100 && \
    update-alternatives --install /usr/bin/clang++      clang++      /usr/bin/clang++-19      100 && \
    update-alternatives --install /usr/bin/clangd       clangd       /usr/bin/clangd-19       100 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 100 && \
    update-alternatives --install /usr/bin/clang-tidy   clang-tidy   /usr/bin/clang-tidy-19   100

# 3) Node.js 22 (NodeSource) for the Vite-based web UI.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# 4) Python deps. Install PyTorch (CUDA-12.9 build, lined up with the TensorRT
#    base) so future training/self-play work has it ready. We use the stable
#    index rather than nightly: the nightly cu129 channel currently ships
#    aarch64 wheels only, while stable has x86_64.
RUN pip3 install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu129 \
      torch torchvision torchaudio
RUN pip3 install --no-cache-dir \
      numpy scipy pandas matplotlib tqdm pytest natsort

# 5) Make python/ importable for any python in the container, without going
#    through PYTHONPATH (which conflicts with pytest's rootdir resolution).
RUN python3 -c "import site; \
    open(site.getsitepackages()[0]+'/scribblez.pth','w').write('/workspace/repo/python\n')"

# 6) Macondo's own data-path env var, so its child process finds the kwg/leaves
#    under <mount>/macondo/data when the engine launches it as a subprocess.
#    The engine itself defaults to /workspace/mount/lexica and
#    /workspace/mount/macondo/bin/shell directly (overridable via --lexica-dir
#    and --macondo on the command line).
ENV MACONDO_CONFIG_DATA_PATH=/workspace/mount/macondo/data

COPY devuser-setup.sh /usr/local/bin/devuser-setup.sh
COPY entrypoint.sh    /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/devuser-setup.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

ARG VERSION="0.2.0"
LABEL version=$VERSION
ENV DOCKER_IMAGE_VERSION=$VERSION

CMD ["bash"]
