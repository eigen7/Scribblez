# Scribblez cloud worker image.
#
# Slim headless runtime for rented cloud machines (Runpod CPU pods). Contains
# dependencies ONLY -- no repo code, no compiled binaries, no lexica -- so the
# registry image stays stable and code iteration never requires an image
# rebuild or re-pull. At pod start the baked-in bootstrap.py downloads a
# code+binary bundle (pushed by py/scripts/cloud_push_binaries.py) from the R2
# bucket into /workspace/repo and hands off to the bundle's worker entrypoint.
# See docker-setup/README.md and docs/cloud_compute.md.
#
# Built and pushed by ./build_and_push_worker_image.py, which requires the local dev
# image (default "scribblez") to be present: the engine binaries dynamically
# link CUDA/TensorRT (the NN inference code is compiled in unconditionally,
# though CPU workloads never execute it), and those libraries are copied out
# of the dev image rather than shipped via a multi-gigabyte NVIDIA base image.
ARG DEV_IMAGE=scribblez
FROM ${DEV_IMAGE} AS devlibs

FROM ubuntu:24.04

# Runtime deps:
#   - boost/protobuf/numpy: shared libraries the engine binaries and the
#     libscribblez_ffi ctypes wrapper link against (same Ubuntu 24.04 ABI as
#     the dev image they were compiled in -- except for libstdc++, which the
#     dev image upgrades past the distribution's; see the last section).
#   - g++: only for bootstrap.py's CPU-arch detection (`g++ -march=native
#     -Q --help=target`), mirroring py/build.py's detect_host_arch().
#   - git/curl/ca-certificates: runtime data fetches (Macondo strategy files,
#     lexica) from their public upstreams.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git g++ \
      python3 python3-numpy \
      libboost-program-options1.83.0 libboost-filesystem1.83.0 \
      libboost-json1.83.0 libprotobuf32t64 zlib1g && \
    rm -rf /var/lib/apt/lists/*

# rclone (pinned; Ubuntu's packaged 1.60 mishandles Cloudflare R2 uploads).
RUN curl -fsSL https://downloads.rclone.org/v1.74.3/rclone-v1.74.3-linux-amd64.zip \
      -o /tmp/rclone.zip && \
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/rclone.zip').extractall('/tmp')" && \
    install -m 755 /tmp/rclone-v1.74.3-linux-amd64/rclone /usr/local/bin/rclone && \
    rm -rf /tmp/rclone.zip /tmp/rclone-v1.74.3-linux-amd64

# NVIDIA/onnx runtime libraries the engine binaries link against, copied from
# the dev image (versions pinned by its TensorRT 25.06 base). COPY would
# flatten the `.so.10 -> .so.10.11.0` symlinks into duplicate full copies, so
# the real files are copied and the soname symlinks recreated by hand.
COPY --from=devlibs /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.11.0 \
                    /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10.11.0 \
                    /usr/lib/x86_64-linux-gnu/libonnx_proto.so.1 \
                    /usr/lib/x86_64-linux-gnu/
COPY --from=devlibs /usr/local/cuda/lib64/libcudart.so.12.9.79 /usr/lib/x86_64-linux-gnu/
RUN ln -s libnvinfer.so.10.11.0      /usr/lib/x86_64-linux-gnu/libnvinfer.so.10 && \
    ln -s libnvonnxparser.so.10.11.0 /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10 && \
    ln -s libcudart.so.12.9.79       /usr/lib/x86_64-linux-gnu/libcudart.so.12 && \
    ldconfig

# The bundle unpacks at /workspace/repo; its py/ tree is the import root.
ENV PYTHONPATH=/workspace/repo/py
ENV PYTHONUNBUFFERED=1

COPY bootstrap.py /opt/scribblez/bootstrap.py
ENTRYPOINT ["python3", "/opt/scribblez/bootstrap.py"]

# Temporary home for install targets that belong above, kept last so adding one
# rebuilds a single late layer instead of invalidating everything below it.
# These items should occasionally be moved up.
#
# The C++ runtime the bundle was compiled against. The dev image builds with
# gcc-16 from the ubuntu-toolchain-r PPA (C++26 reflection), whose libstdc++ is
# newer than Ubuntu 24.04's: binaries built there need GLIBCXX_3.4.35 while the
# distribution ships 3.4.33, so every bundle failed to load here with
# "version `GLIBCXX_3.4.35' not found" the moment the dev image's toolchain
# moved. Installing the runtime from the same PPA the dev image takes its
# compiler from keeps apt in charge of the library (rather than copying files
# out of the dev image, as the NVIDIA libraries above must be -- those have no
# apt source here) and needs 3 MB, where the compiler that produced them is
# 800. The version is deliberately unpinned: the two images are built together
# (build_docker_image.py), and libstdc++ is backward compatible, so newest-from-
# the-PPA is right by construction.
#
# The key is stored armored (.asc), which apt reads directly -- no gnupg in
# this image just to dearmor it.
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5" \
      -o /etc/apt/keyrings/ubuntu-toolchain-r.asc && \
    echo "deb [signed-by=/etc/apt/keyrings/ubuntu-toolchain-r.asc] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu noble main" \
      > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libstdc++6 libgcc-s1 && \
    rm -rf /var/lib/apt/lists/*
