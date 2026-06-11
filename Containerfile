# ── Build ─────────────────────────────────────────────────────────────────────
# Ubuntu 24.04 ships Vulkan SDK 1.3.275 — new enough for whisper.cpp's use of
# vk::LayerSettingEXT (added in 1.3.261). Debian Bookworm only has 1.3.239.
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        build-essential \
        cmake \
        clang \
        libvulkan-dev \
        glslc \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /build
COPY . .
RUN cargo build --release

# ── Runtime ───────────────────────────────────────────────────────────────────
# Default: CPU inference + Vulkan GPU (whisper.cpp) via host driver injection.
#
# NVIDIA GPU (llamafile CUDA):
#   docker build --build-arg RUNTIME=nvidia/cuda:12.6.3-devel-ubuntu24.04 .
#
# llamafile JIT-compiles ggml-cuda.so at first start using nvcc; the devel
# image ships nvcc + CUDA headers. Without them CUDA silently falls back to CPU.
# build-essential is included in both variants so nvcc can link its output.
ARG RUNTIME=ubuntu:24.04
FROM ${RUNTIME}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libvulkan1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/murmur .

# Download models on first start; mount this volume to persist them across restarts.
VOLUME ["/app/models"]
EXPOSE 8000

# Override MURMUR_USERNAME / MURMUR_PASSWORD at runtime via -e.
# If MURMUR_PASSWORD is unset a random one is printed to stdout on startup.
ENV MURMUR_USERNAME=admin

ENTRYPOINT ["/bin/sh", "-c", \
    "[ -f /app/models/libonnxruntime.so ] || /app/murmur --download-models && exec /app/murmur"]
