# ── Build ─────────────────────────────────────────────────────────────────────
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        libvulkan-dev \
        glslc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
RUN cargo build --release

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
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
