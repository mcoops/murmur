default: build

# Debug build
build:
    cargo build

# Release build
release:
    cargo build --release

# Download all models/assets into target/release/models/ for local testing
download-models: release
    ./target/release/murmur --download-models

# Run the server (release) — requires models to be present (run download-models first)
run: release
    ./target/release/murmur

# Run in debug mode
dev:
    cargo build && ./target/debug/murmur

# Lint
check:
    cargo check && cargo clippy

# List downloaded models
models:
    ls -lh target/release/models/ 2>/dev/null || echo "No models yet — run: just download-models"

# Clean
clean:
    cargo clean
