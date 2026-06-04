default: build

# Debug build (both binaries)
build:
    cargo build

# Release build (both binaries)
release:
    cargo build --release

# Run the server (release)
run: release
    ./target/release/whisper-app

# Run the server in debug mode
dev:
    cargo build && ./target/debug/whisper-app

# Lint
check:
    cargo check && cargo clippy

# Cross-compile for Windows using cargo-xwin (requires: apt install clang lld ninja-build)
# NOTE: cross-compilation is blocked by a CRT mismatch — ort prebuilts use /MD (dynamic CRT)
# while sherpa-onnx static prebuilts use /MT (static CRT). Build natively on Windows instead:
#   cargo build --release
# Or use GitHub Actions with a windows-latest runner.
build-windows:
    cargo xwin build --release --target x86_64-pc-windows-msvc

# List downloaded models
models:
    ls -lh models/ 2>/dev/null || echo "No models yet — start the app to auto-download"

# Clean
clean:
    cargo clean
