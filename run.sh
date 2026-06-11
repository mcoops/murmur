#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source "$HOME/.cargo/env"

if [ ! -f target/release/whisper-app ]; then
  echo "==> Building release binary (first time only, ~1 min)..."
  cargo build --release
fi

exec ./target/release/whisper-app
