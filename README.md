# murmur

Local speech transcription and speaker diarization. Upload audio, transcribe with [whisper.cpp](https://github.com/ggerganov/whisper.cpp), optionally identify speakers with [pyannote](https://github.com/pyannote/pyannote-audio) via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Runs entirely on your machine — no data leaves.

On first launch the app downloads the required models (~300 MB) and opens `http://localhost:8000` in your browser.

---

## Supported audio formats

| Format | Notes |
|---|---|
| MP3 | All common bitrates |
| WAV | PCM (8/16/24/32-bit), ADPCM, IEEE float |
| FLAC | |
| M4A | AAC and ALAC (Apple Lossless) |
| OGG | Vorbis only — **OGG Opus is not supported** |

OGG files containing Opus audio will produce an explicit error at decode time. Convert to MP3 or OGG Vorbis first (e.g. `ffmpeg -i input.opus -c:a libvorbis output.ogg`).

Upload size limit is 2 GB.

---

## Building on Linux

### Prerequisites

```
# Debian / Ubuntu
sudo apt install build-essential cmake libclang-dev

# Fedora / RHEL
sudo dnf install gcc gcc-c++ cmake clang-devel

# Arch
sudo pacman -S base-devel cmake clang
```

You also need a Rust toolchain. If you don't have one:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build

```
cargo build --release
```

The `sherpa-onnx` native libraries are downloaded automatically during the build. The first build takes several minutes because whisper.cpp is compiled from source.

The binary is written to `target/release/murmur`.

### Run

```
./target/release/murmur
```

Models are downloaded to a `models/` directory next to the binary on first run.

---

## Building on Windows

### Prerequisites

**1. Rust**

Download and run the installer from [rustup.rs](https://rustup.rs). Accept the default, which installs the `x86_64-pc-windows-msvc` toolchain.

**2. Visual Studio Build Tools**

Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the **Desktop development with C++** workload. This provides MSVC and cmake.

**3. LLVM (for libclang)**

whisper-rs uses bindgen, which requires libclang. The easiest install:

```
winget install LLVM.LLVM
```

Or download the installer from [llvm.org](https://releases.llvm.org). During install, choose **Add LLVM to the system PATH**.

If the build reports it cannot find libclang, set the environment variable:

```
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
```

### Build

Open a **Developer Command Prompt for VS** (or any prompt where `cl.exe` is on PATH), then:

```
cargo build --release
```

The binary is written to `target\release\murmur.exe`.

### Run

```
target\release\murmur.exe
```

Models are downloaded to a `models\` directory next to the binary on first run.

---

## Container

Pre-built images are published to `ghcr.io/mcoops/whisper-rs` for `linux/amd64` and `linux/arm64`.

### CPU / Vulkan (default)

Whisper transcription can use any Vulkan-capable GPU (NVIDIA, AMD, Intel) when the host driver is visible inside the container. No special build flag is needed.

```bash
docker build -t murmur .

docker run -d \
  --name murmur \
  -v murmur-models:/app/models \
  -p 8000:8000 \
  -e MURMUR_PASSWORD=changeme \
  murmur
```

### NVIDIA GPU (llamafile CUDA)

The LLM summarisation backend (llamafile) supports CUDA on NVIDIA hardware. Because llamafile JIT-compiles its CUDA kernels at first startup using `nvcc`, the runtime image must include the full CUDA devel toolchain:

```bash
docker build \
  --build-arg RUNTIME=nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  -t murmur-nvidia .
```

Run with the NVIDIA container toolkit (CDI mode requires toolkit ≥ 1.14):

```bash
# CDI mode (recommended)
docker run -d \
  --name murmur \
  --device nvidia.com/gpu=0 \
  -v murmur-models:/app/models \
  -p 8000:8000 \
  -e MURMUR_PASSWORD=changeme \
  murmur-nvidia

# Legacy runtime mode
docker run -d \
  --name murmur \
  --gpus all \
  -v murmur-models:/app/models \
  -p 8000:8000 \
  -e MURMUR_PASSWORD=changeme \
  murmur-nvidia
```

On first start llamafile compiles `ggml-cuda.so` — this takes a few minutes. Subsequent starts load the cached library from the models volume.

> **Note:** llamafile requires CUDA compute capability ≥ 7.5 (Turing, RTX 20xx / Quadro RTX / Tesla T4 and newer). Older NVIDIA cards (Pascal / GTX 10xx) fall back to CPU.

### Authentication

The web UI and all API endpoints require a username and password:

| Variable | Default | Notes |
|---|---|---|
| `MURMUR_USERNAME` | `admin` | |
| `MURMUR_PASSWORD` | *(random)* | Printed to stdout on first start if not set |

---

## Notes

- Only one transcription runs at a time. A second request while one is in progress returns HTTP 429.
- Diarization requires transcription to complete first. Re-transcribing while diarization is running returns HTTP 409.
- All uploaded audio is held in RAM, never written to persistent disk.
