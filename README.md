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

## Notes

- Only one transcription runs at a time. A second request while one is in progress returns HTTP 429.
- Diarization requires transcription to complete first. Re-transcribing while diarization is running returns HTTP 409.
- All uploaded audio is held in RAM, never written to persistent disk.
