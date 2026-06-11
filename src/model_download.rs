use std::io::Write as _;
use std::path::{Path, PathBuf};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

const ORT_VERSION: &str = "1.24.2";
struct Model {
    filename: &'static str,
    url:      &'static str,
    kind:     Kind,
}

enum Kind {
    Direct,
    TarBz2 { inner: &'static str },
}

const MODELS: &[Model] = &[
    Model {
        filename: "ggml-small.bin",
        url:      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin",
        kind:     Kind::Direct,
    },
    Model {
        filename: "ggml-silero-vad.bin",
        url:      "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
        kind:     Kind::Direct,
    },
    Model {
        filename: "sherpa-onnx-pyannote-seg.onnx",
        url:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        kind:     Kind::TarBz2 { inner: "model.onnx" },
    },
    Model {
        filename: "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
        url:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
        kind:     Kind::Direct,
    },
];

pub fn ort_dylib_path(models_dir: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    return models_dir.join("onnxruntime.dll");
    #[cfg(not(target_os = "windows"))]
    return models_dir.join("libonnxruntime.so");
}

pub fn llamafile_path(models_dir: &Path) -> PathBuf {
    // On Windows the APE binary must have a .exe extension to be executable directly.
    #[cfg(target_os = "windows")]
    return models_dir.join("Qwen3.5-2B-Q8_0.exe");
    #[cfg(not(target_os = "windows"))]
    return models_dir.join("Qwen3.5-2B-Q8_0.llamafile");
}

enum Archive {
    TarBz2,
    TarGz,
    #[cfg(target_os = "windows")]
    Zip,
}

struct Download {
    dest:    PathBuf,
    url:     String,
    inner:   Option<String>,
    archive: Option<Archive>,
}

pub async fn ensure_models(models_dir: &Path) -> anyhow::Result<()> {
    let mut downloads: Vec<Download> = Vec::new();

    for m in MODELS {
        let dest = models_dir.join(m.filename);
        if dest.exists() { continue; }
        downloads.push(match &m.kind {
            Kind::Direct => Download { dest, url: m.url.into(), inner: None, archive: None },
            Kind::TarBz2 { inner } => Download {
                dest, url: m.url.into(),
                inner: Some(inner.to_string()), archive: Some(Archive::TarBz2),
            },
        });
    }

    let dylib = ort_dylib_path(models_dir);
    if !dylib.exists() {
        #[cfg(target_os = "windows")]
        downloads.push(Download {
            dest:    dylib,
            url:     format!("https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip"),
            inner:   Some(format!("onnxruntime-win-x64-{ORT_VERSION}/lib/onnxruntime.dll")),
            archive: Some(Archive::Zip),
        });
        #[cfg(not(target_os = "windows"))]
        {
            let arch = match std::env::consts::ARCH {
                "aarch64" => "linux-aarch64",
                _         => "linux-x64",
            };
            downloads.push(Download {
                dest:    dylib,
                url:     format!("https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-{arch}-{ORT_VERSION}.tgz"),
                inner:   Some(format!("onnxruntime-{arch}-{ORT_VERSION}/lib/libonnxruntime.so.{ORT_VERSION}")),
                archive: Some(Archive::TarGz),
            });
        }
    }

    let llamafile = llamafile_path(models_dir);
    if !llamafile.exists() {
        downloads.push(Download {
            dest:    llamafile,
            url:     "https://huggingface.co/mozilla-ai/llamafile_0.10/resolve/main/Qwen3.5-2B-Q8_0.llamafile".into(),
            inner:   None,
            archive: None,
        });
    }

    if downloads.is_empty() { return Ok(()); }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" Downloading required models…");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Use Schannel on Windows rather than ring's TLS assembly.
    #[cfg(windows)]
    let client = reqwest::Client::builder().user_agent("murmur/1.0").use_native_tls().build()?;
    #[cfg(not(windows))]
    let client = reqwest::Client::builder().user_agent("murmur/1.0").build()?;

    for dl in &downloads {
        let name = dl.dest.file_name().unwrap_or_default().to_string_lossy();
        println!("\n → {name}");

        match (&dl.archive, &dl.inner) {
            (None, _) => {
                stream_to_file(&client, &dl.url, &dl.dest).await?;
            }
            (Some(Archive::TarBz2), Some(inner)) => {
                let tmp = models_dir.join("_dl.tar.bz2");
                stream_to_file(&client, &dl.url, &tmp).await?;
                extract_tar(&tmp, inner, &dl.dest, false)?;
                std::fs::remove_file(&tmp)?;
            }
            (Some(Archive::TarGz), Some(inner)) => {
                let tmp = models_dir.join("_dl.tar.gz");
                stream_to_file(&client, &dl.url, &tmp).await?;
                extract_tar(&tmp, inner, &dl.dest, true)?;
                std::fs::remove_file(&tmp)?;
            }
            #[cfg(target_os = "windows")]
            (Some(Archive::Zip), Some(inner)) => {
                let tmp = models_dir.join("_dl.zip");
                stream_to_file(&client, &dl.url, &tmp).await?;
                extract_zip(&tmp, inner, &dl.dest)?;
                std::fs::remove_file(&tmp)?;
            }
            _ => anyhow::bail!("invalid download config"),
        }
        println!("   done");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let llamafile = llamafile_path(models_dir);
        if llamafile.exists() {
            std::fs::set_permissions(&llamafile, std::fs::Permissions::from_mode(0o755))?;
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    Ok(())
}

async fn stream_to_file(client: &reqwest::Client, url: &str, dest: &Path) -> anyhow::Result<()> {
    let resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length();
    let tmp = dest.with_extension("part");

    let mut file = tokio::fs::File::create(&tmp).await?;
    let mut downloaded = 0u64;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        match total {
            Some(t) => print_progress(downloaded, t),
            None    => {
                let _ = write!(std::io::stdout(), "\r   {:.1} MB", mb(downloaded));
                let _ = std::io::stdout().flush();
            }
        }
    }
    let _ = writeln!(std::io::stdout());
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&tmp, dest).await?;
    Ok(())
}

fn extract_tar(archive: &Path, inner_path: &str, dest: &Path, gzip: bool) -> anyhow::Result<()> {
    let inner_name = Path::new(inner_path).file_name()
        .and_then(|n| n.to_str()).unwrap_or(inner_path);
    let file = std::fs::File::open(archive)?;
    if gzip {
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
        find_and_unpack(&mut tar, inner_name, dest)
    } else {
        let mut tar = tar::Archive::new(bzip2::read::BzDecoder::new(file));
        find_and_unpack(&mut tar, inner_name, dest)
    }
}


fn find_and_unpack<R: std::io::Read>(
    tar: &mut tar::Archive<R>,
    inner_name: &str,
    dest: &Path,
) -> anyhow::Result<()> {
    for entry in tar.entries()? {
        let mut entry = entry?;
        if entry.path()?.file_name().and_then(|n| n.to_str()) == Some(inner_name) {
            entry.unpack(dest)?;
            return Ok(());
        }
    }
    anyhow::bail!("'{inner_name}' not found in archive")
}

#[cfg(target_os = "windows")]
fn extract_zip(archive: &Path, inner_path: &str, dest: &Path) -> anyhow::Result<()> {
    let inner_name = Path::new(inner_path).file_name()
        .and_then(|n| n.to_str()).unwrap_or(inner_path);
    let file = std::fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(file)?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if entry.name().ends_with(inner_name) {
            let mut out = std::fs::File::create(dest)?;
            std::io::copy(&mut entry, &mut out)?;
            return Ok(());
        }
    }
    anyhow::bail!("'{inner_name}' not found in zip")
}


fn print_progress(downloaded: u64, total: u64) {
    const WIDTH: usize = 40;
    let pct  = (downloaded as f64 / total as f64).min(1.0);
    let fill = (pct * WIDTH as f64) as usize;
    let bar  = format!("{}{}", "█".repeat(fill), "░".repeat(WIDTH - fill));
    let _ = write!(
        std::io::stdout(),
        "\r   [{bar}] {:.0}/{:.0} MB  ({:.0}%)",
        mb(downloaded), mb(total), pct * 100.0,
    );
    let _ = std::io::stdout().flush();
}

fn mb(bytes: u64) -> f64 { bytes as f64 / 1_000_000.0 }
