mod audio;
mod diarize;
mod export;
mod job;
mod model_download;
mod models;
mod routes;
mod transcribe;
mod worker;

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use tokio::sync::Semaphore;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use job::JobStore;
use models::ModelCache;

// ── AppState ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<JobStore>,
    /// permits=1: only one transcription runs at a time
    pub transcribe_semaphore: Arc<Semaphore>,
    pub model_cache: Arc<ModelCache>,
    pub models_dir: PathBuf,
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    // Worker subprocess: short-circuit before any UI/setup.
    if std::env::args().nth(1).as_deref() == Some("--worker") {
        worker::run_worker();
        return;
    }

    // Single-thread runtime: eliminates background IOCP worker threads that
    // crash silently on Windows (mio 1.x / STATUS_ACCESS_VIOLATION).
    // Fine for a local single-user app.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");

    if let Err(e) = rt.block_on(run()) {
        let msg = format!("Failed to start: {e:#}");
        eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!(" {msg}");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Write to a log file so the error is visible even if the console
        // window closes before the user can read it (common on Windows
        // when launching via double-click).
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let log = dir.join("murmur-error.log");
                let _ = std::fs::write(&log, format!("{msg}\n"));
                eprintln!("\n(details written to {})", log.display());
            }
        }

        eprintln!("\nPress Enter to close...");
        let _ = std::io::stdin().read_line(&mut String::new());
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("murmur=debug".parse()?))
        .init();

    let models_dir = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("cannot resolve executable path: {e}"))?
        .parent()
        .ok_or_else(|| anyhow::anyhow!("executable has no parent directory"))?
        .join("models");

    std::fs::create_dir_all(&models_dir)?;
    model_download::ensure_models(&models_dir).await?;

    // Load ONNX Runtime dylib for pyannote segmentation (ort load-dynamic mode).
    let ort_dylib = model_download::ort_dylib_path(&models_dir);
    ort::init_from(&ort_dylib)?.commit();

    // Audio is stored in memory on each Job — nothing uploaded ever touches disk.
    // Transcription and diarization write NamedTempFiles to /tmp (tmpfs on Linux)
    // and delete them immediately after use.

    let state = AppState {
        jobs: Arc::new(JobStore::new()),
        transcribe_semaphore: Arc::new(Semaphore::new(1)),
        model_cache: Arc::new(ModelCache::new()),
        models_dir,
    };

    let app = Router::new()
        .merge(routes::router())
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await?;
    tracing::info!("listening on http://127.0.0.1:8000");
    let _ = open::that("http://localhost:8000");
    axum::serve(listener, app).await?;
    Ok(())
}
