mod audio;
mod auth;
mod diarize;
mod export;
mod job;
mod llama_server;
mod model_download;
mod models;
mod routes;
mod summarize;
mod transcribe;
mod worker;

use std::path::PathBuf;
use std::sync::Arc;

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
    pub summary_tokens: u32,
    /// Port of the persistent llamafile server process (None if unavailable).
    pub llama_port:  Option<u16>,
    pub llama_alive: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub http_client: reqwest::Client,
    pub auth: auth::AuthConfig,
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

    if std::env::args().nth(1).as_deref() == Some("--download-models") {
        model_download::ensure_models(&models_dir).await?;
        return Ok(());
    }

    // Load ONNX Runtime dylib for pyannote segmentation (ort load-dynamic mode).
    let ort_dylib = model_download::ort_dylib_path(&models_dir);
    if !ort_dylib.exists() {
        anyhow::bail!(
            "Models not found at {}.\n\
             Run:  murmur --download-models",
            models_dir.display()
        );
    }
    ort::init_from(&ort_dylib)?.commit();

    // Audio is stored in memory on each Job — nothing uploaded ever touches disk.
    // Transcription and diarization write NamedTempFiles to /tmp (tmpfs on Linux)
    // and delete them immediately after use.

    let summary_tokens = std::env::var("SUMMARY_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048u32);

    // Spawn the llamafile server once at startup so the model stays loaded.
    // `_llama_server` is kept alive until `run()` returns (program exit).
    let (_llama_server, llama_port, llama_alive) = if summarize::available(&models_dir) {
        match llama_server::LlamaServer::spawn(&models_dir) {
            Ok(srv) => {
                let port  = srv.port;
                let alive = srv.alive.clone();
                (Some(srv), Some(port), Some(alive))
            }
            Err(e) => {
                tracing::warn!("could not start llamafile server: {e}");
                (None, None, None)
            }
        }
    } else {
        (None, None, None)
    };

    let username = std::env::var("MURMUR_USERNAME").unwrap_or_else(|_| "admin".to_string());
    let password = std::env::var("MURMUR_PASSWORD").unwrap_or_else(|_| {
        let generated = uuid::Uuid::new_v4().to_string()[..12].to_string();
        tracing::warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        tracing::warn!(" No MURMUR_PASSWORD set — generated password: {generated}");
        tracing::warn!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        generated
    });

    let state = AppState {
        jobs: Arc::new(JobStore::new()),
        transcribe_semaphore: Arc::new(Semaphore::new(1)),
        model_cache: Arc::new(ModelCache::new()),
        models_dir,
        summary_tokens,
        llama_port,
        llama_alive,
        http_client: reqwest::Client::new(),
        auth: auth::AuthConfig::new(username, password),
    };

    let app = routes::router(state.clone())
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    let listener = match tokio::net::TcpListener::bind("0.0.0.0:8000").await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            tracing::info!("port 8000 already in use — opening browser to existing instance");
            let _ = open::that("http://localhost:8000");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };
    tracing::info!("listening on http://0.0.0.0:8000");
    let _ = open::that("http://localhost:8000");
    axum::serve(listener, app).await?;
    Ok(())
}
