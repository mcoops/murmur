mod config;
mod upload;
mod transcribe;
mod progress;
mod diarize;
mod export;
mod status;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;

use crate::AppState;

static INDEX_HTML: &str = include_str!("../../static/index.html");

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(|| async { axum::response::Html(INDEX_HTML) }))
        .route("/static/index.html", get(|| async { axum::response::Redirect::permanent("/") }))
        .route("/config", get(config::get_config))
        .route("/upload", post(upload::upload).layer(DefaultBodyLimit::max(2 * 1024 * 1024 * 1024)))
        .route("/transcribe/:job_id", post(transcribe::transcribe))
        .route("/progress/:job_id", get(progress::progress))
        .route("/diarize/:job_id", post(diarize::diarize))
        .route("/diarize-status/:job_id", get(diarize::diarize_status))
        .route("/export/:job_id/:fmt", post(export::export))
        .route("/status/:job_id", get(status::status))
}
