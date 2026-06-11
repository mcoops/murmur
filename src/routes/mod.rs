mod config;
mod upload;
mod transcribe;
mod progress;
mod diarize;
mod export;
mod status;
mod summarize;

use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;

use crate::AppState;

static INDEX_HTML: &str = include_str!("../../static/index.html");

pub fn router(state: crate::AppState) -> Router<AppState> {
    let protected = Router::new()
        .route("/config", get(config::get_config))
        .route("/upload", post(upload::upload).layer(DefaultBodyLimit::max(2 * 1024 * 1024 * 1024)))
        .route("/transcribe/:job_id", post(transcribe::transcribe))
        .route("/progress/:job_id", get(progress::progress))
        .route("/diarize/:job_id", post(diarize::diarize))
        .route("/diarize-status/:job_id", get(diarize::diarize_status))
        .route("/export/:job_id/:fmt", post(export::export))
        .route("/status/:job_id", get(status::status))
        .route("/summarize/:job_id", post(summarize::summarize))
        .route("/summarize-stream/:job_id", get(summarize::summarize_stream))
        .route_layer(middleware::from_fn_with_state(state, crate::auth::require_auth));

    Router::new()
        .route("/", get(|| async { axum::response::Html(INDEX_HTML) }))
        .route("/static/index.html", get(|| async { axum::response::Redirect::permanent("/") }))
        .route("/login", post(crate::auth::login))
        .route("/logout", post(crate::auth::logout))
        .route("/auth/check", get(crate::auth::check))
        .merge(protected)
}
