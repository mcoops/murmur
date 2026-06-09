use std::collections::HashMap;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::job::SummaryStatus;
use crate::AppState;

#[derive(Deserialize)]
pub struct SummarizeRequest {
    /// Current speaker name mapping — applied to segments before summarizing.
    pub speaker_names: Option<HashMap<String, String>>,
}

pub async fn summarize(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(req): Json<SummarizeRequest>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    if !crate::summarize::available(&state.models_dir) {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "Summarization model not available".into()));
    }

    let segments = {
        let mut j = job.lock().unwrap();

        if j.summary_status == SummaryStatus::Running {
            return Err((StatusCode::TOO_MANY_REQUESTS, "Summarization already in progress".into()));
        }
        if j.segments.is_empty() {
            return Err((StatusCode::BAD_REQUEST, "No segments to summarize".into()));
        }

        if let Some(ref names) = req.speaker_names {
            for seg in &mut j.segments {
                if let Some(ref sp) = seg.speaker {
                    if let Some(name) = names.get(sp) {
                        seg.speaker_name = Some(name.clone());
                    }
                }
            }
        }

        j.summary_status = SummaryStatus::Running;
        j.summary_start = Some(Instant::now());
        j.summary = None;
        j.summary_error = None;

        j.segments.clone()
    };

    let models_dir    = state.models_dir.clone();
    let tokens_cap    = state.summary_tokens;

    tokio::task::spawn_blocking(move || {
        tracing::info!("starting summarization");
        match crate::summarize::summarize(&segments, &models_dir, tokens_cap) {
            Ok(text) => {
                tracing::info!("summarization done");
                let mut j = job.lock().unwrap();
                j.summary = Some(text);
                j.summary_status = SummaryStatus::Done;
            }
            Err(e) => {
                tracing::error!(error = %e, "summarization failed");
                let mut j = job.lock().unwrap();
                j.summary_status = SummaryStatus::Error;
                j.summary_error = Some(e.to_string());
            }
        }
    });

    Ok((StatusCode::ACCEPTED, Json(json!({ "status": "started" }))))
}
