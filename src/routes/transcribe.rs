use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::job::{DiarizeStatus, JobStatus};
use crate::AppState;

const ALLOWED_MODELS: &[&str] = &[
    "tiny",    "tiny-q8_0",    "tiny-q5_1",
    "base",    "base-q8_0",    "base-q5_1",
    "small",   "small-q8_0",   "small-q5_1",
    "medium",  "medium-q8_0",  "medium-q5_0",
    "large-v3", "large-v3-q8_0", "large-v3-q5_0",
    "large-v3-turbo", "large-v3-turbo-q8_0", "large-v3-turbo-q5_0",
    "distil-medium.en", "distil-large-v3",
];

#[derive(Deserialize)]
pub struct TranscribeRequest {
    #[serde(default = "default_model")]
    pub model: String,
}

fn default_model() -> String {
    "small".into()
}

pub async fn transcribe(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(req): Json<TranscribeRequest>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    {
        let j = job.lock().unwrap();
        if matches!(j.status, JobStatus::Running | JobStatus::Queued) {
            return Err((StatusCode::TOO_MANY_REQUESTS, "Transcription already in progress".into()));
        }
        if j.diarize_status == DiarizeStatus::Running {
            return Err((StatusCode::CONFLICT, "Diarization is in progress; wait for it to finish before re-transcribing".into()));
        }
    }

    let model = if ALLOWED_MODELS.contains(&req.model.as_str()) {
        req.model.clone()
    } else {
        "small".into()
    };

    // Try to acquire the single transcription slot immediately.
    // Returns 429 if another transcription is running.
    let permit = state
        .transcribe_semaphore
        .clone()
        .try_acquire_owned()
        .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "A transcription is already in progress".into()))?;

    {
        let mut j = job.lock().unwrap();
        j.reset();
        j.model_name = model.clone();
        j.status = JobStatus::Running;
    }

    tokio::spawn(crate::transcribe::run_transcription(
        job,
        permit,
        state.models_dir.clone(),
        state.model_cache.clone(),
    ));

    Ok((
        StatusCode::ACCEPTED,
        Json(json!({ "status": "started", "model": model })),
    ))
}
