use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::job::{DiarizeStatus, JobStatus};
use crate::AppState;

#[derive(Deserialize)]
pub struct DiarizeRequest {
    pub num_speakers: Option<u32>,
}


pub async fn diarize(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(req): Json<DiarizeRequest>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    {
        let j = job.lock().unwrap();
        if j.status != JobStatus::Done {
            return Err((StatusCode::BAD_REQUEST, "Transcription must complete before diarization".into()));
        }
        if j.diarize_status == DiarizeStatus::Running {
            return Err((StatusCode::TOO_MANY_REQUESTS, "Diarization already in progress".into()));
        }
    }

    if !crate::diarize::models_available(&state.models_dir) {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Diarization models not found. See PHASES.md for download instructions.".into(),
        ));
    }

    tokio::spawn(crate::diarize::run_diarization(
        job,
        req.num_speakers,
        state.models_dir.clone(),
    ));

    Ok((StatusCode::ACCEPTED, Json(json!({ "status": "started" }))))
}

pub async fn diarize_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    let j = job.lock().unwrap();
    let elapsed = j
        .diarize_start
        .map(|t| (t.elapsed().as_secs_f32() * 10.0).round() / 10.0)
        .unwrap_or(0.0);

    let mut resp = json!({
        "diarize_status": j.diarize_status,
        "diarize_error": j.diarize_error,
        "speakers": j.diarize_speakers,
        "stage": j.diarize_stage,
        "progress": j.diarize_progress,
        "elapsed": elapsed,
    });

    if j.diarize_status == DiarizeStatus::Done {
        resp["segments"] = serde_json::to_value(&j.segments).unwrap();
    }

    Ok(Json(resp))
}
