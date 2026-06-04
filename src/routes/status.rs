use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

use crate::AppState;

pub async fn status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    let j = job.lock().unwrap();
    Ok(Json(json!({
        "job_id": j.job_id,
        "status": j.status,
        "model": j.model_name,
        "segment_count": j.segments.len(),
        "error": j.error,
    })))
}
