use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};
use std::path::Path;

use crate::AppState;

const ALLOWED_EXTENSIONS: &[&str] = &[".mp3", ".wav", ".m4a", ".ogg", ".flac"];

pub async fn upload(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut field = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "no file field".into()))?;

    let filename = field.file_name().unwrap_or("upload").to_string();

    let ext = Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_lowercase()))
        .unwrap_or_default();

    if !ALLOWED_EXTENSIONS.contains(&ext.as_str()) {
        return Err((StatusCode::BAD_REQUEST, format!("Unsupported file type: {ext}")));
    }

    // Stream chunks directly into a Vec<u8> — never touches disk.
    let mut data: Vec<u8> = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    {
        data.extend_from_slice(&chunk);
    }

    let size = data.len();
    let job = state.jobs.insert(filename.clone(), "small".into());
    let job_id = {
        let mut j = job.lock().unwrap();
        j.audio_data = data;
        j.audio_ext = ext;
        j.status = crate::job::JobStatus::Uploaded;
        j.job_id.clone()
    };

    Ok(Json(json!({
        "job_id": job_id,
        "filename": filename,
        "size": size,
    })))
}
