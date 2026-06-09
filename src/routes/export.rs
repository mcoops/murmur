use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::job::Segment;
use crate::AppState;

#[derive(Deserialize)]
pub struct ExportRequest {
    pub segments: Vec<Segment>,
    pub summary:  Option<String>,
}

pub async fn export(
    State(state): State<AppState>,
    Path((job_id, fmt)): Path<(String, String)>,
    Json(req): Json<ExportRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    let base_name = {
        let j = job.lock().unwrap();
        let raw = std::path::Path::new(&j.filename)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        // Strip characters that would break the Content-Disposition quoted-string.
        raw.chars()
            .map(|c| if matches!(c, '"' | '\\' | '\r' | '\n') { '_' } else { c })
            .collect::<String>()
    };

    let (content, media_type, filename) = match fmt.as_str() {
        "txt" => (
            crate::export::to_txt(&req.segments, req.summary.as_deref()),
            "text/plain; charset=utf-8",
            format!("{base_name}.txt"),
        ),
        "srt" => (
            crate::export::to_srt(&req.segments),
            "text/srt; charset=utf-8",
            format!("{base_name}.srt"),
        ),
        "json" => (
            crate::export::to_json(&req.segments),
            "application/json",
            format!("{base_name}.json"),
        ),
        _ => return Err((StatusCode::BAD_REQUEST, format!("Unknown format: {fmt}"))),
    };

    Ok((
        [(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        )],
        [(header::CONTENT_TYPE, media_type)],
        content,
    ))
}
