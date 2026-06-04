use axum::extract::{Path, Request, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::AppState;

pub async fn audio(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    req: Request,
) -> Result<Response, (StatusCode, String)> {
    let (data, ext) = {
        let j = state
            .jobs
            .get(&job_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;
        let j = j.lock().unwrap();
        if j.audio_data.is_empty() {
            return Err((StatusCode::NOT_FOUND, "Audio not available".into()));
        }
        (j.audio_data.clone(), j.audio_ext.clone())
    };

    let mime = match ext.as_str() {
        ".mp3"  => "audio/mpeg",
        ".wav"  => "audio/wav",
        ".m4a"  => "audio/mp4",
        ".ogg"  => "audio/ogg",
        ".flac" => "audio/flac",
        _       => "application/octet-stream",
    };

    let total = data.len() as u64;

    // Parse optional Range header for audio seeking.
    if let Some(range_val) = req.headers().get(header::RANGE) {
        if let Ok(range_str) = range_val.to_str() {
            if let Some((start, end)) = parse_range(range_str, total) {
                let body = data[start as usize..=end as usize].to_vec();
                let content_range = format!("bytes {start}-{end}/{total}");
                return Ok((
                    StatusCode::PARTIAL_CONTENT,
                    [
                        (header::CONTENT_TYPE, mime.to_string()),
                        (header::CONTENT_RANGE, content_range),
                        (header::ACCEPT_RANGES, "bytes".into()),
                        (header::CONTENT_LENGTH, body.len().to_string()),
                    ],
                    body,
                ).into_response());
            }
        }
    }

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, mime.to_string()),
            (header::ACCEPT_RANGES, "bytes".into()),
            (header::CONTENT_LENGTH, total.to_string()),
        ],
        data,
    ).into_response())
}

/// Parse "bytes=start-end" or "bytes=start-" from a Range header.
fn parse_range(s: &str, total: u64) -> Option<(u64, u64)> {
    let s = s.strip_prefix("bytes=")?;
    let (start_str, end_str) = s.split_once('-')?;
    let start: u64 = start_str.parse().ok()?;
    let end: u64 = if end_str.is_empty() {
        total.saturating_sub(1)
    } else {
        end_str.parse().ok()?
    };
    if start <= end && end < total { Some((start, end)) } else { None }
}
