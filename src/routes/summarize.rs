use std::collections::HashMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures_util::stream::{self, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::job::{SummaryEvent, SummaryStatus};
use crate::AppState;

#[derive(Deserialize)]
pub struct SummarizeRequest {
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

    let llama_port = state.llama_port.ok_or_else(|| {
        (StatusCode::SERVICE_UNAVAILABLE, "Summarization model not available".into())
    })?;

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
        let (tx, _) = broadcast::channel(512);
        j.summary_tx = tx;

        j.segments.clone()
    };

    let client     = state.http_client.clone();
    let tokens_cap = state.summary_tokens;
    let alive      = state.llama_alive.clone();

    tokio::spawn(async move {
        tracing::info!("starting summarization");

        let job_for_token = job.clone();
        let on_token = move |token: String| {
            let mut j = job_for_token.lock().unwrap();
            j.summary.get_or_insert_with(String::new).push_str(&token);
            let _ = j.summary_tx.send(SummaryEvent::Token { text: token });
        };

        match crate::summarize::summarize(&segments, llama_port, tokens_cap, client, alive, on_token).await {
            Ok(text) => {
                tracing::info!("summarization done");
                let mut j = job.lock().unwrap();
                j.summary = Some(text);
                j.summary_status = SummaryStatus::Done;
                let _ = j.summary_tx.send(SummaryEvent::Done);
            }
            Err(e) => {
                tracing::error!(error = %e, "summarization failed");
                let mut j = job.lock().unwrap();
                j.summary_status = SummaryStatus::Error;
                j.summary_error = Some(e.to_string());
                let _ = j.summary_tx.send(SummaryEvent::Error { message: e.to_string() });
            }
        }
    });

    Ok((StatusCode::ACCEPTED, Json(json!({ "status": "started" }))))
}

// ── SSE stream ────────────────────────────────────────────────────────────────

type SseStream = Pin<Box<dyn futures_util::Stream<Item = Result<Event, Infallible>> + Send>>;

pub async fn summarize_stream(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    let (past, is_terminal, maybe_rx) = {
        let j = job.lock().unwrap();
        let rx = if j.summary_status == SummaryStatus::Running {
            Some(j.summary_tx.subscribe())
        } else {
            None
        };
        // Replay whatever text has already arrived so a late subscriber catches up.
        let mut events: Vec<SummaryEvent> = Vec::new();
        if let Some(ref text) = j.summary {
            if !text.is_empty() {
                events.push(SummaryEvent::Token { text: text.clone() });
            }
        }
        let terminal = match j.summary_status {
            SummaryStatus::Done => { events.push(SummaryEvent::Done); true }
            SummaryStatus::Error => {
                events.push(SummaryEvent::Error {
                    message: j.summary_error.clone().unwrap_or_default(),
                });
                true
            }
            _ => false,
        };
        (events, terminal, rx)
    };

    let replay = stream::iter(past.into_iter().map(to_sse));

    let stream: SseStream = if is_terminal {
        Box::pin(replay)
    } else if let Some(rx) = maybe_rx {
        let live = BroadcastStream::new(rx)
            .filter_map(|r| async move { r.ok() })
            .map(to_sse);
        Box::pin(replay.chain(live))
    } else {
        Box::pin(replay)
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn to_sse(ev: SummaryEvent) -> Result<Event, Infallible> {
    let name = match &ev {
        SummaryEvent::Token { .. } => "token",
        SummaryEvent::Done        => "done",
        SummaryEvent::Error { .. } => "error",
    };
    Ok(Event::default().event(name).data(serde_json::to_string(&ev).unwrap()))
}
