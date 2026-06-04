use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use futures_util::stream::{self, StreamExt};
use std::convert::Infallible;
use std::pin::Pin;
use tokio_stream::wrappers::BroadcastStream;

use crate::job::{JobStatus, SegmentEvent};
use crate::AppState;

type SseStream = Pin<Box<dyn futures_util::Stream<Item = Result<Event, Infallible>> + Send>>;

pub async fn progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Job not found".into()))?;

    // Subscribe and collect past events under one lock so there is no window
    // between reading historical state and subscribing to future events.
    let (past_events, is_terminal, maybe_rx) = {
        let j = job.lock().unwrap();
        let rx = if !matches!(j.status, JobStatus::Done | JobStatus::Error) {
            Some(j.tx.subscribe())
        } else {
            None
        };
        let mut events: Vec<SegmentEvent> = j
            .segments
            .iter()
            .cloned()
            .map(SegmentEvent::Segment)
            .collect();

        let terminal = match j.status {
            JobStatus::Done => {
                events.push(SegmentEvent::Done { total_segments: j.segments.len() });
                true
            }
            JobStatus::Error => {
                let msg = j.error.clone().unwrap_or_default();
                events.push(SegmentEvent::Error { message: msg });
                true
            }
            _ => false,
        };
        (events, terminal, rx)
    };

    let replay = stream::iter(past_events.into_iter().map(event_to_sse));

    let stream: SseStream = if is_terminal {
        Box::pin(replay)
    } else {
        let live = BroadcastStream::new(maybe_rx.unwrap())
            .filter_map(|r| async move { r.ok() })
            .map(event_to_sse);
        Box::pin(replay.chain(live))
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn event_to_sse(ev: SegmentEvent) -> Result<Event, Infallible> {
    let name = match &ev {
        SegmentEvent::Segment(_) => "segment",
        SegmentEvent::Done { .. } => "done",
        SegmentEvent::Error { .. } => "error",
    };
    Ok(Event::default()
        .event(name)
        .data(serde_json::to_string(&ev).unwrap()))
}
