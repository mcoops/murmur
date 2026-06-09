use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use uuid::Uuid;

// ── Segment ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
}

// ── SSE events broadcast from transcription thread ────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SegmentEvent {
    Segment(Segment),
    Done { total_segments: usize },
    Error { message: String },
}

// ── Job status enums ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Uploaded,
    Queued,
    Running,
    Done,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiarizeStatus {
    Idle,
    Running,
    Done,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SummaryStatus {
    #[default]
    Idle,
    Running,
    Done,
    Error,
}

// ── Job ───────────────────────────────────────────────────────────────────────

pub struct Job {
    pub job_id: String,
    /// Raw audio bytes — never written to persistent disk.
    pub audio_data: Vec<u8>,
    pub audio_ext: String,    // e.g. ".mp3", for MIME type and temp file suffix
    pub filename: String,
    pub model_name: String,
    pub status: JobStatus,
    pub segments: Vec<Segment>,
    pub error: Option<String>,

    // broadcast channel: SSE handlers subscribe, transcription thread sends
    pub tx: broadcast::Sender<SegmentEvent>,

    pub diarize_status: DiarizeStatus,
    pub diarize_error: Option<String>,
    pub diarize_speakers: Vec<String>,
    pub diarize_stage: String,
    pub diarize_progress: f32,
    pub diarize_start: Option<Instant>,

    pub summary_status: SummaryStatus,
    pub summary: Option<String>,
    pub summary_error: Option<String>,
    pub summary_start: Option<Instant>,
}

impl Job {
    pub fn reset(&mut self) {
        let (tx, _) = broadcast::channel(256);
        self.tx = tx;
        self.status = JobStatus::Uploaded;
        self.segments.clear();
        self.error = None;
        self.diarize_status = DiarizeStatus::Idle;
        self.diarize_error = None;
        self.diarize_speakers.clear();
        self.diarize_stage.clear();
        self.diarize_progress = 0.0;
        self.diarize_start = None;
        self.summary_status = SummaryStatus::Idle;
        self.summary = None;
        self.summary_error = None;
        self.summary_start = None;
    }

    pub fn new(job_id: String, filename: String, model_name: String) -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            job_id,
            audio_data: Vec::new(),
            audio_ext: String::new(),
            filename,
            model_name,
            status: JobStatus::Uploaded,
            segments: Vec::new(),
            error: None,
            tx,
            diarize_status: DiarizeStatus::Idle,
            diarize_error: None,
            diarize_speakers: Vec::new(),
            diarize_stage: String::new(),
            diarize_progress: 0.0,
            diarize_start: None,
            summary_status: SummaryStatus::Idle,
            summary: None,
            summary_error: None,
            summary_start: None,
        }
    }
}

// ── JobStore ──────────────────────────────────────────────────────────────────

// Oldest job is evicted when the store reaches this size, bounding memory use.
const MAX_JOBS: usize = 20;

struct StoreInner {
    map:   HashMap<String, Arc<Mutex<Job>>>,
    order: VecDeque<String>, // insertion order for eviction
}

pub struct JobStore {
    inner: Mutex<StoreInner>,
}

impl JobStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(StoreInner {
                map:   HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    pub fn insert(&self, filename: String, model_name: String) -> Arc<Mutex<Job>> {
        let job_id = Uuid::new_v4().simple().to_string();
        let job = Arc::new(Mutex::new(Job::new(job_id.clone(), filename, model_name)));
        let mut inner = self.inner.lock().unwrap();
        if inner.map.len() >= MAX_JOBS {
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
            }
        }
        inner.map.insert(job_id.clone(), Arc::clone(&job));
        inner.order.push_back(job_id);
        job
    }

    pub fn get(&self, job_id: &str) -> Option<Arc<Mutex<Job>>> {
        self.inner.lock().unwrap().map.get(job_id).cloned()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_roundtrip() {
        let store = JobStore::new();
        let job = store.insert("a.mp3".into(), "small".into());
        let id = job.lock().unwrap().job_id.clone();

        let retrieved = store.get(&id).expect("job should exist");
        assert_eq!(retrieved.lock().unwrap().job_id, id);
    }

    #[test]
    fn get_missing_returns_none() {
        let store = JobStore::new();
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn status_defaults_to_uploaded() {
        let store = JobStore::new();
        let job = store.insert("b.mp3".into(), "small".into());
        assert_eq!(job.lock().unwrap().status, JobStatus::Uploaded);
    }

    #[test]
    fn diarize_status_defaults_to_idle() {
        let store = JobStore::new();
        let job = store.insert("c.mp3".into(), "small".into());
        assert_eq!(job.lock().unwrap().diarize_status, DiarizeStatus::Idle);
    }

    #[test]
    fn broadcast_sender_survives_no_receivers() {
        let store = JobStore::new();
        let job = store.insert("d.mp3".into(), "small".into());
        let tx = job.lock().unwrap().tx.clone();
        // sending with no receivers should not panic
        let _ = tx.send(SegmentEvent::Done { total_segments: 0 });
    }
}
