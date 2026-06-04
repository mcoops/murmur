use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tokio::sync::OwnedSemaphorePermit;
use whisper_rs::{FullParams, SamplingStrategy, WhisperVadParams};

use crate::audio;
use crate::job::{Job, JobStatus, Segment, SegmentEvent};
use crate::models::ModelCache;

/// Runs a full transcription for the given job.
///
/// The `_permit` is the semaphore permit that serialises transcriptions; it is
/// dropped when this function returns, which unblocks the next queued request.
pub async fn run_transcription(
    job: Arc<Mutex<Job>>,
    _permit: OwnedSemaphorePermit,
    models_dir: PathBuf,
    model_cache: Arc<ModelCache>,
) {
    let (audio_data, audio_ext, model_name, tx) = {
        let j = job.lock().unwrap();
        (j.audio_data.clone(), j.audio_ext.clone(), j.model_name.clone(), j.tx.clone())
    };

    tracing::info!(model = %model_name, "transcription starting");

    // All CPU work goes inside spawn_blocking so the tokio runtime stays free.
    let tx_thread = tx.clone();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Segment>> {
        // Write audio to a NamedTempFile — lives in /tmp (tmpfs on Linux, RAM-backed).
        // Dropped at end of this closure = deleted immediately after use.
        let mut tmp = tempfile::Builder::new()
            .suffix(&audio_ext)
            .tempfile()?;
        std::io::Write::write_all(&mut tmp, &audio_data)?;

        let (raw_samples, sample_rate) = audio::decode_to_f32_mono(tmp.path())?;
        let samples = audio::resample_to_16k(raw_samples, sample_rate)?;
        drop(tmp); // delete temp file as soon as we have the samples

        // Load (or retrieve cached) whisper context and create a per-transcription state.
        let mut state = model_cache.get_or_load(&model_name, &models_dir)?;

        let n_threads = (std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) / 2)
            .clamp(4, 8) as i32;

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });
        params.set_n_threads(n_threads);
        params.set_language(None);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Enable Silero VAD to skip silence — equivalent to faster-whisper's vad_filter=True.
        // Falls back gracefully if the model file is missing.
        let vad_path = models_dir.join("ggml-silero-vad.bin");
        if vad_path.exists() {
            if let Some(path_str) = vad_path.to_str() {
                let mut vad_params = WhisperVadParams::new();
                vad_params.set_min_silence_duration(500); // ms — match faster-whisper default
                vad_params.set_speech_pad(400);           // ms — pad edges to avoid clipping
                params.set_vad_model_path(Some(path_str));
                params.set_vad_params(vad_params);
                params.enable_vad(true);
                tracing::debug!("VAD enabled");
            }
        } else {
            tracing::debug!("VAD model not found, skipping silence detection");
        }

        // set_segment_callback_safe_lossy fires for each new segment whisper.cpp produces
        // (roughly once per 30 s of audio on CPU). The broadcast send streams the segment
        // to any SSE subscribers already connected.
        let tx_cb = tx_thread.clone();
        params.set_segment_callback_safe_lossy(move |data: whisper_rs::SegmentCallbackData| {
            let seg = Segment {
                id: data.segment as usize,
                start: data.start_timestamp as f32 / 100.0,
                end: data.end_timestamp as f32 / 100.0,
                text: data.text.trim().to_string(),
                speaker: None,
                speaker_name: None,
            };
            let _ = tx_cb.send(SegmentEvent::Segment(seg));
        });

        state
            .full(params, &samples)
            .map_err(|e| anyhow::anyhow!("whisper full() failed: {e:?}"))?;

        // Collect the canonical segment list from the completed state.
        // We re-collect here (rather than relying solely on the callback) so that
        // the stored job.segments is always authoritative and replayable.
        let segments: Vec<Segment> = state
            .as_iter()
            .enumerate()
            .map(|(id, seg)| Segment {
                id,
                start: seg.start_timestamp() as f32 / 100.0,
                end: seg.end_timestamp() as f32 / 100.0,
                text: seg.to_string().trim().to_string(),
                speaker: None,
                speaker_name: None,
            })
            .collect();

        Ok(segments)
    })
    .await;

    match result {
        Ok(Ok(segments)) => {
            let total = segments.len();
            tracing::info!(segments = total, "transcription done");
            // Store segments and set status under the same lock so that the SSE
            // replay path never sees status=Done with an empty segment list.
            {
                let mut j = job.lock().unwrap();
                j.segments = segments;
                j.status = JobStatus::Done;
            }
            let _ = tx.send(SegmentEvent::Done { total_segments: total });
        }
        Ok(Err(e)) => {
            tracing::error!(error = %e, "transcription error");
            set_error(&job, e.to_string(), &tx);
        }
        Err(e) => {
            tracing::error!(error = %e, "transcription task panicked");
            set_error(&job, format!("task panicked: {e}"), &tx);
        }
    }
    // _permit dropped here → semaphore slot released
}

fn set_error(
    job: &Arc<Mutex<Job>>,
    msg: String,
    tx: &tokio::sync::broadcast::Sender<SegmentEvent>,
) {
    {
        let mut j = job.lock().unwrap();
        j.status = JobStatus::Error;
        j.error = Some(msg.clone());
    }
    let _ = tx.send(SegmentEvent::Error { message: msg });
}
