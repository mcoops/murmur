use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Deserialize;

use crate::audio::assign_speakers;
use crate::job::{DiarizeStatus, Job};

// ── Model paths ───────────────────────────────────────────────────────────────

pub fn emb_model_path(models_dir: &Path) -> PathBuf {
    // Preference order: ERes2NetV2 > ERes2Net-English > WeSpeaker ResNet34
    for name in &[
        "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
        "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
        "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        "sherpa-onnx-wespeaker-emb.onnx",
    ] {
        let p = models_dir.join(name);
        if p.exists() { return p; }
    }
    models_dir.join("sherpa-onnx-wespeaker-emb.onnx")
}

pub fn seg_model_path(models_dir: &Path) -> PathBuf {
    models_dir.join("sherpa-onnx-pyannote-seg.onnx")
}

pub fn models_available(models_dir: &Path) -> bool {
    seg_model_path(models_dir).exists() && emb_model_path(models_dir).exists()
}

// ── Worker subprocess protocol ────────────────────────────────────────────────

#[derive(Deserialize)]
struct WorkerTurn {
    start: f32,
    end: f32,
    speaker: String,
}

#[derive(Deserialize)]
struct WorkerResponse {
    ok: bool,
    turns: Option<Vec<WorkerTurn>>,
    error: Option<String>,
}

// ── run_diarization ───────────────────────────────────────────────────────────

pub async fn run_diarization(
    job: Arc<Mutex<Job>>,
    num_speakers: Option<u32>,
    models_dir: PathBuf,
) {
    {
        let mut j = job.lock().unwrap();
        j.diarize_status = DiarizeStatus::Running;
        j.diarize_stage = "Identifying speakers…".into();
        j.diarize_start = Some(Instant::now());
    }

    let (audio_data, audio_ext, existing_segments) = {
        let mut j = job.lock().unwrap();
        let audio_data = std::mem::take(&mut j.audio_data);
        let audio_ext  = j.audio_ext.clone();
        let segments   = j.segments.clone();
        (audio_data, audio_ext, segments)
    };

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<(f32, f32, String)>> {
        // Write audio to a NamedTempFile (deleted on drop = after worker exits).
        let mut tmp = tempfile::Builder::new().suffix(&audio_ext).tempfile()?;
        tmp.write_all(&audio_data)?;
        tmp.flush()?;
        let tmp_path = tmp.path().to_path_buf();

        let exe = std::env::current_exe()?;

        let request = serde_json::json!({
            "audio_path":   tmp_path.to_string_lossy(),
            "emb_model":    emb_model_path(&models_dir).to_string_lossy(),
            "seg_model":    seg_model_path(&models_dir).to_string_lossy(),
            "num_speakers": num_speakers.map(|n| n as i32),
        });

        tracing::info!("spawning diarize worker");
        let output = std::process::Command::new(&exe)
            .arg("--worker")
            .env("ORT_DYLIB_PATH", crate::model_download::ort_dylib_path(&models_dir))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.as_mut().unwrap().write_all(request.to_string().as_bytes())?;
                drop(child.stdin.take()); // close stdin so worker knows input is done
                child.wait_with_output()
            })
            .map_err(|e| anyhow::anyhow!("failed to run diarize-worker: {e}"))?;

        if !output.status.success() {
            anyhow::bail!(
                "diarize-worker exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let resp: WorkerResponse = serde_json::from_slice(&output.stdout)
            .map_err(|e| anyhow::anyhow!("bad worker response: {e}\nraw: {}", String::from_utf8_lossy(&output.stdout)))?;

        if !resp.ok {
            anyhow::bail!("{}", resp.error.unwrap_or_else(|| "unknown diarization error".into()));
        }

        let turns = resp.turns.unwrap_or_default()
            .into_iter()
            .map(|t| (t.start, t.end, t.speaker))
            .collect();

        Ok(turns)
    })
    .await;

    match result {
        Ok(Ok(turns)) => {
            let updated = assign_speakers(&existing_segments, &turns);
            let mut speakers: Vec<String> = Vec::new();
            for seg in &updated {
                if let Some(sp) = &seg.speaker {
                    if !speakers.contains(sp) {
                        speakers.push(sp.clone());
                    }
                }
            }
            tracing::info!(speakers = speakers.len(), "diarization done");
            let mut j = job.lock().unwrap();
            j.segments = updated;
            j.diarize_speakers = speakers;
            j.diarize_status = DiarizeStatus::Done;
            j.diarize_stage = "Done".into();
            j.diarize_progress = 1.0;
        }
        Ok(Err(e)) => {
            tracing::error!(error = %e, "diarization error");
            let mut j = job.lock().unwrap();
            j.diarize_status = DiarizeStatus::Error;
            j.diarize_error = Some(e.to_string());
        }
        Err(e) => {
            tracing::error!(error = %e, "diarization task panicked");
            let mut j = job.lock().unwrap();
            j.diarize_status = DiarizeStatus::Error;
            j.diarize_error = Some(format!("task panicked: {e}"));
        }
    }
}
