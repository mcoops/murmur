use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::AppState;

pub async fn get_config(State(state): State<AppState>) -> Json<Value> {
    let available_models   = discover_models(&state.models_dir);
    let diarize_available  = crate::diarize::models_available(&state.models_dir);
    let summarize_available = crate::summarize::available(&state.models_dir);

    Json(json!({
        "whisper_available": true,
        "sherpa_diarize_available": diarize_available,
        "summarize_available": summarize_available,
        "available_models": available_models,
    }))
}

fn discover_models(models_dir: &std::path::Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return vec![];
    };
    let mut models = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if s.starts_with("ggml-") && s.ends_with(".bin") {
            let model = s.trim_start_matches("ggml-").trim_end_matches(".bin").to_string();
            models.push(model);
        }
    }
    models.sort();
    models
}
