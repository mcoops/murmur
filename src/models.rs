use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState};

#[derive(Default)]
pub struct ModelCache {
    // Keyed by model name (e.g. "small"). Loading is slow (seconds for large models),
    // so we cache the context and create a fresh state per transcription.
    models: Mutex<HashMap<String, WhisperContext>>,
}

impl ModelCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a ready-to-use `WhisperState` for the given model.
    /// Loads and caches the context on first call; subsequent calls are fast.
    /// Must be called from a blocking thread (whisper context loading is synchronous).
    pub fn get_or_load(&self, model_name: &str, models_dir: &Path) -> anyhow::Result<WhisperState> {
        let mut models = self.models.lock().unwrap();

        if !models.contains_key(model_name) {
            let path = models_dir.join(format!("ggml-{model_name}.bin"));
            if !path.exists() {
                anyhow::bail!(
                    "Model file not found: {}  —  download ggml-{model_name}.bin from \
                     https://huggingface.co/ggerganov/whisper.cpp",
                    path.display()
                );
            }
            tracing::info!(model = model_name, "loading whisper model");
            let mut wparams = WhisperContextParameters::default();
            #[cfg(target_os = "windows")]
            wparams.use_gpu(false);
            let ctx = WhisperContext::new_with_params(path, wparams)
                .map_err(|e| anyhow::anyhow!("failed to load model '{model_name}': {e:?}"))?;
            tracing::info!(model = model_name, "model loaded and cached");
            models.insert(model_name.to_string(), ctx);
        }

        let ctx = models.get(model_name).unwrap();
        let state = ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("failed to create whisper state: {e:?}"))?;
        Ok(state)
        // Lock released here. WhisperState holds its own Arc<WhisperInnerContext>,
        // so the context stays alive even if the cache entry were removed.
    }
}
