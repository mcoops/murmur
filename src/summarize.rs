use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::StreamExt;

use crate::job::Segment;
use crate::model_download::llamafile_path;

pub fn available(models_dir: &Path) -> bool {
    llamafile_path(models_dir).exists()
}

pub async fn summarize(
    segments: &[Segment],
    port: u16,
    tokens_cap: u32,
    client: reqwest::Client,
    alive: Option<Arc<AtomicBool>>,
    mut on_token: impl FnMut(String) + Send,
) -> anyhow::Result<String> {
    if segments.is_empty() {
        anyhow::bail!("no segments to summarize");
    }

    let transcript = format_transcript(segments);
    let duration_mins = segments.iter().map(|s| s.end).fold(0.0f32, f32::max) / 60.0;
    let max_tokens = ((duration_mins * 200.0).round() as u32).max(512).min(tokens_cap);

    let date = today();
    let system_msg = format!(
        "You are an experienced law enforcement intelligence officer. Write a formal intelligence \
        report based on the following recorded debrief between a Handler and a Human Source.\n\n\
        HOUSE STYLE — follow exactly:\n\
        - Begin with the date on its own line at the very top: {date}.\n\
        - Write in the third person and the past tense, in a measured, professional register.\n\
        - Refer to the two participants only as \"HS\" (the human source) and \"Handler\". \
        Do not invent or use any personal names for them.\n\
        - Use flowing narrative paragraphs. Do NOT use bullet points, numbered lists, or section headings.\n\
        - Attribute information with reporting verbs, e.g. \"HS stated that...\", \
        \"Handler advised...\", \"HS confirmed...\", \"HS further disclosed...\".\n\
        - Report only what is supported by the transcript. Do not speculate, embellish, or add \
        facts that are not present, and do not include analytical commentary or recommendations \
        unless the speakers themselves made them.\n\
        - Preserve operationally relevant detail exactly as stated: names, nicknames, locations, \
        vehicles and registrations, dates, times, quantities, and the relationships between people.\n\
        - Be concise and avoid repetition.\n\n\
        Produce only the report text. Do not add a preamble, a title (other than the date line), \
        or any closing remarks."
    );

    let body = serde_json::json!({
        "messages": [
            {"role": "system", "content": &system_msg},
            {"role": "user",   "content": transcript},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": true,
        // Disable Qwen3 chain-of-thought if the server supports it.
        "chat_template_kwargs": {"thinking": false},
    });

    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let resp = post_with_retry(&client, &url, &body, alive.as_deref()).await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("llamafile server error {status}: {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut full = String::new();

    'outer: while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));

        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_string();
            buf = buf[pos + 1..].to_string();

            let Some(json_str) = line.strip_prefix("data: ") else { continue };
            if json_str == "[DONE]" { break 'outer; }

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(token) = val["choices"][0]["delta"]["content"].as_str() {
                    if !token.is_empty() {
                        full.push_str(token);
                        on_token(token.to_string());
                    }
                }
            }
        }
    }

    let result = strip_thinking(&full).trim().replace("\r\n", "\n").to_string();
    Ok(result)
}

/// Retry the POST until the llamafile server accepts it (model loading takes time).
/// Fails fast if the server process has already exited.
async fn post_with_retry(
    client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
    alive: Option<&AtomicBool>,
) -> anyhow::Result<reqwest::Response> {
    // 5 minutes: first run needs Cosmopolitan APE self-extraction + model mmap.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    loop {
        if alive.is_some_and(|a| !a.load(Ordering::Acquire)) {
            anyhow::bail!("llamafile server process exited — check stderr for details");
        }
        match client.post(url).json(body).send().await {
            Ok(resp) => return Ok(resp),
            Err(e) if std::time::Instant::now() < deadline => {
                tracing::debug!("llamafile not ready yet ({e}), retrying…");
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
            Err(e) => anyhow::bail!("llamafile server unreachable after 5 min: {e}"),
        }
    }
}

/// Strip a leading <think>...</think> block emitted by Qwen3 thinking mode.
fn strip_thinking(s: &str) -> &str {
    if let Some(end) = s.find("</think>") {
        s[end + "</think>".len()..].trim_start()
    } else {
        s
    }
}

fn today() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() / 86400;
    // Howard Hinnant's civil_from_days
    let z   = days as i64 + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp  = (5 * doy + 2) / 153;
    let d   = doy - (153 * mp + 2) / 5 + 1;
    let m   = if mp < 10 { mp + 3 } else { mp - 9 };
    let y   = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    const MONTHS: [&str; 12] = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    ];
    format!("{} {} {}", d, MONTHS[(m - 1) as usize], y)
}

fn format_transcript(segments: &[Segment]) -> String {
    segments.iter().map(|s| {
        let speaker = s.speaker_name.as_deref()
            .or(s.speaker.as_deref())
            .unwrap_or("Speaker");
        format!("{speaker}: {}", s.text.trim())
    }).collect::<Vec<_>>().join("\n")
}
