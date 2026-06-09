use std::io::Read;
use std::path::Path;

use crate::job::Segment;
use crate::model_download::llama_cli_path;

pub fn model_path(models_dir: &Path) -> std::path::PathBuf {
    models_dir.join("Qwen3-1.7B-Q4_K_M.gguf")
}

pub fn available(models_dir: &Path) -> bool {
    llama_cli_path(models_dir).exists() && model_path(models_dir).exists()
}

pub fn summarize(segments: &[Segment], models_dir: &Path, tokens_cap: u32) -> anyhow::Result<String> {
    if segments.is_empty() {
        anyhow::bail!("no segments to summarize");
    }

    let transcript = format_transcript(segments);

    // Scale output tokens with audio duration: ~200 tokens per minute, min 512, capped by tokens_cap.
    let duration_mins = segments.iter().map(|s| s.end).fold(0.0f32, f32::max) / 60.0;
    let max_tokens = ((duration_mins * 200.0).round() as u32).max(512).min(tokens_cap);
    // Context window must fit the prompt tokens (~3k for a 10-min transcript) plus output.
    let context = (max_tokens + 3500).max(4096);

    // Full prompt with chat tokens passed via -p.
    // /no_think is omitted — we strip <think>…</think> from output instead, which is
    // more reliable than passing /no_think via stdin (where llama-cli treats leading / as commands).
    let prompt = format!(
        "<|im_start|>user\n\
        You are summarizing a conversation transcript. Write in third person. \
        Write one paragraph per speaker, starting each paragraph with the speaker's name. \
        Separate paragraphs with a blank line. Do not use bullet points or headers. \
        Cover the main subjects and arguments each speaker raises, any notable facts, \
        stories, or examples they share, and any agreements, disagreements, or conclusions. \
        Be specific — include names, places, and details where mentioned. \
        Begin your response directly — no title, header, or preamble.\n\n\
        {transcript}<|im_end|>\n\
        <|im_start|>assistant\n"
    );

    let cli   = llama_cli_path(models_dir);
    let model = model_path(models_dir);

    let mut cmd = std::process::Command::new(&cli);
    cmd.args([
        "-m", &model.to_string_lossy(),
        "-p", &prompt,
        "-n", &max_tokens.to_string(),
        "--temp", "0.7",
        "-c", &context.to_string(),
        "-t", "8",
        "--simple-io",   // route /dev/tty output through stdout so we can capture + kill cleanly
        "--log-disable",
    ]);
    #[cfg(not(target_os = "windows"))]
    cmd.env("LD_LIBRARY_PATH", models_dir);

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS: child gets no console at all, so it cannot send or
        // receive GenerateConsoleCtrlEvent signals that would otherwise kill murmur.
        cmd.creation_flags(0x0000_0008); // DETACHED_PROCESS
    }

    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to run llama-cli: {e}"))?;

    let mut stdout = child.stdout.take().unwrap();
    let mut raw = Vec::<u8>::new();
    let mut buf = [0u8; 4096];
    const MAX_BYTES: usize = 128 * 1024;
    const STATS: &[u8] = b"[ Prompt:";

    loop {
        let n = stdout.read(&mut buf)?;
        if n == 0 { break; }
        raw.extend_from_slice(&buf[..n]);
        if find_bytes(&raw, STATS).is_some() { break; }
        if raw.len() >= MAX_BYTES { break; }
    }

    let _ = child.kill();
    let mut stderr_bytes = Vec::new();
    if let Some(mut stderr) = child.stderr.take() {
        let _ = stderr.read_to_end(&mut stderr_bytes);
    }
    let status = child.wait()?;

    if raw.is_empty() {
        let stderr_str = String::from_utf8_lossy(&stderr_bytes);
        anyhow::bail!("llama-cli produced no output ({})\n{}", status, stderr_str.trim());
    }

    let raw_str = String::from_utf8_lossy(&raw);
    let raw = raw_str.as_ref();
    let stats_pos = raw.find("[ Prompt:").unwrap_or(raw.len());

    // Possible output structures (--simple-io + -p in chat or completion mode):
    //   With thinking   : [banner] > [echo] [Start thinking]…[End thinking]\n\n[RESPONSE][ Prompt:]
    //   Chat no-think   : [banner] > [echo] [RESPONSE] [ Prompt: ]
    //   Completion mode : [echo incl. <|im_start|>assistant\n] [RESPONSE] [ Prompt: ]
    let text = if let Some(pos) = raw.find("[End thinking]") {
        // Model emitted a thinking block; response comes after it
        raw[pos + "[End thinking]".len()..stats_pos].trim()
    } else if let Some(pos) = raw.rfind("<|im_start|>assistant") {
        // Completion mode: prompt echo ends with the assistant marker
        let after = raw[pos + "<|im_start|>assistant".len()..].trim_start_matches('\n');
        let end = after.find("[ Prompt:").or_else(|| after.find("<|im_end|>")).unwrap_or(after.len());
        after[..end].trim()
    } else if let Some(pos) = raw[..stats_pos].rfind("\n> ") {
        // Chat mode without thinking: response after the last prompt marker
        raw[pos + "\n> ".len()..stats_pos].trim()
    } else {
        raw[..stats_pos].trim()
    };

    Ok(text.replace("\r\n", "\n").to_string())
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn format_transcript(segments: &[Segment]) -> String {
    segments.iter().map(|s| {
        let speaker = s.speaker_name.as_deref()
            .or(s.speaker.as_deref())
            .unwrap_or("Speaker");
        format!("{speaker}: {}", s.text.trim())
    }).collect::<Vec<_>>().join("\n")
}
