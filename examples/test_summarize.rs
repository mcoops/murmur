/// Test summarization standalone:
///   cargo run --example test_summarize -- [txt] [models_dir]
///
/// Parses a speaker-labelled txt transcript, builds the prompt, runs llama-cli,
/// and prints the raw output so you can verify parsing before wiring in the full stack.
use std::io::Read;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let txt     = args.next().unwrap_or_else(|| "test/full.txt".into());
    let models  = PathBuf::from(args.next().unwrap_or_else(|| "target/release/models".into()));

    let text = std::fs::read_to_string(&txt)?;
    let transcript = parse_transcript(&text);

    let prompt = format!(
        "<|im_start|>user\n\
        You are summarizing a conversation transcript. Write in third person, describing \
        what each speaker says and discusses. Structure your summary as flowing paragraphs — \
        not bullet points. Cover: the main subjects and arguments raised by each speaker, \
        any notable facts, stories, or examples they share, any agreements, disagreements, \
        or questions exchanged, and any conclusions or action items reached. \
        Be specific — include names, places, and details where mentioned. \
        Begin your response directly with the summary — no title, header, or preamble.\n\n\
        {transcript}<|im_end|>\n\
        <|im_start|>assistant\n"
    );

    let cli   = models.join("llama-cli");
    let model = models.join("Qwen3-1.7B-Q4_K_M.gguf");

    eprintln!("cli:   {}", cli.display());
    eprintln!("model: {}", model.display());
    eprintln!("prompt chars: {}", prompt.len());
    eprintln!("running …\n");

    let mut cmd = std::process::Command::new(&cli);
    cmd.args([
        "-m", &model.to_string_lossy(),
        "-p", &prompt,
        "-n", "1024",
        "--temp", "0.7",
        "-c", "4096",
        "-t", "8",
        "--simple-io",
        "--log-disable",
    ]);
    #[cfg(not(target_os = "windows"))]
    cmd.env("LD_LIBRARY_PATH", &models);

    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    let mut stdout = child.stdout.take().unwrap();
    let mut raw = Vec::<u8>::new();
    let mut buf = [0u8; 4096];
    const MAX_BYTES: usize = 128 * 1024;
    const STATS: &[u8] = b"[ Prompt:";

    let start = std::time::Instant::now();
    loop {
        let n = stdout.read(&mut buf)?;
        if n == 0 { break; }
        raw.extend_from_slice(&buf[..n]);
        if raw.windows(STATS.len()).any(|w| w == STATS) { break; }
        if raw.len() >= MAX_BYTES { break; }
    }
    let _ = child.kill();
    let _ = child.wait();
    let elapsed = start.elapsed();

    let raw_str = String::from_utf8_lossy(&raw);

    eprintln!("=== raw output ({} bytes, {:.1}s) ===", raw.len(), elapsed.as_secs_f32());
    eprintln!("{raw_str}");
    eprintln!("=== end raw ===\n");

    // Parse
    let raw = raw_str.as_ref();
    let stats_pos = raw.find("[ Prompt:").unwrap_or(raw.len());

    // llama-cli renders <think> as [Start thinking]…[End thinking]
    let text = if let Some(pos) = raw.find("[End thinking]") {
        eprintln!("parse mode: thinking (found [End thinking])");
        raw[pos + "[End thinking]".len()..stats_pos].trim()
    } else if let Some(pos) = raw.rfind("<|im_start|>assistant") {
        eprintln!("parse mode: completion (found im_start)");
        let after = raw[pos + "<|im_start|>assistant".len()..].trim_start_matches('\n');
        let end = after.find("[ Prompt:").or_else(|| after.find("<|im_end|>")).unwrap_or(after.len());
        after[..end].trim()
    } else if let Some(pos) = raw[..stats_pos].rfind("\n> ") {
        eprintln!("parse mode: chat no-think (found \\n> )");
        raw[pos + "\n> ".len()..stats_pos].trim()
    } else {
        eprintln!("parse mode: fallback");
        raw[..stats_pos].trim()
    };

    eprintln!("=== parsed summary ({:.1}s) ===", elapsed.as_secs_f32());
    println!("{text}");

    Ok(())
}

fn parse_transcript(text: &str) -> String {
    let mut out = Vec::<String>::new();
    let mut speaker: Option<&str> = None;
    let mut lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("SPEAKER_") && t.ends_with(':') {
            if let Some(sp) = speaker {
                let body = lines.join(" ").trim().to_string();
                if !body.is_empty() {
                    out.push(format!("{}: {}", sp.trim_end_matches(':'), body));
                }
            }
            speaker = Some(t);
            lines.clear();
        } else if !t.is_empty() {
            lines.push(t);
        }
    }
    if let Some(sp) = speaker {
        let body = lines.join(" ").trim().to_string();
        if !body.is_empty() {
            out.push(format!("{}: {}", sp.trim_end_matches(':'), body));
        }
    }

    out.join("\n")
}
