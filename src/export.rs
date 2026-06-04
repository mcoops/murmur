use crate::job::Segment;

pub fn to_txt(segments: &[Segment]) -> String {
    let has_speakers = segments.iter().any(|s| s.speaker.is_some());

    if !has_speakers {
        return segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current_speaker: Option<String> = None;

    for s in segments {
        let display = s
            .speaker_name
            .clone()
            .or_else(|| s.speaker.clone())
            .unwrap_or_else(|| "Unknown".into());

        if Some(&display) != current_speaker.as_ref() {
            if !lines.is_empty() {
                lines.push(String::new());
            }
            lines.push(format!("{display}:"));
            current_speaker = Some(display);
        }
        lines.push(s.text.clone());
    }

    lines.join("\n")
}

pub fn to_srt(segments: &[Segment]) -> String {
    let has_speakers = segments.iter().any(|s| s.speaker.is_some());

    segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let start = srt_time(s.start);
            let end = srt_time(s.end);
            let text = if has_speakers {
                let display = s
                    .speaker_name
                    .clone()
                    .or_else(|| s.speaker.clone())
                    .unwrap_or_else(|| "Unknown".into());
                format!("[{display}] {}", s.text)
            } else {
                s.text.clone()
            };
            format!("{}\n{start} --> {end}\n{text}", i + 1)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn to_json(segments: &[Segment]) -> String {
    let val = serde_json::json!({ "segments": segments, "version": 1 });
    serde_json::to_string_pretty(&val).unwrap()
}

fn srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Segment;

    fn seg(id: usize, start: f32, end: f32, text: &str) -> Segment {
        Segment { id, start, end, text: text.into(), speaker: None, speaker_name: None }
    }

    fn seg_spk(id: usize, start: f32, end: f32, text: &str, speaker: &str) -> Segment {
        Segment { id, start, end, text: text.into(), speaker: Some(speaker.into()), speaker_name: None }
    }

    #[test]
    fn txt_no_speakers() {
        let segs = vec![seg(0, 0.0, 1.0, "Hello"), seg(1, 1.0, 2.0, "World")];
        assert_eq!(to_txt(&segs), "Hello\n\nWorld");
    }

    #[test]
    fn txt_with_speakers_groups_consecutive() {
        let segs = vec![
            seg_spk(0, 0.0, 1.0, "Hi", "SPEAKER_00"),
            seg_spk(1, 1.0, 2.0, "there", "SPEAKER_00"),
            seg_spk(2, 2.0, 3.0, "Hello", "SPEAKER_01"),
        ];
        let out = to_txt(&segs);
        assert!(out.contains("SPEAKER_00:\nHi\nthere"));
        assert!(out.contains("SPEAKER_01:\nHello"));
    }

    #[test]
    fn srt_time_formatting() {
        assert_eq!(srt_time(3661.5), "01:01:01,500");
        assert_eq!(srt_time(0.0), "00:00:00,000");
    }

    #[test]
    fn srt_no_speakers() {
        let segs = vec![seg(0, 0.0, 1.5, "Hello")];
        let out = to_srt(&segs);
        assert!(out.contains("1\n00:00:00,000 --> 00:00:01,500\nHello"));
    }
}
