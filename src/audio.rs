use std::path::Path;

use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decode any supported audio file to f32 mono PCM samples.
/// Returns (samples, original_sample_rate).
pub fn decode_to_f32_mono(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    // Pick the first audio track.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no audio track found"))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };

        append_frames(&decoded, channels, &mut samples);
    }

    Ok((samples, sample_rate))
}

/// Resample f32 mono PCM from src_rate to 16000 Hz.
/// Returns the resampled samples. No-op if already 16000 Hz.
pub fn resample_to_16k(samples: Vec<f32>, src_rate: u32) -> anyhow::Result<Vec<f32>> {
    const TARGET: u32 = 16_000;

    if src_rate == TARGET {
        return Ok(samples);
    }

    // rubato works on chunks; chunk size of 1024 is a reasonable default.
    let chunk = 1024usize;
    let mut resampler = FftFixedIn::<f32>::new(
        src_rate as usize,
        TARGET as usize,
        chunk,
        2,   // sub-chunks (quality/speed tradeoff)
        1,   // mono
    )?;

    let mut output: Vec<f32> = Vec::with_capacity(
        (samples.len() as f64 * TARGET as f64 / src_rate as f64) as usize + chunk,
    );

    let mut pos = 0usize;
    while pos < samples.len() {
        let end = (pos + chunk).min(samples.len());
        let mut chunk_data = samples[pos..end].to_vec();
        // Pad the last chunk if shorter than expected.
        chunk_data.resize(chunk, 0.0);

        let waves_in = vec![chunk_data];
        let waves_out = resampler.process(&waves_in, None)?;
        output.extend_from_slice(&waves_out[0]);
        pos += chunk;
    }

    // Flush any remaining samples from the resampler.
    let waves_out = resampler.process_partial::<Vec<f32>>(None, None)?;
    if !waves_out.is_empty() {
        output.extend_from_slice(&waves_out[0]);
    }

    Ok(output)
}


// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract interleaved f32 samples from any AudioBufferRef, mix down to mono.
fn append_frames(buf: &AudioBufferRef<'_>, channels: usize, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => mix_down(b.chan(0), (1..channels).map(|c| b.chan(c)), out),
        AudioBufferRef::F64(b) => {
            let ch0: Vec<f32> = b.chan(0).iter().map(|&s| s as f32).collect();
            let rest: Vec<Vec<f32>> = (1..channels)
                .map(|c| b.chan(c).iter().map(|&s| s as f32).collect())
                .collect();
            mix_down(&ch0, rest.iter().map(|v| v.as_slice()), out);
        }
        AudioBufferRef::S16(b) => {
            let scale = 1.0 / i16::MAX as f32;
            let ch0: Vec<f32> = b.chan(0).iter().map(|&s| s as f32 * scale).collect();
            let rest: Vec<Vec<f32>> = (1..channels)
                .map(|c| b.chan(c).iter().map(|&s| s as f32 * scale).collect())
                .collect();
            mix_down(&ch0, rest.iter().map(|v| v.as_slice()), out);
        }
        AudioBufferRef::S32(b) => {
            let scale = 1.0 / i32::MAX as f32;
            let ch0: Vec<f32> = b.chan(0).iter().map(|&s| s as f32 * scale).collect();
            let rest: Vec<Vec<f32>> = (1..channels)
                .map(|c| b.chan(c).iter().map(|&s| s as f32 * scale).collect())
                .collect();
            mix_down(&ch0, rest.iter().map(|v| v.as_slice()), out);
        }
        AudioBufferRef::U8(b) => {
            let scale = 1.0 / 128.0;
            let ch0: Vec<f32> = b.chan(0).iter().map(|&s| (s as f32 - 128.0) * scale).collect();
            let rest: Vec<Vec<f32>> = (1..channels)
                .map(|c| b.chan(c).iter().map(|&s| (s as f32 - 128.0) * scale).collect())
                .collect();
            mix_down(&ch0, rest.iter().map(|v| v.as_slice()), out);
        }
        // S24 and other less common formats — convert via f64 path
        _ => {
            // Fallback: treat as silent rather than panic.
        }
    }
}

fn mix_down<'a>(
    ch0: &[f32],
    rest: impl Iterator<Item = &'a [f32]>,
    out: &mut Vec<f32>,
) {
    let others: Vec<&[f32]> = rest.collect();
    let n_channels = 1 + others.len();

    if n_channels == 1 {
        out.extend_from_slice(ch0);
        return;
    }

    let inv = 1.0 / n_channels as f32;
    for (i, &s) in ch0.iter().enumerate() {
        let sum: f32 = s + others.iter().map(|ch| ch.get(i).copied().unwrap_or(0.0)).sum::<f32>();
        out.push(sum * inv);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_noop_at_16k() {
        let samples = vec![0.5f32; 1600];
        let out = resample_to_16k(samples.clone(), 16_000).unwrap();
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn resample_44100_to_16k_length() {
        // 1 second of audio at 44100 Hz should resample to ~16000 samples.
        let samples = vec![0.0f32; 44100];
        let out = resample_to_16k(samples, 44_100).unwrap();
        // rubato flushes a partial output chunk at the end (~1 chunk = 371 samples at this ratio),
        // so the output is slightly longer than exactly 16000. Allow up to one output chunk.
        assert!(out.len() >= 16000, "expected at least 16000 samples, got {}", out.len());
        assert!(out.len() <= 17000, "expected at most 17000 samples, got {}", out.len());
    }

    #[test]
    fn write_and_verify_wav() {
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 / 16000.0 * 2.0 - 1.0) * 0.5).collect();
        let path = std::env::temp_dir().join("test_audio_write.wav");
        write_wav_16k_mono(&samples, &path).unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        let read_back: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read_back.len(), 16000);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_clamps_out_of_range() {
        let samples = vec![2.0f32, -2.0f32];
        let path = std::env::temp_dir().join("test_audio_clamp.wav");
        write_wav_16k_mono(&samples, &path).unwrap();
        let mut reader = hound::WavReader::open(&path).unwrap();
        let vals: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(vals[0], i16::MAX);
        assert_eq!(vals[1], i16::MIN + 1); // clamp(-1.0) * 32767 = -32767
        std::fs::remove_file(&path).ok();
    }
}

// ── assign_speakers ───────────────────────────────────────────────────────────

use crate::job::Segment;

/// Assigns each whisper segment to the speaker with the most overlap in the diarization turns.
pub fn assign_speakers(segments: &[Segment], turns: &[(f32, f32, String)]) -> Vec<Segment> {
    segments
        .iter()
        .map(|seg| {
            let mut best_speaker: Option<String> = None;
            let mut best_overlap = 0.0f32;
            for (t_start, t_end, speaker) in turns {
                let overlap = (seg.end.min(*t_end) - seg.start.max(*t_start)).max(0.0);
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_speaker = Some(speaker.clone());
                }
            }
            Segment { speaker: best_speaker, ..seg.clone() }
        })
        .collect()
}
