/// Diarization worker — pyannote segmentation + per-segment embedding + AHC.
///
///   Phase 1: Slide pyannote ONNX model in 10s windows → find pure single-speaker frames
///            → group into speech segments (Vec<(start_sample, end_sample)>)
///   Phase 2: Extract one ERes2NetV2 embedding per segment ≥ 0.5s
///   Phase 3: Complete-linkage AHC on segment embeddings → global speaker labels
///   Phase 4: Map labels back to time ranges → output turns

use std::io::{self, Read};
use serde::{Deserialize, Serialize};
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

const SAMPLE_RATE: i32 = 16_000;
const MIN_ENERGY: f32  = 1e-4;
const MIN_TURN_S: f32  = 1.5;   // fixed-window fallback only

// Pyannote sliding window parameters
const SEG_WIN_SAMPLES: usize  = 160_000; // 10s at 16kHz
const SEG_STEP_SAMPLES: usize =  80_000; // 5s step (50% overlap)
const SEG_FRAME_SHIFT: usize  =     270; // samples per output frame
const MIN_SEG_SAMPLES: usize  =   3_200; // 0.2s — pyannote boundaries are clean, keep short responses
const MERGE_GAP_S: f32        =    1.0;  // merge same-speaker segments separated by ≤ this gap

// Fallback fixed-window params (used when seg_model unavailable)
const WINDOW_S: f32 = 2.5;
const STEP_S: f32   = 1.25;
const SMOOTH_K: usize = 3;

#[derive(Deserialize)]
struct Request {
    audio_path:   String,
    emb_model:    String,
    num_speakers: Option<i32>,
    #[serde(default)]
    seg_model:    String,
}

#[derive(Serialize)]
struct Turn { start: f32, end: f32, speaker: String }

#[derive(Serialize)]
struct Response {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")] turns: Option<Vec<Turn>>,
    #[serde(skip_serializing_if = "Option::is_none")] error: Option<String>,
}

pub fn run_worker() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("stdin");
    let resp = match run(&input) {
        Ok(t)  => Response { ok: true,  turns: Some(t), error: None },
        Err(e) => Response { ok: false, turns: None,    error: Some(e.to_string()) },
    };
    println!("{}", serde_json::to_string(&resp).unwrap());
}

fn run(input: &str) -> anyhow::Result<Vec<Turn>> {
    let req: Request = serde_json::from_str(input)?;
    // None = auto-detect from dendrogram; Some(k) = use exactly k speakers
    let num_speakers: Option<usize> = req.num_speakers.map(|n| n.max(1) as usize);

    let samples = decode_audio(std::path::Path::new(&req.audio_path))?;

    let extractor = SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig {
        model:       Some(req.emb_model.clone()),
        num_threads: 4,
        debug:       false,
        provider:    Some("cpu".into()),
    }).ok_or_else(|| anyhow::anyhow!("failed to create embedding extractor"))?;

    let use_pyannote = !req.seg_model.is_empty()
        && std::path::Path::new(&req.seg_model).exists();

    if use_pyannote {
        eprintln!("[diarize] using pyannote segmentation: {}", req.seg_model);
        run_pyannote_pipeline(&samples, &extractor, num_speakers, &req.seg_model)
    } else {
        eprintln!("[diarize] seg_model not found, using fixed-window fallback");
        run_fixed_window_pipeline(&samples, &extractor, num_speakers.unwrap_or(2))
    }
}

// ── Phase 1: Pyannote segmentation ───────────────────────────────────────────

/// Returns contiguous speech segments as (start_sample, end_sample) pairs.
/// Uses pyannote ONNX model; picks frames where exactly one speaker is active
/// (powerset classes 1, 2, 3 = pure single-speaker).
fn pyannote_segments(samples: &[f32], seg_model_path: &str) -> anyhow::Result<Vec<(usize, usize)>> {
    use ort::{session::Session, value::Tensor};

    let mut session = Session::builder()?.commit_from_file(seg_model_path)?;

    let n = samples.len();
    let frames_per_win = (SEG_WIN_SAMPLES + SEG_FRAME_SHIFT - 1) / SEG_FRAME_SHIFT;
    let total_frames = (n + SEG_FRAME_SHIFT - 1) / SEG_FRAME_SHIFT + frames_per_win;
    let mut single_prob = vec![0.0f32; total_frames];

    let mut win_start = 0usize;
    while win_start < n {
        let win_end = (win_start + SEG_WIN_SAMPLES).min(n);
        let mut chunk = vec![0.0f32; SEG_WIN_SAMPLES];
        chunk[..win_end - win_start].copy_from_slice(&samples[win_start..win_end]);

        let input = Tensor::from_array(([1usize, 1, SEG_WIN_SAMPLES], chunk.into_boxed_slice()))?;
        let outputs = session.run(ort::inputs!["x" => input])?;
        let (shape, flat) = outputs["y"].try_extract_tensor::<f32>()?;
        let n_frames  = shape[1] as usize;
        let n_classes = shape[2] as usize;
        // Powerset size: n_classes = 1 + n*(n+1)/2  →  n = (-1 + sqrt(1+8*(n_classes-1))) / 2
        // Single-speaker classes occupy positions 1..=n_speakers in the powerset.
        let n_speakers = if n_classes > 1 {
            ((-1.0 + (1.0 + 8.0 * (n_classes - 1) as f64).sqrt()) / 2.0).round() as usize
        } else {
            0
        };

        let frame_offset = win_start / SEG_FRAME_SHIFT;

        for fi in 0..n_frames {
            let base = fi * n_classes;
            if base + n_classes > flat.len() { break; }
            let probs = &flat[base..base + n_classes];
            let argmax = probs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            let is_single = argmax >= 1 && argmax <= n_speakers;
            let gfi = frame_offset + fi;
            if gfi < single_prob.len() && is_single {
                single_prob[gfi] = 1.0;
            }
        }

        if win_start + SEG_WIN_SAMPLES >= n { break; }
        win_start += SEG_STEP_SAMPLES;
    }

    // Convert frame-level binary labels to sample-level segments
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut seg_start: Option<usize> = None;

    for (fi, &v) in single_prob.iter().enumerate() {
        let sample_pos = fi * SEG_FRAME_SHIFT;
        if sample_pos >= n { break; }
        let active = v > 0.5;
        match (seg_start, active) {
            (None, true) => seg_start = Some(sample_pos),
            (Some(s), false) => {
                let end = sample_pos.min(n);
                if end - s >= MIN_SEG_SAMPLES {
                    segments.push((s, end));
                }
                seg_start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = seg_start {
        if n - s >= MIN_SEG_SAMPLES {
            segments.push((s, n));
        }
    }

    eprintln!("[diarize] pyannote found {} segments", segments.len());
    Ok(segments)
}

// ── Pyannote pipeline ─────────────────────────────────────────────────────────

fn run_pyannote_pipeline(
    samples: &[f32],
    extractor: &SpeakerEmbeddingExtractor,
    num_speakers: Option<usize>,
    seg_model_path: &str,
) -> anyhow::Result<Vec<Turn>> {
    let segments = pyannote_segments(samples, seg_model_path)?;

    if segments.is_empty() {
        anyhow::bail!("no speech detected");
    }

    // Extract one embedding per segment
    struct Seg { start: f32, end: f32, emb: Vec<f32> }
    let mut segs: Vec<Seg> = Vec::new();

    for (s_start, s_end) in &segments {
        let chunk = &samples[*s_start..*s_end];
        let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
        if rms < MIN_ENERGY { continue; }

        let stream = extractor.create_stream()
            .ok_or_else(|| anyhow::anyhow!("failed to create stream"))?;
        stream.accept_waveform(SAMPLE_RATE, chunk);
        if let Some(emb) = extractor.compute(&stream) {
            segs.push(Seg {
                start: *s_start as f32 / SAMPLE_RATE as f32,
                end:   *s_end   as f32 / SAMPLE_RATE as f32,
                emb,
            });
        }
    }

    if segs.is_empty() {
        anyhow::bail!("no usable speech segments");
    }

    eprintln!("[diarize] extracted {} segment embeddings", segs.len());

    let emb_refs: Vec<&[f32]> = segs.iter().map(|s| s.emb.as_slice()).collect();
    let k = match num_speakers {
        Some(k) => k.min(segs.len()),
        None    => estimate_speakers(&emb_refs, 8),
    };
    eprintln!("[diarize] clustering into {} speakers", k);
    let labels = ahc_complete(&emb_refs, k);

    // Build turns from segments — merge adjacent same-speaker segments with small gaps
    let mut turns: Vec<Turn> = segs.iter().zip(&labels).map(|(seg, &lbl)| Turn {
        start:   (seg.start * 1000.0).round() / 1000.0,
        end:     (seg.end   * 1000.0).round() / 1000.0,
        speaker: format!("SPEAKER_{lbl:02}"),
    }).collect();

    turns = merge_adjacent(turns);
    Ok(turns)
}

fn merge_adjacent(mut turns: Vec<Turn>) -> Vec<Turn> {
    if turns.len() < 2 { return turns; }
    let mut out: Vec<Turn> = Vec::new();
    out.push(turns.remove(0));
    for t in turns {
        let last = out.last_mut().unwrap();
        if last.speaker == t.speaker && t.start - last.end <= MERGE_GAP_S {
            last.end = t.end;
        } else {
            out.push(t);
        }
    }
    out
}

// ── Fixed-window fallback pipeline ────────────────────────────────────────────

fn run_fixed_window_pipeline(
    samples: &[f32],
    extractor: &SpeakerEmbeddingExtractor,
    num_speakers: usize,
) -> anyhow::Result<Vec<Turn>> {
    let win_samples  = (WINDOW_S * SAMPLE_RATE as f32) as usize;
    let step_samples = (STEP_S   * SAMPLE_RATE as f32) as usize;

    struct Window { start: f32, end: f32, emb: Vec<f32> }
    let mut windows: Vec<Window> = Vec::new();

    let mut pos = 0usize;
    while pos + win_samples <= samples.len() {
        let chunk = &samples[pos..pos + win_samples];
        let rms = (chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        if rms >= MIN_ENERGY {
            let stream = extractor.create_stream()
                .ok_or_else(|| anyhow::anyhow!("failed to create stream"))?;
            stream.accept_waveform(SAMPLE_RATE, chunk);
            if let Some(emb) = extractor.compute(&stream) {
                windows.push(Window {
                    start: pos as f32 / SAMPLE_RATE as f32,
                    end:   (pos + win_samples) as f32 / SAMPLE_RATE as f32,
                    emb,
                });
            }
        }
        pos += step_samples;
    }

    if windows.is_empty() {
        anyhow::bail!("no speech detected");
    }

    let k = num_speakers.min(windows.len());
    let emb_refs: Vec<&[f32]> = windows.iter().map(|w| w.emb.as_slice()).collect();
    let labels = spectral_cluster(&emb_refs, k);
    let labels = smooth(&labels, SMOOTH_K);

    let mut turns: Vec<Turn> = Vec::new();
    let mut cur_label = labels[0];
    let mut cur_start = windows[0].start;
    let mut cur_end   = windows[0].end;

    for (i, win) in windows.iter().enumerate().skip(1) {
        if labels[i] == cur_label && win.start <= cur_end + STEP_S + 0.05 {
            cur_end = win.end;
        } else {
            turns.push(Turn {
                start:   (cur_start * 1000.0).round() / 1000.0,
                end:     (cur_end   * 1000.0).round() / 1000.0,
                speaker: format!("SPEAKER_{cur_label:02}"),
            });
            cur_label = labels[i];
            cur_start = win.start;
            cur_end   = win.end;
        }
    }
    turns.push(Turn {
        start:   (cur_start * 1000.0).round() / 1000.0,
        end:     (cur_end   * 1000.0).round() / 1000.0,
        speaker: format!("SPEAKER_{cur_label:02}"),
    });

    Ok(merge_short_turns(turns, MIN_TURN_S))
}

// ── Clustering ────────────────────────────────────────────────────────────────

const MAX_AHC_WINDOWS: usize = 800;

fn spectral_cluster(embeddings: &[&[f32]], k: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n <= k { return (0..n).collect(); }
    if k == 1 { return vec![0; n]; }
    if n <= MAX_AHC_WINDOWS {
        ahc_complete(embeddings, k)
    } else {
        kmeans_cosine(embeddings, k)
    }
}


/// Estimate speaker count from the AHC dendrogram.
/// Runs AHC to completion, finds the largest jump in merge distances —
/// that jump is where genuinely different speakers start being forced together.
fn estimate_speakers(embeddings: &[&[f32]], max_k: usize) -> usize {
    let n = embeddings.len();
    if n <= 2 { return n; }

    let normed: Vec<Vec<f32>> = embeddings.iter().map(|e| l2_norm(e)).collect();
    let mut d = vec![vec![f32::MAX; n]; n];
    for i in 0..n {
        d[i][i] = 0.0;
        for j in (i + 1)..n {
            let dist = 1.0 - dot(&normed[i], &normed[j]);
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }

    let mut active = vec![true; n];
    let mut n_act = n;
    // merge_dists[i] = cost of the merge that produced n-(i+1) clusters
    let mut merge_dists: Vec<f32> = Vec::with_capacity(n - 1);

    while n_act > 1 {
        let mut best = (f32::MAX, 0usize, 1usize);
        for i in 0..n {
            if !active[i] { continue; }
            for j in (i + 1)..n {
                if !active[j] { continue; }
                if d[i][j] < best.0 { best = (d[i][j], i, j); }
            }
        }
        let (dist, ai, aj) = best;
        merge_dists.push(dist);

        for l in 0..n {
            if !active[l] || l == ai || l == aj { continue; }
            let nd = d[ai][l].max(d[aj][l]);
            d[ai][l] = nd;
            d[l][ai] = nd;
        }
        active[aj] = false;
        n_act -= 1;
    }

    // Find the largest gap between consecutive merge distances.
    // Gap after merge i → optimal k = n - (i+1) clusters (state just before the expensive merge).
    let max_k = max_k.min(n);
    let mut best_gap = f32::NEG_INFINITY;
    let mut best_k = 2usize.min(max_k);

    for i in 0..merge_dists.len().saturating_sub(1) {
        let gap = merge_dists[i + 1] - merge_dists[i];
        let k_at_gap = n - (i + 1);
        if k_at_gap >= 2 && k_at_gap <= max_k && gap > best_gap {
            best_gap = gap;
            best_k = k_at_gap;
        }
    }

    eprintln!("[diarize] auto-detected {} speakers (largest dendrogram gap: {:.3})", best_k, best_gap);
    best_k
}

fn ahc_complete(embeddings: &[&[f32]], k: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n <= k { return (0..n).collect(); }
    if k == 1 { return vec![0; n]; }

    let normed: Vec<Vec<f32>> = embeddings.iter().map(|e| l2_norm(e)).collect();

    let mut d = vec![vec![f32::MAX; n]; n];
    for i in 0..n {
        d[i][i] = 0.0;
        for j in (i + 1)..n {
            let dist = 1.0 - dot(&normed[i], &normed[j]);
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }

    let mut label  = (0..n).collect::<Vec<_>>();
    let mut active = vec![true; n];
    let mut n_act  = n;

    while n_act > k {
        let mut best = (f32::MAX, 0usize, 1usize);
        for i in 0..n {
            if !active[i] { continue; }
            for j in (i + 1)..n {
                if !active[j] { continue; }
                if d[i][j] < best.0 { best = (d[i][j], i, j); }
            }
        }
        let (_, ai, aj) = best;

        for l in 0..n {
            if !active[l] || l == ai || l == aj { continue; }
            let nd = d[ai][l].max(d[aj][l]);
            d[ai][l] = nd;
            d[l][ai] = nd;
        }
        active[aj] = false;
        n_act -= 1;
        for i in 0..n {
            if label[i] == aj { label[i] = ai; }
        }
    }

    let actives: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
    let mut remap = vec![0usize; n];
    for (new_id, &old_id) in actives.iter().enumerate() {
        remap[old_id] = new_id;
    }
    label.iter().map(|&l| remap[l]).collect()
}

fn kmeans_cosine(embeddings: &[&[f32]], k: usize) -> Vec<usize> {
    let n = embeddings.len();
    let normed: Vec<Vec<f32>> = embeddings.iter().map(|e| l2_norm(e)).collect();
    let mut centroids: Vec<Vec<f32>> = vec![normed[0].clone()];
    while centroids.len() < k {
        let next = (0..n).max_by(|&a, &b| {
            let da = centroids.iter().map(|c| 1.0 - dot(&normed[a], c)).fold(f32::MIN, f32::max);
            let db = centroids.iter().map(|c| 1.0 - dot(&normed[b], c)).fold(f32::MIN, f32::max);
            da.partial_cmp(&db).unwrap()
        }).unwrap();
        centroids.push(normed[next].clone());
    }
    let mut labels = vec![0usize; n];
    for _ in 0..150 {
        let mut changed = false;
        for i in 0..n {
            let best = (0..k).max_by(|&a, &b|
                dot(&normed[i], &centroids[a]).partial_cmp(&dot(&normed[i], &centroids[b])).unwrap()
            ).unwrap();
            if labels[i] != best { labels[i] = best; changed = true; }
        }
        if !changed { break; }
        let dim = centroids[0].len();
        let mut sums   = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, e) in normed.iter().enumerate() {
            sums[labels[i]].iter_mut().zip(e).for_each(|(s, &x)| *s += x);
            counts[labels[i]] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                let m = l2_mag(&sums[ci]);
                centroids[ci] = if m > 1e-9 { sums[ci].iter().map(|&x| x / m).collect() }
                                else { sums[ci].clone() };
            } else {
                // Re-seed empty cluster with the point most distant from its nearest centroid.
                let seed = (0..n).max_by(|&a, &b| {
                    let da = centroids.iter().map(|c| 1.0 - dot(&normed[a], c)).fold(f32::MIN, f32::max);
                    let db = centroids.iter().map(|c| 1.0 - dot(&normed[b], c)).fold(f32::MIN, f32::max);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                }).unwrap_or(0);
                centroids[ci] = normed[seed].clone();
            }
        }
    }
    labels
}

fn smooth(labels: &[usize], k: usize) -> Vec<usize> {
    let n = labels.len();
    let half = k / 2;
    (0..n).map(|i| {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let mut counts = std::collections::HashMap::new();
        for &l in &labels[lo..hi] { *counts.entry(l).or_insert(0usize) += 1; }
        counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0
    }).collect()
}

fn merge_short_turns(mut turns: Vec<Turn>, min_s: f32) -> Vec<Turn> {
    loop {
        let short_idx = turns.iter().enumerate()
            .filter(|(_, t)| t.end - t.start < min_s)
            .min_by(|a, b| (a.1.end - a.1.start).partial_cmp(&(b.1.end - b.1.start)).unwrap())
            .map(|(i, _)| i);
        let Some(i) = short_idx else { break; };
        let n = turns.len();
        let absorb_into = match (i > 0, i + 1 < n) {
            (true, true) => {
                let left  = turns[i - 1].end - turns[i - 1].start;
                let right = turns[i + 1].end - turns[i + 1].start;
                if left >= right { i - 1 } else { i + 1 }
            }
            (true, false) => i - 1,
            (false, true) => i + 1,
            (false, false) => break,
        };
        let short = turns.remove(i);
        let adj_i = if absorb_into > i { absorb_into - 1 } else { absorb_into };
        turns[adj_i].start = turns[adj_i].start.min(short.start);
        turns[adj_i].end   = turns[adj_i].end.max(short.end);
    }
    turns
}

// ── Math ──────────────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(&x, &y)| x * y).sum() }
fn l2_mag(v: &[f32]) -> f32 { v.iter().map(|&x| x * x).sum::<f32>().sqrt() }
fn l2_norm(v: &[f32]) -> Vec<f32> {
    let m = l2_mag(v);
    if m < 1e-9 { v.to_vec() } else { v.iter().map(|&x| x / m).collect() }
}

// ── Audio decode (symphonia + rubato) ─────────────────────────────────────────

fn decode_audio(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SE;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss  = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) { hint.with_extension(ext); }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())?;
    let mut format = probed.format;

    let track = format.tracks().iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no audio track"))?;
    let track_id = track.id;
    let src_rate = track.codec_params.sample_rate
        .ok_or_else(|| anyhow::anyhow!("audio track has no sample rate"))?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;
    let mut raw: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p)  => p,
            Err(SE::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SE::ResetRequired) => continue,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id { continue; }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d, Err(SE::DecodeError(_)) => continue, Err(e) => return Err(e.into()),
        };
        let inv = 1.0 / channels as f32;
        match decoded {
            AudioBufferRef::F32(b) => mix(b.chan(0), (1..channels).map(|c| b.chan(c)).collect(), inv, &mut raw),
            AudioBufferRef::F64(b) => {
                let c0: Vec<f32> = b.chan(0).iter().map(|&x| x as f32).collect();
                let rest: Vec<Vec<f32>> = (1..channels).map(|c| b.chan(c).iter().map(|&x| x as f32).collect()).collect();
                mix(&c0, rest.iter().map(|v| v.as_slice()).collect(), inv, &mut raw);
            }
            AudioBufferRef::S16(b) => {
                let s = 1.0 / i16::MAX as f32;
                let c0: Vec<f32> = b.chan(0).iter().map(|&x| x as f32 * s).collect();
                let rest: Vec<Vec<f32>> = (1..channels).map(|c| b.chan(c).iter().map(|&x| x as f32 * s).collect()).collect();
                mix(&c0, rest.iter().map(|v| v.as_slice()).collect(), inv, &mut raw);
            }
            AudioBufferRef::S32(b) => {
                let s = 1.0 / i32::MAX as f32;
                let c0: Vec<f32> = b.chan(0).iter().map(|&x| x as f32 * s).collect();
                let rest: Vec<Vec<f32>> = (1..channels).map(|c| b.chan(c).iter().map(|&x| x as f32 * s).collect()).collect();
                mix(&c0, rest.iter().map(|v| v.as_slice()).collect(), inv, &mut raw);
            }
            AudioBufferRef::U8(b) => {
                let s = 1.0 / 128.0f32;
                let c0: Vec<f32> = b.chan(0).iter().map(|&x| (x as f32 - 128.0) * s).collect();
                let rest: Vec<Vec<f32>> = (1..channels).map(|c| b.chan(c).iter().map(|&x| (x as f32 - 128.0) * s).collect()).collect();
                mix(&c0, rest.iter().map(|v| v.as_slice()).collect(), inv, &mut raw);
            }
            _ => {}
        }
    }

    if src_rate == 16000 { return Ok(raw); }

    let chunk = 1024usize;
    let mut resampler = FftFixedIn::<f32>::new(src_rate as usize, 16000, chunk, 2, 1)?;
    let mut out = Vec::with_capacity((raw.len() as f64 * 16000.0 / src_rate as f64) as usize + chunk);
    let mut pos = 0;
    while pos < raw.len() {
        let end = (pos + chunk).min(raw.len());
        let mut ch = raw[pos..end].to_vec();
        ch.resize(chunk, 0.0);
        let w = resampler.process(&[ch], None)?;
        out.extend_from_slice(&w[0]);
        pos += chunk;
    }
    let w = resampler.process_partial::<Vec<f32>>(None, None)?;
    if !w.is_empty() { out.extend_from_slice(&w[0]); }
    Ok(out)
}

fn mix<'a>(ch0: &[f32], rest: Vec<&'a [f32]>, inv: f32, out: &mut Vec<f32>) {
    if rest.is_empty() { out.extend_from_slice(ch0); return; }
    for (i, &s) in ch0.iter().enumerate() {
        let sum: f32 = s + rest.iter().map(|ch| ch.get(i).copied().unwrap_or(0.0)).sum::<f32>();
        out.push(sum * inv);
    }
}
