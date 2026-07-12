//! Comprehensive benchmark: detect faces, extract embeddings, and save an
//! annotated image with bounding boxes, 5-point landmarks, and per-face
//! scores — all with detailed timing breakdowns across multiple runs.
//!
//! Usage:
//! ```bash
//! cargo run --release --example detect_and_recognize -- \
//!     path/to/det_640.onnx \
//!     path/to/w600k_r50.onnx \
//!     path/to/input.jpg \
//!     [output.jpg] \
//!     [num_runs] \
//!     [font.ttf]
//! ```
//!
//! - `num_runs` defaults to 10. A warm-up iteration runs first and is excluded
//!   from reported statistics.
//! - `font.ttf` is optional. When provided, score labels are drawn on the
//!   image; otherwise scores are only printed to stdout.

use std::time::{Duration, Instant};

use ab_glyph::FontVec;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use insight_face_rs::{DetectedFace, FaceDetector, FaceEmbedding, FaceRecognizer};

const RED: Rgb<u8> = Rgb([255, 0, 0]);
const GREEN: Rgb<u8> = Rgb([0, 255, 0]);

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let det_model = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing detection model path"))?;
    let rec_model = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing recognition model path"))?;
    let image_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing input image path"))?;
    let output_path = args.next().unwrap_or_else(|| "output.jpg".to_string());
    let num_runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let font_path = args.next();

    // ------------------------------------------------------------------
    // 1. Load models
    // ------------------------------------------------------------------
    println!("=== insight-face-rs Benchmark ===\n");
    let load_start = Instant::now();
    let mut detector = FaceDetector::new(&det_model, None, None, None)?;
    let mut recognizer = FaceRecognizer::new(&rec_model, None)?;
    let load_time = load_start.elapsed();
    println!("models loaded in {:8.2} ms", ms(load_time));
    println!("  detection  : {det_model}");
    println!("  recognition: {rec_model}");
    println!();

    // ------------------------------------------------------------------
    // 2. Load image once (not measured in runs)
    // ------------------------------------------------------------------
    let img: RgbImage = image::open(&image_path)?.to_rgb8();
    println!(
        "image: {}  ({} × {})",
        image_path,
        img.width(),
        img.height()
    );
    println!();

    // ------------------------------------------------------------------
    // 3. Warm-up run (excluded from statistics)
    // ------------------------------------------------------------------
    println!("--- warm-up (1 iteration) ---");
    warmup(&mut detector, &mut recognizer, &img)?;
    println!();

    // ------------------------------------------------------------------
    // 4. Benchmark loop
    // ------------------------------------------------------------------
    println!("--- benchmark ({num_runs} iterations) ---");

    let mut detect_times = Vec::with_capacity(num_runs);
    let mut recog_times = Vec::with_capacity(num_runs);
    let mut total_times = Vec::with_capacity(num_runs);
    let mut face_counts = Vec::with_capacity(num_runs);

    // The last run's data is retained for drawing / detailed output.
    let mut last_faces: Vec<DetectedFace> = Vec::new();
    let mut last_embeddings: Vec<FaceEmbedding> = Vec::new();

    for run in 0..num_runs {
        // --- detection ---
        let t0 = Instant::now();
        let faces: Vec<DetectedFace> = detector.detect(&img)?;
        let dt = t0.elapsed();

        // --- recognition ---
        let t1 = Instant::now();
        let embeddings: Vec<FaceEmbedding> = recognizer.extract_embedding(&img, &faces)?;
        let rt = t1.elapsed();
        let total = t0.elapsed();

        detect_times.push(dt);
        recog_times.push(rt);
        total_times.push(total);
        face_counts.push(faces.len());

        println!(
            "  run {:3}  detect={:>8.2} ms  recog={:>8.2} ms  total={:>8.2} ms  faces={}",
            run + 1,
            ms(dt),
            ms(rt),
            ms(total),
            faces.len()
        );

        if run == num_runs - 1 {
            last_faces = faces;
            last_embeddings = embeddings;
        }
    }

    // ------------------------------------------------------------------
    // 5. Statistics
    // ------------------------------------------------------------------
    println!();
    println!("--- statistics (over {num_runs} runs) ---");

    print_stats("detection", &detect_times);
    print_stats("recognition", &recog_times);
    print_stats("total      ", &total_times);

    let avg_faces: f64 =
        face_counts.iter().map(|&c| c as f64).sum::<f64>() / face_counts.len() as f64;
    println!(
        "  {:<12}  avg={:>7.1}  faces per image",
        "face count", avg_faces,
    );
    println!();

    // ------------------------------------------------------------------
    // 6. Per-face details (from the last run)
    // ------------------------------------------------------------------
    println!("--- per-face details (last run) ---");
    for (i, face) in last_faces.iter().enumerate() {
        let b = &face.bbox;
        println!(
            "  face {i}: score={:.4}  bbox=({:.0},{:.0},{:.0},{:.0})  area={:.0}",
            face.score,
            b.x1,
            b.y1,
            b.x2,
            b.y2,
            b.area()
        );
        println!("          landmarks:");
        for (kp_i, kp) in face.landmarks.0.iter().enumerate() {
            println!("            {kp_i}: ({:.1}, {:.1})", kp[0], kp[1]);
        }
    }
    println!();

    // ------------------------------------------------------------------
    // 7. Pairwise cosine similarities (from the last run)
    // ------------------------------------------------------------------
    if last_embeddings.len() >= 2 {
        println!("--- pairwise cosine similarity ---");
        for i in 0..last_embeddings.len() {
            for j in (i + 1)..last_embeddings.len() {
                let sim = cosine_similarity(&last_embeddings[i], &last_embeddings[j]);
                println!(
                    "  face {i} ↔ face {j}:  sim = {:.6}  (1.0 = identical)",
                    sim
                );
            }
        }
        println!();
    }

    // ------------------------------------------------------------------
    // 8. Draw annotated image (from the last run)
    // ------------------------------------------------------------------
    let font: Option<FontVec> = match font_path {
        Some(path) => {
            let data = std::fs::read(&path)?;
            Some(
                FontVec::try_from_vec(data)
                    .map_err(|_| anyhow::anyhow!("failed to parse font: {}", path))?,
            )
        }
        None => None,
    };

    let mut out = img.clone();
    for (i, face) in last_faces.iter().enumerate() {
        let b = &face.bbox;
        let x = b.x1 as i32;
        let y = b.y1 as i32;
        let w = (b.x2 - b.x1) as u32;
        let h = (b.y2 - b.y1) as u32;
        draw_hollow_rect_mut(&mut out, Rect::at(x, y).of_size(w, h), RED);

        for kp in &face.landmarks.0 {
            draw_cross_mut(&mut out, GREEN, kp[0] as i32, kp[1] as i32);
        }

        if let Some(f) = font.as_ref() {
            let label = format!("face {i}: {:.2}", face.score);
            draw_text_mut(&mut out, RED, x, y.saturating_sub(16), 16.0, f, &label);
        } else {
            println!(
                "  face {i}: score={:.3}  bbox=({:.0},{:.0},{:.0},{:.0})",
                face.score, b.x1, b.y1, b.x2, b.y2
            );
        }
    }
    out.save(&output_path)?;
    println!("annotated image saved to {output_path}");

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Single warm-up to prime GPU/CPU caches.
fn warmup(
    detector: &mut FaceDetector,
    recognizer: &mut FaceRecognizer,
    img: &RgbImage,
) -> anyhow::Result<()> {
    let faces = detector.detect(img)?;
    let _embs = recognizer.extract_embedding(img, &faces)?;
    println!("  detected {} face(s)", faces.len());
    Ok(())
}

/// Cosine similarity of two 512-d embeddings.
fn cosine_similarity(a: &FaceEmbedding, b: &FaceEmbedding) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Duration → fractional milliseconds.
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Print min / max / avg / median / P95 for a sequence of timings.
fn print_stats(label: &str, times: &[Duration]) {
    let n = times.len();
    if n == 0 {
        return;
    }
    let mut ms_vals: Vec<f64> = times.iter().copied().map(ms).collect();
    ms_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let min = ms_vals[0];
    let max = ms_vals[n - 1];
    let avg = ms_vals.iter().sum::<f64>() / n as f64;
    let median = ms_vals[n / 2];
    let p95 = ms_vals[(n as f64 * 0.95).ceil() as usize - 1];

    println!(
        "  {label}  \
         min={min:>8.2}  max={max:>8.2}  avg={avg:>8.2}  \
         median={median:>8.2}  P95={p95:>8.2}  ms"
    );
}
