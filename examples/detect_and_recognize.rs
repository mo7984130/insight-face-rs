//! Example: detect faces in an image, extract embeddings, and save an annotated
//! image with bounding boxes, 5-point landmarks, and per-face scores.
//!
//! Usage:
//! ```bash
//! cargo run --release --example detect_and_recognize -- \
//!     path/to/det_640.onnx \
//!     path/to/w600k_r50.onnx \
//!     path/to/input.jpg \
//!     [path/to/output.jpg] \
//!     [path/to/font.ttf]
//! ```
//!
//! The font argument is optional. When provided, score labels are drawn on the
//! image; otherwise scores are only printed to stdout.

use ab_glyph::FontVec;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use insight_face_rs::{FaceDetector, FaceEmbedding, FaceRecognizer};

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
    let font_path = args.next();

    // 1. Load models (pass None to use default parameters).
    let mut detector = FaceDetector::new(det_model, None, None, None)?;
    let mut recognizer = FaceRecognizer::new(rec_model, None)?;

    // 2. Read the image and detect faces.
    let img: RgbImage = image::open(&image_path)?.to_rgb8();
    let faces = detector.detect(&img)?;
    println!("detected {} face(s) in {}", faces.len(), image_path);

    // 3. Prepare an annotated copy of the image.
    let mut out = img.clone();

    // 4. Load an optional font for drawing score labels.
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

    // 5. Draw bounding boxes, landmarks and scores.
    for (i, face) in faces.iter().enumerate() {
        let b = &face.bbox;
        let x = b.x1 as i32;
        let y = b.y1 as i32;
        let w = (b.x2 - b.x1) as u32;
        let h = (b.y2 - b.y1) as u32;
        draw_hollow_rect_mut(&mut out, Rect::at(x, y).of_size(w, h), RED);

        for kp in &face.landmarks.0 {
            draw_cross_mut(&mut out, GREEN, kp[0] as i32, kp[1] as i32);
        }

        if let Some(font) = font.as_ref() {
            let label = format!("face {}: {:.2}", i, face.score);
            draw_text_mut(&mut out, RED, x, y.saturating_sub(16), 16.0, font, &label);
        } else {
            println!("  face {}: score={:.3}", i, face.score);
        }
    }
    out.save(&output_path)?;
    println!("saved annotated image to {}", output_path);

    // 6. Extract a 512-d embedding for each detected face, then compare the
    //    first two by cosine similarity.
    let embeddings: Vec<FaceEmbedding> = recognizer.extract_embedding(img, faces)?;
    if embeddings.len() >= 2 {
        let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
        println!(
            "cosine similarity between face 0 and face 1: {:.4} (closer to 1 means more similar)",
            sim
        );
    }

    Ok(())
}

/// Compute the cosine similarity of two embeddings. `FaceEmbedding` derefs to
/// `&[f32]`, so it can be iterated directly.
fn cosine_similarity(a: &FaceEmbedding, b: &FaceEmbedding) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
