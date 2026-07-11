pub mod detection;
mod error;
pub mod recognition;
use std::path::Path;

pub use error::Error;
pub use error::Result;
pub(crate) mod config;
pub(crate) mod model;
pub mod types;

pub use detection::FaceDetector;
use image::RgbImage;
pub use recognition::FaceRecognizer;
pub use types::{BoundingBox, DetectdFace, Face, FaceEmbedding, FaceLandmarks};

pub struct FaceEngine {
    det: FaceDetector,
    rec: FaceRecognizer,
}
impl FaceEngine {
    pub fn new(det_model_path: impl AsRef<Path>, rec_model_path: impl AsRef<Path>) -> Result<Self> {
        let det = FaceDetector::new(det_model_path, None, None, None)?;
        let rec = FaceRecognizer::new(rec_model_path, None)?;
        Ok(Self { det, rec })
    }

    pub fn run(&mut self, img: &RgbImage) -> Result<Vec<Face>> {
        let faces = self.det.detect(img)?;
        let embeddings = self.rec.extract_embedding(img, &faces)?;
        let results = faces
            .into_iter()
            .zip(embeddings.into_iter())
            .map(|(face, embedding)| Face::from(face, embedding))
            .collect();
        Ok(results)
    }
}
