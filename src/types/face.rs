use crate::types::{BoundingBox, DetectedFace, FaceEmbedding, FaceLandmarks};

pub struct Face {
    pub score: f32,
    pub bbox: BoundingBox,
    pub landmarks: FaceLandmarks,
    pub embedding: FaceEmbedding,
}
impl Face {
    pub fn from(face: DetectedFace, embedding: FaceEmbedding) -> Self {
        Self {
            score: face.score,
            bbox: face.bbox,
            landmarks: face.landmarks,
            embedding: embedding,
        }
    }
}
