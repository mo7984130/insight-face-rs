use crate::types::{BoundingBox, FaceLandmarks};

pub struct DetectedFace {
    pub bbox: BoundingBox,
    pub landmarks: FaceLandmarks,
    pub score: f32,
}
