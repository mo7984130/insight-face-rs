use crate::types::{BoundingBox, FaceLandmarks};

/// A single face produced by [`FaceDetector::detect`](crate::FaceDetector).
///
/// Both [`bbox`](Self::bbox) and [`landmarks`](Self::landmarks) use
/// coordinates normalized to `[0, 1]` relative to the original image.
pub struct DetectedFace {
    pub bbox: BoundingBox,
    pub landmarks: FaceLandmarks,
    pub score: f32,
}
