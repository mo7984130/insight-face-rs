pub mod detection;
mod error;
pub mod recognition;
pub use error::Error;
pub use error::Result;
pub(crate) mod config;
pub(crate) mod model;
pub mod types;

pub use detection::FaceDetector;
pub use recognition::FaceRecognizer;
pub use types::{BoundingBox, DetectdFace, FaceEmbedding, FaceLandmarks, FaceResult};
