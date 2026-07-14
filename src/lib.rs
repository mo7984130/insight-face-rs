pub mod detection;
mod error;
mod face_engine;
pub(crate) mod init;
pub mod recognition;

pub use error::Error;
pub use error::Result;
pub(crate) mod config;
pub(crate) mod model;
pub mod types;

pub use detection::FaceDetector;
pub use face_engine::FaceEngine;
pub use recognition::FaceRecognizer;
pub use types::{BoundingBox, DetectedFace, Face, FaceEmbedding, FaceLandmarks};
