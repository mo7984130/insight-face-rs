use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(from = "[[f32; 2]; 5]", into = "[[f32; 2]; 5]")]
pub struct FaceLandmarks(pub [[f32; 2]; 5]);

impl From<[[f32; 2]; 5]> for FaceLandmarks {
    fn from(value: [[f32; 2]; 5]) -> Self {
        Self(value)
    }
}

impl From<FaceLandmarks> for [[f32; 2]; 5] {
    fn from(value: FaceLandmarks) -> Self {
        value.0
    }
}
