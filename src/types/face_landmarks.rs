use serde::{Deserialize, Serialize};

/// Five facial landmarks (e.g. eyes, nose, mouth corners).
///
/// Coordinates returned by [`FaceDetector::detect`](crate::FaceDetector) are
/// normalized to `[0, 1]` relative to the original image: `x` coordinates are
/// divided by the image width and `y` coordinates by the image height. Use
/// [`FaceLandmarks::to_absolute`] to convert back to pixels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(from = "[[f32; 2]; 5]", into = "[[f32; 2]; 5]")]
pub struct FaceLandmarks(pub [[f32; 2]; 5]);

impl FaceLandmarks {
    /// Convert these normalized `[0, 1]` coordinates to absolute pixels for an
    /// image of the given size.
    pub fn to_absolute(&self, width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        let mut out = [[0f32; 2]; 5];
        for (i, kp) in self.0.iter().enumerate() {
            out[i] = [kp[0] * w, kp[1] * h];
        }
        Self(out)
    }

    /// Convert these absolute pixel coordinates to normalized `[0, 1]`
    /// coordinates relative to an image of the given size.
    pub fn to_relative(&self, width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        let mut out = [[0f32; 2]; 5];
        for (i, kp) in self.0.iter().enumerate() {
            out[i] = [kp[0] / w, kp[1] / h];
        }
        Self(out)
    }
}

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
