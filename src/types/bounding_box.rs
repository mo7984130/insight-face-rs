use serde::{Deserialize, Serialize};

/// Axis-aligned bounding box in `(x1, y1, x2, y2)` form.
///
/// Coordinates returned by [`FaceDetector::detect`](crate::FaceDetector) are
/// normalized to `[0, 1]` relative to the original image: `x` coordinates are
/// divided by the image width and `y` coordinates by the image height. Use
/// [`BoundingBox::to_absolute`] to convert back to pixels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(from = "[f32; 4]", into = "[f32; 4]")]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}
impl BoundingBox {
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    pub fn inter_area(&self, other: &Self) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.max(other.y2);

        let w = (x2 - x1).max(0.0);
        let h = (y2 - y1).max(0.0);
        w * h
    }

    pub fn union_area(&self, other: &Self, inter_area: f32) -> f32 {
        self.area() + other.area() - inter_area
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let inter_area = self.inter_area(other);
        let union_area = self.union_area(other, inter_area);
        inter_area / union_area
    }

    /// Convert these normalized `[0, 1]` coordinates to absolute pixels for an
    /// image of the given size.
    pub fn to_absolute(&self, width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        Self {
            x1: self.x1 * w,
            y1: self.y1 * h,
            x2: self.x2 * w,
            y2: self.y2 * h,
        }
    }

    /// Convert these absolute pixel coordinates to normalized `[0, 1]`
    /// coordinates relative to an image of the given size.
    pub fn to_relative(&self, width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        Self {
            x1: self.x1 / w,
            y1: self.y1 / h,
            x2: self.x2 / w,
            y2: self.y2 / h,
        }
    }
}

impl From<[f32; 4]> for BoundingBox {
    fn from(value: [f32; 4]) -> Self {
        BoundingBox {
            x1: value[0],
            y1: value[1],
            x2: value[2],
            y2: value[3],
        }
    }
}
impl From<BoundingBox> for [f32; 4] {
    fn from(value: BoundingBox) -> Self {
        [value.x1, value.y1, value.x2, value.y2]
    }
}
