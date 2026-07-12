use serde::{Deserialize, Serialize};

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
