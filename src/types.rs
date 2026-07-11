pub struct DetectdFace {
    pub bbox: BoundingBox,
    pub landmarks: FaceLandmarks,
    pub score: f32,
}

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

pub struct FaceLandmarks(pub [[f32; 2]; 5]);

pub struct FaceEmbedding(pub [f32; 512]);
impl std::ops::Deref for FaceEmbedding {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Face {
    pub score: f32,
    pub bbox: BoundingBox,
    pub landmarks: FaceLandmarks,
    pub embedding: FaceEmbedding,
}
impl Face {
    pub fn from(face: DetectdFace, embedding: FaceEmbedding) -> Self {
        Self {
            score: face.score,
            bbox: face.bbox,
            landmarks: face.landmarks,
            embedding: embedding,
        }
    }
}
