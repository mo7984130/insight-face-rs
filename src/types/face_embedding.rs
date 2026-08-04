use serde::Deserialize;
use serde::Serialize;
use serde_big_array::BigArray;

pub const DIMS: usize = 512;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceEmbedding(#[serde(with = "BigArray")] pub [f32; DIMS]);
impl std::ops::Deref for FaceEmbedding {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "pgvector")]
impl From<pgvector::Vector> for FaceEmbedding {
    fn from(value: pgvector::Vector) -> Self {
        assert_eq!(
            value.as_slice().len(),
            512,
            "Vector must have exactly {} dimensions for FaceEmbedding",
            DIMS
        );
        let mut embedding = [0.0f32; DIMS];
        for (i, val) in value.as_slice().iter().enumerate() {
            embedding[i] = *val;
        }
        FaceEmbedding(embedding)
    }
}

#[cfg(feature = "pgvector")]
impl From<FaceEmbedding> for pgvector::Vector {
    fn from(value: FaceEmbedding) -> Self {
        pgvector::Vector::from(value.to_vec())
    }
}

impl FaceEmbedding {
    #[inline]
    pub fn normalize(mut self) -> Self {
        let norm = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm > 0.0 {
            let inv_norm = 1.0 / norm;

            for x in &mut self.0 {
                *x *= inv_norm;
            }
        }

        self
    }

    /// score 加权和: 返回 `(weight, centroid)`
    ///
    /// - `weight` = Σscore(f64, 对应 `photo_person.weight` 列)
    /// - `centroid` = Σ(score × embedding)(未归一化, f64 累加后一次性截断为 f32)
    ///
    /// 约定: centroid 存储 **score 加权未归一化向量和**,读取/检索时再 normalize。
    /// 归一化会丢失模长, 无法由归一化质心反推加权和, 因此增量维护必须基于原始和。
    pub fn weighted_sum<'a>(
        items: impl IntoIterator<Item = (f32, &'a FaceEmbedding)>,
    ) -> (f64, FaceEmbedding) {
        let mut weight = 0.0f64;
        let mut sum = [0.0f64; DIMS];
        for (score, embedding) in items {
            weight += score as f64;
            for (i, value) in embedding.iter().enumerate() {
                sum[i] += score as f64 * *value as f64;
            }
        }
        (weight, FaceEmbedding(sum.map(|v| v as f32)))
    }

    /// 向量和: `self + other`(合并人物用)
    pub fn add(&self, other: &Self) -> Self {
        let mut out = [0.0f32; DIMS];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *o = x + y;
        }
        FaceEmbedding(out)
    }

    /// 向量增量加: `self + score × e`(转移人脸到新人物用)
    pub fn add_scaled(&self, e: &Self, score: f32) -> Self {
        let mut out = [0.0f32; DIMS];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(e.0.iter())) {
            *o = x + score * y;
        }
        FaceEmbedding(out)
    }

    /// 向量增量减: `self − score × e`(从旧人物移除人脸用)
    pub fn sub_scaled(&self, e: &Self, score: f32) -> Self {
        let mut out = [0.0f32; DIMS];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(e.0.iter())) {
            *o = x - score * y;
        }
        FaceEmbedding(out)
    }
}
