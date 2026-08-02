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
}
