const DIMS: usize = 512;

pub struct FaceEmbedding(pub [f32; DIMS]);
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
