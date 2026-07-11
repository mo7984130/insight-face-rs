use std::path::Path;

use image::{Rgb, RgbImage};
use imageproc::geometric_transformations::{Border, Projection, warp_into};
use nalgebra::{Matrix2, Matrix2x3, Vector2};
use ndarray::Array4;
use ort::value::Tensor;

use crate::config::recognizer::*;
use crate::model::OnnxModel;
use crate::types::{DetectdFace, FaceEmbedding};
use crate::{Error, Result};

pub struct FaceRecognizer {
    session: OnnxModel,
    input_size: u32,
}

impl FaceRecognizer {
    pub fn new(model_path: impl AsRef<Path>, input_size: Option<u32>) -> Result<Self> {
        let session = OnnxModel::new(model_path)?;
        Ok(Self {
            session,
            input_size: input_size.unwrap_or(INPUT_SIZE),
        })
    }

    pub fn extract_embedding(
        &mut self,
        img: &RgbImage,
        faces: &Vec<DetectdFace>,
    ) -> Result<Vec<FaceEmbedding>> {
        let mut results = Vec::with_capacity(faces.len());

        for face in faces {
            let m = Self::estimate_similarity_transform(&face.landmarks.0, &ARCFACE_TEMPLATE);
            let aligned = Self::align_face(self.input_size, &img, &m)?;
            // let _ = aligned.save(format!("./test-imgs/outputs/{}.jpg", face.score));
            let blob = Self::to_blob(self.input_size, &aligned);

            let outputs = self.session.run(Tensor::from_array(blob)?)?;
            let view = outputs[0].try_extract_array::<f32>()?;
            let slice = view.as_slice().ok_or_else(|| {
                Error::ModelRunError(
                    "The memory layout of the output is not continuous".to_string(),
                )
            })?;
            let embedding: [f32; 512] = std::array::from_fn(|i| slice[i]);

            results.push(FaceEmbedding(embedding));
        }

        Ok(results)
    }

    fn estimate_similarity_transform(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> Matrix2x3<f32> {
        let n = src.len() as f32;

        let mut src_mean = Vector2::zeros();
        let mut dst_mean = Vector2::zeros();
        for i in 0..5 {
            src_mean += Vector2::new(src[i][0], src[i][1]);
            dst_mean += Vector2::new(dst[i][0], dst[i][1]);
        }
        src_mean /= n;
        dst_mean /= n;

        let mut sigma2 = 0.0f32;
        let mut cov = Matrix2::zeros();
        for i in 0..5 {
            let sx = Vector2::new(src[i][0], src[i][1]) - src_mean;
            let dx = Vector2::new(dst[i][0], dst[i][1]) - dst_mean;
            sigma2 += sx.dot(&sx);
            cov += dx * sx.transpose();
        }
        sigma2 /= n;
        cov /= n;

        let svd = cov.svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let s = svd.singular_values;

        let det_cov = cov.determinant();
        let mut d = Matrix2::identity();
        if det_cov < 0.0 || (det_cov == 0.0 && u.determinant() * vt.determinant() < 0.0) {
            d[(1, 1)] = -1.0;
        }

        let r = u * d * vt;
        let trace_ds = s[0] * d[(0, 0)] + s[1] * d[(1, 1)];
        let scale = trace_ds / sigma2;
        let t = dst_mean - scale * (r * src_mean);

        let mut m = Matrix2x3::zeros();
        m.fixed_view_mut::<2, 2>(0, 0).copy_from(&(scale * r));
        m.set_column(2, &t);
        m
    }

    fn align_face(input_size: u32, img: &RgbImage, m: &Matrix2x3<f32>) -> Result<RgbImage> {
        let projection = Projection::from_matrix([
            m[(0, 0)],
            m[(0, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(1, 1)],
            m[(1, 2)],
            0.0,
            0.0,
            1.0,
        ])
        .ok_or_else(|| {
            Error::AlignFaceError("The transformation matrix is not invertible".to_string())
        })?;

        let mut aligned = RgbImage::new(input_size, input_size);

        warp_into(
            img,
            projection,
            imageproc::geometric_transformations::Interpolation::Bilinear,
            Border::Constant(Rgb([0u8, 0u8, 0u8])),
            &mut aligned,
        );

        Ok(aligned)
    }

    fn to_blob(input_size: u32, aligned: &RgbImage) -> Array4<f32> {
        let mut blob = Array4::<f32>::zeros((1, 3, input_size as usize, input_size as usize));

        for y in 0..input_size {
            for x in 0..input_size {
                let Rgb([r, g, b]) = *aligned.get_pixel(x, y);
                blob[[0, 0, y as usize, x as usize]] = (r as f32 - 127.5) / 128.0;
                blob[[0, 1, y as usize, x as usize]] = (g as f32 - 127.5) / 128.0;
                blob[[0, 2, y as usize, x as usize]] = (b as f32 - 127.5) / 128.0;
            }
        }
        blob
    }
}
