use crate::{
    Result,
    config::{
        common::{ANCHOR_NUM, STRIDES},
        detection::*,
    },
    init::init_ort,
    model::OnnxModel,
    types::{BoundingBox, DetectedFace, FaceLandmarks},
};
use image::{RgbImage, imageops};
use ndarray::{Array4, ArrayViewD};
use ort::{session::SessionOutputs, value::Tensor};
use std::path::Path;

pub struct FaceDetector {
    session: OnnxModel,
    input_size: u32,
    score_threshold: f32,
    nms_threshold: f32,
}

impl FaceDetector {
    pub fn new(
        model_path: impl AsRef<Path>,
        input_size: Option<u32>,
        score_threshold: Option<f32>,
        nms_threshold: Option<f32>,
    ) -> Result<Self> {
        init_ort()?;

        let session = OnnxModel::new(model_path)?;
        Ok(Self {
            session,
            input_size: input_size.unwrap_or(INPUT_SIZE),
            score_threshold: score_threshold.unwrap_or(SCORE_THRESHOLD),
            nms_threshold: nms_threshold.unwrap_or(NMS_THRESHOLD),
        })
    }

    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<DetectedFace>> {
        let (scale, img) = self.preprocess_img(img)?;
        let outputs = self.session.run(Tensor::from_array(img)?)?;
        let faces = Self::process_outputs(outputs, scale, self.input_size, self.score_threshold)?;
        Ok(Self::nms(faces, self.nms_threshold))
    }

    fn preprocess_img(&self, img: &RgbImage) -> Result<(f32, Array4<f32>)> {
        let input_size = self.input_size as usize;
        let (orig_w, orig_h) = img.dimensions();
        let scale = input_size as f32 / orig_w.max(orig_h) as f32;
        let w = (orig_w as f32 * scale) as u32;
        let h = (orig_h as f32 * scale) as u32;

        let resized = imageops::resize(img, w, h, imageops::FilterType::Triangle);

        let num_pixels = input_size * input_size;
        const PAD_VAL: f32 = (0.0 - 127.5) / 128.0;
        let mut chw_data = vec![PAD_VAL; num_pixels * 3];
        let (r_plane, rest) = chw_data.split_at_mut(num_pixels);
        let (g_plane, b_plane) = rest.split_at_mut(num_pixels);

        let raw = resized.as_raw();
        let w_usize = w as usize;
        const INV_SCALE: f32 = 1.0 / 128.0;

        for y in 0..h as usize {
            let row_start = y * w_usize * 3;
            let src_row = &raw[row_start..row_start + w_usize * 3];
            let dst_offset = y * input_size;
            for (x, px) in src_row.chunks_exact(3).enumerate() {
                let idx = dst_offset + x;
                r_plane[idx] = (px[0] as f32 - 127.5) * INV_SCALE;
                g_plane[idx] = (px[1] as f32 - 127.5) * INV_SCALE;
                b_plane[idx] = (px[2] as f32 - 127.5) * INV_SCALE;
            }
        }

        let arr = Array4::from_shape_vec((1, 3, input_size, input_size), chw_data)?;
        Ok((scale, arr))
    }

    fn process_outputs(
        outputs: SessionOutputs,
        scale: f32,
        input_size: u32,
        score_threshold: f32,
    ) -> Result<Vec<DetectedFace>> {
        let scale = 1.0 / scale;

        let output_scores = [&outputs[0], &outputs[1], &outputs[2]];
        let output_bboxes = [&outputs[3], &outputs[4], &outputs[5]];
        let output_kps = [&outputs[6], &outputs[7], &outputs[8]];

        let mut detections: Vec<DetectedFace> = Vec::new();

        for (level, stride) in STRIDES.iter().enumerate() {
            let feat_size = input_size / stride;

            let scores = {
                let scores: ArrayViewD<f32> = output_scores[level].try_extract_array()?;
                let len = scores.len();
                scores.into_shape_with_order((len,))?
            };

            let bboxes = {
                let bboxes: ArrayViewD<f32> = output_bboxes[level].try_extract_array()?;
                let len = bboxes.len();
                bboxes.into_shape_with_order((len / 4, 4))?
            };
            let kps = {
                let kps: ArrayViewD<f32> = output_kps[level].try_extract_array()?;
                let len = kps.len();
                kps.into_shape_with_order((len / 10, 10))?
            };

            let mut i = 0usize;
            let scaling_factor = (*stride as f32) * scale;
            for y in 0..feat_size {
                let anchor_y = y as f32;
                for x in 0..feat_size {
                    let anchor_x = x as f32;
                    for _ in 0..ANCHOR_NUM {
                        let score = scores[i];

                        if score <= score_threshold {
                            i += 1;
                            continue;
                        }

                        let bbox = bboxes.row(i);
                        let kp = kps.row(i);

                        let x1 = (anchor_x - bbox[0]) * scaling_factor;
                        let y1 = (anchor_y - bbox[1]) * scaling_factor;
                        let x2 = (anchor_x + bbox[2]) * scaling_factor;
                        let y2 = (anchor_y + bbox[3]) * scaling_factor;

                        let mut kps = [[0f32; 2]; 5];
                        for p in 0..5 {
                            kps[p][0] = (kp[p * 2] + anchor_x) * scaling_factor;
                            kps[p][1] = (kp[p * 2 + 1] + anchor_y) * scaling_factor;
                        }

                        detections.push(DetectedFace {
                            bbox: BoundingBox { x1, y1, x2, y2 },
                            landmarks: FaceLandmarks(kps),
                            score,
                        });

                        i += 1;
                    }
                }
            }
        }

        Ok(detections)
    }

    fn nms(mut detections: Vec<DetectedFace>, iou_threshold: f32) -> Vec<DetectedFace> {
        detections.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

        let n = detections.len();
        let mut suppressed = vec![false; n];
        let mut kept_idx = Vec::with_capacity(n);

        for i in 0..n {
            if suppressed[i] {
                continue;
            }
            kept_idx.push(i);

            let bbox_i = &detections[i].bbox;
            for j in (i + 1)..n {
                if suppressed[j] {
                    continue;
                }
                if bbox_i.iou(&detections[j].bbox) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }

        let mut kept = Vec::with_capacity(kept_idx.len());
        let mut iter = detections.into_iter().enumerate();
        for idx in kept_idx {
            for (i, d) in iter.by_ref() {
                if i == idx {
                    kept.push(d);
                    break;
                }
            }
        }
        kept
    }
}
