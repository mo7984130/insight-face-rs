# insight-face-rs

> English | [中文](#中文)

Rust implementation of InsightFace-style face analysis, powered by ONNX Runtime
(`ort`). It provides lightweight face detection and face recognition
(embedding extraction) capabilities.

- **Face detection** — a single-stage, SCRFD-style detector that outputs bounding
  boxes (`bbox`) and 5-point facial landmarks (`landmarks`).
- **Face recognition** — an ArcFace-style recognizer that encodes an aligned face
  into a 512-dimensional embedding.
- Supports CUDA / CPU execution providers (CUDA is preferred automatically).

## Models

This crate does **not** bundle model weights. You need to supply two ONNX models
yourself:

| Purpose | Type | Input | Output |
| --- | --- | --- | --- |
| Detection (`FaceDetector`) | SCRFD family | `(1, 3, 640, 640)` BGR, normalized `(x-127.5)/128` | 9 branches: 3 levels of scores / bboxes / kps |
| Recognition (`FaceRecognizer`) | ArcFace family | `(1, 3, 112, 112)` BGR, normalized `(x-127.5)/128` | `(1, 512)` feature vector |

Input tensors use NCHW layout and RGB channel order (the crate performs the
`(x-127.5)/128` normalization internally).

## Installation

```toml
[dependencies]
insight-face-rs = "0.1"
# Pin the ort version explicitly (the CUDA EP requires a matching CUDA runtime)
ort = ">=2.0.0-rc.12, <3"
```

> Using the CUDA execution provider requires a matching CUDA / cuDNN runtime in
> your environment; otherwise execution falls back to CPU.

## Quick start

A full runnable example is available at
[`examples/detect_and_recognize.rs`](examples/detect_and_recognize.rs).

```rust
use image::RgbImage;
use insight_face_rs::{FaceDetector, FaceEmbedding, FaceRecognizer};

fn main() -> anyhow::Result<()> {
    // 1. Load models
    let mut detector = FaceDetector::new(
        "models/det_640.onnx",
        None, // input_size, defaults to 640
        None, // score_threshold, defaults to 0.6
        None, // nms_threshold, defaults to 0.4
    )?;
    let mut recognizer = FaceRecognizer::new("models/w600k_r50.onnx", None)?;

    // 2. Read the image and detect faces
    let img = image::open("examples/person.jpg")?.to_rgb8();
    let faces = detector.detect(&img)?;
    println!("detected {} face(s)", faces.len());

    // 3. Extract a 512-d embedding for each detected face
    let embeddings: Vec<FaceEmbedding> = recognizer.extract_embedding(img, faces)?;

    // 4. Compute cosine similarity (FaceEmbedding derefs to &[f32])
    let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
    println!("similarity: {sim:.4}");

    Ok(())
}

fn cosine_similarity(a: &FaceEmbedding, b: &FaceEmbedding) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}
```

## API overview

- `FaceDetector::new(model_path, input_size, score_threshold, nms_threshold)` —
  loads the detection model; the last three arguments accept `None` to use the
  defaults.
- `FaceDetector::detect(&mut self, img: &RgbImage) -> Result<Vec<DetectdFace>>` —
  returns each face's `bbox`, `landmarks` (5 points) and `score`; NMS is already
  applied.
- `FaceRecognizer::new(model_path, input_size)` — loads the recognition model.
- `FaceRecognizer::extract_embedding(&mut self, img, faces) -> Result<Vec<FaceEmbedding>>` —
  aligns each face by its landmarks internally and outputs 512-d vectors.
- `FaceEmbedding` implements `Deref<Target = [f32]>`, so it can be used directly
  as a slice for similarity computation.

## License

MIT

---

# 中文

基于 ONNX Runtime (`ort`) 实现的 InsightFace 风格人脸分析工具库，提供轻量的人脸检测与
人脸识别（特征提取）能力。

- **人脸检测**：基于 SCRFD 风格的单阶段检测器，输出人脸包围框 (`bbox`) 与五点关键点 (`landmarks`)。
- **人脸识别**：基于 ArcFace 风格的识别模型，将对齐后的人脸编码为 512 维向量 (`embedding`)。
- 支持 CUDA / CPU 执行提供器（自动优先使用 CUDA）。

## 模型

本库不内置模型权重，需要你自己准备两个 ONNX 模型：

| 用途 | 类型 | 输入 | 输出 |
| --- | --- | --- | --- |
| 检测 `FaceDetector` | SCRFD 系列 | `(1, 3, 640, 640)` BGR, 归一化 `(x-127.5)/128` | 9 个分支：3 层 scores / bboxes / kps |
| 识别 `FaceRecognizer` | ArcFace 系列 | `(1, 3, 112, 112)` BGR, 归一化 `(x-127.5)/128` | `(1, 512)` 特征向量 |

输入张量为 NCHW 布局、RGB 顺序（代码内部已做 `(x-127.5)/128` 归一化）。

## 安装

```toml
[dependencies]
insight-face-rs = "0.1"
# 显式约束 ort 版本（CUDA EP 需要对应 CUDA 运行时）
ort = ">=2.0.0-rc.12, <3"
```

> 使用 CUDA 执行提供器需要在环境中安装匹配的 CUDA / cuDNN 运行时；
> 否则会回退到 CPU。

## 快速开始

完整可运行示例见 [`examples/detect_and_recognize.rs`](examples/detect_and_recognize.rs)。

```rust
use image::RgbImage;
use insight_face_rs::{FaceDetector, FaceEmbedding, FaceRecognizer};

fn main() -> anyhow::Result<()> {
    // 1. 加载模型
    let mut detector = FaceDetector::new(
        "models/det_640.onnx",
        None, // input_size，默认 640
        None, // score_threshold，默认 0.6
        None, // nms_threshold，默认 0.4
    )?;
    let mut recognizer = FaceRecognizer::new("models/w600k_r50.onnx", None)?;

    // 2. 读取图像并检测人脸
    let img = image::open("examples/person.jpg")?.to_rgb8();
    let faces = detector.detect(&img)?;
    println!("检测到 {} 张人脸", faces.len());

    // 3. 对检测到的每张人脸提取 512 维特征
    let embeddings: Vec<FaceEmbedding> = recognizer.extract_embedding(img, faces)?;

    // 4. 计算余弦相似度（FaceEmbedding 解引用为 &[f32]）
    let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
    println!("相似度: {sim:.4}");

    Ok(())
}

fn cosine_similarity(a: &FaceEmbedding, b: &FaceEmbedding) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}
```

## API 概览

- `FaceDetector::new(model_path, input_size, score_threshold, nms_threshold)` —
  加载检测模型，后三个参数均可传 `None` 使用默认值。
- `FaceDetector::detect(&mut self, img: &RgbImage) -> Result<Vec<DetectdFace>>` —
  返回每张脸的 `bbox`、`landmarks`（5 点）与 `score`，已做 NMS 抑制。
- `FaceRecognizer::new(model_path, input_size)` — 加载识别模型。
- `FaceRecognizer::extract_embedding(&mut self, img, faces) -> Result<Vec<FaceEmbedding>>` —
  内部根据关键点对齐人脸并输出 512 维向量。
- `FaceEmbedding` 实现了 `Deref<Target = [f32]>`，可直接当作切片进行相似度计算。

## 许可证

MIT
