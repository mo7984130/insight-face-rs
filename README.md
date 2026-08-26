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
insight-face-rs = "1.0"
```

### Backend selection

Two ONNX backends are available; the default is **`backend-onnxruntime`**:

| Feature | Description | Pros | Cons |
| --- | --- | --- | --- |
| `backend-onnxruntime`<br>(default) | Full ONNX Runtime via `ort/download-binaries` | Broad ONNX op support, dynamic input shapes | Larger binary, downloads prebuilt libs at build time |
| `backend-tract` | Pure-Rust `tract`-based interpreter | Lighter binary, no native deps | Limited op support, **requires static input shape** (see below) |

#### Switch to `backend-tract`

```toml
[dependencies]
insight-face-rs = { version = "3.0", default-features = false, features = ["backend-tract"] }
```

> ⚠ **tract limitation**: The `tract` backend cannot load ONNX models with dynamic
> (symbolic) input dimensions. If you get `Translating node #0 … Source
> ToTypedTranslator` during model loading, your model has dynamic shapes. Use
> [onnxsim](https://github.com/daquexian/onnx-simplifier) to fix the input shape:
> ```bash
> pip install onnxsim
> python -m onnxsim model.onnx model_fixed.onnx \
>     --overwrite-input-shape input.1:1,3,640,640
> ```
> Or simply switch back to the default `backend-onnxruntime`, which handles
> dynamic shapes natively.

If you choose to use a CUDA execution provider, ensure a matching CUDA / cuDNN
runtime is installed in your environment; otherwise execution falls back to CPU.

## Quick start

A full runnable example is available at
[`examples/detect_and_recognize.rs`](examples/detect_and_recognize.rs).

```rust
use std::time::Duration;

use insight_face_rs::{FaceEngine, FaceEngineConfig};

fn main() -> anyhow::Result<()> {
    // 1. Configure and load models
    let config = FaceEngineConfig::new(
        "models/det_640.onnx",
        "models/w600k_r50.onnx",
        Duration::from_secs(60),
    );
    let engine = FaceEngine::new(&config)?;

    // 2. Read the image and detect and recognize faces
    let img = image::open("examples/person.jpg")?.to_rgb8();
    let faces = engine.run(&img)?;
    println!("detected {} face(s)", faces.len());

    // 3. Compute cosine similarity between two face embeddings
    let sim = faces[0].embedding.cosine_similarity(&faces[1].embedding);
    println!("similarity: {sim:.4}");

    Ok(())
}
```

## API overview

- `FaceEngineConfig::new(det_model_path, rec_model_path, idle_timeout)` —
  configures model paths and the idle timeout used for reclaiming model memory.
  Set `idle_timeout` to `Duration::ZERO` to disable automatic reclamation.
- `FaceEngine::new(&config)` — loads both models immediately;
  `FaceEngine::new_without_load(&config)` defers loading until the first run.
- `FaceEngine::run(&self, img: &RgbImage) -> Result<Vec<Face>>` — detects and
  recognizes every face, returning its detection data and 512-d embedding.
- `FaceEngine::unload()` and `FaceEngine::reclaim_if_idle()` release model
  memory manually or after the configured idle timeout.
- `FaceDetector::new(model_path, input_size, score_threshold, nms_threshold)` —
  loads the detection model; the last three arguments accept `None` to use the
  defaults.
- `FaceDetector::detect(&mut self, img: &RgbImage) -> Result<Vec<DetectedFace>>` —
  returns each face's `bbox`, `landmarks` (5 points) and `score`; NMS is already
  applied. `bbox` and `landmarks` coordinates are normalized to `[0, 1]`
  relative to the original image (`x` / width, `y` / height). Use
  `BoundingBox::to_absolute(w, h)` / `FaceLandmarks::to_absolute(w, h)` to
  convert back to pixels.
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
insight-face-rs = "1.0"
```

### 后端选择

提供了两种 ONNX 后端，默认使用 **`backend-onnxruntime`**：

| 功能 | 描述 | 优点 | 缺点 |
| --- | --- | --- | --- |
| `backend-onnxruntime`<br>(默认) | 完整 ONNX Runtime，通过 `ort/download-binaries` | 算子支持全面，支持动态输入 shape | 二进制较大，构建时会下载预编译库 |
| `backend-tract` | 纯 Rust 的 `tract` 解释器 | 更轻量，无原生依赖 | 算子支持有限，**要求静态输入 shape**（见下文） |

#### 切换为 `backend-tract`

```toml
[dependencies]
insight-face-rs = { version = "3.0", default-features = false, features = ["backend-tract"] }
```

> ⚠ **tract 限制**： `tract` 后端无法加载带有动态（符号）输入维度的 ONNX 模型。
> 如果在加载模型时出现 `Translating node #0 … Source ToTypedTranslator`，说明模型含有动态 shape。
> 可以通过 [onnxsim](https://github.com/daquexian/onnx-simplifier) 固定输入 shape：
> ```bash
> pip install onnxsim
> python -m onnxsim model.onnx model_fixed.onnx \
>     --overwrite-input-shape input.1:1,3,640,640
> ```
> 或直接切换回默认的 `backend-onnxruntime`，它对动态 shape 有原生支持。

如果使用 CUDA 执行提供器，需要在环境中安装匹配的 CUDA / cuDNN 运行时；
否则会回退到 CPU。

## 快速开始

完整可运行示例见 [`examples/detect_and_recognize.rs`](examples/detect_and_recognize.rs)。

```rust
use std::time::Duration;

use insight_face_rs::{FaceEngine, FaceEngineConfig};

fn main() -> anyhow::Result<()> {
    // 1. 配置并加载模型
    let config = FaceEngineConfig::new(
        "models/det_640.onnx",
        "models/w600k_r50.onnx",
        Duration::from_secs(60),
    );
    let engine = FaceEngine::new(&config)?;

    // 2. 读取图像并完成人脸检测与识别
    let img = image::open("examples/person.jpg")?.to_rgb8();
    let faces = engine.run(&img)?;
    println!("检测到 {} 张人脸", faces.len());

    // 3. 计算两张人脸特征的余弦相似度
    let sim = faces[0].embedding.cosine_similarity(&faces[1].embedding);
    println!("相似度: {sim:.4}");

    Ok(())
}
```

## API 概览

- `FaceEngineConfig::new(det_model_path, rec_model_path, idle_timeout)` —
  配置模型路径和空闲回收模型内存的超时时间；传入 `Duration::ZERO` 可禁用自动回收。
- `FaceEngine::new(&config)` — 立即加载两个模型；
  `FaceEngine::new_without_load(&config)` 延迟至首次推理时加载。
- `FaceEngine::run(&self, img: &RgbImage) -> Result<Vec<Face>>` —
  完成人脸检测与识别，返回检测结果及对应的 512 维特征。
- `FaceEngine::unload()` 与 `FaceEngine::reclaim_if_idle()` —
  分别用于手动卸载模型和在超过配置的空闲时间后回收模型内存。
- `FaceDetector::new(model_path, input_size, score_threshold, nms_threshold)` —
  加载检测模型，后三个参数均可传 `None` 使用默认值。
- `FaceDetector::detect(&mut self, img: &RgbImage) -> Result<Vec<DetectedFace>>` —
  返回每张脸的 `bbox`、`landmarks`（5 点）与 `score`，已做 NMS 抑制。
  `bbox` 与 `landmarks` 坐标为相对于原图的归一化值（`[0, 1]`：`x` 除以宽、`y` 除以高）。
  需要像素坐标时可用 `BoundingBox::to_absolute(w, h)` / `FaceLandmarks::to_absolute(w, h)` 换算。
- `FaceRecognizer::new(model_path, input_size)` — 加载识别模型。
- `FaceRecognizer::extract_embedding(&mut self, img, faces) -> Result<Vec<FaceEmbedding>>` —
  内部根据关键点对齐人脸并输出 512 维向量。
- `FaceEmbedding` 实现了 `Deref<Target = [f32]>`，可直接当作切片进行相似度计算。

## 许可证

MIT
