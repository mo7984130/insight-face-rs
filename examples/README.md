# Examples

## `detect_and_recognize`

Comprehensive benchmark that detects faces, extracts 512-D embeddings, and saves
an annotated image — with **detailed timing breakdowns across multiple runs**.

```bash
cargo run --release --example detect_and_recognize -- \
    models/det_10g.onnx models/w600k_r50.onnx input.jpg \
    [output.jpg] [num_runs] [font.ttf]
```

### Features

- **Warm-up** run excluded from statistics
- Multiple iterations (default `10`, configurable) with per-run timing:
  - `detect` — face detection
  - `recog` — embedding extraction
  - `total` — combined pipeline
- **Aggregate statistics**: min / max / avg / median / P95
- **Per-face details**: score, bounding box, area, 5-point landmarks
- **Pairwise cosine similarity** matrix between all detected faces
- **Annotated output image** with bounding boxes, landmarks, and scores

### Output example

```
=== insight-face-rs Benchmark ===

models loaded in   234.56 ms
  detection  : models/det_10g.onnx
  recognition: models/w600k_r50.onnx

image: test.jpg  (1920 × 1080)

--- warm-up (1 iteration) ---
  detected 3 face(s)

--- benchmark (10 iterations) ---
  run   1  detect=   12.34 ms  recog=    5.67 ms  total=   18.01 ms  faces=3
  run   2  detect=   11.98 ms  recog=    5.43 ms  total=   17.41 ms  faces=3
  ...

--- statistics (over 10 runs) ---
  detection     min=   11.87  max=   13.01  avg=   12.34  median=   12.20  P95=   12.90  ms
  recognition   min=    5.23  max=    6.01  avg=    5.56  median=    5.48  P95=    5.89  ms
  total         min=   17.10  max=   19.02  avg=   17.90  median=   17.68  P95=   18.79  ms
  face count    avg=    3.0  faces per image

--- per-face details (last run) ---
  face 0: score=0.9734  bbox=(120,85,340,375)  area=63800
          landmarks: [[120, 150], [200, 145], [220, 220], [150, 260], [280, 255]]
  face 1: score=0.8912  bbox=(450,200,620,450)  area=42500
          ...

--- pairwise cosine similarity ---
  face 0 ↔ face 1:  sim = 0.234567  (1.0 = identical)
  face 0 ↔ face 2:  sim = 0.123456  (1.0 = identical)
  face 1 ↔ face 2:  sim = 0.345678  (1.0 = identical)

annotated image saved to output.jpg
```

The full source is at
<https://github.com/mo7984130/insight-face-rs/blob/main/examples/detect_and_recognize.rs>.
