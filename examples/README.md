# Insights

- Run example:
  ```bash
  cargo run --example detect_and_recognize -- det_640.onnx w600k_r50.onnx input.jpg
  ```
  Output contains:
  - bounding boxes of faces
  - 5-point landmarks for each face
  - confidence scores (and optional text labels when a font is passed)
  - images saved to `output.jpg`

- When font is available, the face ID and score will be drawn onto the image.
  No score will be printed to stdout when font is provided.

- The example also computes the cosine similarity (embedding vectors) between the
  first two faces as a simple demo of `FaceEmbedding` usage (you may replace this
  with custom logic).

The whole example script and the repository are available at
https://github.com/mo7984130/insight-face-rs.
