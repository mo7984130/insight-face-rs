# Examples

## `detect_and_recognize`

Detect faces, extract 512-D embeddings, and save an annotated image.

```bash
cargo run --example detect_and_recognize -- det_640.onnx w600k_r50.onnx input.jpg [output.jpg] [font.ttf]
```

- Draws **bounding boxes** and **5-point landmarks** for every detected face.
- Prints or draws **confidence scores** (text on image requires a `.ttf` font path).
- Computes **cosine similarity** between the first two face embeddings.
- Output: annotated image (default `output.jpg`).

The full source is at
<https://github.com/mo7984130/insight-face-rs/blob/main/examples/detect_and_recognize.rs>.
