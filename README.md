# hahog-rust

HAHOG (Hessian-and-gradient) feature extraction in Rust, with OpenCV image and
keypoint types and SIFT-style calling conventions. Ported from OpenSfM/VLFeat.

## Setup

Requires native OpenCV, Clang and a C++ toolchain for the `opencv` bindings.

```toml
[dependencies]
hahog-rust = { path = "../hahog-rust" }
opencv = { version = "0.98.1", default-features = false, features = ["imgproc", "clang-runtime"] }
```

## Usage

```rust
use hahog_rust::Hahog;
use opencv::core;

# fn extract(image: &core::Mat) -> hahog_rust::Result<()> {
let mut detector = Hahog::new();
let mut keypoints = core::Vector::<core::KeyPoint>::new();
let mut descriptors = core::Mat::default();
detector.detect_and_compute_def(
    image,
    &core::no_array(),
    &mut keypoints,
    &mut descriptors,
)?;
# Ok(())
# }
```

- Accepts 8-bit grayscale, BGR or BGRA images at least 32 × 32 pixels, including
  strided ROIs. Empty images clear outputs; featureless images yield no detections.
- Returns pixel-coordinate keypoints and 128-byte RootSIFT-style descriptor rows
  (`CV_8UC1`). Match descriptors with **L2 distance**.
- Optional masks are `CV_8UC1`, match the image size, and select centers at nonzero pixels.
- Detection uses fixed thresholds and keeps up to 4000 detections before
  orientation expansion, so the final keypoint count can be larger.

Use `detect` for keypoints only and `compute` for supplied keypoints. Combined
extraction preserves more frame precision than separate calls. `Hahog` follows
OpenCV's method conventions but does not implement C++ `Feature2D` inheritance.
Run `cargo doc --open` for the complete API contract.

## License

BSD-3-Clause; adapted portions retain their upstream BSD terms.
See [attribution and license notices](licenses/README.md).
