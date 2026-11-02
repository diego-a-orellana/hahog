```
▗▖ ▗▖ ▗▄▖ ▗▖ ▗▖ ▗▄▖  ▗▄▄▖
▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   
▐▛▀▜▌▐▛▀▜▌▐▛▀▜▌▐▌ ▐▌▐▌▝▜▌
▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▝▚▄▞▘▝▚▄▞▘
                v1.0.0 🐖         
```

Hessian-and-gradient feature extraction for OpenCV images.

HAHOG detects distinctive image points and describes their local appearance for
feature matching. This Rust port of OpenSfM/VLFeat uses OpenCV's `Mat` and
`KeyPoint` types with SIFT-style calling conventions. It returns oriented
keypoints and 128-byte RootSIFT-style descriptors.

## Usage

Requires Rust 1.87+, native OpenCV, Clang and a C++ toolchain for the `opencv` bindings.

Add both crates to your dependencies:

```toml
[dependencies]
hahog = "1.0.0"
opencv = { version = "0.98.1", default-features = false, features = ["imgproc", "clang-runtime"] }
```

Extract features from an existing image:

```rust
use hahog::{Hahog, Result};
use opencv::core;

fn extract(image: &core::Mat) -> Result<(core::Vector<core::KeyPoint>, core::Mat)> {
    let mut detector = Hahog::new();
    let mut keypoints = core::Vector::<core::KeyPoint>::new();
    let mut descriptors = core::Mat::default();

    detector.detect_and_compute_def(
        image,
        &core::no_array(),
        &mut keypoints,
        &mut descriptors,
    )?;

    Ok((keypoints, descriptors))
}
```

Descriptor row `i` belongs to keypoint `i`. Match descriptors with **L2 distance**.
Replace `core::no_array()` with a mask to select keypoint centers.

## Inputs and outputs

| Value | Contract |
| --- | --- |
| Image | 8-bit grayscale, BGR or BGRA; at least 32 × 32 pixels. Strided ROIs are supported. |
| Mask | Optional `CV_8UC1` matrix matching the image size; nonzero pixels select centers. |
| Keypoints | Original-image pixel coordinates, diameters in pixels and clockwise angles in degrees. |
| Descriptors | One 128-byte row per keypoint, stored as `CV_8UC1`. |

Detection uses fixed thresholds and keeps up to 4000 detections before orientation
assignment. A center can have multiple orientations, so the final keypoint count
can be larger. Masks filter centers after this global budget is applied;
orientation and descriptor patches may sample outside the selected region.

Use `detect` for keypoints only and `compute` to describe supplied keypoints.
Supplied angles of `-1` trigger orientation assignment, which can add keypoints or
remove flat patches. Response, octave and class ID are retained. Combined
extraction preserves more frame precision than separate detection and computation.

Empty images clear outputs after mask validation; featureless images yield no
detections. In supplied-keypoint mode, the mask is ignored. `Hahog` follows
OpenCV's method conventions but does not implement C++ `Feature2D` inheritance.
See the [API documentation](https://docs.rs/hahog) for the full contract.

## Development

See [CONTRIBUTING.md](https://github.com/diego-a-orellana/hahog/blob/release/1.0.0/CONTRIBUTING.md)
for checks, benchmarks and reference fixtures.

## License

BSD-3-Clause. See [LICENSE](https://github.com/diego-a-orellana/hahog/blob/release/1.0.0/LICENSE).
Adapted portions retain their upstream BSD terms; see
[attribution and license notices](https://github.com/diego-a-orellana/hahog/blob/release/1.0.0/licenses/README.md).
