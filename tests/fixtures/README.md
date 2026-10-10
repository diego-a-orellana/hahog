# Synthetic extraction references

These fixtures test HAHOG extraction independently of an application or dataset.
`tests/reference.rs` constructs each image deterministically and calls the public
API. No files are loaded at runtime: expected results use `include_str!` and
`include_bytes!`.

| Case | Purpose | Features |
| --- | --- | ---: |
| `texture-32x32` | Minimum supported image size | 4 |
| `texture-65x63` | Odd dimensions and octave downsampling | 23 |
| `border-blobs` | Border patches, larger scales and multiple orientations | 6 |

Each `.csv` contains ordered `x,y,scale,angle_degrees` frames. Each `.bin` contains
exactly 128 unsigned descriptor bytes per frame, with no header or padding.
The tests require identical descriptor bytes and feature order. Centers and scales
allow `1e-5` absolute error; circular angle error allows `1e-4` degrees for the
reference/OpenCV conversion through single-precision degrees.

The checked-in results were generated on macOS arm64 from OpenSfM revision
`c4985ced723888d2529f9b7a1902d4aa682de1d0`, using its native `pyfeatures.hahog`
implementation. `manifest.json` records the extension hash, parameters and data
hashes. Inputs are normalized as `float32(pixel) / float32(255)`; descriptors use
`round_ties_even(clip(362 * sqrt(histogram), 0, 255))`. Texture pixels use an LCG
seeded with 1729; border blobs use filled integer-radius disks. Both generators
are reproduced in `generate.py`.

Only refreshing these fixtures requires the original implementation and NumPy.
From the repository root, use that implementation's Python environment:

```sh
python tests/fixtures/generate.py /path/to/opensfm/pyfeatures.cpython-*.so
cargo test --locked --test reference
```

Review changes against the upstream source and update provenance when refreshing.
Do not replace expected results with this Rust implementation's own output.
