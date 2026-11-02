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
| `supplied-texture` | Supplied angles, border/coarse patches, automatic orientations | 9 |

The three detection cases' `.csv` files contain ordered `x,y,scale,angle_degrees`
frames. Each corresponding `.bin` contains
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

## Supplied frames

`supplied-texture-input.csv` contains seven input frames on the same 65×63
texture; scale is half the OpenCV keypoint diameter and angle `-1` requests
orientation assignment. `supplied-texture.csv` adds an `input_index` column to
the expected output frames, including a frame that expands to three orientations.
`supplied-texture.bin` contains 128 descriptor bytes per output frame.

`generate_supplied.py` loads the original extension's VLFeat C exports with
`ctypes`. It follows `hahog.cc`'s patch extraction, gradient and raw SIFT sequence,
using `vl_covdet_extract_orientations_for_frame` only for missing angles.
The Python adapter converts supplied float32 centers/scales/angles into VLFeat
frames; the native code computes all orientations and descriptors. The test checks
both public supplied-keypoint entry points and preserves input metadata.
`supplied-manifest.json` records extension/input/output hashes and provenance.

```sh
python tests/fixtures/generate_supplied.py /path/to/opensfm/pyfeatures.cpython-*.so
```

This refresh is independent of the three detection goldens and leaves them intact.

## Selection caps and ties

`selection-ties-{4001,6000,6001}.bin` stores 4000 ordered input feature IDs each,
as little-endian `u32` without headers. For input ID `i`, its signed float32 peak
is `((i * 37 + 11) % 17) - 8 + 0.25`. Apply a 6000 cap followed by a 4000 cap;
each cap sorts only when the current count exceeds it. These cases exercise ties
at the cutoff, the 6000 boundary and the second sort after the first truncation.

`generate_selection.py` calls Darwin's actual `qsort` through `ctypes`, using the
48-byte `VlCovDetFeature` layout from `covdet.h` and the comparator from `hahog.cc`.
It needs only Python's standard library on macOS. The expected Darwin ordering
is tested on every platform; regenerate only on Darwin, since other C libraries
can permute ties differently. `selection-manifest.json` records the input formula,
ABI, macOS version and hashes.

```sh
python3 tests/fixtures/generate_selection.py
```
