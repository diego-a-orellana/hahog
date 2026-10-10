# Development

Run the complete test suite with the same OpenCV prerequisites as the library:

```sh
cargo test --locked
cargo test --locked --release
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items
```

All tests run without external datasets, environment variables, Python or OpenSfM.
API tests exercise masks, strided images, color conversion, supplied keypoints and
errors. Numerical tests protect rounding, interpolation and feature ordering.
The integration tests compare extraction against checked-in synthetic reference
outputs; [fixture documentation](tests/fixtures/README.md) explains their provenance
and how maintainers can regenerate them using the original implementation.

`src/api.rs` owns the OpenCV boundary; `src/error.rs` defines public errors.
The modules under `src/algorithm/` implement the extraction stages in Rust.
Keep arithmetic order, mixed precision, fused operations and tie ordering intact:
seemingly equivalent expressions can change descriptor bytes.

Consumers can override OpenCV with a workspace-root `[patch.crates-io]`.
Direct OpenCV dependencies should also name the crates.io package so `Mat` types
resolve to one source. Keep the fork's version compatible with the dependency.
