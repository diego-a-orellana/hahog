# Development

The supported minimum is Rust 1.87, tested with the checked-in lockfile on macOS
arm64 alongside Rust 1.93. Keep the minimum explicit in `Cargo.toml` and retest it
when updating dependencies.

Run these checks with the same native OpenCV/Clang prerequisites as the library:

```sh
cargo +1.87.0 test --locked --all-targets
cargo +1.87.0 test --locked --release
cargo test --locked
cargo test --locked --release
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items
```

Run `cargo bench --locked --bench synthetic` for a deterministic 320×240 grayscale
benchmark. It reports median wall time from seven samples after two warmups for
detection, detection plus descriptors, a central selection mask, and 64 supplied
frames. Inputs are generated before timing; supplied-vector reset and output
allocation are included. Compare on the same machine/toolchain/OpenCV build with
other workloads idle. There are no timing assertions or benchmark dependencies;
`cargo test --all-targets` skips timing runs.

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
