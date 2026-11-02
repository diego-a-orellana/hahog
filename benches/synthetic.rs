//! Deterministic, dependency-free timings for the public extraction paths.
use hahog::Hahog;
use opencv::{core, prelude::*};
use std::{hint::black_box, time::Instant};

/// Fixed synthetic image width in pixels.
const WIDTH: i32 = 320;
/// Fixed synthetic image height in pixels.
const HEIGHT: i32 = 240;
/// Untimed calls per mode before collecting samples.
const WARMUPS: usize = 2;
/// Odd sample count gives one middle duration after sorting.
const SAMPLES: usize = 7;

/// Warm caches, then report the median of independent full API calls.
fn measure(name: &str, mut extract: impl FnMut() -> hahog::Result<usize>) {
    for _ in 0..WARMUPS {
        black_box(extract().expect("benchmark warmup"));
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut features = 0;
    for _ in 0..SAMPLES {
        let start = Instant::now();
        features = black_box(extract().expect("benchmark extraction"));
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    println!(
        "{name:24} median {:9.3} ms  {features:4} features",
        samples[SAMPLES / 2].as_secs_f64() * 1000.
    );
}

/// Build deterministic inputs and benchmark the detection and supplied-frame paths.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cargo launches harness=false benches during all-target tests too, but only
    // `cargo bench` supplies --bench. Keep ordinary tests free of timing runs.
    if !std::env::args().any(|arg| arg == "--bench") {
        println!("timing skipped; run cargo bench --locked --bench synthetic");
        return Ok(());
    }
    let mut image =
        core::Mat::new_rows_cols_with_default(HEIGHT, WIDTH, core::CV_8UC1, core::Scalar::all(0.))?;
    let mut state = 1729u32;
    for pixel in image.data_typed_mut::<u8>()? {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *pixel = (state >> 24) as u8;
    }
    let mut mask =
        core::Mat::new_rows_cols_with_default(HEIGHT, WIDTH, core::CV_8UC1, core::Scalar::all(0.))?;
    for y in HEIGHT / 4..3 * HEIGHT / 4 {
        for x in WIDTH / 4..3 * WIDTH / 4 {
            *mask.at_2d_mut::<u8>(y, x)? = 255;
        }
    }
    let supplied = core::Vector::<core::KeyPoint>::from_iter(
        (0..8)
            .flat_map(|y| (0..8).map(move |x| (x, y)))
            .map(|(x, y)| {
                core::KeyPoint::new_coords(
                    20. + x as f32 * 40.,
                    15. + y as f32 * 30.,
                    6.,
                    123.,
                    0.,
                    0,
                    -1,
                )
            })
            .collect::<opencv::Result<Vec<_>>>()?,
    );
    let mut detector = Hahog::new();
    let mut points = core::Vector::new();
    let mut descriptors = core::Mat::default();
    println!("synthetic {WIDTH}x{HEIGHT} grayscale; {WARMUPS} warmups, {SAMPLES} samples per mode");
    measure("detect", || {
        detector.detect_def(black_box(&image), &mut points)?;
        black_box(&points);
        Ok(points.len())
    });
    for (name, selection) in [
        ("detect + describe", &core::Mat::default()),
        ("masked + describe", &mask),
    ] {
        measure(name, || {
            detector.detect_and_compute_def(
                black_box(&image),
                black_box(selection),
                &mut points,
                &mut descriptors,
            )?;
            black_box((&points, &descriptors));
            Ok(points.len())
        });
    }
    measure("supplied 64 frames", || {
        points = supplied.clone();
        detector.compute(black_box(&image), &mut points, &mut descriptors)?;
        black_box((&points, &descriptors));
        Ok(points.len())
    });
    Ok(())
}
