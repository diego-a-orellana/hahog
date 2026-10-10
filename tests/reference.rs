//! Public extraction against independent reference outputs on synthetic images.
use hahog_rust::Hahog;
use opencv::{core, prelude::*};

/// Generate reproducible grayscale texture without an image codec or random dependency.
fn texture(width: i32, height: i32) -> core::Mat {
    let mut image =
        core::Mat::new_rows_cols_with_default(height, width, core::CV_8UC1, core::Scalar::all(0.))
            .unwrap();
    let mut state = 1729u32;
    for pixel in image.data_typed_mut::<u8>().unwrap() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *pixel = (state >> 24) as u8;
    }
    image
}

/// Compare every frame in order and every descriptor byte with the native reference.
fn assert_reference(image: &core::Mat, frames_csv: &str, expected_descriptors: &[u8]) {
    let mut lines = frames_csv.lines();
    assert_eq!(lines.next(), Some("x,y,scale,angle_degrees"));
    let frames: Vec<[f32; 4]> = lines
        .map(|line| {
            line.split(',')
                .map(|value| value.parse::<f32>().expect("numeric reference frame"))
                .collect::<Vec<_>>()
                .try_into()
                .expect("four reference frame coordinates")
        })
        .collect();
    assert!(
        !frames.is_empty(),
        "reference fixture must exercise extraction"
    );
    assert_eq!(expected_descriptors.len(), frames.len() * 128);

    let mut points = core::Vector::new();
    let mut descriptors = core::Mat::default();
    Hahog::new()
        .detect_and_compute_def(image, &core::no_array(), &mut points, &mut descriptors)
        .unwrap();
    assert_eq!(points.len(), frames.len());
    for (index, (point, expected)) in points.iter().zip(frames).enumerate() {
        let actual = [point.pt().x, point.pt().y, point.size() / 2.];
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() <= 1e-5,
                "frame {index}, coordinate {axis}: {} != {}",
                actual[axis],
                expected[axis]
            );
        }
        let angle_difference = (point.angle() as f64 - expected[3] as f64).rem_euclid(360.);
        assert!(
            angle_difference.min(360. - angle_difference) <= 1e-4,
            "frame {index}: angle {} != {} degrees",
            point.angle(),
            expected[3]
        );
    }
    assert_eq!(descriptors.data_bytes().unwrap(), expected_descriptors);
}

/// Exercise the smallest supported image with nonempty native reference features.
#[test]
fn minimum_image_matches_reference() {
    assert_reference(
        &texture(32, 32),
        include_str!("fixtures/texture-32x32.csv"),
        include_bytes!("fixtures/texture-32x32.bin"),
    );
}

/// Preserve extraction through odd-sized octave downsampling and patch boundaries.
#[test]
fn odd_dimensions_match_reference() {
    assert_reference(
        &texture(65, 63),
        include_str!("fixtures/texture-65x63.csv"),
        include_bytes!("fixtures/texture-65x63.bin"),
    );
}

/// Protect border sampling, coarse scales and orientation expansion on smooth blobs.
#[test]
fn border_features_and_multiple_orientations_match_reference() {
    let mut image =
        core::Mat::new_rows_cols_with_default(97, 129, core::CV_8UC1, core::Scalar::all(0.))
            .unwrap();
    for (cx, cy, radius) in [
        (0, 0, 8),
        (128, 0, 13),
        (0, 96, 11),
        (128, 96, 18),
        (64, 48, 15),
    ] {
        for y in 0i32..97 {
            for x in 0i32..129 {
                if (x - cx).pow(2) + (y - cy).pow(2) <= radius * radius {
                    *image.at_2d_mut::<u8>(y, x).unwrap() = 255;
                }
            }
        }
    }
    assert_reference(
        &image,
        include_str!("fixtures/border-blobs.csv"),
        include_bytes!("fixtures/border-blobs.bin"),
    );
}
