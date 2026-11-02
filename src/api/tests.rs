use super::*;

/// Create deterministic nonuniform grayscale pixels without external fixtures.
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

/// Extract keypoints and descriptors, failing the test on any API error.
fn detect(
    image: &impl core::ToInputArray,
    mask: &impl core::ToInputArray,
) -> (core::Vector<core::KeyPoint>, core::Mat) {
    let mut points = core::Vector::new();
    let mut descriptors = core::Mat::default();
    Hahog::new()
        .detect_and_compute_def(image, mask, &mut points, &mut descriptors)
        .unwrap();
    (points, descriptors)
}

/// Assert equality of all OpenCV keypoint fields in order.
fn same_points(a: &core::Vector<core::KeyPoint>, b: &core::Vector<core::KeyPoint>) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(
            (
                a.pt(),
                a.size(),
                a.angle(),
                a.response(),
                a.octave(),
                a.class_id()
            ),
            (
                b.pt(),
                b.size(),
                b.angle(),
                b.response(),
                b.octave(),
                b.class_id()
            )
        );
    }
}

/// Verify coordinate conversion and row alignment against the internal frames.
#[test]
fn opencv_rows_use_pixel_centers_diameters_angles_and_l2_descriptors() {
    let image = texture(128, 128);
    let (points, descriptors) = detect(&image, &core::no_array());
    let pyramid = pyramid::Pyramid::build(&image_plane(&image).unwrap());
    let mut workspace = PatchWorkspace::default();
    let rows: Vec<_> = selected_detections(&pyramid)
        .into_iter()
        .flat_map(|detection| orient_detection(&pyramid, detection, &mut workspace))
        .collect();
    assert!(!rows.is_empty());
    assert_eq!(points.len(), rows.len());
    assert_eq!(descriptors.rows() as usize, points.len());
    assert_eq!(descriptors.cols(), 128);
    assert_eq!(descriptors.typ(), core::CV_8UC1);
    for (i, row) in rows.iter().enumerate() {
        let point = points.get(i).unwrap();
        assert_eq!(point.pt(), core::Point2f::new(row.frame.x, row.frame.y));
        let determinant =
            math::determinant2([row.frame.a11, row.frame.a12, row.frame.a21, row.frame.a22]);
        assert_eq!(point.size(), 2. * determinant.abs().sqrt());
        assert!((0. ..360.).contains(&point.angle()));
        assert_eq!(point.octave(), row.octave);
        assert_eq!(
            descriptors
                .row(i as i32)
                .unwrap()
                .data_typed::<u8>()
                .unwrap(),
            describe_frame(&pyramid, row.frame, &mut workspace)
        );
    }
    assert_eq!(Hahog::new().default_norm(), core::NORM_L2);
    assert_eq!(Hahog::new().descriptor_size(), 128);
    assert_eq!(Hahog::new().descriptor_type(), core::CV_8U);
}

/// Reject unsupported dimensions and grayscale types; preserve normalized pixel values.
#[test]
fn image_conversion_validates_dimensions_and_normalizes_pixels() {
    for image in [
        core::Mat::default(),
        texture(31, 32),
        core::Mat::new_rows_cols_with_default(32, 32, core::CV_32FC1, core::Scalar::all(0.))
            .unwrap(),
    ] {
        assert!(validate_image(&image).is_err());
    }
    let blank = core::Mat::new_rows_cols_with_default(32, 32, core::CV_8UC1, core::Scalar::all(0.))
        .unwrap();
    let plane = image_plane(&blank).unwrap();
    assert_eq!((plane.width, plane.height), (32, 32));
    assert_eq!(plane.values, vec![0.; 32 * 32]);
    let color = core::Mat::new_rows_cols_with_default(32, 32, core::CV_8UC3, core::Scalar::all(0.))
        .unwrap();
    assert!(matches!(
        grayscale_plane(&color),
        Err(Error::GrayscaleType { .. })
    ));
}

/// Masking must preserve the selected rows of unmasked extraction exactly.
#[test]
fn mask_filters_centers_without_reordering_descriptor_rows() {
    let image = texture(128, 128);
    let (all, all_descriptors) = detect(&image, &core::no_array());
    let mut mask =
        core::Mat::new_rows_cols_with_default(128, 128, core::CV_8UC1, core::Scalar::all(0.))
            .unwrap();
    for y in 0..128 {
        for x in 0..64 {
            *mask.at_2d_mut::<u8>(y, x).unwrap() = 255;
        }
    }
    let (selected, descriptors) = detect(&image, &mask);
    assert!(!selected.is_empty() && selected.len() < all.len());
    let mut row = 0;
    for (i, point) in all.iter().enumerate() {
        if point.pt().x.round_ties_even() < 64. {
            assert_eq!(point.pt(), selected.get(row).unwrap().pt());
            assert_eq!(
                all_descriptors
                    .row(i as i32)
                    .unwrap()
                    .data_typed::<u8>()
                    .unwrap(),
                descriptors
                    .row(row as i32)
                    .unwrap()
                    .data_typed::<u8>()
                    .unwrap()
            );
            row += 1;
        }
    }
    assert_eq!(row, selected.len());

    mask.data_typed_mut::<u8>().unwrap().fill(0);
    let (points, descriptors) = detect(&image, &mask);
    assert!(points.is_empty());
    assert!(descriptors.empty());
}

/// Row strides and grayscale color conversions must preserve extraction results.
#[test]
fn strided_rois_and_bgr_images_match_contiguous_grayscale() {
    use opencv::imgproc;
    let image = texture(144, 128);
    let roi = core::Mat::roi(&image, core::Rect::new(8, 8, 128, 112)).unwrap();
    assert!(!roi.is_continuous());
    let mut gray = core::Mat::default();
    roi.copy_to(&mut gray).unwrap();
    let (reference, values) = detect(&gray, &core::no_array());
    let (points, descriptors) = detect(&roi, &core::no_array());
    same_points(&reference, &points);
    assert_eq!(
        values.data_bytes().unwrap(),
        descriptors.data_bytes().unwrap()
    );
    for conversion in [imgproc::COLOR_GRAY2BGR, imgproc::COLOR_GRAY2BGRA] {
        let mut color = core::Mat::default();
        imgproc::cvt_color_def(&gray, &mut color, conversion).unwrap();
        let (points, descriptors) = detect(&color, &core::no_array());
        same_points(&reference, &points);
        assert_eq!(
            values.data_bytes().unwrap(),
            descriptors.data_bytes().unwrap()
        );
    }
}

/// Distinguish the BGR convention from RGB on unequal color channels.
#[test]
fn bgr_conversion_uses_blue_first_channel_order() {
    let source = texture(384, 128);
    let color = source.reshape(3, 128).unwrap();
    let mut gray = core::Mat::default();
    opencv::imgproc::cvt_color_def(&color, &mut gray, opencv::imgproc::COLOR_BGR2GRAY).unwrap();
    let (reference, expected) = detect(&gray, &core::no_array());
    let (points, descriptors) = detect(&color, &core::no_array());
    same_points(&reference, &points);
    assert_eq!(
        descriptors.data_bytes().unwrap(),
        expected.data_bytes().unwrap()
    );
}

/// Empty extraction must clear caller-owned keypoints and descriptors.
#[test]
fn empty_and_featureless_images_clear_prior_outputs() {
    let blank = core::Mat::new_rows_cols_with_default(64, 64, core::CV_8UC1, core::Scalar::all(0.))
        .unwrap();
    for image in [core::Mat::default(), blank] {
        let mut points =
            core::Vector::from_iter([core::KeyPoint::new_coords_def(10., 10., 3.).unwrap()]);
        let mut descriptors = core::Mat::from_slice_2d(&[[1u8; 128]]).unwrap();
        Hahog::new()
            .detect_and_compute_def(&image, &core::no_array(), &mut points, &mut descriptors)
            .unwrap();
        assert!(points.is_empty());
        assert!(descriptors.empty());
    }
}

/// Invalid nonempty masks must win over empty-image clearing in both detection APIs.
#[test]
fn empty_images_validate_nonempty_masks_before_clearing_outputs() {
    let image = core::Mat::default();
    for (mask_type, expected) in [
        (
            core::CV_32FC1,
            Error::MaskType {
                actual: core::CV_32FC1,
            },
        ),
        (
            core::CV_8UC1,
            Error::MaskShape {
                dimensions: 2,
                width: 32,
                height: 32,
                image_width: 0,
                image_height: 0,
            },
        ),
    ] {
        let mask = core::Mat::new_rows_cols_with_default(32, 32, mask_type, core::Scalar::all(0.))
            .unwrap();
        let original =
            core::Vector::from_iter([
                core::KeyPoint::new_coords(10., 10., 3., 123., 0.25, 7, 42).unwrap()
            ]);
        let mut points = original.clone();
        let error = Hahog::new().detect(&image, &mut points, &mask).unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        same_points(&points, &original);

        let mut descriptors = core::Mat::from_slice_2d(&[[13u8; 128]]).unwrap();
        let error = Hahog::new()
            .detect_and_compute_def(&image, &mask, &mut points, &mut descriptors)
            .unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        same_points(&points, &original);
        assert_eq!(descriptors.data_bytes().unwrap(), [13u8; 128]);
    }
}

/// A zero mask ROI must clear stale outputs even when surrounding pixels are selected.
#[test]
fn zero_mask_shortcuts_clear_outputs_with_a_strided_mask() {
    let image = texture(64, 64);
    let mut storage =
        core::Mat::new_rows_cols_with_default(66, 66, core::CV_8UC1, core::Scalar::all(255.))
            .unwrap();
    for y in 1..65 {
        for x in 1..65 {
            *storage.at_2d_mut::<u8>(y, x).unwrap() = 0;
        }
    }
    let mask = core::Mat::roi(&storage, core::Rect::new(1, 1, 64, 64)).unwrap();
    assert!(!mask.is_continuous());
    let original = core::Vector::from_iter([core::KeyPoint::new_coords_def(10., 10., 3.).unwrap()]);
    let mut points = original.clone();
    Hahog::new().detect(&image, &mut points, &mask).unwrap();
    assert!(points.is_empty());

    points = original;
    let mut descriptors = core::Mat::from_slice_2d(&[[13u8; 128]]).unwrap();
    Hahog::new()
        .detect_and_compute_def(&image, &mask, &mut points, &mut descriptors)
        .unwrap();
    assert!(points.is_empty());
    assert!(descriptors.empty());
}

/// Zero-valued masks must not bypass validation or change mask-before-image precedence.
#[test]
fn zero_mask_shortcuts_preserve_validation_and_outputs() {
    for (width, height, image_type, mask_width, mask_type, expected) in [
        (
            64,
            64,
            core::CV_32FC1,
            64,
            core::CV_8UC1,
            Error::ImageType {
                actual: core::CV_32FC1,
            },
        ),
        (
            16,
            32,
            core::CV_32FC1,
            16,
            core::CV_8UC1,
            Error::ImageShape {
                dimensions: 2,
                width: 16,
                height: 32,
            },
        ),
        (
            64,
            64,
            core::CV_32FC1,
            32,
            core::CV_8UC1,
            Error::MaskShape {
                dimensions: 2,
                width: 32,
                height: 64,
                image_width: 64,
                image_height: 64,
            },
        ),
        (
            16,
            32,
            core::CV_32FC1,
            64,
            core::CV_32FC1,
            Error::MaskType {
                actual: core::CV_32FC1,
            },
        ),
    ] {
        let image =
            core::Mat::new_rows_cols_with_default(height, width, image_type, core::Scalar::all(0.))
                .unwrap();
        let mask = core::Mat::new_rows_cols_with_default(
            height,
            mask_width,
            mask_type,
            core::Scalar::all(0.),
        )
        .unwrap();
        let original =
            core::Vector::from_iter([core::KeyPoint::new_coords_def(10., 10., 3.).unwrap()]);
        let mut points = original.clone();
        let mut descriptors = core::Mat::from_slice_2d(&[[13u8; 128]]).unwrap();
        let error = Hahog::new().detect(&image, &mut points, &mask).unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        same_points(&points, &original);
        let error = Hahog::new()
            .detect_and_compute_def(&image, &mask, &mut points, &mut descriptors)
            .unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        same_points(&points, &original);
        assert_eq!(descriptors.data_bytes().unwrap(), [13u8; 128]);
    }
}

/// Empty supplied frames still validate images, ignore masks and clear old descriptors.
#[test]
fn empty_provided_keypoints_validate_images_and_clear_descriptors() {
    let irrelevant_mask =
        core::Mat::new_rows_cols_with_default(1, 1, core::CV_32FC1, core::Scalar::all(0.)).unwrap();
    let mut points = core::Vector::new();
    for image in [
        core::Mat::default(),
        texture(64, 64),
        core::Mat::new_rows_cols_with_default(64, 64, core::CV_8UC3, core::Scalar::all(127.))
            .unwrap(),
        core::Mat::new_rows_cols_with_default(64, 64, core::CV_8UC4, core::Scalar::all(127.))
            .unwrap(),
    ] {
        let mut descriptors = core::Mat::from_slice_2d(&[[13u8; 128]]).unwrap();
        Hahog::new()
            .detect_and_compute(
                &image,
                &irrelevant_mask,
                &mut points,
                &mut descriptors,
                true,
            )
            .unwrap();
        assert!(points.is_empty());
        assert!(descriptors.empty());
    }
    for (width, expected) in [
        (
            16,
            Error::ImageShape {
                dimensions: 2,
                width: 16,
                height: 64,
            },
        ),
        (
            64,
            Error::ImageType {
                actual: core::CV_32FC1,
            },
        ),
    ] {
        let image =
            core::Mat::new_rows_cols_with_default(64, width, core::CV_32FC1, core::Scalar::all(0.))
                .unwrap();
        let mut descriptors = core::Mat::from_slice_2d(&[[13u8; 128]]).unwrap();
        let error = Hahog::new()
            .detect_and_compute(
                &image,
                &irrelevant_mask,
                &mut points,
                &mut descriptors,
                true,
            )
            .unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        assert!(points.is_empty());
        assert_eq!(descriptors.data_bytes().unwrap(), [13u8; 128]);
    }
}

/// Borrowed descriptor rows must copy into strided outputs before committing keypoints.
#[test]
fn descriptor_copy_supports_strided_output_and_preserves_points_on_failure() {
    let image = texture(64, 64);
    let original = core::Vector::from_iter([
        core::KeyPoint::new_coords(32., 32., 8., -1., 0.25, 7, 42).unwrap(),
        core::KeyPoint::new_coords(40., 36., 6., 123., 0.5, 3, 19).unwrap(),
    ]);
    let mut expected = original.clone();
    let mut descriptors = core::Mat::default();
    Hahog::new()
        .compute(&image, &mut expected, &mut descriptors)
        .unwrap();
    assert!(expected.len() >= 2);
    let mut storage = core::Mat::new_rows_cols_with_default(
        descriptors.rows() + 2,
        132,
        core::CV_8UC1,
        core::Scalar::all(165.),
    )
    .unwrap();
    let mut points = original.clone();
    {
        let mut output =
            core::Mat::roi_mut(&mut storage, core::Rect::new(2, 1, 128, descriptors.rows()))
                .unwrap();
        assert!(!output.is_continuous());
        Hahog::new()
            .compute(&image, &mut points, &mut output)
            .unwrap();
        same_points(&points, &expected);
        for row in 0..descriptors.rows() {
            assert_eq!(
                output.row(row).unwrap().data_bytes().unwrap(),
                descriptors.row(row).unwrap().data_bytes().unwrap(),
            );
        }
    }
    for y in 0..storage.rows() {
        for x in 0..storage.cols() {
            if y == 0 || y == storage.rows() - 1 || !(2..130).contains(&x) {
                assert_eq!(*storage.at_2d::<u8>(y, x).unwrap(), 165);
            }
        }
    }
    points = original.clone();
    assert!(Hahog::new()
        .compute(&image, &mut points, &mut core::no_array())
        .is_err());
    same_points(&points, &original);
}

/// Verify explicit and inferred orientations and the provided-keypoint mask contract.
#[test]
fn provided_keypoints_preserve_fields_ignore_mask_and_assign_missing_angles() {
    let image = texture(128, 128);
    let mask =
        core::Mat::new_rows_cols_with_default(128, 128, core::CV_8UC1, core::Scalar::all(0.))
            .unwrap();
    let original = core::KeyPoint::new_coords(56., 62., 8., 123., 0.25, 7, 42).unwrap();
    let mut points = core::Vector::from_iter([original.clone()]);
    let mut descriptors = core::Mat::default();
    Hahog::new()
        .detect_and_compute(&image, &mask, &mut points, &mut descriptors, true)
        .unwrap();
    same_points(&points, &core::Vector::from_iter([original]));
    assert_eq!((descriptors.rows(), descriptors.cols()), (1, 128));
    let irrelevant_mask =
        core::Mat::new_rows_cols_with_default(1, 1, core::CV_32FC1, core::Scalar::all(0.)).unwrap();
    Hahog::new()
        .detect_and_compute(
            &image,
            &irrelevant_mask,
            &mut points,
            &mut descriptors,
            true,
        )
        .unwrap();
    assert_eq!((descriptors.rows(), descriptors.cols()), (1, 128));
    let plane = grayscale_plane(&image).unwrap();
    let pyramid = pyramid::Pyramid::build(&plane);
    let frame = Frame {
        x: 56.,
        y: 62.,
        a11: 4.,
        a12: 0.,
        a21: 0.,
        a22: 4.,
    };
    let expected = describe_frame(
        &pyramid,
        orientation::rotate(frame, 123f64.to_radians()),
        &mut PatchWorkspace::default(),
    );
    assert_eq!(descriptors.data_typed::<u8>().unwrap(), expected);
    points = core::Vector::from_iter([
        core::KeyPoint::new_coords(56., 62., 8., -1., 0.25, 7, 42).unwrap()
    ]);
    Hahog::new()
        .detect_and_compute(&image, &mask, &mut points, &mut descriptors, true)
        .unwrap();
    assert!(!points.is_empty());
    assert_eq!(descriptors.rows() as usize, points.len());
    for point in points {
        assert!((0. ..360.).contains(&point.angle()));
        assert_eq!(point.pt(), core::Point2f::new(56., 62.));
        assert_eq!(point.size(), 8.);
        assert_eq!(point.response(), 0.25);
        assert_eq!(point.octave(), 7);
        assert_eq!(point.class_id(), 42);
    }
}

/// Reject invalid images, masks and frames before committing staged outputs.
#[test]
fn malformed_inputs_return_errors_without_replacing_outputs() {
    let image = texture(64, 64);
    let mut points =
        core::Vector::from_iter([core::KeyPoint::new_coords_def(10., 10., 3.).unwrap()]);
    let mut descriptors = core::Mat::from_slice_2d(&[[1u8; 128]]).unwrap();
    for (mask, expected) in [
        (
            texture(32, 32),
            crate::Error::MaskShape {
                dimensions: 2,
                width: 32,
                height: 32,
                image_width: 64,
                image_height: 64,
            },
        ),
        (
            core::Mat::new_rows_cols_with_default(64, 64, core::CV_32FC1, core::Scalar::all(0.))
                .unwrap(),
            crate::Error::MaskType {
                actual: core::CV_32FC1,
            },
        ),
    ] {
        let error = Hahog::new()
            .detect_and_compute_def(&image, &mask, &mut points, &mut descriptors)
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
        assert_eq!(points.len(), 1);
        assert_eq!(descriptors.data_typed::<u8>().unwrap(), [1u8; 128]);
    }
    for (x, size, angle) in [
        (f32::NAN, 3., 0.),
        (64., 3., 0.),
        (10., 0., 0.),
        (10., 129., 0.),
        (10., 3., 360.),
    ] {
        points =
            core::Vector::from_iter([
                core::KeyPoint::new_coords(x, 10., size, angle, 0., 0, -1).unwrap()
            ]);
        assert!(Hahog::new()
            .detect_and_compute(
                &image,
                &core::no_array(),
                &mut points,
                &mut descriptors,
                true
            )
            .is_err());
    }
    let float =
        core::Mat::new_rows_cols_with_default(64, 64, core::CV_32FC1, core::Scalar::all(0.))
            .unwrap();
    for bad_image in [float, texture(16, 32)] {
        assert!(Hahog::new()
            .detect_and_compute_def(&bad_image, &core::no_array(), &mut points, &mut descriptors)
            .is_err());
    }
}

/// Detection alone must retain combined extraction order and validation behavior.
#[test]
fn detection_only_matches_combined_keypoints_and_mask_order() {
    let image = texture(128, 128);
    let mut detector = Hahog::new();
    let mut points = core::Vector::new();
    detector.detect_def(&image, &mut points).unwrap();
    let (expected, _) = detect(&image, &core::no_array());
    assert!(!points.is_empty());
    same_points(&points, &expected);

    let mut mask = texture(128, 128);
    for y in 0..mask.rows() {
        for x in 64..mask.cols() {
            *mask.at_2d_mut::<u8>(y, x).unwrap() = 0;
        }
    }
    detector.detect(&image, &mut points, &mask).unwrap();
    let (expected, _) = detect(&image, &mask);
    assert!(!points.is_empty());
    same_points(&points, &expected);

    let error = detector
        .detect(&image, &mut points, &texture(32, 32))
        .unwrap_err();
    assert!(matches!(
        error,
        Error::MaskShape {
            width: 32,
            image_width: 128,
            ..
        }
    ));
    same_points(&points, &expected);
    detector
        .detect_def(&core::Mat::default(), &mut points)
        .unwrap();
    assert!(points.is_empty());
}

/// Shared traversal must stage detected points until a generic descriptor output succeeds.
#[test]
fn detected_keypoints_are_preserved_when_descriptor_output_fails() {
    let image = texture(128, 128);
    let original =
        core::Vector::from_iter([
            core::KeyPoint::new_coords(10., 10., 3., 123., 0.25, 7, 42).unwrap()
        ]);
    let mut points = original.clone();
    assert!(Hahog::new()
        .detect_and_compute_def(
            &image,
            &core::no_array(),
            &mut points,
            &mut core::no_array()
        )
        .is_err());
    same_points(&points, &original);

    Hahog::new().detect_def(&image, &mut points).unwrap();
    let (expected, _) = detect(&image, &core::no_array());
    assert!(!expected.is_empty());
    same_points(&points, &expected);
}

/// The compute shorthand must match combined extraction with supplied frames.
#[test]
fn compute_matches_provided_keypoint_mode() {
    let image = texture(128, 128);
    let mut detector = Hahog::new();
    let mut points = core::Vector::from_iter([
        core::KeyPoint::new_coords(56., 62., 8., 123., 0.25, 7, 42).unwrap(),
        core::KeyPoint::new_coords(70., 50., 6., -1., 0.5, 3, 19).unwrap(),
    ]);
    let mut expected = points.clone();
    let mut descriptors = core::Mat::default();
    let mut expected_descriptors = core::Mat::default();
    detector
        .compute(&image, &mut points, &mut descriptors)
        .unwrap();
    detector
        .detect_and_compute(
            &image,
            &core::no_array(),
            &mut expected,
            &mut expected_descriptors,
            true,
        )
        .unwrap();
    assert!(!points.is_empty());
    same_points(&points, &expected);
    assert_eq!(descriptors.rows() as usize, points.len());
    assert_eq!(
        descriptors.data_bytes().unwrap(),
        expected_descriptors.data_bytes().unwrap()
    );

    detector
        .compute(&core::Mat::default(), &mut points, &mut descriptors)
        .unwrap();
    assert!(points.is_empty());
    assert!(descriptors.empty());
}

/// Invalid frames must identify their index and preserve every caller-owned field.
#[test]
fn provided_keypoint_validation_preserves_outputs_and_reports_the_failing_index() {
    let image = texture(64, 64);
    let valid = core::KeyPoint::new_coords(10., 10., 3., 0., 0., 0, 5).unwrap();
    for (x, y, size, angle, expected) in [
        (f32::NAN, 10., 3., 0., Error::KeypointCenter { index: 1 }),
        (
            10.,
            f32::INFINITY,
            3.,
            0.,
            Error::KeypointCenter { index: 1 },
        ),
        (10., -1., 3., 0., Error::KeypointCenter { index: 1 }),
        (10., 64., 3., 0., Error::KeypointCenter { index: 1 }),
        (10., 10., f32::NAN, 0., Error::KeypointSize { index: 1 }),
        (
            10.,
            10.,
            f32::INFINITY,
            0.,
            Error::KeypointSize { index: 1 },
        ),
        (
            10.,
            10.,
            f32::MIN_POSITIVE,
            0.,
            Error::KeypointSize { index: 1 },
        ),
        (10., 10., 3., f32::NAN, Error::KeypointAngle { index: 1 }),
        (10., 10., 3., -2., Error::KeypointAngle { index: 1 }),
    ] {
        let mut points = core::Vector::from_iter([
            valid.clone(),
            core::KeyPoint::new_coords(x, y, size, angle, 0., 0, 9).unwrap(),
        ]);
        let mut descriptors = core::Mat::from_slice_2d(&[[13u8; 128]]).unwrap();
        let error = Hahog::new()
            .compute(&image, &mut points, &mut descriptors)
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
        assert_eq!(error.to_string(), expected.to_string());
        assert_eq!(points.len(), 2);
        let unchanged = points.get(1).unwrap();
        assert_eq!(unchanged.pt().x.to_bits(), x.to_bits());
        assert_eq!(unchanged.pt().y.to_bits(), y.to_bits());
        assert_eq!(unchanged.size().to_bits(), size.to_bits());
        assert_eq!(unchanged.angle().to_bits(), angle.to_bits());
        assert_eq!(unchanged.class_id(), 9);
        assert_eq!(descriptors.data_bytes().unwrap(), [13u8; 128]);
    }
}

/// Wrapping an OpenCV error must preserve its details and error chain.
#[test]
fn opencv_errors_retain_the_status_code_and_source() {
    use std::error::Error as _;
    let error = Error::from(opencv::Error::new(core::StsBadArg, "invalid test matrix"));
    let Error::OpenCv(ref source) = error else {
        panic!("expected the original OpenCV error");
    };
    assert_eq!(source.code, core::StsBadArg);
    assert_eq!(source.message, "invalid test matrix");
    assert!(error.source().unwrap().is::<opencv::Error>());
}
