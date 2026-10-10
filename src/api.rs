//! OpenCV-shaped API with pixel-coordinate keypoints and quantized histogram descriptors.
use crate::algorithm::{
    describe_frame, math, orientation, oriented_detections, pyramid, Frame, Plane,
};
use crate::{Error, Result};
use opencv::{core, imgproc, prelude::*};

/// Rust HAHOG detector with the calling convention of OpenCV's `Feature2D` methods.
/// Descriptors have 128 unsigned-byte bins and use L2 distance, not Hamming distance.
///
/// Images must be two-dimensional, at least 32 × 32 pixels, and have type
/// `CV_8UC1`, `CV_8UC3` (BGR) or `CV_8UC4` (BGRA). Strided ROIs are supported.
/// No resizing is performed. Empty images clear outputs; featureless images
/// produce no detections. Invalid inputs return [`Error`] before replacing outputs.
///
/// Keypoints use pixel centers, diameter `2 × scale`, clockwise angles in
/// `[0, 360)` degrees and zero-based pyramid octaves. This Rust type does not
/// implement C++ `Feature2D` inheritance.
#[derive(Debug, Default)]
pub struct Hahog;

impl Hahog {
    /// Construct the fixed, reference-compatible Hessian-and-gradient detector.
    /// Uses peak threshold `1e-5`, edge threshold `10`, and a limit of 4000
    /// detections before orientation expansion. These settings are not configurable.
    pub fn new() -> Self {
        Self
    }

    /// Detect oriented keypoints without computing descriptors.
    /// A nonempty mask selects centers and must match the image size and type `CV_8UC1`.
    /// Nonzero pixels select centers rounded to the nearest pixel with ties to even.
    /// Invalid inputs leave `keypoints` unchanged; an empty image clears them.
    pub fn detect(
        &mut self,
        image: &impl core::ToInputArray,
        keypoints: &mut core::Vector<core::KeyPoint>,
        mask: &impl core::ToInputArray,
    ) -> Result<()> {
        let image = image.input_array()?.get_mat(-1)?;
        let mask = mask.input_array()?.get_mat(-1)?;
        validate_mask(&image, &mask)?;
        let mut points = core::Vector::new();
        if !image.empty() {
            let pyramid = pyramid::Pyramid::build(&image_plane(&image)?);
            for detection in oriented_detections(&pyramid) {
                if mask_selects_center(&mask, detection.frame, image.cols(), image.rows())? {
                    points.push(keypoint_from_frame(
                        detection.frame,
                        detection.peak,
                        detection.octave,
                    )?);
                }
            }
        }
        *keypoints = points;
        Ok(())
    }

    /// Detect oriented keypoints using the whole image.
    pub fn detect_def(
        &mut self,
        image: &impl core::ToInputArray,
        keypoints: &mut core::Vector<core::KeyPoint>,
    ) -> Result<()> {
        self.detect(image, keypoints, &core::no_array())
    }

    /// Describe supplied keypoints, assigning dominant orientations to angles of -1.
    /// Orientation assignment can expand the keypoint vector or remove flat patches.
    /// Centers must be finite and inside the image. Diameters must be finite and
    /// between `2 * f32::MIN_POSITIVE` and twice the larger image dimension.
    /// Angles must be `-1` or finite values in `[0, 360)` degrees. Supplied response,
    /// octave and class ID fields are retained.
    /// Invalid inputs leave both outputs unchanged.
    /// Reconstructing frames from keypoint diameters and angles can change descriptor
    /// rounding. Use [`Self::detect_and_compute`] to preserve the native detector frames.
    pub fn compute(
        &mut self,
        image: &impl core::ToInputArray,
        keypoints: &mut core::Vector<core::KeyPoint>,
        descriptors: &mut impl core::ToOutputArray,
    ) -> Result<()> {
        self.detect_and_compute(image, &core::no_array(), keypoints, descriptors, true)
    }

    /// Detect and describe OpenCV pixels, or describe supplied pixel-coordinate frames.
    /// With `use_provided_keypoints = false`, applies the mask contract of [`Self::detect`].
    /// With `true`, ignores the mask and applies the frame contract of [`Self::compute`].
    /// Each output keypoint has one corresponding 128-byte descriptor row.
    pub fn detect_and_compute(
        &mut self,
        image: &impl core::ToInputArray,
        mask: &impl core::ToInputArray,
        keypoints: &mut core::Vector<core::KeyPoint>,
        descriptors: &mut impl core::ToOutputArray,
        use_provided_keypoints: bool,
    ) -> Result<()> {
        let image = image.input_array()?.get_mat(-1)?;
        let (points, rows) = if use_provided_keypoints {
            compute_provided(&image, keypoints)?
        } else {
            compute_detected(&image, mask)?
        };
        let matrix = if rows.is_empty() {
            core::Mat::default()
        } else {
            core::Mat::from_slice_2d(&rows)?
        };
        matrix.copy_to(descriptors)?;
        *keypoints = points;
        Ok(())
    }

    /// Detect new keypoints and compute descriptors with OpenCV's default arguments.
    pub fn detect_and_compute_def(
        &mut self,
        image: &impl core::ToInputArray,
        mask: &impl core::ToInputArray,
        keypoints: &mut core::Vector<core::KeyPoint>,
        descriptors: &mut impl core::ToOutputArray,
    ) -> Result<()> {
        self.detect_and_compute(image, mask, keypoints, descriptors, false)
    }

    /// Return the number of histogram bins per descriptor row.
    pub fn descriptor_size(&self) -> i32 {
        128
    }

    /// Return the OpenCV type of the quantized descriptor matrix.
    pub fn descriptor_type(&self) -> i32 {
        core::CV_8U
    }

    /// Return the distance metric for quantized gradient histograms.
    pub fn default_norm(&self) -> i32 {
        core::NORM_L2
    }
}

/// Aligned OpenCV keypoints and descriptor rows, staged before replacing caller outputs.
type Computed = (core::Vector<core::KeyPoint>, Vec<[u8; 128]>);

/// Detect new keypoints after validating the selection mask and image pixels.
fn compute_detected(image: &core::Mat, mask: &impl core::ToInputArray) -> Result<Computed> {
    let mask = mask.input_array()?.get_mat(-1)?;
    validate_mask(image, &mask)?;
    if image.empty() {
        return Ok((core::Vector::new(), Vec::new()));
    }
    let plane = image_plane(image)?;
    let pyramid = pyramid::Pyramid::build(&plane);
    let mut points = core::Vector::new();
    let mut rows = Vec::new();
    for detection in oriented_detections(&pyramid) {
        if !mask_selects_center(
            &mask,
            detection.frame,
            plane.width as i32,
            plane.height as i32,
        )? {
            continue;
        }
        points.push(keypoint_from_frame(
            detection.frame,
            detection.peak,
            detection.octave,
        )?);
        rows.push(describe_frame(&pyramid, detection.frame));
    }
    Ok((points, rows))
}

/// Describe supplied keypoints without reading or validating a selection mask.
fn compute_provided(
    image: &core::Mat,
    keypoints: &core::Vector<core::KeyPoint>,
) -> Result<Computed> {
    if image.empty() {
        return Ok((core::Vector::new(), Vec::new()));
    }
    let plane = image_plane(image)?;
    if keypoints.is_empty() {
        return Ok((core::Vector::new(), Vec::new()));
    }
    validate_provided(&plane, keypoints)?;
    let pyramid = pyramid::Pyramid::build(&plane);
    let mut points = core::Vector::new();
    let mut rows = Vec::new();
    for point in keypoints {
        let frame = Frame {
            x: point.pt().x,
            y: point.pt().y,
            a11: point.size() / 2.,
            a12: 0.,
            a21: 0.,
            a22: point.size() / 2.,
        };
        let angles = if point.angle() == -1. {
            orientation::assign(&pyramid, frame)
                .into_iter()
                .map(|v| v[0])
                .collect()
        } else {
            vec![(point.angle() as f64).to_radians()]
        };
        for angle in angles {
            let mut oriented = point.clone();
            if point.angle() == -1. {
                oriented.set_angle(math::positive_degrees(angle));
            }
            rows.push(describe_frame(&pyramid, orientation::rotate(frame, angle)));
            points.push(oriented);
        }
    }
    Ok((points, rows))
}

/// Reject masks that cannot select pixel centers in the supplied image.
fn validate_mask(image: &core::Mat, mask: &core::Mat) -> Result<()> {
    if mask.empty() {
        return Ok(());
    }
    if mask.typ() != core::CV_8UC1 {
        return Err(Error::MaskType { actual: mask.typ() });
    }
    if mask.dims() != 2 || mask.size()? != image.size()? {
        return Err(Error::MaskShape {
            dimensions: mask.dims(),
            width: mask.cols(),
            height: mask.rows(),
            image_width: image.cols(),
            image_height: image.rows(),
        });
    }
    Ok(())
}

/// Validate and normalize pixels, converting BGR/BGRA only when necessary.
fn image_plane(image: &core::Mat) -> Result<Plane> {
    if image.dims() != 2 || image.rows() < 32 || image.cols() < 32 {
        return Err(Error::ImageShape {
            dimensions: image.dims(),
            width: image.cols(),
            height: image.rows(),
        });
    }
    let mut gray = core::Mat::default();
    match image.typ() {
        core::CV_8UC1 => return grayscale_plane(image),
        core::CV_8UC3 => imgproc::cvt_color_def(image, &mut gray, imgproc::COLOR_BGR2GRAY)?,
        core::CV_8UC4 => imgproc::cvt_color_def(image, &mut gray, imgproc::COLOR_BGRA2GRAY)?,
        _ => {
            return Err(Error::ImageType {
                actual: image.typ(),
            });
        }
    }
    grayscale_plane(&gray)
}

/// Test the nearest mask pixel with ties-to-even rounding and clamped image coordinates.
fn mask_selects_center(mask: &core::Mat, frame: Frame, width: i32, height: i32) -> Result<bool> {
    if mask.empty() {
        return Ok(true);
    }
    let x = (frame.x.round_ties_even() as i32).clamp(0, width - 1);
    let y = (frame.y.round_ties_even() as i32).clamp(0, height - 1);
    Ok(*mask.at_2d::<u8>(y, x)? != 0)
}

/// Convert an oriented frame to OpenCV's diameter and clockwise-degree conventions.
fn keypoint_from_frame(frame: Frame, response: f32, octave: i32) -> Result<core::KeyPoint> {
    let scale = math::determinant2([frame.a11, frame.a12, frame.a21, frame.a22])
        .abs()
        .sqrt();
    let angle = math::positive_degrees(frame.a21.atan2(frame.a11) as f64);
    Ok(core::KeyPoint::new_coords(
        frame.x,
        frame.y,
        2. * scale,
        angle,
        response,
        octave,
        -1,
    )?)
}

/// Validate supplied center, diameter and orientation contracts before producing output.
fn validate_provided(plane: &Plane, provided: &core::Vector<core::KeyPoint>) -> Result<()> {
    let maximum_size = 2. * plane.width.max(plane.height) as f32;
    for (index, point) in provided.iter().enumerate() {
        let p = point.pt();
        if !p.x.is_finite()
            || !p.y.is_finite()
            || p.x < 0.
            || p.x >= plane.width as f32
            || p.y < 0.
            || p.y >= plane.height as f32
        {
            return Err(Error::KeypointCenter { index });
        }
        if !point.size().is_finite()
            || point.size() < 2. * f32::MIN_POSITIVE
            || point.size() > maximum_size
        {
            return Err(Error::KeypointSize { index });
        }
        if !point.angle().is_finite()
            || (point.angle() != -1. && !(0. ..360.).contains(&point.angle()))
        {
            return Err(Error::KeypointAngle { index });
        }
    }
    Ok(())
}

/// Copy CV_8UC1 pixels, including strided ROIs, into a dense normalized image.
fn grayscale_plane(gray: &core::Mat) -> Result<Plane> {
    if gray.typ() != core::CV_8UC1 {
        return Err(Error::GrayscaleType { actual: gray.typ() });
    }
    let values = if gray.is_continuous() {
        gray.data_typed::<u8>()?
            .iter()
            .map(|&v| v as f32 / 255.)
            .collect()
    } else {
        let mut values = Vec::with_capacity(gray.total());
        for y in 0..gray.rows() {
            values.extend(
                gray.row(y)?
                    .data_typed::<u8>()?
                    .iter()
                    .map(|&v| v as f32 / 255.),
            );
        }
        values
    };
    Ok(Plane {
        width: gray.cols() as usize,
        height: gray.rows() as usize,
        values,
    })
}

#[cfg(test)]
mod tests;
