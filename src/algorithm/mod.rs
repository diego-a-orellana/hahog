//! Pure Rust extraction stages and their shared pixel and frame representations.
//! Arithmetic order, precision and feature ordering follow OpenSfM/VLFeat.
//! See `licenses/README.md` for source attribution and retained notices.

mod descriptor;
mod detector;
pub(crate) mod math;
pub(crate) mod orientation;
mod patch;
pub(crate) mod pyramid;
mod selection;

/// Dense row-major pixels; input intensities are normalized to [0, 1].
#[derive(Clone, Debug)]
pub(crate) struct Plane {
    /// Number of columns.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Exactly `width * height` pixels, without row padding.
    pub values: Vec<f32>,
}

/// Pixel center and row-major affine basis, before API-specific scale and angle conversion.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    /// Horizontal center in original-image pixels.
    pub x: f32,
    /// Vertical center in original-image pixels.
    pub y: f32,
    /// Horizontal component of the first scaled basis vector.
    pub a11: f32,
    /// Horizontal component of the second scaled basis vector.
    pub a12: f32,
    /// Vertical component of the first scaled basis vector.
    pub a21: f32,
    /// Vertical component of the second scaled basis vector.
    pub a22: f32,
}

/// Select scale extrema using the full-image peak budget before mask filtering.
pub(crate) fn selected_detections(pyramid: &pyramid::Pyramid) -> Vec<detector::Detection> {
    let mut detections = detector::detect(pyramid);
    selection::select_strongest(&mut detections);
    detections
}

/// Expand one retained center into its dominant orientations in peak order.
pub(crate) fn orient_detection(
    pyramid: &pyramid::Pyramid,
    detection: detector::Detection,
) -> impl Iterator<Item = detector::Detection> {
    orientation::assign(pyramid, detection.frame)
        .into_iter()
        .map(move |angle| detector::Detection {
            frame: orientation::rotate(detection.frame, angle[0]),
            ..detection.clone()
        })
}

/// Quantize the RootSIFT-style histogram of an oriented, isotropic frame.
pub(crate) fn describe_frame(pyramid: &pyramid::Pyramid, frame: Frame) -> [u8; 128] {
    let (patch, _) = patch::sample(pyramid, frame, 15, 7.5);
    descriptor::quantize(descriptor::describe(&patch))
}
