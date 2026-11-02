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

pub(crate) use descriptor::DESCRIPTOR_LEN;
pub(crate) use detector::Detection;

/// Dense row-major pixels; input intensities are normalized to [0, 1].
/// The default is empty scratch storage; pixel kernels require initialized dimensions.
#[derive(Debug, Default)]
pub(crate) struct Plane {
    /// Number of columns.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Exactly `width * height` pixels, without row padding.
    pub values: Vec<f32>,
}

impl Plane {
    /// Take ownership of dense pixels, checking their dimensions once at construction.
    pub(crate) fn from_values(width: usize, height: usize, values: Vec<f32>) -> Self {
        assert_eq!(width.checked_mul(height), Some(values.len()));
        Self {
            width,
            height,
            values,
        }
    }

    /// Set dimensions and zero every pixel while retaining the existing allocation.
    pub(super) fn reset(&mut self, width: usize, height: usize) {
        let length = width
            .checked_mul(height)
            .expect("plane dimensions overflow");
        self.values.clear();
        self.values.resize(length, 0.);
        self.width = width;
        self.height = height;
    }
}

/// Buffers reused across orientation and descriptor patches within one extraction.
#[derive(Default)]
pub(crate) struct PatchWorkspace {
    /// Canonical pixels sampled at the current frame.
    patch: Plane,
    /// Smoothed orientation pixels; descriptor patches need no additional blur.
    smoothed: Plane,
    /// Intermediate convolution pixels and kernel coefficients.
    blur: pyramid::GaussianBlur,
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
    workspace: &mut PatchWorkspace,
) -> impl Iterator<Item = detector::Detection> {
    orientation::assign(pyramid, detection.frame, workspace).map(move |angle| detector::Detection {
        frame: orientation::rotate(detection.frame, angle),
        ..detection
    })
}

/// Quantize the RootSIFT-style histogram of an oriented, isotropic frame.
pub(crate) fn describe_frame(
    pyramid: &pyramid::Pyramid,
    frame: Frame,
    workspace: &mut PatchWorkspace,
) -> [u8; DESCRIPTOR_LEN] {
    patch::sample(
        pyramid,
        frame,
        descriptor::PATCH_RADIUS,
        descriptor::PATCH_EXTENT,
        &mut workspace.patch,
    );
    descriptor::quantize(descriptor::describe(&workspace.patch))
}
