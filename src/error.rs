//! Validation failures and errors from the OpenCV image boundary.

/// A failure to validate, detect or describe HAHOG features.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The image is not two-dimensional or is smaller than 32 × 32 pixels.
    #[error("HAHOG needs a 2D image at least 32 x 32; got {dimensions}D, {width} x {height}")]
    ImageShape {
        /// Number of matrix dimensions.
        dimensions: i32,
        /// Number of image columns.
        width: i32,
        /// Number of image rows.
        height: i32,
    },
    /// The input does not contain 8-bit grayscale, BGR or BGRA pixels.
    #[error("HAHOG requires CV_8UC1, CV_8UC3 or CV_8UC4 pixels; got type {actual}")]
    ImageType {
        /// OpenCV matrix type code.
        actual: i32,
    },
    /// An internal grayscale image has an unexpected type.
    #[error("HAHOG grayscale descriptions require CV_8UC1 pixels; got type {actual}")]
    GrayscaleType {
        /// OpenCV matrix type code.
        actual: i32,
    },
    /// A nonempty mask is not an 8-bit, single-channel matrix.
    #[error("HAHOG masks require CV_8UC1 pixels; got type {actual}")]
    MaskType {
        /// OpenCV matrix type code.
        actual: i32,
    },
    /// A nonempty mask is not two-dimensional or does not match the image size.
    #[error("HAHOG mask is {dimensions}D, {width} x {height}; expected 2D, {image_width} x {image_height}")]
    MaskShape {
        /// Number of mask dimensions.
        dimensions: i32,
        /// Number of mask columns.
        width: i32,
        /// Number of mask rows.
        height: i32,
        /// Required number of columns.
        image_width: i32,
        /// Required number of rows.
        image_height: i32,
    },
    /// A provided keypoint has a nonfinite center or lies outside the image.
    #[error("HAHOG keypoint {index} has a nonfinite or out-of-image center")]
    KeypointCenter {
        /// Zero-based index in the supplied keypoints.
        index: usize,
    },
    /// A provided diameter is nonfinite, too small or larger than twice the image extent.
    #[error("HAHOG keypoint {index} has an unsupported diameter")]
    KeypointSize {
        /// Zero-based index in the supplied keypoints.
        index: usize,
    },
    /// A provided angle is neither -1 nor a finite value in [0, 360) degrees.
    #[error("HAHOG keypoint {index} angle must be finite and in [0, 360), or -1")]
    KeypointAngle {
        /// Zero-based index in the supplied keypoints.
        index: usize,
    },
    /// An OpenCV operation failed; the original status code and message are retained.
    #[error("OpenCV operation failed: {0}")]
    OpenCv(#[from] opencv::Error),
}

/// Result of a HAHOG operation.
pub type Result<T> = std::result::Result<T, Error>;
