//! Gaussian octaves and separable convolution with the reference's accumulation order.
use super::{math, Plane};
/// Scale subdivisions per octave, excluding the two neighbouring detection levels.
pub(super) const SUBDIVISIONS_PER_OCTAVE: usize = 3;
/// Stored levels include one neighbour on either side of the detection levels.
pub(super) const LEVELS_PER_OCTAVE: usize = SUBDIVISIONS_PER_OCTAVE + 2;
/// Subdivision represented by the first stored level.
pub(super) const FIRST_SUBDIVISION: i32 = -1;

/// Base smoothing scale in original-image pixels: 1.6 × 2^(1/3).
pub(crate) fn base_sigma() -> f64 {
    1.6 * 2f64.powf(1. / SUBDIVISIONS_PER_OCTAVE as f64)
}

/// Gaussian scale in original-image pixels for an octave and subdivision.
pub(crate) fn sigma(octave: usize, subdivision: i32) -> f64 {
    base_sigma() * 2f64.powf(octave as f64 + subdivision as f64 / SUBDIVISIONS_PER_OCTAVE as f64)
}

/// Reusable intermediate pixels and coefficients for Gaussian convolution.
#[derive(Default)]
pub(super) struct GaussianBlur {
    /// Vertical-pass pixels, reset before each convolution.
    intermediate: Vec<f32>,
    /// Normalized weights for the current smoothing scale.
    kernel: Vec<f32>,
}

impl GaussianBlur {
    /// Blur a nonempty plane with nonnegative sigma and replicated borders.
    /// Output storage is reused; each pixel retains vertical-first, fused sums.
    pub(super) fn apply(&mut self, input: &Plane, sigma: f64, output: &mut Plane) {
        let w = input.width;
        let h = input.height;
        debug_assert!(w > 0 && h > 0);
        debug_assert_eq!(input.values.len(), w * h);
        self.intermediate.clear();
        self.intermediate.resize(w * h, 0.);
        output.reset(w, h);
        math::gaussian_kernel(sigma, &mut self.kernel);
        let tmp = &mut self.intermediate;
        let out = &mut output.values;
        let kernel = &self.kernel;
        // Iterate across rows inside each tap to expose contiguous, independent updates.
        // Retaining tap order preserves the reference rounding for every pixel.
        for y in 0..h {
            let destination = &mut tmp[y * w..(y + 1) * w];
            for (i, &k) in kernel.iter().enumerate() {
                let yy = (y as isize + i as isize - (kernel.len() / 2) as isize)
                    .clamp(0, h as isize - 1) as usize;
                let source = &input.values[yy * w..(yy + 1) * w];
                for (out, &value) in destination.iter_mut().zip(source) {
                    *out = value.mul_add(k, *out);
                }
            }
        }
        for y in 0..h {
            let source = &tmp[y * w..(y + 1) * w];
            let destination = &mut out[y * w..(y + 1) * w];
            for (i, &k) in kernel.iter().enumerate() {
                let shift = i as isize - (kernel.len() / 2) as isize;
                let start = (-shift).clamp(0, w as isize) as usize;
                let end = (w as isize - shift).clamp(0, w as isize) as usize;
                for out in &mut destination[..start] {
                    *out = source[0].mul_add(k, *out);
                }
                if start < end {
                    let values =
                        &source[(start as isize + shift) as usize..(end as isize + shift) as usize];
                    for (out, &value) in destination[start..end].iter_mut().zip(values) {
                        *out = value.mul_add(k, *out);
                    }
                }
                for out in &mut destination[end..] {
                    *out = source[w - 1].mul_add(k, *out);
                }
            }
        }
    }
}

/// Gaussian scale space with five levels per octave, ordered from fine to coarse.
pub(crate) struct Pyramid {
    /// Subdivisions -1 through 3 in each octave; level index is subdivision + 1.
    pub octaves: Vec<[Plane; LEVELS_PER_OCTAVE]>,
}
impl Pyramid {
    /// Build octaves from a normalized image at least 32 × 32, assuming input sigma 0.5.
    pub fn build(image: &Plane) -> Self {
        let last = (((image.width.min(image.height) - 1) as f64 / 15.).ln()
            * std::f64::consts::LOG2_E)
            .floor() as usize;
        let mut octaves: Vec<[Plane; LEVELS_PER_OCTAVE]> = Vec::with_capacity(last + 1);
        let mut blur = GaussianBlur::default();
        for octave in 0..=last {
            let initial = if octave == 0 {
                let delta = (sigma(0, FIRST_SUBDIVISION).powi(2) - 0.25).sqrt();
                let mut level = Plane::default();
                blur.apply(image, delta, &mut level);
                level
            } else {
                let p = &octaves[octave - 1][SUBDIVISIONS_PER_OCTAVE];
                let w = p.width / 2;
                let h = p.height / 2;
                Plane::from_values(
                    w,
                    h,
                    (0..h)
                        .flat_map(|y| (0..w).map(move |x| p.values[2 * y * p.width + 2 * x]))
                        .collect(),
                )
            };
            let mut levels = std::array::from_fn(|_| Plane::default());
            levels[0] = initial;
            for level_index in 1..LEVELS_PER_OCTAVE {
                let subdivision = level_index as i32 + FIRST_SUBDIVISION;
                let a = sigma(octave, subdivision);
                let b = sigma(octave, subdivision - 1);
                let delta = ((a * a - b * b) as f32).sqrt() as f64 / 2f64.powi(octave as i32);
                let (previous, next) = levels.split_at_mut(level_index);
                blur.apply(&previous[level_index - 1], delta, &mut next[0]);
            }
            octaves.push(levels);
        }
        Self { octaves }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reuse must not retain pixels or coefficients when dimensions and sigma change.
    #[test]
    fn reused_blur_matches_fresh_storage_across_sizes_and_scales() {
        let mut blur = GaussianBlur::default();
        let mut output = Plane::default();
        for (width, height, sigma) in [(9, 7, 2.3), (2, 1, 4.), (1, 3, 0.), (9, 7, 2.3)] {
            let input = Plane {
                width,
                height,
                values: (0..width * height)
                    .map(|i| ((i * 7 + 3) % 17) as f32 / 16.)
                    .collect(),
            };
            let mut expected = Plane::default();
            GaussianBlur::default().apply(&input, sigma, &mut expected);
            blur.apply(&input, sigma, &mut output);
            assert_eq!((output.width, output.height), (width, height));
            assert_eq!(output.values.len(), expected.values.len());
            for (actual, expected) in output.values.iter().zip(&expected.values) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}
