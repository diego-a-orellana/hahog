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
    /// Vertical-pass pixels, overwritten by each convolution.
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
        // Keep small patch blurs on the simpler row loops.
        if w >= 64 {
            self.apply_blocked(input, sigma, output);
            return;
        }
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

    /// Accumulate wide rows in blocks to avoid storing every partial sum.
    fn apply_blocked(&mut self, input: &Plane, sigma: f64, output: &mut Plane) {
        const BLOCK: usize = 16;
        let (w, h) = (input.width, input.height);
        math::gaussian_kernel(sigma, &mut self.kernel);
        let radius = self.kernel.len() / 2;
        // Every pixel is assigned below, so reused storage need not be cleared.
        self.intermediate.resize(w * h, 0.);
        output.values.resize(w * h, 0.);
        output.width = w;
        output.height = h;

        let mut rows = Vec::with_capacity(self.kernel.len());
        for y in 0..h {
            rows.clear();
            for i in 0..self.kernel.len() {
                let yy =
                    (y as isize + i as isize - radius as isize).clamp(0, h as isize - 1) as usize;
                rows.push(&input.values[yy * w..(yy + 1) * w]);
            }
            let row = &mut self.intermediate[y * w..(y + 1) * w];
            let mut chunks = row.chunks_exact_mut(BLOCK);
            for (block, out) in chunks.by_ref().enumerate() {
                let x = block * BLOCK;
                let mut sums = [0f32; BLOCK];
                // Interleave independent pixels, retaining each pixel's FMA tap order.
                for (source, &k) in rows.iter().zip(&self.kernel) {
                    let source = &source[x..x + BLOCK];
                    for lane in 0..BLOCK {
                        sums[lane] = source[lane].mul_add(k, sums[lane]);
                    }
                }
                out.copy_from_slice(&sums);
            }
            for (offset, out) in chunks.into_remainder().iter_mut().enumerate() {
                let x = w / BLOCK * BLOCK + offset;
                let mut sum = 0f32;
                for (source, &k) in rows.iter().zip(&self.kernel) {
                    sum = source[x].mul_add(k, sum);
                }
                *out = sum;
            }
        }
        for y in 0..h {
            let source = &self.intermediate[y * w..(y + 1) * w];
            let row = &mut output.values[y * w..(y + 1) * w];
            // Keep border ranges disjoint even when the kernel exceeds the image.
            let start = radius.min(w);
            let end = w.saturating_sub(radius).max(start);
            for x in (0..start).chain(end..w) {
                let mut sum = 0f32;
                for (i, &k) in self.kernel.iter().enumerate() {
                    let xx = (x as isize + i as isize - radius as isize).clamp(0, w as isize - 1)
                        as usize;
                    sum = source[xx].mul_add(k, sum);
                }
                row[x] = sum;
            }
            let mut chunks = row[start..end].chunks_exact_mut(BLOCK);
            for (block, out) in chunks.by_ref().enumerate() {
                let x = start + block * BLOCK;
                let mut sums = [0f32; BLOCK];
                for (i, &k) in self.kernel.iter().enumerate() {
                    let source = &source[x + i - radius..x + i - radius + BLOCK];
                    for lane in 0..BLOCK {
                        sums[lane] = source[lane].mul_add(k, sums[lane]);
                    }
                }
                out.copy_from_slice(&sums);
            }
            for (offset, out) in chunks.into_remainder().iter_mut().enumerate() {
                let x = start + (end - start) / BLOCK * BLOCK + offset;
                let mut sum = 0f32;
                for (i, &k) in self.kernel.iter().enumerate() {
                    sum = source[x + i - radius].mul_add(k, sum);
                }
                *out = sum;
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

    /// Scalar vertical-first convolution with replicated borders and ordered FMA taps.
    fn scalar_blur(input: &Plane, sigma: f64) -> Plane {
        let (w, h) = (input.width, input.height);
        let mut kernel = Vec::new();
        math::gaussian_kernel(sigma, &mut kernel);
        let radius = (kernel.len() / 2) as isize;
        let mut intermediate = vec![0f32; w * h];
        let mut values = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let yy = (y as isize + i as isize - radius).clamp(0, h as isize - 1) as usize;
                    sum = input.values[yy * w + x].mul_add(k, sum);
                }
                intermediate[y * w + x] = sum;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let xx = (x as isize + i as isize - radius).clamp(0, w as isize - 1) as usize;
                    sum = intermediate[y * w + xx].mul_add(k, sum);
                }
                values[y * w + x] = sum;
            }
        }
        Plane::from_values(w, h, values)
    }

    /// Cover both paths, full blocks, tails, oversized kernels, and reused storage.
    #[test]
    fn reused_blur_matches_scalar_bits_at_borders_and_block_boundaries() {
        let mut blur = GaussianBlur::default();
        let mut output = Plane::default();
        for (case, width) in [
            1, 63, 64, 65, 77, 78, 79, 80, 81, 82, 83, 128, 129, 64, 2, 81,
        ]
        .into_iter()
        .enumerate()
        {
            for height in [1, 5] {
                let input = Plane::from_values(
                    width,
                    height,
                    (0..width * height)
                        .map(|i| ((i * 31 + case * 7) % 257) as f32 / 256.)
                        .collect(),
                );
                for sigma in [0., 1., 2.3, 11., 32.] {
                    let expected = scalar_blur(&input, sigma);
                    blur.apply(&input, sigma, &mut output);
                    assert_eq!((output.width, output.height), (width, height));
                    assert_eq!(output.values.len(), expected.values.len());
                    for (i, (actual, expected)) in
                        output.values.iter().zip(&expected.values).enumerate()
                    {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{width}x{height}, sigma {sigma}, pixel {i}"
                        );
                    }
                }
            }
        }
    }
}
