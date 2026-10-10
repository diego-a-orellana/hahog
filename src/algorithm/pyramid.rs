//! Gaussian octaves and scalar convolution with the reference's accumulation order.
use super::{math, Plane};
/// Base smoothing scale in original-image pixels: 1.6 × 2^(1/3).
pub(crate) fn base_sigma() -> f64 {
    1.6 * 2f64.powf(1. / 3.)
}

/// Gaussian scale in original-image pixels for octave `o` and subdivision `s`.
pub(crate) fn sigma(o: usize, s: i32) -> f64 {
    base_sigma() * 2f64.powf(o as f64 + s as f64 / 3.)
}

/// Apply an isotropic blur with nonnegative sigma, clamped borders and vertical-first sums.
pub(crate) fn gaussian_blur(p: &Plane, sigma: f64) -> Plane {
    let w = p.width;
    let h = p.height;
    let mut tmp = vec![0.; w * h];
    let mut out = vec![0.; w * h];
    let kernel = math::gaussian_kernel(sigma);
    for y in 0..h {
        for x in 0..w {
            let mut a = 0f32;
            for (i, &k) in kernel.iter().enumerate() {
                let yy = (y as isize + i as isize - (kernel.len() / 2) as isize)
                    .clamp(0, h as isize - 1) as usize;
                a = p.values[yy * w + x].mul_add(k, a);
            }
            tmp[y * w + x] = a;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut a = 0f32;
            for (i, &k) in kernel.iter().enumerate() {
                let xx = (x as isize + i as isize - (kernel.len() / 2) as isize)
                    .clamp(0, w as isize - 1) as usize;
                a = tmp[y * w + xx].mul_add(k, a);
            }
            out[y * w + x] = a;
        }
    }
    Plane {
        width: w,
        height: h,
        values: out,
    }
}

/// Gaussian scale space with five levels per octave, ordered from fine to coarse.
pub(crate) struct Pyramid {
    /// Subdivisions -1 through 3 in each octave; level index is subdivision + 1.
    pub octaves: Vec<Vec<Plane>>,
}
impl Pyramid {
    /// Build octaves from a normalized image at least 32 × 32, assuming input sigma 0.5.
    pub fn build(image: &Plane) -> Self {
        let last = (((image.width.min(image.height) - 1) as f64 / 15.).ln()
            * std::f64::consts::LOG2_E)
            .floor() as usize;
        let mut octaves: Vec<Vec<Plane>> = Vec::with_capacity(last + 1);
        for o in 0..=last {
            let initial = if o == 0 {
                let delta = (sigma(0, -1).powi(2) - 0.25).sqrt();
                gaussian_blur(image, delta)
            } else {
                let p = &octaves[o - 1][3];
                let w = p.width / 2;
                let h = p.height / 2;
                Plane {
                    width: w,
                    height: h,
                    values: (0..h)
                        .flat_map(|y| (0..w).map(move |x| p.values[2 * y * p.width + 2 * x]))
                        .collect(),
                }
            };
            let mut levels = vec![initial];
            for s in 0..=3 {
                let a = sigma(o, s);
                let b = sigma(o, s - 1);
                let delta = ((a * a - b * b) as f32).sqrt() as f64 / 2f64.powi(o as i32);
                levels.push(gaussian_blur(levels.last().unwrap(), delta));
            }
            octaves.push(levels);
        }
        Self { octaves }
    }
}
