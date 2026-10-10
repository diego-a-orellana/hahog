//! Gradient sampling and the fixed 4x4x8 normalized histogram descriptor.
use super::{math, Plane};
use std::sync::OnceLock;

/// Image-independent spatial contributions to the descriptor's trilinear histogram.
struct SpatialBins {
    /// Gaussian window weight at this canonical patch pixel.
    window: f32,
    /// Lower horizontal histogram bin, relative to the patch center.
    x: i32,
    /// Lower vertical histogram bin, relative to the patch center.
    y: i32,
    /// Contributions to the lower and upper horizontal bins.
    x_weights: [f32; 2],
    /// Contributions to the lower and upper vertical bins.
    y_weights: [f32; 2],
}

/// Return row-major [magnitude, radians] pairs using one-sided border differences.
pub(super) fn gradients(plane: &Plane) -> Vec<[f32; 2]> {
    let w = plane.width;
    let h = plane.height;
    let at = |x, y| plane.values[y * w + x];
    (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let gx = if x == 0 {
                    at(1, y) - at(0, y)
                } else if x + 1 == w {
                    at(x, y) - at(x - 1, y)
                } else {
                    ((at(x + 1, y) - at(x - 1, y)) as f64 * 0.5) as f32
                };
                let gy = if y == 0 {
                    at(x, 1) - at(x, 0)
                } else if y + 1 == h {
                    at(x, y) - at(x, y - 1)
                } else {
                    ((at(x, y + 1) - at(x, y - 1)) as f64 * 0.5) as f32
                };
                [
                    math::vlfeat_sqrt(math::squared_norm2_fma(gx, gy)),
                    math::mod_tau(
                        (math::vlfeat_atan2(gy, gx) as f64 + std::f64::consts::TAU) as f32,
                    ),
                ]
            })
        })
        .collect()
}

/// Describe a 31×31 canonical patch with scale 2 and orientation pi/2.
pub(super) fn describe(g: &[[f32; 2]]) -> [f32; 128] {
    assert_eq!(g.len(), 961);
    let mut d = [0f32; 128];
    let angle0 = std::f64::consts::FRAC_PI_2;
    // Keep factors separate: combining them changes single-precision rounding.
    static SPATIAL_BINS: OnceLock<[SpatialBins; 961]> = OnceLock::new();
    let spatial_bins = SPATIAL_BINS.get_or_init(|| {
        let st = angle0.sin();
        let ct = angle0.cos();
        let sbp = 6. + f64::EPSILON;
        std::array::from_fn(|i| {
            let x = i % 31;
            let y = i / 31;
            let dx = x as f32 - 15.;
            let dy = y as f32 - 15.;
            let nx = ((ct.mul_add(dx as f64, st * dy as f64)) / sbp) as f32;
            let ny = (((-st).mul_add(dx as f64, ct * dy as f64)) / sbp) as f32;
            let win = math::vlfeat_exp_neg((math::squared_norm2_fma(nx, ny)) as f64 / 8.) as f32;
            let bx = (nx as f64 - 0.5).floor() as i32;
            let by = (ny as f64 - 0.5).floor() as i32;
            let rx = (nx as f64 - (bx as f64 + 0.5)) as f32;
            let ry = (ny as f64 - (by as f64 + 0.5)) as f32;
            SpatialBins {
                window: win,
                x: bx,
                y: by,
                x_weights: std::array::from_fn(|ix| (1. - ix as f32 - rx).abs()),
                y_weights: std::array::from_fn(|iy| (1. - iy as f32 - ry).abs()),
            }
        })
    });
    for (&[modulus, angle], spatial) in g.iter().zip(spatial_bins) {
        let theta = math::mod_tau((angle as f64 - angle0) as f32);
        let nt = (8. * theta as f64 / std::f64::consts::TAU) as f32;
        let bt = nt.floor() as i32;
        let rt = nt - bt as f32;
        for (ix, &wx) in spatial.x_weights.iter().enumerate() {
            for (iy, &wy) in spatial.y_weights.iter().enumerate() {
                for it in 0..2 {
                    let xx = spatial.x + ix as i32;
                    let yy = spatial.y + iy as i32;
                    if (-2..2).contains(&xx) && (-2..2).contains(&yy) {
                        let weight =
                            spatial.window * modulus * wx * wy * (1. - it as f32 - rt).abs();
                        d[((yy + 2) * 32 + (xx + 2) * 8 + (bt + it) % 8) as usize] += weight;
                    }
                }
            }
        }
    }
    math::normalize(&mut d);
    for v in &mut d {
        *v = v.min(0.2);
    }
    math::normalize(&mut d);
    d
}

/// Apply the reference square-root transform, scale by 362 and round ties to even.
pub(super) fn quantize(d: [f32; 128]) -> [u8; 128] {
    d.map(|v| (362. * v.sqrt()).clamp(0., 255.).round_ties_even() as u8)
}
