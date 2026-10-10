//! Smoothed 36-bin gradient peaks, including additional orientations above 80% of maximum.
use super::{
    descriptor, math, patch,
    pyramid::{self, Pyramid},
    Frame,
};
use std::sync::OnceLock;
/// Return up to four [radians, strength] peaks, strongest first; flat patches return none.
pub(crate) fn assign(pyramid: &Pyramid, frame: Frame) -> Vec<[f64; 2]> {
    // The 41×41 Gaussian window is independent of the input frame.
    static MASK: OnceLock<Vec<f32>> = OnceLock::new();
    let mask = MASK.get_or_init(|| {
        (-20..=20)
            .flat_map(|y| {
                (-20..=20).map(move |x| {
                    let dx = x as f64 * (18. / 41.) / 3.;
                    let dy = y as f64 * (18. / 41.) / 3.;
                    (-0.5 * math::squared_norm2_fma(dx, dy)).exp() as f32
                })
            })
            .collect()
    });
    let (patch, sigma) = patch::sample(pyramid, frame, 20, 9.);
    let delta = (1. - sigma * sigma).max(0.).sqrt() / (9. / 20.);
    let patch = pyramid::gaussian_blur(&patch, delta);
    let gradients = descriptor::gradients(&patch);
    let mut hist = [0f64; 36];
    let extent = std::f64::consts::TAU / 36.;
    for (g, &w) in gradients.iter().zip(mask) {
        let x = g[1] as f64 / extent;
        let bin = x.floor() as usize;
        let r = x - bin as f64;
        let m = g[0] as f64 * w as f64;
        hist[bin % 36] = (1. - r).mul_add(m, hist[bin % 36]);
        hist[(bin + 1) % 36] = r.mul_add(m, hist[(bin + 1) % 36]);
    }
    for _ in 0..6 {
        let before = hist;
        for i in 0..36 {
            hist[i] = (before[(i + 35) % 36] + before[i] + before[(i + 1) % 36]) / 3.;
        }
    }
    let maximum = hist.iter().copied().fold(0f64, f64::max);
    let mut peaks = Vec::new();
    for i in 0..36 {
        let h = hist[i];
        let m = hist[(i + 35) % 36];
        let p = hist[(i + 1) % 36];
        if h > 0.8 * maximum && h > m && h > p {
            let d = math::quadratic_peak_offset(m, h, p);
            peaks.push([extent * (i as f64 + d), h]);
            if peaks.len() == 4 {
                break;
            }
        }
    }
    peaks.sort_by(|a, b| b[1].total_cmp(&a[1]));
    peaks
}

/// Rotate the frame's basis clockwise in image coordinates by an angle in radians.
pub(crate) fn rotate(f: Frame, angle: f64) -> Frame {
    let [a11, a12, a21, a22] = math::rotate2([f.a11, f.a12, f.a21, f.a22], angle);
    Frame {
        a11,
        a12,
        a21,
        a22,
        ..f
    }
}
