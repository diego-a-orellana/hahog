//! Smoothed 36-bin gradient peaks, including additional orientations above 80% of maximum.
use super::{descriptor, math, patch, pyramid::Pyramid, Frame, PatchWorkspace};
use std::sync::OnceLock;

/// Samples on either side of the canonical orientation patch center.
const PATCH_RADIUS: usize = 20;
/// Half-width of the orientation patch in canonical frame units.
const PATCH_EXTENT: f64 = 9.;
/// Circular gradient-histogram bins.
const HISTOGRAM_BINS: usize = 36;
/// Reference limit, applied in histogram scan order before sorting by strength.
const MAX_ORIENTATIONS: usize = 4;

/// Return the first four qualifying peaks, ordered by strength, as angles in radians.
/// Flat patches return none; the limit is applied before sorting, as in the reference.
/// The iterator owns its peaks so the workspace can immediately be reused.
pub(crate) fn assign(
    pyramid: &Pyramid,
    frame: Frame,
    workspace: &mut PatchWorkspace,
) -> impl Iterator<Item = f64> {
    // The 41×41 Gaussian window is independent of the input frame.
    static MASK: OnceLock<Vec<f32>> = OnceLock::new();
    let mask = MASK.get_or_init(|| {
        (-(PATCH_RADIUS as i32)..=PATCH_RADIUS as i32)
            .flat_map(|y| {
                (-(PATCH_RADIUS as i32)..=PATCH_RADIUS as i32).map(move |x| {
                    let dx = x as f64 * (2. * PATCH_EXTENT / (2 * PATCH_RADIUS + 1) as f64) / 3.;
                    let dy = y as f64 * (2. * PATCH_EXTENT / (2 * PATCH_RADIUS + 1) as f64) / 3.;
                    (-0.5 * math::squared_norm2_fma(dx, dy)).exp() as f32
                })
            })
            .collect()
    });
    let sigma = patch::sample(
        pyramid,
        frame,
        PATCH_RADIUS,
        PATCH_EXTENT,
        &mut workspace.patch,
    );
    let delta = (1. - sigma * sigma).max(0.).sqrt() / (PATCH_EXTENT / PATCH_RADIUS as f64);
    workspace
        .blur
        .apply(&workspace.patch, delta, &mut workspace.smoothed);
    let mut hist = [0f64; HISTOGRAM_BINS];
    let extent = std::f64::consts::TAU / HISTOGRAM_BINS as f64;
    descriptor::fill_gradients(&workspace.smoothed, &mut workspace.gradients);
    for (g, &w) in workspace.gradients.iter().zip(mask) {
        let x = g[1] as f64 / extent;
        let bin = x.floor() as usize;
        let r = x - bin as f64;
        let m = g[0] as f64 * w as f64;
        hist[bin % HISTOGRAM_BINS] = (1. - r).mul_add(m, hist[bin % HISTOGRAM_BINS]);
        hist[(bin + 1) % HISTOGRAM_BINS] = r.mul_add(m, hist[(bin + 1) % HISTOGRAM_BINS]);
    }
    for _ in 0..6 {
        let before = hist;
        for i in 0..HISTOGRAM_BINS {
            hist[i] = (before[(i + HISTOGRAM_BINS - 1) % HISTOGRAM_BINS]
                + before[i]
                + before[(i + 1) % HISTOGRAM_BINS])
                / 3.;
        }
    }
    peak_angles(&hist)
}

/// Select strict peaks above 80% of maximum, retaining reference scan-order ties.
fn peak_angles(hist: &[f64; HISTOGRAM_BINS]) -> impl Iterator<Item = f64> {
    let extent = std::f64::consts::TAU / HISTOGRAM_BINS as f64;
    let maximum = hist.iter().copied().fold(0f64, f64::max);
    let mut peaks = [[0f64; 2]; MAX_ORIENTATIONS];
    let mut count = 0;
    for i in 0..HISTOGRAM_BINS {
        let h = hist[i];
        let m = hist[(i + HISTOGRAM_BINS - 1) % HISTOGRAM_BINS];
        let p = hist[(i + 1) % HISTOGRAM_BINS];
        if h > 0.8 * maximum && h > m && h > p {
            let d = math::quadratic_peak_offset(m, h, p);
            peaks[count] = [extent * (i as f64 + d), h];
            count += 1;
            if count == peaks.len() {
                break;
            }
        }
    }
    peaks[..count].sort_by(|a, b| b[1].total_cmp(&a[1]));
    peaks.into_iter().take(count).map(|peak| peak[0])
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Later, stronger peaks must not displace the first four qualifying peaks.
    #[test]
    fn peak_limit_precedes_sorting_and_equal_strengths_keep_scan_order() {
        let mut hist = [0.; HISTOGRAM_BINS];
        for (bin, strength) in [(1, 9.), (5, 9.), (9, 9.5), (13, 9.75), (17, 10.)] {
            hist[bin] = strength;
        }
        let expected =
            [13, 9, 1, 5].map(|bin| std::f64::consts::TAU / HISTOGRAM_BINS as f64 * bin as f64);
        assert_eq!(peak_angles(&hist).collect::<Vec<_>>(), expected);
        assert_eq!(peak_angles(&[0.; HISTOGRAM_BINS]).count(), 0);
    }
}
